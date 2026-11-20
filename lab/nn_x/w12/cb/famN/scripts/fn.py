"""famN common code (family N of nn_x/w12/cb/PREDECLARED.txt: CatBoost-native feature processing at reduced scale).

CPU only, 12 threads (cbx enforces it). Nothing outside nn_x/w12/cb/famN/ is written.
  subset(n_fit, n_sel)  : the FIXED reduced-scale split: n_fit feeds drawn from the train-side FIT feeds of xm and n_sel feeds drawn from the
                          train-side SELECTION feeds of xm (cbx.trainside_split; user-grouped, no val label). Nested: a smaller n is a prefix
                          of the same permutation, so the 6K subset is inside the 30K subset.
  Data                  : one process-wide cache of the row blocks (X, label, vectors, token strings) of a set restricted to a feed mask.
  build_pool            : catboost.Pool with the 177 features (category_id native) + optional native embedding columns, text columns and
                          numeric extra columns. Training pools keep the feeds in file order (= time order, checked) and shuffle the rows
                          inside each feed (has_time=True then gives a time-aware ordered statistic with no label-related row order).
  run_arm2              : line-for-line the protocol of cbx.run_arm (fit on the fit feeds, early stopping on the train-side selection feeds by
                          NDCG on clicks, prediction at the selected tree count, cbx.Evaluator) with the pool hook that cbx.run_arm lacks.
"""
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
import cbx  # noqa: E402
import numpy as np  # noqa: E402
from cbx import log  # noqa: E402

ROOT = cbx.ROOT
HERE = ROOT / "nn_x/w12/cb/famN"; PRED = HERE / "preds"; LOGS = HERE / "logs"; CACHE = HERE / "cache"
SUB_SEED = 20261115
SHUF_SEED = 7
CLU = ROOT / "data/clu_work"


def subset(n_fit=30000, n_sel=5000):
    """(fit_mask, sel_mask) per xm feed. n_fit / n_sel None = all train-side fit / selection feeds (full scale)."""
    tr = cbx.load("xm"); fit, sel = cbx.trainside_split(tr)
    rg = np.random.default_rng(SUB_SEED)
    of = rg.permutation(np.where(fit)[0]); osel = rg.permutation(np.where(sel)[0])
    f = np.zeros(tr.n_feeds, bool); f[of if n_fit is None else of[:n_fit]] = True
    s = np.zeros(tr.n_feeds, bool); s[osel if n_sel is None else osel[:n_sel]] = True
    assert not (f & s).any()
    uf = tr.uid_feed[f]; assert len(np.intersect1d(uf[uf != 0], tr.uid_feed[s])) == 0, "user overlap between fit and selection"
    return f, s


# ----------------------------------------------------------------------------------------------------------------
# row blocks
# ----------------------------------------------------------------------------------------------------------------
class Block:
    """Rows of data set `ds` restricted to the feed mask `feeds` (None = all), in file order (feeds ascending = time order)."""

    def __init__(self, ds, feeds=None):
        self.ds = ds
        self.fmask = np.ones(ds.n_feeds, bool) if feeds is None else np.asarray(feeds, bool)
        self.idx = None if feeds is None else np.flatnonzero(np.repeat(self.fmask, ds.M))
        self.n = ds.n_rows if feeds is None else len(self.idx)
        self.M = ds.M[self.fmask]
        self.gid = np.repeat(np.arange(len(self.M), dtype=np.int32), self.M)            # contiguous group ids inside the block
        self.fpos = np.flatnonzero(self.fmask)                                          # feed positions inside ds
        self.frow = self.fpos[self.gid]                                                  # per row: feed position inside ds
        self._c = {}

    def take(self, a):
        return np.asarray(a) if self.idx is None else np.asarray(a[self.idx])

    def cached(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    @property
    def X(self):
        return self.cached("X", lambda: np.ascontiguousarray(self.take(self.ds.X), np.float32))

    def label(self, kind):
        if kind == "y":
            return self.cached("y", lambda: self.take(self.ds.y))
        if kind in ("oof", "mean11"):
            return self.cached("t_" + kind, lambda: self.take(self.ds.teacher(kind, "centred")))
        raise KeyError(kind)

    # ---- vectors (label-free) ----
    def vec(self, kind):
        """float32 (n, d): 'pair' = (u/256)*(e/256) (row sum = cosine), 'ehm' = (e - mean history vector)/256, 'ue' = [u ; e]/256,
        'cand' = e/256. Missing vectors are zero rows (as stored by cbx)."""
        def build():
            e = self.take(self.ds.emb_cand_i8).astype(np.float32) / 256.0
            if kind == "cand":
                return e
            if kind == "pair":
                u = np.asarray(self.ds.emb_feed("user"), np.float32)[self.frow] / 256.0
                return e * u
            if kind == "ehm":
                h = np.asarray(self.ds.emb_feed("hmean"), np.float32)[self.frow] / 256.0
                return e - h
            if kind == "ue":
                u = np.asarray(self.ds.emb_feed("user"), np.float32)[self.frow] / 256.0
                return np.concatenate([u, e], 1)
            raise KeyError(kind)
        return build()   # not cached (0.5 GB per million rows at 128-d): callers keep what they need

    def vec_pca(self, kind, dim=32):
        """PCA projection (label-free; fitted once on a 400K-row sample of the 30K fit subset, cached in famN/cache)."""
        return self.cached(f"pca{dim}_{kind}", lambda: pca_apply(kind, self.vec(kind), dim))

    # ---- token sets (label-free) ----
    def cand_code(self, what):
        """per row int32: 'cat' = native category id (0 = unknown), 'clu' = 256-cluster of the candidate vector (-1 = no vector)."""
        if what == "cat":
            return self.take(self.ds.cat).astype(np.int32)
        f = np.load(CLU / f"cand_cl_256_{self.ds.split}.npy", mmap_mode="r")
        rows = self.ds.rows
        if rows is None:
            return self.take(f).astype(np.int32)
        return np.asarray(f[rows if self.idx is None else rows[self.idx]], np.int32)

    def hist_last(self, what, k=20):
        """per feed of the block: (n_feeds, k) int32, the k most recent history events' category / 256-cluster, -1 = no event."""
        sp = self.ds.split
        h = np.load(cbx.FLAT / sp / "hist_category_id.npy", mmap_mode="r") if what == "cat" else np.load(CLU / f"hist_cl_256_{sp}.npy", mmap_mode="r")
        fd = cbx.feeds(sp); hoff = fd["hoff"][self.ds.fidx][self.fmask]; H = self.ds.H[self.fmask]
        out = np.full((len(H), k), -1, np.int32)
        for j in range(k):
            m = H > j
            out[m, j] = h[hoff[m] + j]          # history is stored newest first
        return out

    def text(self, what, k=20):
        """per row token-set string: candidate code crossed with each DISTINCT code among the user's k most recent history events,
        e.g. 'c3_7 c3_1' ('c3_none' when the feed has no usable history event). One string per (feed, candidate code), shared by rows."""
        def build():
            c = self.cand_code(what); hl = self.hist_last(what, k); px = "c" if what == "cat" else "k"
            key = self.gid.astype(np.int64) * 100000 + (c.astype(np.int64) + 1)
            uk, inv = np.unique(key, return_inverse=True)
            gf = (uk // 100000).astype(np.int64); cc = (uk % 100000 - 1).astype(np.int64)
            strs = np.empty(len(uk), object)
            last_g = -1; hs = ()
            for i in range(len(uk)):
                g = gf[i]
                if g != last_g:
                    row = hl[g]; hs = sorted(set(row[row >= 0].tolist())); last_g = g
                ci = cc[i]
                strs[i] = " ".join([f"{px}{ci}_{x}" for x in hs]) if hs else f"{px}{ci}_none"
            return strs[inv]
        return self.cached(f"text_{what}_{k}", build)


_BLOCKS = {}


def block(name, feeds=None, key=None):
    k = (name, key)
    if k not in _BLOCKS:
        _BLOCKS[k] = Block(cbx.load(name), feeds)
    return _BLOCKS[k]


# ----------------------------------------------------------------------------------------------------------------
# PCA (label-free)
# ----------------------------------------------------------------------------------------------------------------
def pca_fit(kind, dim=32):
    p = CACHE / f"pca{dim}_{kind}.npz"
    if not p.exists():
        f, _ = subset(30000, 5000)
        tr = cbx.load("xm"); rg = np.random.default_rng(1)
        ff = np.zeros(tr.n_feeds, bool); ff[rg.choice(np.where(f)[0], 3000, replace=False)] = True
        v = Block(tr, ff).vec(kind).astype(np.float64)
        mu = v.mean(0); C = np.cov((v - mu).T); w, V = np.linalg.eigh(C); o = np.argsort(-w)[:dim]
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"pca{dim}_{kind}.tmp.{os.getpid()}.npz"
        np.savez(tmp, mu=mu.astype(np.float32), W=V[:, o].astype(np.float32), ev=w[o], total=w.sum(), n=len(v)); os.replace(tmp, p)
    z = np.load(p)
    return z["mu"], z["W"], float(z["ev"].sum() / z["total"])


def pca_apply(kind, v, dim=32):
    mu, W, _ = pca_fit(kind, dim)
    return np.ascontiguousarray((v - mu) @ W, np.float32)


# ----------------------------------------------------------------------------------------------------------------
# pools
# ----------------------------------------------------------------------------------------------------------------
def build_pool(b, label=None, shuffle=False, emb=None, text=None, extra=None, cols=None, cat="native"):
    """catboost.Pool of block b.
    label  : None | 'y' | 'oof' | 'mean11' | per-row array aligned to the block
    shuffle: True -> rows permuted inside each feed (fixed seed); feeds keep their file (= time) order. Training pools only.
    emb    : list of (column name, vector kind, pca dim or None) -> native CatBoost embedding columns
    text   : list of 'cat' / 'clu' -> native text columns t_cat / t_clu (token sets)
    extra  : dict name -> per-row float array aligned to the block (numeric columns)"""
    import pandas as pd
    from catboost import Pool
    names = b.ds.names if cols is None else list(cols)
    perm = None
    if shuffle:
        perm = np.lexsort((np.random.default_rng(SHUF_SEED).random(b.n), b.gid))
    pm = (lambda a: a if perm is None else a[perm])
    X = b.X if cols is None else b.X[:, [b.ds.names.index(c) for c in names]]
    df = cbx.to_df(pm(X), names, cat)
    kw = dict(group_id=pm(b.gid), thread_count=cbx.THREADS)
    if label is not None:
        kw["label"] = pm(b.label(label) if isinstance(label, str) else np.asarray(label))
    if cat == "native" and cbx.CAT_FEATURE in names:
        kw["cat_features"] = [cbx.CAT_FEATURE]
    for k, v in (extra or {}).items():
        v = np.asarray(v); assert len(v) == b.n, f"extra {k}: {len(v)} rows, block has {b.n}"
        df[k] = pm(v).astype(np.float32)
    if text:
        kw["text_features"] = [f"t_{w}" for w in text]
        for w in text:
            df[f"t_{w}"] = pd.Series(pm(b.text(w)), dtype=object, index=df.index)
    if emb:
        kw["embedding_features"] = [n for n, _, _ in emb]
        kw["embedding_features_data"] = {n: pm(b.vec(kind) if d is None else b.vec_pca(kind, d)) for n, kind, d in emb}
    return Pool(df, **kw)


def file_mb(p):
    return os.path.getsize(p) / 1048576 if os.path.exists(p) else float("nan")


def run_arm2(tag, params, ptr, pes, pred_pools, out_dir=PRED, ev=None, budget_min=None, verbose=50, extra=None, keep_model_mb=300, test_pool=None):
    """Protocol of cbx.run_arm with ready pools. params: dict for the generic catboost.CatBoost constructor (needed for
    embedding_calcers / text parameters). ptr = fit pool, pes = train-side selection pool (click labels), pred_pools = dict(val=Pool, hold=Pool)."""
    from catboost import CatBoost
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    assert params.get("thread_count") == cbx.THREADS and params.get("task_type", "CPU") == "CPU"
    m = CatBoost(params)
    cb = cbx.Budget(budget_min, out_dir / f"{tag}.curve.json", str(out_dir / f"{tag}.STOP"))
    tf = time.time()
    m.fit(ptr, eval_set=pes, verbose=verbose, callbacks=[cb])
    fit_s = time.time() - tf
    cb.dump()
    best = int(m.get_best_iteration()) if m.get_best_iteration() is not None else m.tree_count_ - 1
    k = min(best + 1, m.tree_count_)
    log(f"[{tag}] fit {fit_s:.0f}s, {len(cb.rows)} iterations ({fit_s / max(len(cb.rows), 1):.2f} s/iter), best iteration {best}, budget stop {cb.stopped}")
    mp = out_dir / f"{tag}.cbm"
    m.save_model(str(mp)); size = file_mb(mp)
    if size > keep_model_mb:
        os.remove(mp)
    tp = time.time()
    pv = np.asarray(m.predict(pred_pools["val"], ntree_end=k, thread_count=cbx.THREADS), np.float32)
    ph = np.asarray(m.predict(pred_pools["hold"], ntree_end=k, thread_count=cbx.THREADS), np.float32)
    pt = None if test_pool is None else np.asarray(m.predict(test_pool, ntree_end=k, thread_count=cbx.THREADS), np.float32)
    pred_s = time.time() - tp
    try:
        imp = sorted(zip(m.get_feature_importance(type="PredictionValuesChange", thread_count=cbx.THREADS), m.feature_names_), reverse=True)
        top = [(n, round(float(v), 2)) for v, n in imp[:12]]
        nonbase = [(n, round(float(v), 2)) for v, n in imp if n not in cbx.feature_names()]
    except Exception as e:  # noqa: BLE001
        top, nonbase = [("ERR " + str(e)[:100], 0.0)], []
    info = dict(n_fit_rows=int(ptr.num_row()), n_sel_rows=int(pes.num_row()), best_iter=best, trees_built=len(cb.rows), budget_stop=bool(cb.stopped),
                fit_seconds=fit_s, s_per_iter=fit_s / max(len(cb.rows), 1), predict_seconds=pred_s, model_mb=size, peak_rss_gb=cbx.rss_gb()[1],
                params={k_: str(v) for k_, v in params.items()}, top=top, new_feature_importance=nonbase, selection="trainside")
    info.update(extra or {})
    ev = ev or cbx.Evaluator()
    r = ev.evaluate(tag, pv, ph, pt, out_dir=out_dir, extra=info)
    return r, m


def nats_line(tag, pv, ph):
    """the shared SCREEN (nn_x/w12/cb/nats.py), in process."""
    sys.path.insert(0, str(ROOT / "nn_x/w12/cb"))
    import nats
    return nats.screen(pv, ph, tag=tag)
