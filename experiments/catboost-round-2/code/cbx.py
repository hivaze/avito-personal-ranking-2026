# Loader and evaluator used by every arm of the round: the training sets (134,118 random-order
# train feeds without the 30,000 holdout feeds), the 177 tabular features, CatBoost pools, a
# train-side split for choosing the number of trees without any val label, guards that refuse val
# and holdout labels during fitting, and the evaluator against the pool of 16 wide distilled
# students (train-only branch, code name W16).
# How it was run: imported by the arm scripts (CPU only, 12 threads); its quick-use block is in
# the docstring.
# What it produced: caches under lab/nn_x/w12/cb/infra/cache/ and, through run_arm, the prediction
# and evaluation files of each arm.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/cb/infra/cbx.py
"""cbx: ONE loader + ONE evaluator for the round-2 CatBoost / tree-member arms (track nn_x/w12/cb, KEY = infra).

CPU only. Read-only on every project file; caches live under nn_x/w12/cb/infra/cache/ (built on first use).
Rules implemented here (nn_x/w12/cb/PREDECLARED.txt): member rule, law value, paired per-feed SEs.

QUICK USE (see README.md next to this file for the full block)
    import sys; sys.path.insert(0, "<repo>/nn_x/w12/cb/infra"); import cbx
    tr = cbx.load("xm")                       # 134,118 exploration feeds (p_random > 0.5, M >= 50, minus the 30K holdout)
    fit, sel = cbx.trainside_split(tr)        # user-grouped feed masks: ~90% fit / ~10% train-side selection (no val label)
    pool = cbx.make_pool(tr, feeds=fit, label=tr.y)        # catboost.Pool, 177 features, category_id native categorical
    ...
    ev = cbx.Evaluator()                      # val + 30K holdout labels, W16 train-only pool
    res = ev.evaluate("my_arm", pred_val, pred_hold, pred_test, out_dir="<repo>/nn_x/w12/cb/<KEY>/preds")

SETS (cbx.load(name)); row order inside a set = flat file order of its split, feeds ascending
    xm        train  134,118 feeds / 19,741,607 rows   exploration, M >= 50  (= cb26 / W25 / long_bets training set)
    xs        train  350,905 feeds /  5,361,945 rows   exploration-like natural short feeds, M < 50 (raw p_random > 0.5)
    prod<N>k  train  N*1000 feeds                       uniform sample (fixed seed) of production feeds = raw p_random <= 0.5 (not NaN)
    hold      train   30,000 feeds /  4,630,966 rows   the 30K holdout (EVALUATION ONLY; row order = nn_x/w5/preds/holdout)
    val       val     14,423 feeds /  1,389,971 rows   row order = preds/*_val.npy
    test      test    28,595 feeds /  2,904,806 rows   row order = preds/*_test.npy
  cbx.concat(a, b, ...) joins train sets for data-scale arms (xm + xs = the W25 "485,023 feeds" set).

LEAK GUARDS
  - ds.y raises for 'val' and 'hold' unless cbx.ALLOW_EVAL_LABELS is set True by the caller (only the Evaluator and the
    record-reproduction mode "valA" do that). 'test' has no labels.
  - no training set contains a holdout feed (asserted at build time and in smoke.py).
  - teacher targets: t11 is IN-SAMPLE on train feeds (fine as a TARGET, not as an INPUT); toof is out-of-fold by user fold.
"""
import os
import sys

if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
    raise RuntimeError("cbx is CPU only: export CUDA_VISIBLE_DEVICES=\"\" before starting python (hard rule 1)")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_NT = max(1, min(12, int(os.environ.get("CBX_THREADS", "12") or 12)))  # hard cap 12 (rule 1); CBX_THREADS=4 for light checks
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    if int(os.environ.get(_k, str(_NT)) or _NT) > _NT:
        os.environ[_k] = str(_NT)
    os.environ.setdefault(_k, str(_NT))
sys.dont_write_bytecode = True  # never write .pyc next to files we only read
if os.environ.get("CBX_CACHE_EXTRA") and not os.environ.get("NUMBA_CACHE_DIR"):
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.environ["CBX_CACHE_EXTRA"], "numba")  # compiled kernels go to the agent's own dir

import json
import time
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

nb.set_num_threads(min(nb.get_num_threads(), _NT))  # also caps numba when it was imported before cbx

ROOT = Path("<repo>")
HERE = ROOT / "nn_x/w12/cb/infra"
CACHE = HERE / "cache"
# Agents with another KEY: export CBX_CACHE_EXTRA=<repo>/nn_x/w12/cb/<KEY>/cache so that any cache that is not
# pre-built in infra/cache (e.g. a new prod<N>k set) is written into YOUR directory. Reads look in infra/cache first.
CACHE_EXTRA = Path(os.environ["CBX_CACHE_EXTRA"]) if os.environ.get("CBX_CACHE_EXTRA") else None
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
FNEW = ROOT / "data/feats_new"
EMB_TABLE = {s: ROOT / f"data/competition/{s}/item_embeddings" for s in ("train", "val", "test")}
KD = ROOT / "nn_x/w12/kd/out"
W5P = ROOT / "nn_x/w5/preds"
THREADS = _NT
EMB_DIM = 128
ALLOW_EVAL_LABELS = False
W16 = [f"w5_e10_kd05w192_s{s}" for s in range(951, 959)] + [f"w5_e10_kd05t25w192_s{s}" for s in range(951, 959)]
PROD_SEED = 20261115
# frontier law on record (PREDECLARED.txt): gain of the equal mix over the better pool = A (1 - rho) - B gap
LAW_RECORD = (0.0186, 0.378)
LAW_REFIT = (0.0103, 0.304)

_T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# feature list (frozen copy of the cb26 member set; smoke.py re-derives it from nn_x/cb26/cb_member.member_features)
# ----------------------------------------------------------------------------------------------------------------
def feature_names():
    """The 177 tabular features of the CatBoost of record, in the record's column order."""
    p = CACHE / "features_177.json"
    if not p.exists():
        names = json.loads((ROOT / "nn_x/cb26/data/e3_xm_yeti.json").read_text())["features"]
        assert len(names) == 177 and len(set(names)) == 177
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(names))
    return json.loads(p.read_text())


CAT_FEATURE = "category_id"


def feat_path(split, name):
    p = FEATS / split / f"{name}.npy"
    return p if p.exists() else FNEW / split / f"{name}.npy"


# ----------------------------------------------------------------------------------------------------------------
# feed tables and set definitions
# ----------------------------------------------------------------------------------------------------------------
_FEEDS = {}


def feeds(split):
    """dict of per-feed numpy arrays of a split: M, H, off (cand_off), hoff (hist_off), uid, ts, feed_id."""
    if split not in _FEEDS:
        f = pl.read_parquet(FLAT / split / "feeds.parquet")
        d = dict(M=f["M"].to_numpy().astype(np.int64), H=f["H"].to_numpy().astype(np.int64),
                 off=f["cand_off"].to_numpy().astype(np.int64), hoff=f["hist_off"].to_numpy().astype(np.int64),
                 uid=f["user_id"].to_numpy().astype(np.int64), ts=f["timestamp"].to_numpy().astype(np.int64),
                 feed_id=f["feed_id"].to_numpy().astype(np.int64))
        assert (d["off"] == np.r_[0, np.cumsum(d["M"])[:-1]]).all() and (d["hoff"] == np.r_[0, np.cumsum(d["H"])[:-1]]).all()
        _FEEDS[split] = d
    return _FEEDS[split]


_TRAIN_SETS = {}


def train_sets():
    """Train feed-index sets. p_raw = raw p_random (NaN for 400,000 feeds); p = the nn/train.py value (NaN -> 0, M < 50 -> 0).
    hold = 30,000 feeds drawn (rng 12345) from p > 0.9 & M >= 50  (nn/train.py; nn_x/cb26/cbdata.hold_index)
    xm   = p > 0.5 & M >= 50, minus hold                           (134,118)
    xs   = p_raw > 0.5 & M < 50                                    (350,905 natural short exploration-like feeds)
    prod = p_raw <= 0.5 and not NaN                                (2,236,185 feeds; never overlaps hold / xm / xs)
    The 400,000 feeds with NaN p_random (used to fit the proxy, reports/eda_labels_position.md:211) are in no training set here except
    through nn/train.py's convention (NaN -> 0) that defines hold / xm. p_random for M < 50 is documented as UNRELIABLE (same file):
    xs is "exploration-LIKE", not a clean exploration set."""
    if not _TRAIN_SETS:
        f = feeds("train")
        M = f["M"]
        pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
        assert (pr["fidx"] == np.arange(len(M))).all()
        p_raw = pr["p_random"].astype(np.float32)
        p = np.nan_to_num(p_raw, nan=0.0)
        p[M < 50] = 0.0
        cand = np.where((p > 0.9) & (M >= 50))[0]
        hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
        xm = np.setdiff1d(np.where((p > 0.5) & (M >= 50))[0], hold)
        praw0 = np.nan_to_num(p_raw, nan=0.0)
        xs = np.where((praw0 > 0.5) & (M < 50))[0]
        prod = np.where(np.isfinite(p_raw) & ~(p_raw > 0.5))[0]  # NaN = the 400,000 feeds used to fit the p_random proxy: regime unknown
        assert len(xm) == 134118 and len(hold) == 30000 and not np.isin(xm, hold).any()
        assert not np.isin(xs, hold).any() and not np.isin(prod, hold).any() and not np.isin(prod, xm).any() and not np.isin(prod, xs).any()
        _TRAIN_SETS.update(p_raw=p_raw, p=p, hold=hold, xm=xm, xs=xs, prod=prod)
    return _TRAIN_SETS


def set_feeds(name):
    """(split, sorted feed indices in that split) for a set name."""
    if name in ("val", "test"):
        return name, np.arange(len(feeds(name)["M"]))
    ts = train_sets()
    if name in ("xm", "xs", "hold"):
        return "train", ts[name]
    if name.startswith("prod") and name.endswith("k"):
        n = int(name[4:-1]) * 1000
        assert 0 < n <= len(ts["prod"])
        return "train", np.sort(np.random.default_rng(PROD_SEED).choice(ts["prod"], n, replace=False))
    raise KeyError(name)


def xfold_of(uid, fidx):
    """The 5 user folds of the cross-fitted teacher (nn_x/w12/kd/assemble_oof.py): user 0 feeds are spread by feed index."""
    h = (uid * 2654435761) % 4294967296
    fold = np.where(uid == 0, fidx % 5, h % 5)
    sub = np.where(uid == 0, (fidx // 5) % 2, (h // 5) % 2)
    return fold.astype(np.int8), sub.astype(np.int8)


# ----------------------------------------------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------------------------------------------
@nb.njit(parallel=True, cache=True)
def _ndcg(s, y, off, M):
    # verbatim copy of nn_x/w12/recheck/pool_composition/pc_common._ndcg (pessimistic ties, feeds without positives -> 1)
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m]; yv = y[o:o + m]
        npos = 0
        for k in range(m):
            if yv[k] > 0: npos += 1
        if npos == 0: continue
        # pessimistic ties: among equal scores negatives first -> sort key (-score, target)
        order = np.argsort(-sv.astype(np.float64) + 0.0, kind='mergesort')
        dcg = 0.0
        # tie handling: walk runs of equal score, put negatives first inside a run
        i = 0
        while i < m:
            j = i
            while j + 1 < m and sv[order[j + 1]] == sv[order[i]]: j += 1
            if j == i:
                if yv[order[i]] > 0: dcg += 1.0 / np.log2(i + 2.0)
            else:
                cp = 0
                for k in range(i, j + 1):
                    if yv[order[k]] > 0: cp += 1
                for k in range(j - cp + 1, j + 1): dcg += 1.0 / np.log2(k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


def ndcg_fast(s, y, off, M):
    """Per-feed NDCG (organisers' definition). s float32 scores, y float32 labels, off / M int64 per feed."""
    return _ndcg(np.ascontiguousarray(s, np.float32), np.ascontiguousarray(y, np.float32),
                 np.ascontiguousarray(off, np.int64), np.ascontiguousarray(M, np.int64))


@nb.njit(parallel=True, cache=True)
def _feed_z(off, M, x):
    # per-feed z = (x - mean) / (std + 1e-9), identical to src/stack_val.feed_rank_z[:, 1]
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m):
            out[o + k] = (v[k] - mu) / sd
    return out


def feed_z(off, M, x):
    return _feed_z(np.ascontiguousarray(off, np.int64), np.ascontiguousarray(M, np.int64), np.ascontiguousarray(x, np.float32))


@nb.njit(parallel=True, cache=True)
def _feed_corr(off, M, a, b):
    # within-feed Pearson correlation per feed (0 when a feed has zero variance in a or b)
    out = np.zeros(len(M), np.float64)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        x = a[o:o + m].astype(np.float64); y = b[o:o + m].astype(np.float64)
        x = x - x.mean(); y = y - y.mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        if d > 0: out[f] = (x * y).sum() / d
    return out


@nb.njit(parallel=True, cache=True)
def _feed_softmax(off, M, x, T):
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        mx = -1e30
        for k in range(m):
            if x[o + k] > mx: mx = x[o + k]
        s = 0.0
        for k in range(m):
            s += np.exp((x[o + k] - mx) / T)
        for k in range(m):
            out[o + k] = np.exp((x[o + k] - mx) / T) / s
    return out


@nb.njit(parallel=True, cache=True)
def _feed_center(off, M, x):
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        mu = 0.0
        for k in range(m): mu += x[o + k]
        mu /= m
        for k in range(m): out[o + k] = x[o + k] - mu
    return out


@nb.njit(parallel=True, cache=True)
def _hist_means(table, hidx, hoff, H, out_all, out5, out20, n_all, n5, n20):
    # history is stored newest first (hpos 0 = most recent event). Events without an embedding (idx < 0) are skipped:
    # "last k" = the k most recent events that HAVE an embedding.
    for i in nb.prange(len(H)):
        o, h = hoff[i], H[i]
        c = 0
        s = np.zeros(128, np.float64); s5 = np.zeros(128, np.float64); s20 = np.zeros(128, np.float64)
        c5 = 0; c20 = 0
        for k in range(h):
            e = hidx[o + k]
            if e < 0: continue
            row = table[e]
            c += 1
            if c <= 5:
                c5 = c
                for d in range(128): s5[d] += row[d]
            if c <= 20:
                c20 = c
                for d in range(128): s20[d] += row[d]
            for d in range(128): s[d] += row[d]
        n_all[i] = c; n5[i] = c5; n20[i] = c20
        for d in range(128):
            out_all[i, d] = s[d] / c if c > 0 else 0.0
            out5[i, d] = s5[d] / c5 if c5 > 0 else 0.0
            out20[i, d] = s20[d] / c20 if c20 > 0 else 0.0


@nb.njit(parallel=True, cache=True)
def _gather_rows_i8(table, idx, out):
    for i in nb.prange(len(idx)):
        e = idx[i]
        if e >= 0:
            for d in range(128): out[i, d] = table[e, d]
        else:
            for d in range(128): out[i, d] = 0


def emb_table(split):
    mt = json.loads((EMB_TABLE[split] / "data.meta").read_text())
    assert mt["dtype"] == "int8" and mt["shape"][1] == EMB_DIM
    return np.memmap(EMB_TABLE[split] / "data.memmap", dtype=np.int8, mode="r", shape=tuple(mt["shape"]))


# ----------------------------------------------------------------------------------------------------------------
# dataset handle
# ----------------------------------------------------------------------------------------------------------------
def _rows_of(off, M):
    """flat row indices of feeds (off, M), feeds in the given order."""
    start = np.repeat(off - np.r_[0, np.cumsum(M)[:-1]], M)
    return start + np.arange(int(M.sum()), dtype=np.int64)


class DS:
    """One data set. Per-row arrays are in flat file order of the split restricted to the set's feeds (ascending).
    Per-feed: fidx (index into the split's feeds.parquet), M, off (offset inside THIS set), uid_feed, ts_feed, H, p_random.
    Per-row : X (float32 memmap, n x 177), gid (group id = position of the feed inside the set, int32), item_id (int32),
              cat (int8 native category id 0..53; 0 = unknown), y (labels), t11 / toof (teacher logits, train split only).
    """

    def __init__(self, name):
        self.name = name
        self.split, self.fidx = set_feeds(name)
        f = feeds(self.split)
        self.M = f["M"][self.fidx]
        self.off = np.r_[0, np.cumsum(self.M)[:-1]].astype(np.int64)
        self.n_feeds = len(self.fidx)
        self.n_rows = int(self.M.sum())
        self.identity = self.split != "train"
        self.uid_feed = f["uid"][self.fidx]
        self.ts_feed = f["ts"][self.fidx]
        self.H = f["H"][self.fidx]
        self.feed_id = f["feed_id"][self.fidx]
        self.dir = CACHE / name
        self._mem = {}
        if self.split == "train":
            ts = train_sets()
            self.p_random = ts["p_raw"][self.fidx]
            if name != "hold":
                assert not np.isin(self.fidx, ts["hold"]).any(), "holdout feed inside a training set"
            self.xfold, self.xsub = xfold_of(self.uid_feed, self.fidx)
        self.names = feature_names()

    # ---- helpers ----
    def __repr__(self):
        return f"DS({self.name}: split={self.split} feeds={self.n_feeds} rows={self.n_rows})"

    @property
    def rows(self):
        """flat row indices into data/flat/<split>/cand_*.npy (None = identity for val / test)."""
        if self.identity:
            return None
        if "rows" not in self._mem:
            self._mem["rows"] = _rows_of(feeds(self.split)["off"][self.fidx], self.M)
        return self._mem["rows"]

    @property
    def gid(self):
        if "gid" not in self._mem:
            self._mem["gid"] = np.repeat(np.arange(self.n_feeds, dtype=np.int32), self.M)
        return self._mem["gid"]

    def expand(self, per_feed):
        """per-feed array -> per-row array."""
        return np.asarray(per_feed)[self.gid]

    @property
    def uid(self):
        return self.expand(self.uid_feed)

    @property
    def M_row(self):
        return self.expand(self.M)

    def _cached(self, key, build):
        p = self.dir / f"{key}.npy"
        if not p.exists() and CACHE_EXTRA is not None:
            p = CACHE_EXTRA / self.name / f"{key}.npy"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            arr = build()
            tmp = p.parent / f"{key}.tmp.{os.getpid()}.npy"
            np.save(tmp, arr)
            os.replace(tmp, p)  # atomic: concurrent builders cannot leave a half-written file
            del arr
        return np.load(p, mmap_mode="r")

    def _flat(self, col):
        x = np.load(FLAT / self.split / f"cand_{col}.npy", mmap_mode="r")
        return np.asarray(x) if self.identity else np.asarray(x[self.rows])

    # ---- per-row columns ----
    @property
    def X(self):
        """float32 memmap (n_rows, 177), column order = feature_names(). category_id is stored as its float code."""
        def build():
            from concurrent.futures import ThreadPoolExecutor
            out = np.empty((self.n_rows, len(self.names)), np.float32)
            rows = self.rows

            def one(j):
                x = np.load(feat_path(self.split, self.names[j]), mmap_mode="r")
                assert x.shape[0] == int(feeds(self.split)["M"].sum())
                out[:, j] = x if rows is None else x[rows]
            with ThreadPoolExecutor(THREADS) as ex:
                list(ex.map(one, range(len(self.names))))
            return out
        return self._cached("X177", build)

    @property
    def item_id(self):
        return self._cached("item_id", lambda: self._flat("item_id").astype(np.int32))

    @property
    def cat(self):
        return self._cached("cat", lambda: self._flat("category_id").astype(np.int8))

    @property
    def emb_idx(self):
        return self._cached("emb_idx", lambda: self._flat("emb_idx").astype(np.int32))

    @property
    def pos(self):
        """train only: production position of the candidate (int16). NEVER a model input for val / test (absent there)."""
        assert self.split == "train"
        return self._cached("pos", lambda: self._flat("pos").astype(np.int16))

    @property
    def y(self):
        """labels as float32 (0 / 1). Guarded for the evaluation sets."""
        if self.split == "test":
            raise PermissionError("test has no labels")
        if self.name in ("val", "hold") and not ALLOW_EVAL_LABELS:
            raise PermissionError(f"labels of '{self.name}' are evaluation labels: use cbx.Evaluator (or set cbx.ALLOW_EVAL_LABELS = True "
                                  "only for the record-reproduction mode)")
        return self._cached("y", lambda: self._flat("target").astype(np.float32))

    # ---- teacher targets (train split only) ----
    def teacher(self, which="mean11", form="logit", T=1.0):
        """Teacher targets aligned to this set's rows (float32).
        which: 'mean11' = nn_x/w12/kd/out/teacher_mean11.npy  (mean logit of the 11 hard-label train-only members r6_cd_m02_r101-107 +
                          w5_e10_base_s951-954, ALREADY centred per feed; within-feed sd about 1.14; IN-SAMPLE on training feeds, honest on 'hold')
               'oof'    = nn_x/w12/kd/out/teacher_oof.npy     (mean RAW logit of the 3 w5_e10_xf<k>_s951-953 models that excluded the feed's
                          user fold k = cbx.xfold_of; NOT centred: per-feed offset about +114, within-feed sd about 1.12)
        form : 'logit' (as stored), 'centred' (per-feed mean removed), 'soft' (within-feed softmax(logit / T): the NN students' soft label)."""
        assert self.split == "train", "teacher files cover train rows only"
        f = {"mean11": "teacher_mean11.npy", "oof": "teacher_oof.npy"}[which]
        t = self._cached(f"teacher_{which}", lambda: np.asarray(np.load(KD / f, mmap_mode="r")[self.rows], np.float32))
        if form == "logit":
            return t
        if form == "centred":
            return _feed_center(self.off, self.M, np.ascontiguousarray(t))
        if form == "soft":
            return _feed_softmax(self.off, self.M, np.ascontiguousarray(t), float(T))
        raise KeyError(form)

    # ---- raw embeddings ----
    @property
    def emb_cand_i8(self):
        """int8 memmap (n_rows, 128): the candidate item embedding (row of the split's item table); all-zero row when the candidate
        has no embedding (emb_idx = -1 <=> feature emb_missing = 1)."""
        def build():
            out = np.empty((self.n_rows, EMB_DIM), np.int8)
            _gather_rows_i8(emb_table(self.split), np.ascontiguousarray(self.emb_idx), out)
            return out
        return self._cached("emb_cand_i8", build)

    def _hist_dir(self):
        if (self.dir / "emb_hmean.npy").exists() or CACHE_EXTRA is None:
            return self.dir
        return CACHE_EXTRA / self.name

    def _hist(self):
        hd = self._hist_dir()
        if not (hd / "emb_hmean.npy").exists():
            hd.mkdir(parents=True, exist_ok=True)
            f = feeds(self.split)
            hidx = np.load(FLAT / self.split / "hist_emb_idx.npy", mmap_mode="r")
            hoff = f["hoff"][self.fidx]
            n = self.n_feeds
            oa, o5, o20 = (np.zeros((n, EMB_DIM), np.float32) for _ in range(3))
            na, n5, n20 = (np.zeros(n, np.int16) for _ in range(3))
            _hist_means(emb_table(self.split), hidx, hoff, self.H, oa, o5, o20, na, n5, n20)
            for k, v in (("emb_h5", o5), ("emb_h20", o20), ("emb_hn_all", na), ("emb_hn_5", n5), ("emb_hn_20", n20), ("emb_hmean", oa)):
                tmp = hd / f"{k}.tmp.{os.getpid()}.npy"
                np.save(tmp, v); os.replace(tmp, hd / f"{k}.npy")

    def emb_feed(self, kind):
        """Per-FEED embedding (n_feeds, 128) float32.
        'user'  : the feed's user embedding (int8 in the file, cast to float32; an all-zero vector = no user embedding)
        'hmean' : mean item embedding over ALL history events that have an embedding
        'h5'    : mean over the 5 most recent history events that have an embedding
        'h20'   : mean over the 20 most recent history events that have an embedding
        Feeds with no usable history event get an all-zero vector; emb_feed_n(kind) gives the number of events averaged."""
        if kind == "user":
            return self._cached("emb_user", lambda: np.asarray(np.load(FLAT / self.split / "user_emb.npy", mmap_mode="r")[self.fidx], np.float32))
        assert kind in ("hmean", "h5", "h20")
        self._hist()
        return np.load(self._hist_dir() / f"emb_{kind}.npy", mmap_mode="r")

    def emb_feed_n(self, kind):
        self._hist()
        return np.load(self._hist_dir() / ({"hmean": "emb_hn_all", "h5": "emb_hn_5", "h20": "emb_hn_20"}[kind] + ".npy"))

    def emb_rows(self, kind, dtype=np.float16, feeds_mask=None):
        """Per-ROW embedding matrix (n_rows, 128) of `dtype`, in RAM (not cached): kind in cand / user / hmean / h5 / h20.
        Raw scale (int8 units: item norm about 256, user norm about 250). float16 is exact for 'cand' and 'user' (integers) and keeps
        3-4 significant digits of the history means. Memory per kind for xm: 5.05 GB float16, 10.1 GB float32.
        feeds_mask: optional boolean per-feed mask (rows of the selected feeds only)."""
        rm = None if feeds_mask is None else np.repeat(np.asarray(feeds_mask, bool), self.M)
        if kind == "cand":
            e = self.emb_cand_i8
            return np.asarray(e if rm is None else e[rm], dtype)
        e = np.asarray(self.emb_feed(kind), dtype)
        g = self.gid if rm is None else self.gid[rm]
        return e[g]


_DS = {}


def load(name):
    """Data-set handle (cached per process). Names: xm, xs, prod<N>k, hold, val, test."""
    if name not in _DS:
        _DS[name] = DS(name)
    return _DS[name]


class Concat:
    """Row-wise union of train sets (e.g. cbx.concat(cbx.load('xm'), cbx.load('xs'))). Arrays are materialised in RAM on access."""

    def __init__(self, parts):
        assert all(p.split == "train" and p.name != "hold" for p in parts)
        self.parts = parts
        self.name = "+".join(p.name for p in parts)
        self.split = "train"
        self.names = parts[0].names
        self.M = np.concatenate([p.M for p in parts])
        self.off = np.r_[0, np.cumsum(self.M)[:-1]].astype(np.int64)
        self.fidx = np.concatenate([p.fidx for p in parts])
        assert len(np.unique(self.fidx)) == len(self.fidx), "overlapping sets"
        assert not np.isin(self.fidx, train_sets()["hold"]).any()
        self.n_feeds = len(self.M); self.n_rows = int(self.M.sum())
        self.uid_feed = np.concatenate([p.uid_feed for p in parts]); self.ts_feed = np.concatenate([p.ts_feed for p in parts])
        self.H = np.concatenate([p.H for p in parts]); self.p_random = np.concatenate([p.p_random for p in parts])
        self.xfold = np.concatenate([p.xfold for p in parts]); self.xsub = np.concatenate([p.xsub for p in parts])
        self.gid = np.repeat(np.arange(self.n_feeds, dtype=np.int32), self.M)

    def __repr__(self):
        return f"Concat({self.name}: feeds={self.n_feeds} rows={self.n_rows})"

    def expand(self, per_feed):
        return np.asarray(per_feed)[self.gid]

    def _cat(self, attr):
        return np.concatenate([np.asarray(getattr(p, attr)) for p in self.parts])

    X = property(lambda s: s._cat("X")); y = property(lambda s: s._cat("y")); item_id = property(lambda s: s._cat("item_id"))
    cat = property(lambda s: s._cat("cat")); emb_cand_i8 = property(lambda s: s._cat("emb_cand_i8")); uid = property(lambda s: s.expand(s.uid_feed))

    def teacher(self, which="mean11", form="logit", T=1.0):
        return np.concatenate([np.asarray(p.teacher(which, form, T)) for p in self.parts])

    def emb_feed(self, kind):
        return np.concatenate([np.asarray(p.emb_feed(kind)) for p in self.parts])

    def emb_rows(self, kind, dtype=np.float16, feeds_mask=None):
        if feeds_mask is None:
            return np.concatenate([p.emb_rows(kind, dtype) for p in self.parts])
        fm = np.asarray(feeds_mask, bool); cut = np.cumsum([0] + [p.n_feeds for p in self.parts])
        return np.concatenate([p.emb_rows(kind, dtype, fm[cut[i]:cut[i + 1]]) for i, p in enumerate(self.parts)])


def concat(*parts):
    return Concat(list(parts))


# ----------------------------------------------------------------------------------------------------------------
# NN scores as INPUTS (stacking / baseline arms): out-of-fold on train rows, out-of-sample on val / holdout / test
# ----------------------------------------------------------------------------------------------------------------
NN11 = [f"r6_cd_m02_r{i}" for i in range(101, 108)] + [f"w5_e10_base_s{i}" for i in range(951, 955)]


def _pred_file(member, set_name):
    base = W5P if member.startswith("w5_") else ROOT / "preds"
    return base / (f"holdout/nn_{member}_holdout.npy" if set_name == "hold" else f"nn_{member}_trainonly_{set_name}.npy")


def nn_input(ds, which="auto"):
    """Per-row NN logit, CENTRED per feed (float32), usable as a model INPUT or a CatBoost baseline without leaking labels.
      train sets (xm / xs / prod*)      : which = 'oof'  -> the cross-fitted teacher (3 models that never saw the feed's user fold)
      evaluation sets (val / hold / test): which = 'xf<k>' (k = 0..4: mean of the 3 fold-k models; the like-for-like partner of 'oof'),
                                          'xf15' (mean of all 15 fold models) or 'mean11' (the 11 members behind teacher_mean11);
                                          every one of these models is out-of-sample for val, test and the holdout.
      'auto' = 'oof' on train sets, 'xf15' on evaluation sets.
    Scale: raw logit units, within-feed sd about 1.1 to 1.2 for a single model or a 3-model mean. NEVER use teacher('mean11') as an
    input on training feeds (in-sample)."""
    ev_set = ds.name in ("val", "hold", "test")
    if which == "auto":
        which = "xf15" if ev_set else "oof"
    if not ev_set:
        assert which == "oof", "train rows: only the out-of-fold teacher is a legal NN input"
        return ds.teacher("oof", "centred")
    if which == "mean11":
        members = NN11
    elif which == "xf15":
        members = [f"w5_e10_xf{k}_s{s}" for k in range(5) for s in (951, 952, 953)]
    elif which.startswith("xf") and len(which) == 3:
        members = [f"w5_e10_{which}_s{s}" for s in (951, 952, 953)]
    else:
        raise KeyError(which)

    def build():
        acc = np.zeros(ds.n_rows, np.float64)
        for m in members:
            x = np.load(_pred_file(m, ds.name)); assert x.shape == (ds.n_rows,)
            acc += x
        return _feed_center(ds.off, ds.M, np.ascontiguousarray((acc / len(members)), np.float32))
    return ds._cached(f"nn_{which}", build)


# ----------------------------------------------------------------------------------------------------------------
# train-side model selection (no val label)
# ----------------------------------------------------------------------------------------------------------------
def trainside_split(ds, frac=0.1):
    """User-grouped split of a TRAIN set into (fit_mask, sel_mask) per feed, for early stopping / iteration-count selection
    WITHOUT val labels. A user is a "selection user" when it falls in teacher fold 0, sub-half 0 (frac 0.1, about 10% of users)
    or in teacher fold 0 (frac 0.2). Then
        fit_mask = feeds of all other users                      (no feed of a selection user is ever fitted, in any set)
        sel_mask = feeds of selection users that belong to `xm`  (the SAME 13,527 M >= 50 exploration feeds whatever the training
                   set is; for a set without xm feeds, e.g. 'xs' or 'prod200k' alone, sel_mask is empty: cbx.run_arm then takes the
                   selection pool from cbx.load('xm'))
    The folds are the folds of the cross-fitted teacher (cbx.xfold_of), so teacher_oof on selection feeds comes from models that
    never saw those users. User-0 feeds are assigned by feed index."""
    assert ds.split == "train"
    if abs(frac - 0.2) < 1e-9:
        hs = ds.xfold == 0
    elif abs(frac - 0.1) < 1e-9:
        hs = (ds.xfold == 0) & (ds.xsub == 0)
    else:
        raise ValueError("frac must be 0.1 or 0.2")
    return ~hs, hs & np.isin(ds.fidx, train_sets()["xm"])


# ----------------------------------------------------------------------------------------------------------------
# CatBoost pool helper (same construction as nn_x/cb26/cb_member.to_df: DataFrame view + int32 category column)
# ----------------------------------------------------------------------------------------------------------------
def to_df(X, names, cat="native"):
    """pandas frame over X (no copy of the float block). cat: 'native' (category_id as int32 categorical column, as the record),
    'numeric' (left as float code)."""
    import pandas as pd
    df = pd.DataFrame(X, columns=names, copy=False)
    if cat == "native" and CAT_FEATURE in names:
        df[CAT_FEATURE] = np.nan_to_num(df[CAT_FEATURE].to_numpy(), nan=-1).astype(np.int32)
    return df


def make_pool(ds, feeds=None, label=None, cat="native", weight=None, baseline=None, embeddings=None, emb_dtype=np.float32, cols=None,
              extra=None, extra_cat=None):
    """catboost.Pool for a set (or for the feeds selected by the boolean per-feed mask `feeds`).
    label     : per-row array aligned to ds rows (it is sub-selected with the same mask), or None (prediction pool)
    cat       : 'native' -> category_id is a categorical feature (the record's setting); 'numeric' -> plain float feature
    embeddings: list of kinds among cand / user / hmean / h5 / h20 -> passed as CatBoost embedding features (embedding_features_data)
    cols      : optional list of feature names (subset of the 177)
    extra     : dict name -> per-row array aligned to ds rows, or callable(ds) -> such an array (so that one pool_kw serves train,
                val, holdout and test pools); built-in names need no array: "item_id", "user_id"
    extra_cat : names of `extra` columns to declare categorical (integer ids; CatBoost then builds its ordered target statistics
                on the TRAINING pool only, which is the leak-safe form). e.g. extra={"item_id": None}, extra_cat=["item_id"]
    Group ids are the set's feed positions (contiguous, file order)."""
    from catboost import Pool
    names = ds.names if cols is None else list(cols)
    X = ds.X
    rm = None if feeds is None else np.repeat(np.asarray(feeds, bool), ds.M)
    if cols is not None:
        ci = [ds.names.index(c) for c in names]
        X = np.asarray(X[:, ci] if rm is None else X[rm][:, ci], np.float32)
    else:
        X = np.asarray(X) if rm is None else np.asarray(X[rm])
    gid = ds.gid if rm is None else ds.gid[rm]
    sub = (lambda a: None if a is None else (np.asarray(a) if rm is None else np.asarray(a)[rm]))
    kw = dict(label=sub(label), group_id=gid, weight=sub(weight), baseline=sub(baseline), thread_count=THREADS)
    if cat == "native" and CAT_FEATURE in names:
        kw["cat_features"] = [CAT_FEATURE]
    if embeddings:
        kw["embedding_features"] = [f"emb_{k}" for k in embeddings]
        kw["embedding_features_data"] = {f"emb_{k}": ds.emb_rows(k, emb_dtype, feeds) for k in embeddings}
    df = to_df(X, names, cat)
    for k, v in (extra or {}).items():
        if v is None:
            v = {"item_id": lambda d: d.item_id, "user_id": lambda d: d.uid}[k]
        v = np.asarray(v(ds) if callable(v) else v)
        assert len(v) == ds.n_rows, f"extra column {k}: {len(v)} rows, set has {ds.n_rows}"
        v = v if rm is None else v[rm]
        if extra_cat and k in extra_cat:
            df[k] = v.astype(np.int64)
            kw["cat_features"] = kw.get("cat_features", []) + [k]
        else:
            df[k] = v.astype(np.float32)
    return Pool(df, **kw)


def idle_cpus(n=THREADS, dt=0.5):
    """the n least busy CPUs (of those this process may use) over a dt-second window of /proc/stat."""
    def snap():
        d = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split(); d[int(p[0][3:])] = (int(p[4]) + int(p[5]), sum(int(x) for x in p[1:9]))
        return d
    a = snap(); time.sleep(dt); b = snap()
    allowed = os.sched_getaffinity(0)
    busy = sorted((1.0 - (b[c][0] - a[c][0]) / max(b[c][1] - a[c][1], 1), c) for c in allowed if c in a and c in b)
    return {c for _, c in busy[:n]}


class pinned:
    """Context manager: pin the calling thread (and every thread it creates inside the block) to THREADS idle cores.
    NEEDED for catboost calls that ignore thread_count: Pool.quantize() uses EVERY core of the machine (measured here:
    many cores busy on a 0.9 s call), which breaks the 12-thread rule. Usage:  with cbx.pinned(): pool.quantize(border_count=254)"""

    def __enter__(self):
        self.old = os.sched_getaffinity(0)
        os.sched_setaffinity(0, idle_cpus(THREADS))
        return self

    def __exit__(self, *exc):
        os.sched_setaffinity(0, self.old)
        return False


def quantize(pool, **kw):
    """Thread-capped Pool.quantize (see cbx.pinned). kw e.g. border_count=254. A quantized pool can be reused by many fits
    (no per-fit quantization: saves about 20 s per fit on the 19.7M-row set)."""
    with pinned():
        pool.quantize(**kw)
    return pool


# ----------------------------------------------------------------------------------------------------------------
# evaluation: one function for every arm
# ----------------------------------------------------------------------------------------------------------------
def pse(d):
    d = np.asarray(d, np.float64)
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


class Evaluator:
    """Val (14,423 feeds) and 30K-holdout witnesses + the W16 train-only pool.
    Conventions (fixed for the whole track):
      z(.)      = per-feed z-score (src/stack_val.feed_rank_z[:, 1])
      W16 pool  = member-equal mean of the 16 members' z, then z(.) again  -> "z(W16)"
      rho       = mean over feeds of the within-feed Pearson correlation between z(X) and z(W16)
      law value = A (1 - rho_val) - B (NDCG_val(W16) - NDCG_val(X)), (A, B) = (0.0186, 0.378) on record, (0.0103, 0.304) refit
      member rule statistic: d = NDCG(0.9 z(W16) + 0.1 z(X)) - NDCG(z(W16)) per feed on val and on the holdout;
                 z_val, z_hold = mean / SE; pooled z = (z_val + z_hold) / sqrt(2); PASS iff d_val >= 0 and d_hold >= 0 and pooled >= 2.
      (weights 0.05 and 0.2 are reported, never chosen)."""

    def __init__(self):
        global ALLOW_EVAL_LABELS
        self.v, self.h, self.t = load("val"), load("hold"), load("test")
        prev = ALLOW_EVAL_LABELS
        ALLOW_EVAL_LABELS = True
        try:
            self.yv = np.asarray(self.v.y, np.float32)
            self.yh = np.asarray(self.h.y, np.float32)
        finally:
            ALLOW_EVAL_LABELS = prev
        Mh, oh = np.load(W5P / "holdout/holdout_meta.npy")
        assert (Mh == self.h.M).all() and (oh == self.h.off).all(), "holdout row order differs from nn_x/w5/preds/holdout"
        yh_ref = np.load(W5P / "holdout/holdout_y.npy")
        assert (yh_ref == self.yh).all(), "holdout labels differ from nn_x/w5/preds/holdout/holdout_y.npy"
        self.w16 = {k: self._w16(k) for k in ("val", "hold", "test")}
        self.nd16 = dict(val=self.ndcg("val", self.w16["val"]), hold=self.ndcg("hold", self.w16["hold"]))

    def ds(self, sp):
        return {"val": self.v, "hold": self.h, "test": self.t}[sp]

    def z(self, sp, x):
        d = self.ds(sp)
        x = np.asarray(x, np.float32)
        assert x.shape == (d.n_rows,), f"{sp}: expected {d.n_rows} rows, got {x.shape}"
        assert np.isfinite(x).all(), f"{sp}: non-finite predictions"
        return feed_z(d.off, d.M, x)

    def ndcg(self, sp, s):
        d = self.ds(sp)
        return ndcg_fast(s, self.yv if sp == "val" else self.yh, d.off, d.M)

    def _w16(self, sp):
        p = CACHE / f"w16_z_{sp}.npy"
        if not p.exists():
            acc = np.zeros(self.ds(sp).n_rows, np.float64)
            for m in W16:
                f = {"val": W5P / f"nn_{m}_trainonly_val.npy", "test": W5P / f"nn_{m}_trainonly_test.npy",
                     "hold": W5P / f"holdout/nn_{m}_holdout.npy"}[sp]
                acc += self.z(sp, np.load(f))
            tmp = CACHE / f"w16_z_{sp}.tmp.{os.getpid()}.npy"
            np.save(tmp, self.z(sp, (acc / len(W16)).astype(np.float32))); os.replace(tmp, p)
        return np.load(p)

    def rho(self, sp, x):
        d = self.ds(sp)
        c = _feed_corr(d.off, d.M, self.z(sp, x), self.w16[sp])
        return float(c.mean())

    def member_rule(self, pv, ph, w=0.1):
        out = {}
        for sp, x in (("val", pv), ("hold", ph)):
            mix = (1 - w) * self.w16[sp] + w * self.z(sp, x)
            d = self.ndcg(sp, mix) - self.nd16[sp]
            m, s = pse(d)
            out[sp] = dict(delta=m, se=s, z=m / s if s > 0 else 0.0, n=len(d))
        out["pooled_z"] = (out["val"]["z"] + out["hold"]["z"]) / np.sqrt(2)
        out["pass"] = bool(out["val"]["delta"] >= 0 and out["hold"]["delta"] >= 0 and out["pooled_z"] >= 2)
        return out

    def evaluate(self, tag, pv, ph, pt=None, out_dir=None, verbose=True, extra=None):
        """Score one model X. pv / ph / pt = raw predictions in the row order of preds/ (val, test) and nn_x/w5/preds/holdout (hold).
        Saves float32 <out_dir>/<tag>_{val,hold,test}.npy and <tag>.eval.json when out_dir is given. Returns the result dict."""
        pv = np.asarray(pv, np.float32); ph = np.asarray(ph, np.float32)
        nv, nh = self.ndcg("val", pv), self.ndcg("hold", ph)
        r = dict(tag=tag, n_val=len(nv), n_hold=len(nh))
        r["val"], r["val_se"] = pse(nv)
        r["hold"], r["hold_se"] = pse(nh)
        r["w16_val"], r["w16_hold"] = float(self.nd16["val"].mean()), float(self.nd16["hold"].mean())
        r["gap_val"], r["gap_val_se"] = pse(self.nd16["val"] - nv)
        r["gap_hold"], r["gap_hold_se"] = pse(self.nd16["hold"] - nh)
        r["rho_val"], r["rho_hold"] = self.rho("val", pv), self.rho("hold", ph)
        # the project's other convention ("wcorr": one Pearson over all rows of the z arrays); printed for comparison only
        r["rho_global_val"] = float(np.corrcoef(self.z("val", pv), self.w16["val"])[0, 1])
        r["rho_global_hold"] = float(np.corrcoef(self.z("hold", ph), self.w16["hold"])[0, 1])
        r["law_record"] = LAW_RECORD[0] * (1 - r["rho_val"]) - LAW_RECORD[1] * r["gap_val"]
        r["law_refit"] = LAW_REFIT[0] * (1 - r["rho_val"]) - LAW_REFIT[1] * r["gap_val"]
        r["law_record_hold"] = LAW_RECORD[0] * (1 - r["rho_hold"]) - LAW_RECORD[1] * r["gap_hold"]
        r["member_rule"] = self.member_rule(pv, ph, 0.1)
        r["mix_w005"] = self.member_rule(pv, ph, 0.05)
        r["mix_w020"] = self.member_rule(pv, ph, 0.2)
        r["stop_rule_val_ge_0.3900"] = bool(r["val"] >= 0.39)
        r["stop_rule_law_ge_-0.002"] = bool(r["law_record"] >= -0.002)
        if extra:
            r["extra"] = extra
        if out_dir is not None:
            out_dir = Path(out_dir)
            assert str(out_dir.resolve()).startswith(str(ROOT / "nn_x/w12/cb")), "write only inside nn_x/w12/cb/<KEY>/"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"{tag}_val.npy", pv); np.save(out_dir / f"{tag}_hold.npy", ph)
            if pt is not None:
                pt = np.asarray(pt, np.float32)
                assert pt.shape == (self.t.n_rows,) and np.isfinite(pt).all()
                np.save(out_dir / f"{tag}_test.npy", pt)
            (out_dir / f"{tag}.eval.json").write_text(json.dumps(r, indent=1))
        if verbose:
            print(self.fmt(r), flush=True)
        return r

    @staticmethod
    def fmt(r):
        m = r["member_rule"]
        return (f"[{r['tag']}] val {r['val']:.5f} +- {r['val_se']:.5f} (n={r['n_val']}) | holdout {r['hold']:.5f} +- {r['hold_se']:.5f} (n={r['n_hold']})\n"
                f"   W16 {r['w16_val']:.5f} / {r['w16_hold']:.5f}; gap {r['gap_val']:.5f} +- {r['gap_val_se']:.5f} / {r['gap_hold']:.5f} +- {r['gap_hold_se']:.5f}; "
                f"rho {r['rho_val']:.4f} / {r['rho_hold']:.4f} (global-Pearson convention {r.get('rho_global_val', float('nan')):.4f} / "
                f"{r.get('rho_global_hold', float('nan')):.4f}); law {r['law_record']:+.5f} (record) {r['law_refit']:+.5f} (refit)\n"
                f"   member rule 0.9 z(W16) + 0.1 z(X): val {m['val']['delta']:+.5f} +- {m['val']['se']:.5f} (z {m['val']['z']:+.2f}) | "
                f"holdout {m['hold']['delta']:+.5f} +- {m['hold']['se']:.5f} (z {m['hold']['z']:+.2f}) | pooled z {m['pooled_z']:+.2f} -> "
                f"{'PASS' if m['pass'] else 'FAIL'}\n"
                f"   (w 0.05: {r['mix_w005']['val']['delta']:+.5f} / {r['mix_w005']['hold']['delta']:+.5f}; "
                f"w 0.2: {r['mix_w020']['val']['delta']:+.5f} / {r['mix_w020']['hold']['delta']:+.5f}) "
                f"stop rule: val >= 0.3900 {r['stop_rule_val_ge_0.3900']}, law >= -0.002 {r['stop_rule_law_ge_-0.002']}")

    def compare(self, a_v, a_h, b_v, b_h, verbose=True, label="A - B"):
        """Progress rule read: paired per-feed NDCG delta of arm A over its baseline B on both witnesses.
        KEPT iff delta_val > 0 and delta_hold > 0 and pooled z >= 2 (and remember: single-seed differences < 0.003 between tree arms are noise)."""
        dv = self.ndcg("val", a_v) - self.ndcg("val", b_v)
        dh = self.ndcg("hold", a_h) - self.ndcg("hold", b_h)
        mv, sv = pse(dv); mh, sh = pse(dh)
        zp = (mv / sv + mh / sh) / np.sqrt(2)
        r = dict(val=mv, val_se=sv, hold=mh, hold_se=sh, pooled_z=float(zp), n_val=len(dv), n_hold=len(dh),
                 kept=bool(mv > 0 and mh > 0 and zp >= 2))
        if verbose:
            print(f"[{label}] val {mv:+.5f} +- {sv:.5f} (n={len(dv)}) | holdout {mh:+.5f} +- {sh:.5f} (n={len(dh)}) | pooled z {zp:+.2f} -> "
                  f"{'KEPT' if r['kept'] else 'not kept'}", flush=True)
        return r


class Budget:
    """CatBoost callback (CPU): stop cleanly when `minutes` of wall clock are used or when the file `stop_file` exists;
    records (iteration, seconds, metrics) per iteration in .rows and dumps them to `curve_file` every 50 iterations."""

    def __init__(self, minutes=None, curve_file=None, stop_file=None):
        self.t0 = time.time(); self.seconds = None if minutes is None else minutes * 60
        self.curve_file = curve_file; self.stop_file = stop_file; self.rows = []; self.stopped = False

    def dump(self):
        if self.curve_file is not None:
            with open(self.curve_file, "w") as f:
                json.dump(self.rows, f)

    def after_iteration(self, info):
        try:
            mets = {f"{ds}:{k}": float(v[-1]) for ds, d in info.metrics.items() for k, v in d.items()}
        except Exception:  # noqa: BLE001
            mets = {}
        self.rows.append((int(info.iteration), time.time() - self.t0, mets))
        if info.iteration % 50 == 0:
            self.dump()
        if (self.seconds is not None and time.time() - self.t0 > self.seconds) or (self.stop_file and os.path.exists(self.stop_file)):
            self.stopped = True
            return False
        return True


def run_arm(tag, model, ds, label, out_dir, sel_frac=0.1, sel_label=None, pool_kw=None, weight=None, baseline=None,
            pred_pool_fn=None, budget_min=None, ev=None, verbose=50, extra=None, save_model=True, max_feeds=None):
    """The standard arm protocol (train-side selection: NO val label is read before the final evaluation).
      1. fit `model` on the train-side fit split of `ds` (cbx.trainside_split), eval_set = the train-side selection split;
         the model must be an UNFITTED CatBoost estimator created with thread_count=cbx.THREADS and with its own
         iterations / od_type / od_wait / eval_metric (recommended: eval_metric="NDCG:type=Base", od_type="Iter").
      2. predict val / holdout / test with ntree_end = best iteration + 1 on the selection split.
      3. cbx.Evaluator().evaluate(...) and save <out_dir>/<tag>_{val,hold,test}.npy, <tag>.eval.json, <tag>.curve.json[, <tag>.cbm].
    label     : per-row training target aligned to ds rows (clicks ds.y, teacher logits ds.teacher(...), ...)
    sel_label : target on the selection split used by the eval metric, aligned to cbx.load("xm") rows; default = the click label
                (so the selection metric is NDCG on clicks whatever the training target is). The selection pool is ALWAYS the
                13,527 xm selection feeds, also when ds is a larger set (cbx.concat(xm, xs), prod200k, ...)
    pool_kw   : dict passed to cbx.make_pool for every pool (cat=..., cols=..., embeddings=[...], emb_dtype=...)
    weight / baseline : per-row arrays aligned to ds rows, or callables f(ds) -> array (needed when ds is not 'xm'); the baseline is
                also applied to the selection pool; give pred_pool_fn when prediction needs a baseline
    pred_pool_fn : optional callable(ds_pred) -> catboost.Pool for val / hold / test (default: cbx.make_pool(ds_pred, **pool_kw))
    budget_min: wall-clock cap for the fit; creating <out_dir>/<tag>.STOP also ends the fit cleanly.
    max_feeds : dry-run switch: use a random subset of that many feeds of ds (fit + selection together).
    Returns the evaluation dict (see Evaluator.evaluate)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pool_kw = dict(pool_kw or {})
    t0 = time.time()
    xm = load("xm")
    fit, _ = trainside_split(ds, sel_frac)
    _, sel = trainside_split(xm, sel_frac)      # the selection set is the same for every arm
    if max_feeds:
        rg = np.random.default_rng(0)
        k1 = np.zeros(ds.n_feeds, bool); k1[rg.choice(ds.n_feeds, min(max_feeds, ds.n_feeds), replace=False)] = True
        k2 = np.zeros(xm.n_feeds, bool); k2[rg.choice(xm.n_feeds, min(max_feeds, xm.n_feeds), replace=False)] = True
        fit = fit & k1; sel = sel & k2

    def per_set(a, d, what):
        if a is None:
            return None
        if callable(a):
            return a(d)
        assert d is ds or ds is xm, f"{what}: give a callable f(ds) when the training set is not 'xm' (the selection pool is built on 'xm')"
        return a
    ptr = make_pool(ds, feeds=fit, label=label, weight=per_set(weight, ds, "weight"), baseline=per_set(baseline, ds, "baseline"), **pool_kw)
    pes = make_pool(xm, feeds=sel, label=xm.y if sel_label is None else sel_label, baseline=per_set(baseline, xm, "baseline"), **pool_kw)
    log(f"[{tag}] fit feeds {int(fit.sum())} rows {ptr.num_row()} | train-side selection feeds {int(sel.sum())} rows {pes.num_row()} | "
        f"pools {time.time() - t0:.0f}s rss {rss_gb()[0]:.1f} GB")
    cb = Budget(budget_min, out_dir / f"{tag}.curve.json", str(out_dir / f"{tag}.STOP"))
    tf = time.time()
    model.fit(ptr, eval_set=pes, verbose=verbose, callbacks=[cb])
    fit_s = time.time() - tf
    cb.dump()
    del ptr, pes
    best = int(model.get_best_iteration()) if model.get_best_iteration() is not None else model.tree_count_ - 1
    k = min(best + 1, model.tree_count_)
    log(f"[{tag}] fit {fit_s:.0f}s, {len(cb.rows)} iterations ({fit_s / max(len(cb.rows), 1):.2f} s/iter), best iteration {best}, budget stop {cb.stopped}")
    if save_model:
        model.save_model(str(out_dir / f"{tag}.cbm"))
    ev = ev or Evaluator()
    mk = pred_pool_fn or (lambda d: make_pool(d, **pool_kw))
    pv, ph, pt = (np.asarray(model.predict(mk(load(n)), ntree_end=k, thread_count=THREADS), np.float32) for n in ("val", "hold", "test"))
    info = dict(n_fit_feeds=int(fit.sum()), n_sel_feeds=int(sel.sum()), best_iter=best, trees_built=len(cb.rows), budget_stop=bool(cb.stopped),
                fit_seconds=fit_s, s_per_iter=fit_s / max(len(cb.rows), 1), total_seconds=time.time() - t0, peak_rss_gb=rss_gb()[1],
                params={k_: str(v) for k_, v in model.get_params().items()}, set=ds.name, selection="trainside")
    info.update(extra or {})
    return ev.evaluate(tag, pv, ph, pt, out_dir=out_dir, extra=info)


def rss_gb():
    """(current RSS, peak RSS) of this process in GB."""
    cur = peak = 0.0
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"): cur = int(line.split()[1]) / 1048576
        if line.startswith("VmHWM"): peak = int(line.split()[1]) / 1048576
    return cur, peak
