"""Shared helpers for the test-time information audit (CPU only, read-only on project data).

Base = G3 pool on val: 0.5*z(ft) + 0.5*z(to), z per feed (src/pmix.zmean convention).
All deltas are paired per-feed NDCG differences with SE = sd/sqrt(n_feeds) (exact organizer metric, pessimistic ties).
"""
import os, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "12")
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = "<repo>"
FLAT = f"{ROOT}/data/flat"
FEATS = f"{ROOT}/data/feats"
PREDS = f"{ROOT}/preds"
OUT = f"{ROOT}/nn_x/audit/testinfo"
BOUND = 1790497920  # public = test feeds with timestamp < BOUND (eda_shift_overlap §5.1)
BIG = 1e6
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def feeds(split):
    return pl.read_parquet(f"{FLAT}/{split}/feeds.parquet")


def arr(split, name, mmap=True):
    return np.load(f"{FLAT}/{split}/{name}.npy", mmap_mode="r" if mmap else None)


def feat(split, name):
    return np.load(f"{FEATS}/{split}/{name}.npy")


class Split:
    def __init__(self, split):
        self.split = split
        f = feeds(split)
        self.f = f
        self.M = f["M"].to_numpy().astype(np.int64)
        self.off = f["cand_off"].to_numpy().astype(np.int64)
        self.H = f["H"].to_numpy().astype(np.int64)
        self.hoff = f["hist_off"].to_numpy().astype(np.int64)
        self.user = f["user_id"].to_numpy()
        self.ts = f["timestamp"].to_numpy().astype(np.int64)
        self.feed_id = f["feed_id"].to_numpy()
        self.n = len(self.M)
        self.N = int(self.M.sum())
        self.fid = np.repeat(np.arange(self.n), self.M)
        self.hfid = np.repeat(np.arange(self.n), self.H)
        self.y = np.asarray(arr(split, "cand_target")).astype(np.float64) if split != "test" else None
        self.item = np.asarray(arr(split, "cand_item_id"))
        self.hitem = np.asarray(arr(split, "hist_item_id"))
        if split == "test":
            self.public = self.ts < BOUND

    def z(self, s):
        s = np.asarray(s, dtype=np.float64)
        mu = np.bincount(self.fid, s, self.n) / self.M
        sd = np.sqrt(np.maximum(np.bincount(self.fid, s * s, self.n) / self.M - mu ** 2, 1e-12))
        return (s - mu[self.fid]) / sd[self.fid]

    def rank_pct(self, s):
        """within-feed percentile rank in [0,1] (ties: average)."""
        s = np.asarray(s, dtype=np.float64)
        order = np.lexsort((s, self.fid))
        r = np.empty(self.N); r[order] = np.arange(self.N) - np.repeat(self.off, self.M)[order]
        return r / np.maximum(self.M[self.fid] - 1, 1)

    def ndcg(self, s):
        return ndcg_per_feed(np.asarray(s, dtype=np.float64), self.y, self.off, self.M)

    def npos(self):
        return np.bincount(self.fid, self.y, self.n)


def g3(V):
    ft = np.load(f"{PREDS}/ens_G3_ft_val.npy"); to = np.load(f"{PREDS}/ens_G3_to_val.npy")
    return 0.5 * V.z(ft) + 0.5 * V.z(to)


def paired(pa, pb, mask=None):
    """returns (delta, se, z, n). pa/pb per-feed NDCG."""
    d = pa - pb
    if mask is not None:
        d = d[mask]
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else float("nan"), float("nan"), float("nan"), n)
    se = d.std(ddof=1) / np.sqrt(n)
    return float(d.mean()), float(se), float(d.mean() / se) if se > 0 else float("nan"), n


def fmt(pa, pb, mask=None, label=""):
    d, se, z, n = paired(pa, pb, mask)
    return f"{label:60s} delta {d:+.5f} ± {se:.5f} (z {z:+.2f}, n={n})"


def pmix_weights():
    sys.path.insert(0, f"{ROOT}/src")
    import pmix
    return pmix.weights()


def wmean_se(x, w):
    m = np.average(x, weights=w)
    neff = w.sum() ** 2 / (w ** 2).sum()
    return float(m), float(np.sqrt(np.average((x - m) ** 2, weights=w) / neff))


def tiebreak(V, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(V.N) * 1e-9


def groups_for_cv(V):
    """user-grouped folds; user 0 feeds are their own groups."""
    g = V.user.astype(np.int64).copy()
    g[g == 0] = -(np.arange(V.n)[g == 0] + 1)
    return g


def user_group_folds(V, k=5, seed=0):
    from sklearn.model_selection import GroupKFold
    g = groups_for_cv(V)
    # shuffle groups deterministically
    rng = np.random.default_rng(seed)
    ug = np.unique(g); perm = rng.permutation(len(ug)); remap = dict(zip(ug, perm))
    gg = np.array([remap[x] for x in g])
    return list(GroupKFold(n_splits=k).split(np.zeros(V.n), groups=gg))


def nested_blend(V, base, sig, pf_base, weights=(0.02, 0.05, 0.1, 0.2, 0.4, 0.8), k=5, seed=0, use_z=True, label=""):
    """Nested (user-grouped) weight selection for score = base + w * z(sig). Returns per-feed OOF NDCG array."""
    s = V.z(sig) if use_z else sig
    folds = user_group_folds(V, k, seed)
    out = np.empty(V.n)
    pfs = {w: V.ndcg(base + w * s) for w in weights}
    chosen = []
    for tr, te in folds:
        best = max(weights, key=lambda w: pfs[w][tr].mean())
        chosen.append(best)
        out[te] = pfs[best][te]
    d, se, z, n = paired(out, pf_base)
    print(f"  nested blend {label:48s} delta {d:+.5f} ± {se:.5f} (z {z:+.2f}) chosen w {chosen}; "
          f"fixed-w deltas: " + " ".join(f"{w}:{pfs[w].mean() - pf_base.mean():+.5f}" for w in weights), flush=True)
    return out


def lgb_residual(V, feats_dict, pf_base, base_names, add_names, k=5, seed=0, rounds=150, label="", verbose=True):
    """LightGBM lambdarank residual harness on val: OOF NDCG of model(base_names) vs model(base_names+add_names).
    User-grouped folds; returns (pf_A, pf_B)."""
    import lightgbm as lgb
    folds = user_group_folds(V, k, seed)
    params = dict(objective="lambdarank", learning_rate=0.08, num_leaves=31, min_data_in_leaf=200, feature_fraction=1.0,
                  bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=30, max_bin=63, num_threads=12,
                  verbose=-1, seed=seed, label_gain=[0, 1], eval_at=[300])
    res = {}
    for tag, names in (("A", base_names), ("B", base_names + add_names)):
        X = np.stack([feats_dict[nm] for nm in names], 1).astype(np.float32)
        oof = np.zeros(V.N)
        for fi, (tr, te) in enumerate(folds):
            trm = np.isin(V.fid, tr); tem = ~trm
            ds = lgb.Dataset(X[trm], V.y[trm], group=V.M[tr], free_raw_data=False)
            bst = lgb.train(params, ds, num_boost_round=rounds)
            oof[tem] = bst.predict(X[tem], num_threads=12)
        res[tag] = V.ndcg(oof + tiebreak(V))
        if verbose:
            log(f"  lgb {label} {tag} ({len(names)} feats) OOF {res[tag].mean():.5f}  vs G3 {fmt(res[tag], pf_base)}")
    if verbose:
        log(f"  lgb {label} B-A {fmt(res['B'], res['A'])}")
    return res["A"], res["B"]


def lgb_resid_init(V, feats_dict, pf_base, base_names, add_names, init, k=5, seed=0, rounds=120, lr=0.03, leaves=15, min_leaf=1000,
                   label="", masks=None):
    """Residual harness that starts boosting FROM the G3 score (init_score), so the A model cannot lose G3's resolution.
    OOF score = g3 + f(features). A = base_names, B = base_names + add_names. Returns (pf_A, pf_B)."""
    import lightgbm as lgb
    folds = user_group_folds(V, k, seed)
    params = dict(objective="lambdarank", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_leaf, feature_fraction=1.0,
                  bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=30, max_bin=255, num_threads=12,
                  verbose=-1, seed=seed, label_gain=[0, 1], eval_at=[300])
    res = {}
    for tag, names in (("A", base_names), ("B", base_names + add_names)):
        X = np.stack([feats_dict[nm] for nm in names], 1).astype(np.float32)
        oof = np.zeros(V.N)
        for fi, (tr, te) in enumerate(folds):
            trm = np.isin(V.fid, tr); tem = ~trm
            ds = lgb.Dataset(X[trm], V.y[trm], group=V.M[tr], init_score=init[trm], free_raw_data=False)
            bst = lgb.train(params, ds, num_boost_round=rounds)
            oof[tem] = init[tem] + bst.predict(X[tem], num_threads=12, raw_score=True)
        res[tag] = V.ndcg(oof)
        log(f"  lgbR {label} {tag} ({len(names)} feats) OOF {res[tag].mean():.5f}  vs G3 {fmt(res[tag], pf_base)}")
    log(f"  lgbR {label} B-A {fmt(res['B'], res['A'])}")
    if masks:
        for mn, mm in masks.items():
            print(f"      B-A on {mn}: {fmt(res['B'], res['A'], mm)}", flush=True)
    return res["A"], res["B"]


HOUR = 3600.0
# ---- cross-user signals (copied from 02) ----
def hist_table(S, sid):
    df = pl.DataFrame({"sid": np.full(len(S.hitem), sid, np.int8), "hfidx": S.hfid.astype(np.int32), "huser": S.user[S.hfid],
                       "hts": S.ts[S.hfid], "item_id": S.hitem,
                       "etype": np.asarray(arr(S.split, "hist_event_type")), "esrc": np.asarray(arr(S.split, "hist_event_source")),
                       "hpos": np.asarray(arr(S.split, "hist_hpos")).astype(np.int16)})
    df = df.with_columns(((pl.col("etype") == 1) & (pl.col("esrc") == 1)).alias("fc"), (pl.col("etype") == 1).alias("t1"))
    return df.group_by(["sid", "hfidx", "huser", "hts", "item_id"]).agg(pl.col("fc").any(), pl.col("t1").any(), pl.col("hpos").min())


def signals(S, sid, HT, tag):
    b = pl.DataFrame({"row": np.arange(S.N, dtype=np.int64), "fidx": S.fid.astype(np.int32), "user_id": S.user[S.fid], "ts": S.ts[S.fid], "item_id": S.item})
    j = b.join(HT, on="item_id").filter(~((pl.col("sid") == sid) & (pl.col("hfidx") == pl.col("fidx"))))
    j = j.filter((pl.col("huser") != pl.col("user_id")) | (pl.col("user_id") == 0))
    j = j.with_columns(((pl.col("hts") - pl.col("ts")) / HOUR).alias("dth"))
    later = pl.col("dth") > 0
    agg = j.group_by("row").agg(
        pl.len().alias(f"{tag}_all"), pl.col("fc").sum().alias(f"{tag}_fc"), pl.col("t1").sum().alias(f"{tag}_t1"),
        later.sum().alias(f"{tag}_later"), (later & pl.col("fc")).sum().alias(f"{tag}_later_fc"),
        (~later).sum().alias(f"{tag}_earlier"), (~later & pl.col("fc")).sum().alias(f"{tag}_earlier_fc"),
        (later & (pl.col("dth") <= 24)).sum().alias(f"{tag}_later24"), (later & (pl.col("dth") <= 48) & pl.col("fc")).sum().alias(f"{tag}_later48_fc"),
        (~later & (pl.col("dth") >= -24)).sum().alias(f"{tag}_earlier24"),
        (-pl.col("dth").abs() / 24.0).exp().sum().alias(f"{tag}_rec24"), (-pl.col("dth").abs() / 72.0).exp().sum().alias(f"{tag}_rec72"),
        ((-pl.col("dth").abs() / 24.0).exp() * pl.col("fc")).sum().alias(f"{tag}_rec24_fc"),
        (pl.col("fc") & (pl.col("hpos") < 8)).sum().alias(f"{tag}_fc_hpos8"))
    out = pl.DataFrame({"row": np.arange(S.N, dtype=np.int64)}).join(agg, on="row", how="left").fill_null(0).sort("row")
    return {c: out[c].to_numpy().astype(np.float32) for c in out.columns if c != "row"}


