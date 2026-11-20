"""Shared helpers for the Avito-platform-nuance audit (CPU only, read-only on project data)."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
PREDS = ROOT / "preds"
OUT = ROOT / "nn_x/audit/avito/out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402

MSK = 3 * 3600
PUBLIC_TS = 1790497920  # first private timestamp (eda_shift_overlap)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    d = dict(fidx=f["fidx"].to_numpy(), ts=f["timestamp"].to_numpy().astype(np.int64),
             user=f["user_id"].to_numpy().astype(np.int64), M=f["M"].to_numpy().astype(np.int64),
             H=f["H"].to_numpy().astype(np.int64), off=f["cand_off"].to_numpy().astype(np.int64),
             hoff=f["hist_off"].to_numpy().astype(np.int64))
    t = d["ts"] + MSK
    d["hour"] = (t // 3600) % 24
    d["day"] = t // 86400
    d["dow"] = (d["day"] + 3) % 7  # Monday = 0
    d["groups"] = np.where(d["user"] == 0, -1 - np.arange(len(d["user"])), d["user"])
    d["n"] = len(d["M"])
    return d


def cand(split, name):
    return np.load(FLAT / split / f"cand_{name}.npy", mmap_mode="r")


def hist(split, name):
    return np.load(FLAT / split / f"hist_{name}.npy", mmap_mode="r")


def feat(split, name):
    return np.load(FEATS / split / f"{name}.npy")


def emb_table(split):
    m = json.load(open(ROOT / f"data/competition/{split}/item_embeddings/data.meta"))
    return np.memmap(ROOT / f"data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r",
                     shape=tuple(m["shape"]))


def g3(split):
    return np.load(PREDS / f"ens_G3_final_{split}.npy").astype(np.float64)


def hour_bucket(h):
    return np.digitize(h, [7, 12, 18])  # 0: 0-6, 1: 7-11, 2: 12-17, 3: 18-23


def feed_z(x, fidx, n):
    x = np.asarray(x, dtype=np.float64)
    c = np.bincount(fidx, minlength=n).astype(np.float64)
    mu = np.bincount(fidx, weights=x, minlength=n) / np.maximum(c, 1)
    v = np.bincount(fidx, weights=(x - mu[fidx]) ** 2, minlength=n) / np.maximum(c, 1)
    sd = np.sqrt(v)
    sd[sd < 1e-9] = 1.0
    return (x - mu[fidx]) / sd[fidx]


def paired(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def fmt(a, b):
    m, se = paired(a, b)
    return f"{m:+.5f} ± {se:.5f} (z {m / se if se > 0 else 0:+.1f})"


def make_folds(groups, nfold=5, seed=0):
    u, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    f = rng.permutation(len(u)) % nfold
    return f[inv]


def feed_auc(score, y, fidx, n, seed=0):
    """Per-feed AUC of score for the positive class (random tie-break); NaN for feeds without both classes."""
    rng = np.random.default_rng(seed)
    s = np.asarray(score, dtype=np.float64) + rng.uniform(0, 1e-9, len(score))
    order = np.lexsort((s, fidx))
    rank = np.empty(len(s), dtype=np.float64)
    pos_in_feed = np.arange(len(s)) - np.repeat(np.cumsum(np.bincount(fidx, minlength=n)) - np.bincount(fidx, minlength=n), np.bincount(fidx, minlength=n))
    rank[order] = pos_in_feed + 1.0
    npos = np.bincount(fidx, weights=y, minlength=n)
    M = np.bincount(fidx, minlength=n).astype(np.float64)
    nneg = M - npos
    srank = np.bincount(fidx, weights=rank * y, minlength=n)
    auc = (srank - npos * (npos + 1) / 2) / np.maximum(npos * nneg, 1)
    auc[(npos == 0) | (nneg == 0)] = np.nan
    return auc


def nested_additive(base, extra, y, off, M, fold, wgrid):
    """score = base + w * extra; w chosen on the other folds by mean per-feed NDCG. Returns (oof per-feed NDCG, base per-feed NDCG, chosen ws, full-fit curve)."""
    n = len(M)
    nd = np.zeros((len(wgrid), n))
    for i, w in enumerate(wgrid):
        nd[i] = ndcg_per_feed(base + w * extra, y, off, M)
    base_nd = ndcg_per_feed(base, y, off, M)
    oof = np.zeros(n)
    ws = []
    for k in np.unique(fold):
        tr = fold != k
        te = fold == k
        i = int(np.argmax(nd[:, tr].mean(1)))
        oof[te] = nd[i, te]
        ws.append(float(wgrid[i]))
    return oof, base_nd, ws, nd.mean(1)


def lgb_oof(X, y, off, M, fold, seed=0, rounds=300, lr=0.05, leaves=31, min_leaf=200, threads=12, cat_idx=None):
    """Grouped OOF LightGBM lambdarank over feeds; X is (rows, feats) float32 in feed order. Returns OOF scores."""
    import lightgbm as lgb
    n = len(M)
    row_fold = np.repeat(fold, M)
    oof = np.zeros(len(y), dtype=np.float64)
    params = dict(objective="lambdarank", metric="None", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_leaf,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=60,
                  num_threads=threads, verbose=-1, seed=seed, max_bin=127)
    for k in np.unique(fold):
        tr_rows = row_fold != k
        te_rows = row_fold == k
        ds = lgb.Dataset(X[tr_rows], label=y[tr_rows], group=M[fold != k], categorical_feature=cat_idx or "auto",
                         free_raw_data=True)
        b = lgb.train(params, ds, num_boost_round=rounds)
        oof[te_rows] = b.predict(X[te_rows], num_threads=threads)
    return oof


def holdout_index():
    """Reproduce nn/train.py's 30K random-like holdout feed list (seed 12345)."""
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    f = pl.read_parquet(FLAT / "train" / "feeds.parquet", columns=["M"])
    Mt = f["M"].to_numpy().astype(np.int64)
    p[Mt < 50] = 0.0
    cand_ = np.where((p > 0.9) & (Mt >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand_, min(30000, len(cand_)), replace=False))
    return hold, p


def g3_holdout_pool():
    """Member-equal z-mean of the G3 train-only members' holdout predictions."""
    names = [x.strip() for x in open(ROOT / "nn_x/pubdiag/build/g3_members.txt").read().replace("\n", ",").split(",") if x.strip()]
    hm = np.load(PREDS / "holdout/holdout_meta.npy")
    M = hm[0].astype(np.int64)
    fidx = np.repeat(np.arange(len(M)), M)
    acc = None
    used = []
    for nm in names:
        p = PREDS / "holdout" / f"nn_{nm}_holdout.npy"
        if not p.exists():
            continue
        z = feed_z(np.load(p), fidx, len(M))
        acc = z if acc is None else acc + z
        used.append(nm)
    return acc / len(used), used, M
