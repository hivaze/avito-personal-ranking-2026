"""S2a: content-kNN quality prior. For each val/test candidate item, the position-debiased click rate of its nearest TRAIN items
in item-embedding space (the item itself excluded), so cold listings inherit the record of similar listings.

DB = unique train candidate items with >= 3 train impressions (n, clicks c, expected clicks e from the global train CTR by
production position). Query = val/test candidate items (their own split's embedding table). faiss IVF (inner product on unit
vectors), k = 50, CPU 16 threads. Neighbours restricted to the same category_id.
Features: kn_coec50 = (sum c + 1) / (sum e + 1) over same-category neighbours (top 50); kn_coec_s90 / kn_coec_s95 = same over
neighbours with cos >= 0.90 / 0.95 (NaN if none); kn_n95 = #neighbours with cos >= 0.95; kn_sim1 = best cosine;
kn_ctr_w = similarity-softmax-weighted (temperature 0.02) mean of per-neighbour smoothed COEC.
Train labels only; identical for val and test.
"""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np, polars as pl, faiss
from pathlib import Path

faiss.omp_set_num_threads(16)
F = Path("<repo>/data/flat"); E = Path("<repo>/data/competition")
OUT = Path("<repo>/eda/errors/synthesis/derived")
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def emb_table(split):
    p = E / split / "item_embeddings" / "data.memmap"
    return np.memmap(p, dtype=np.int8, mode="r", shape=(os.path.getsize(p) // 128, 128))


def unitf(x):
    x = x.astype(np.float32); n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-6)


it = np.load(F / "train/cand_item_id.npy"); t = np.load(F / "train/cand_target.npy"); pos = np.load(F / "train/cand_pos.npy")
ctr = np.bincount(pos.astype(np.int64), weights=t.astype(np.float64)) / np.maximum(np.bincount(pos.astype(np.int64)), 1)
df = pl.DataFrame({"item": it, "t": t.astype(np.float32), "e": ctr[pos].astype(np.float32),
                   "emb": np.load(F / "train/cand_emb_idx.npy"), "cat": np.load(F / "train/cand_category_id.npy").astype(np.int32)})
del it, t, pos
agg = df.group_by("item").agg(pl.len().alias("n"), pl.col("t").sum().alias("c"), pl.col("e").sum().alias("e"),
                              pl.col("emb").max().alias("emb"), pl.col("cat").first().alias("cat"))
del df
db = agg.filter((pl.col("n") >= 3) & (pl.col("emb") >= 0)).sort("item")
log("DB items (n>=3)", len(db), "of", len(agg))
db_item = db["item"].to_numpy(); db_c = db["c"].to_numpy().astype(np.float64); db_e = db["e"].to_numpy().astype(np.float64)
db_cat = db["cat"].to_numpy()
db_coec = (db_c + 1) / (db_e + 1)
tab = emb_table("train")
ei = db["emb"].to_numpy(); o = np.argsort(ei)
X = np.empty((len(ei), 128), np.float32)
CH = 2_000_000
for s in range(0, len(o), CH):
    X[o[s:s + CH]] = unitf(tab[ei[o[s:s + CH]]])
log("DB embeddings loaded")
nlist = 16384
q = faiss.IndexFlatIP(128)
index = faiss.IndexIVFFlat(q, 128, nlist, faiss.METRIC_INNER_PRODUCT)
rs = np.random.default_rng(0).choice(len(X), 600_000, replace=False)
index.train(X[rs]); index.add(X); index.nprobe = 24
log("index built")
del X
K = 50
for split in ("val", "test"):
    cit = np.load(F / split / "cand_item_id.npy"); cem = np.load(F / split / "cand_emb_idx.npy")
    ccat = np.load(F / split / "cand_category_id.npy").astype(np.int32)
    u, inv = np.unique(cit, return_inverse=True)
    first = np.zeros(len(u), np.int64); first[inv[::-1]] = np.arange(len(cit))[::-1]
    qe = cem[first]; qc = ccat[first]
    Q = np.zeros((len(u), 128), np.float32); ok = qe >= 0
    Q[ok] = unitf(emb_table(split)[qe[ok]])
    Dm = np.zeros((len(u), K), np.float32); Im = np.zeros((len(u), K), np.int64)
    for s in range(0, len(u), 500_000):
        Dm[s:s + 500_000], Im[s:s + 500_000] = index.search(Q[s:s + 500_000], K)
    log(split, "search done", len(u))
    valid = (Im >= 0) & ok[:, None]
    Ii = np.where(valid, Im, 0)
    valid &= db_item[Ii] != u[:, None]            # exclude the item itself
    valid &= db_cat[Ii] == qc[:, None]            # same category
    cc = np.where(valid, db_c[Ii], 0); ee = np.where(valid, db_e[Ii], 0); sim = np.where(valid, Dm, -1)
    feats = {}
    feats["kn_coec50"] = (cc.sum(1) + 1) / (ee.sum(1) + 1)
    for thr, nm in ((0.90, "s90"), (0.95, "s95")):
        m = sim >= thr
        feats[f"kn_coec_{nm}"] = np.where(m.any(1), ((cc * m).sum(1) + 1) / ((ee * m).sum(1) + 1), np.nan)
    feats["kn_n95"] = (sim >= 0.95).sum(1).astype(np.float64)
    feats["kn_sim1"] = np.where(valid.any(1), sim.max(1), np.nan)
    w = np.where(valid, np.exp((sim - 1) / 0.02), 0)
    feats["kn_ctr_w"] = np.where(valid.any(1), (w * np.where(valid, db_coec[Ii], 0)).sum(1) / np.maximum(w.sum(1), 1e-30), np.nan)
    (OUT / split).mkdir(parents=True, exist_ok=True)
    for k, v in feats.items():
        np.save(OUT / split / f"{k}.npy", v[inv].astype(np.float32))
    log(split, {k: float(np.nanmean(v[inv])) for k, v in feats.items()}, "s95 coverage", float(np.isfinite(feats["kn_coec_s95"][inv]).mean()))
