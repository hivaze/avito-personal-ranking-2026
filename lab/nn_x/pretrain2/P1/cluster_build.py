"""P1 stage A: label-free trajectory clusters (see PREDECLARED.txt). CPU only, 12 threads.
Fit sample = 150K train latest snapshots per non-zero user (holdout feeds excluded, seed 0).
Keys: ue = L2-normalised user_emb; hp = history profile (cat shares last 32, event-type shares last 32, log1p H,
recency-weighted content PCA16). k-means K in {8, 32} (n_init 4, random_state 0). Assigns fit / holdout / val / test.
Output: nn_x/pretrain2/P1/out/clusters.npz (labels per set and key/K, feed indices, fit-sample descriptors)."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, json
from pathlib import Path
import numpy as np, polars as pl, numba as nb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; COMP = ROOT / "data/competition"
OUT = ROOT / "nn_x/pretrain2/P1/out"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

NFIT = 150_000; NCAT = 54; NET = 10; L32 = 32; L64 = 64; HALF = 8.0


def holdout_index():
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    Mt = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M"])["M"].to_numpy().astype(np.int64)
    p[Mt < 50] = 0.0
    cand = np.where((p > 0.9) & (Mt >= 50))[0]
    return np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))


def emb_table(split):
    meta = json.load(open(COMP / split / "item_embeddings/data.meta"))
    return np.memmap(COMP / split / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


@nb.njit(parallel=True, cache=True)
def _profile(starts, H, cat, et, cvec, has, out_cat, out_et, out_c):
    """starts: offsets into the gathered arrays (rows hpos 0..min(H,64)-1 per feed, newest first)."""
    for i in nb.prange(len(H)):
        s = starts[i]; h = min(H[i], 64); h32 = min(H[i], 32)
        for k in range(h32):
            out_cat[i, cat[s + k]] += 1.0
            out_et[i, et[s + k] - 1] += 1.0
        if h32 > 0:
            for j in range(out_cat.shape[1]):
                out_cat[i, j] = np.sqrt(out_cat[i, j] / h32)
            for j in range(out_et.shape[1]):
                out_et[i, j] /= h32
        wsum = 0.0
        for k in range(h):
            if has[s + k]:
                w = 0.5 ** (k / 8.0)
                wsum += w
                for j in range(cvec.shape[1]):
                    out_c[i, j] += w * cvec[s + k, j]
        if wsum > 0:
            for j in range(cvec.shape[1]):
                out_c[i, j] /= wsum


def profile(split, feeds):
    """Raw history profile blocks for feeds (indices into split's feeds.parquet)."""
    f = pl.read_parquet(FLAT / split / "feeds.parquet", columns=["H", "hist_off"])
    H = f["H"].to_numpy().astype(np.int64)[feeds]; ho = f["hist_off"].to_numpy().astype(np.int64)[feeds]
    n64 = np.minimum(H, L64)
    starts = np.concatenate([[0], np.cumsum(n64)[:-1]]).astype(np.int64)
    rows = np.repeat(ho - starts, n64) + np.arange(n64.sum())
    order = np.argsort(rows, kind="stable"); rs = rows[order]
    def g(name, dt):
        x = np.load(FLAT / split / f"hist_{name}.npy", mmap_mode="r")
        v = np.empty(len(rows), dt); v[order] = np.asarray(x[rs], dtype=dt); return v
    hp = g("hpos", np.int64)
    exp_hp = np.arange(n64.sum()) - np.repeat(starts, n64)
    assert (hp == exp_hp).all(), "hpos order mismatch"
    cat = g("category_id", np.int64); et = g("event_type", np.int64); ei = g("emb_idx", np.int64)
    assert cat.min() >= 0 and cat.max() < NCAT and et.min() >= 1 and et.max() <= NET
    has = ei >= 0
    tab = emb_table(split)
    cvec = np.zeros((len(ei), 128), np.float32)
    idx = np.nonzero(has)[0]; o2 = np.argsort(ei[idx], kind="stable")
    cvec[idx[o2]] = np.asarray(tab[ei[idx][o2]], dtype=np.float32)
    out_cat = np.zeros((len(feeds), NCAT)); out_et = np.zeros((len(feeds), NET)); out_c = np.zeros((len(feeds), 128))
    _profile(starts, H, cat, et, cvec, has, out_cat, out_et, out_c)
    return dict(cat=out_cat, et=out_et, logH=np.log1p(H)[:, None].astype(np.float64), cont=out_c, H=H)


def uemb(split, feeds):
    u = np.load(FLAT / split / "user_emb.npy", mmap_mode="r")
    o = np.argsort(feeds); x = np.empty((len(feeds), 128), np.float32); x[o] = np.asarray(u[feeds[o]], dtype=np.float32)
    nrm = np.linalg.norm(x, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return x / nrm


def main():
    ft = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["fidx", "user_id", "timestamp"])
    hold = holdout_index()
    assert (ft["fidx"].to_numpy() == np.arange(len(ft))).all()
    hm = np.load(ROOT / "preds/holdout/holdout_meta.npy")
    Mt = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M"])["M"].to_numpy()
    assert (Mt[hold] == hm[0]).all(), "holdout index does not match holdout_meta"
    latest = (ft.filter(pl.col("user_id") != 0).sort(["user_id", "timestamp", "fidx"])
                .group_by("user_id", maintain_order=True).agg(pl.col("fidx").last()))["fidx"].to_numpy()
    latest = np.setdiff1d(latest, hold)
    fit = np.sort(np.random.default_rng(0).choice(latest, NFIT, replace=False))
    log(f"train latest snapshots (non-zero users, holdout excluded): {len(latest):,}; fit sample {len(fit):,}; holdout {len(hold):,}")
    sets = {"fit": ("train", fit), "hold": ("train", hold),
            "val": ("val", np.arange(pl.read_parquet(FLAT / "val/feeds.parquet", columns=["M"]).height)),
            "test": ("test", np.arange(pl.read_parquet(FLAT / "test/feeds.parquet", columns=["M"]).height))}
    UE, HP = {}, {}
    for s, (split, fe) in sets.items():
        UE[s] = uemb(split, fe)
        HP[s] = profile(split, fe)
        log(f"{s}: {len(fe):,} feeds; ue norm-zero {(np.abs(UE[s]).sum(1) == 0).sum()}; H median {np.median(HP[s]['H']):.0f}; "
            f"et1 share last32 {HP[s]['et'][:, 0].mean():.3f}; no-content feeds {(np.abs(HP[s]['cont']).sum(1) == 0).sum()}")
    # hp: PCA16 of content on fit, standardise columns on fit, equal block variance
    pca = PCA(16, random_state=0).fit(HP["fit"]["cont"])
    log(f"content PCA16 explained variance {pca.explained_variance_ratio_.sum():.3f}")
    def blocks(d): return [d["cat"], d["et"], d["logH"], pca.transform(d["cont"])]
    fitb = blocks(HP["fit"])
    mus = [b.mean(0) for b in fitb]; sds = [np.where(b.std(0) > 1e-9, b.std(0), 1.0) for b in fitb]
    scl = [np.sqrt(((((b - m) / s) ** 2).mean(0)).sum()) for b, m, s in zip(fitb, mus, sds)]
    def hpx(d): return np.hstack([((b - m) / s) / c for b, m, s, c in zip(blocks(d), mus, sds, scl)]).astype(np.float32)
    X = {"ue": {s: UE[s] for s in sets}, "hp": {s: hpx(HP[s]) for s in sets}}
    res = {f"feeds_{s}": fe for s, (_, fe) in sets.items()}
    for s in sets:
        res[f"H_{s}"] = HP[s]["H"]; res[f"et1_{s}"] = HP[s]["et"][:, 0]
    info = {}
    for key in ["ue", "hp"]:
        for K in [8, 32]:
            km = KMeans(K, n_init=4, random_state=0, max_iter=300).fit(X[key]["fit"])
            for s in sets:
                lab = km.predict(X[key][s]).astype(np.int16)
                res[f"{key}{K}_{s}"] = lab
            sh = {s: np.round(np.bincount(res[f"{key}{K}_{s}"], minlength=K) / len(res[f"{key}{K}_{s}"]), 4).tolist() for s in sets}
            info[f"{key}{K}"] = dict(inertia=float(km.inertia_), shares=sh)
            log(f"{key} K={K}: inertia {km.inertia_:.4g}; min/max share fit {min(sh['fit']):.4f}/{max(sh['fit']):.4f} "
                f"val {min(sh['val']):.4f}/{max(sh['val']):.4f} hold {min(sh['hold']):.4f}/{max(sh['hold']):.4f}")
            np.save(OUT / f"centroids_{key}{K}.npy", km.cluster_centers_)
    np.savez_compressed(OUT / "clusters.npz", **res)
    json.dump(info, open(OUT / "cluster_info.json", "w"), indent=1)
    log("saved", OUT / "clusters.npz")


if __name__ == "__main__":
    main()
