"""Per-feed listwise CE / log M (the RHO scoring quantity) and NDCG on VAL and on the 30K random-like train HOLDOUT for
the RHO models (pool-only reference r6_rhoref_pool_s1, F7 @77500 snapshots r101-104), the cooled F7 members and the
m02 re-cools (F7x). CPU, read-only. Writes out/val_feed_losses.parquet, out/holdout_feed_losses.parquet."""
import numpy as np
import polars as pl
from pathlib import Path
import sys

ROOT = Path("<repo>")
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed

OUT = ROOT / "nn_x/insights/data_usage/out"
P = ROOT / "preds"


def feed_ce(s, y, M, off):
    n = len(M)
    fid = np.repeat(np.arange(n), M)
    s = s.astype(np.float64)
    mx = np.full(n, -np.inf); np.maximum.at(mx, fid, s)
    lse = np.log(np.bincount(fid, weights=np.exp(s - mx[fid]), minlength=n)) + mx
    logp = s - lse[fid]
    npos = np.bincount(fid, weights=y, minlength=n)
    ce = -np.bincount(fid, weights=y * logp, minlength=n) / np.maximum(npos, 1)
    return np.where(npos > 0, ce / np.log(np.maximum(M, 2)), np.nan)


roots = [101, 102, 103, 104]
# ---------------- VAL
fv = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet").sort("fidx")
Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
yv = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
models = {"ref": ROOT / "nn_x/r6/preds/nn_r6_rhoref_pool_s1_trainonly_val.npy"}
for r in roots:
    models[f"snap{r}"] = P / f"nn_v17_pre2s_s{r}@77500_trainonly_val.npy"
    models[f"f7_{r}"] = P / f"nn_v17_pre2s_s{r}_trainonly_val.npy"
    models[f"f7x_{r}"] = P / f"nn_r6_cd_m02_r{r}_trainonly_val.npy"
cols = dict(fidx=np.arange(len(Mv)), M=Mv, npos=np.bincount(np.repeat(np.arange(len(Mv)), Mv), weights=yv).astype(np.int32),
            user_id=fv["user_id"].to_numpy(), ts=fv["timestamp"].to_numpy())
for k, f in models.items():
    s = np.load(f)
    cols[f"L_{k}"] = feed_ce(s, yv, Mv, offv)
    cols[f"N_{k}"] = ndcg_per_feed(s, yv, offv, Mv)
    print(k, f"L {np.nanmean(cols[f'L_{k}']):.4f}  ndcg {cols[f'N_{k}'].mean():.5f}", flush=True)
pl.DataFrame(cols).write_parquet(OUT / "val_feed_losses.parquet")

# ---------------- HOLDOUT (random-like train feeds with shown positions)
ft = pl.read_parquet(ROOT / "data/flat/train/feeds.parquet").sort("fidx")
Mt = ft["M"].to_numpy().astype(np.int64); offt = ft["cand_off"].to_numpy().astype(np.int64)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
prand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); prand[Mt < 50] = 0
cand = np.where((prand > 0.9) & (Mt >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
hm = np.load(P / "holdout/holdout_meta.npy"); hy = np.load(P / "holdout/holdout_y.npy").astype(np.float64)
assert (hm[0] == Mt[hold]).all()
idx = np.concatenate([np.arange(offt[f], offt[f] + Mt[f]) for f in hold])
pos = np.load(ROOT / "data/flat/train/cand_pos.npy", mmap_mode="r")[idx].astype(np.int64)
yt = np.load(ROOT / "data/flat/train/cand_target.npy", mmap_mode="r")[idx].astype(np.float64)
assert (yt == hy).all(), "holdout candidate order != flat train order"
Mh, offh = hm[0].astype(np.int64), hm[1].astype(np.int64)
n = len(Mh); fid = np.repeat(np.arange(n), Mh)
order = np.lexsort((pos, fid)); rank = np.empty(len(pos), np.int64); rank[order] = np.arange(len(pos)) - offh[fid[order]]
rkend = Mh[fid] - 1 - rank
yi = hy > 0
r0 = np.full(n, 10 ** 6); np.minimum.at(r0, fid[yi], rank[yi])
e0 = np.full(n, 10 ** 6); np.minimum.at(e0, fid[yi], rkend[yi])
hcols = dict(fidx=hold, M=Mh, npos=np.bincount(fid, weights=hy, minlength=n).astype(np.int32), r0=r0, e0=e0,
             prand=pr["p_random"][hold])
hmods = {}
for r in roots:
    hmods[f"snap{r}"] = P / f"holdout/nn_v17_pre2s_s{r}@77500_holdout.npy"
    hmods[f"f7_{r}"] = P / f"holdout/nn_v17_pre2s_s{r}_holdout.npy"
    hmods[f"f7x_{r}"] = P / f"holdout/nn_r6_cd_m02_r{r}_holdout.npy"
for k, f in hmods.items():
    s = np.load(f)
    hcols[f"L_{k}"] = feed_ce(s, hy, Mh, offh)
    hcols[f"N_{k}"] = ndcg_per_feed(s, hy, offh, Mh)
    # per-feed Spearman-like agreement of the model score with production order (-rank): Pearson of within-feed ranks
    o = np.lexsort((-s, fid)); sr = np.empty(len(s), np.int64); sr[o] = np.arange(len(s)) - offh[fid[o]]
    a = sr - (Mh[fid] - 1) / 2.0; b = rank - (Mh[fid] - 1) / 2.0
    num = np.bincount(fid, weights=a * b, minlength=n); den = np.bincount(fid, weights=a * a, minlength=n)
    hcols[f"rho_{k}"] = num / den  # Spearman(model rank, shown rank); >0 = model agrees with production order
    print("holdout", k, f"L {np.nanmean(hcols[f'L_{k}']):.4f} ndcg {hcols[f'N_{k}'].mean():.5f} "
          f"spearman(score-rank, shown-rank) {np.mean(hcols[f'rho_{k}']):+.4f}", flush=True)
pl.DataFrame(hcols).write_parquet(OUT / "holdout_feed_losses.parquet")
print("done")
