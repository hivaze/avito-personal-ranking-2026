"""Feature cache for the CPU GBDT data-usage proxy (read-only on inputs; writes out/gb/*.npy).

Train sample (fixed, seed 7): U = 150K feeds uniform over the NN's training feeds (holdout excluded; natural mix),
X = 30K extra pool feeds (p_random > 0.5, M >= 50, not in U, not holdout) for exploration-weighting arms,
HO = 10K of the 30K holdout feeds (train-period random-like, M >= 50) as a second readout.
Per train row: label, within-feed rank of the shown position, rank-from-end, M; per feed: cls, excess (r101..r104 mean;
NaN for non-production), r0, pos_last.
Features: leak-free, NN-like inputs (content, history similarity, history category counts, co-visitation, LUO
popularity, ctx neighbour features); no pre*/prew*, no train-label TE (it_tr_*, tte_*, u_tr_*), no split-count features.
"""
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path("<repo>")
OUT = ROOT / "nn_x/insights/data_usage/out/gb"
OUT.mkdir(parents=True, exist_ok=True)
FEATS = ["f1", "f2", "f3", "f4", "f5", "f1_z", "f3_z", "f4_z", "f5_z", "f1_rk", "f5_rk", "cos_u", "cos_u_z", "dot_u",
         "cos_hall", "cos_hall_z", "cos_h10", "maxsim32", "meansim32", "attn32", "cat_cnt_all", "cat_cnt_20", "cat_cnt_5",
         "cat_share_all", "cat_min_hpos", "cat_cnt_et1", "seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo",
         "it_hist_all_luo_z", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo", "ctx_nd_max",
         "ctx_nd_knn_h", "ctx_nd_knn_u", "ctx_nd_top5h", "ctx_nd_hcent", "ctx_hn_rrw", "ctx_hn_top1w", "ctx_hs_k128",
         "ctx_hs_mutual", "seq_max_all", "seq_cos_last1", "seq_top3", "item_norm", "emb_missing", "M", "H",
         "feed_cat_share", "hist_n_uniq_cat"]
t = pl.read_parquet(ROOT / "nn_x/insights/data_usage/out/train_feeds.parquet")
M = t["M"].to_numpy().astype(np.int64); cls = t["cls"].to_numpy()
off = np.concatenate([[0], np.cumsum(M)[:-1]])
rng = np.random.default_rng(7)
train_ids = np.where(cls != 2)[0]
U = np.sort(rng.choice(train_ids, 150000, replace=False))
poolc = np.setdiff1d(np.where(cls == 1)[0], U)
X = np.sort(rng.choice(poolc, 30000, replace=False))
HO = np.sort(rng.choice(np.where(cls == 2)[0], 10000, replace=False))
# excess (mean of 4 roots) for production feeds
RHO = ROOT / "nn_x/r6/rho"
prod = np.load(RHO / "prod_fidx.npy")
L_ref = np.load(RHO / "L_ref_r6_rhoref_pool_s1.npy")
exc_all = np.full(len(M), np.nan)
exc_all[prod] = np.mean([np.load(RHO / f"L_snap_r{r}.npy") for r in (101, 102, 103, 104)], 0) - L_ref
exc101 = np.full(len(M), np.nan); exc101[prod] = np.load(RHO / "L_snap_r101.npy") - L_ref
pos_all = np.load(ROOT / "data/flat/train/cand_pos.npy", mmap_mode="r")
y_all = np.load(ROOT / "data/flat/train/cand_target.npy", mmap_mode="r")
for name, ids in [("U", U), ("X", X), ("HO", HO)]:
    rows = np.concatenate([np.arange(off[f], off[f] + M[f]) for f in ids])
    g = M[ids]
    fid = np.repeat(np.arange(len(ids)), g)
    goff = np.concatenate([[0], np.cumsum(g)[:-1]])
    pos = np.asarray(pos_all[rows]).astype(np.int64)
    o = np.lexsort((pos, fid)); rank = np.empty(len(rows), np.int64); rank[o] = np.arange(len(rows)) - goff[fid[o]]
    rkend = g[fid] - 1 - rank
    Xm = np.empty((len(rows), len(FEATS)), np.float32)
    for j, fn in enumerate(FEATS):
        Xm[:, j] = np.load(ROOT / f"data/feats/train/{fn}.npy", mmap_mode="r")[rows]
    np.save(OUT / f"{name}_X.npy", Xm)
    np.savez(OUT / f"{name}_meta.npz", fidx=ids, M=g, y=np.asarray(y_all[rows]).astype(np.float32), rank=rank.astype(np.int16),
             rkend=rkend.astype(np.int16), cls=cls[ids], excess=exc_all[ids], excess101=exc101[ids],
             r0=t["r0"].to_numpy()[ids], pos_last=t["pos_last"].to_numpy()[ids], day=t["day"].to_numpy()[ids],
             coldpos=t["coldpos"].to_numpy()[ids], prand=t["prand_nn"].to_numpy()[ids])
    print(name, len(ids), Xm.shape, flush=True)
fv = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet").sort("fidx")
Xv = np.empty((int(fv["M"].sum()), len(FEATS)), np.float32)
for j, fn in enumerate(FEATS):
    Xv[:, j] = np.load(ROOT / f"data/feats/val/{fn}.npy")
np.save(OUT / "val_X.npy", Xv)
np.savez(OUT / "val_meta.npz", M=fv["M"].to_numpy(), y=np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float32),
         user_id=fv["user_id"].to_numpy())
print("val", Xv.shape)
import json
json.dump(FEATS, open(OUT / "feats.json", "w"))
