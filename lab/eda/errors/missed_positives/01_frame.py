"""Build the per-candidate analysis frame for the MISSED POSITIVES lens.

Best model = v5 blend: 0.5 z(nn_final2_val_oof) + 0.5 z(stk2_res2_r3)  (val 0.3869, public 0.3802).
Also keeps the clean train-only NN (nn_final2_trainonly) for leak-free checks.
Outputs frame.npz with: rank (pessimistic, 0-based), relative rank, group labels, leak mask (per feed and per cand).
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
sys.path.insert(0, "<repo>/src")
from common import *
from stack_val import feed_rank_z

OUT = Path("<repo>/eda/errors/missed_positives")
PREDS = Path("<repo>/preds")


def zf(off, M, name):
    x = np.load(PREDS / f"{name}.npy").astype(np.float32)
    return feed_rank_z(off, M, x)[:, 1].astype(np.float64)


def ranks(score, y, off, M):
    n = len(M)
    fid = np.repeat(np.arange(n), M)
    order = np.lexsort((y, -score, fid))  # pessimistic: negatives first on ties
    r = np.empty(len(score), np.int64)
    r[order] = np.arange(len(order)) - np.repeat(off, M)
    return r


def leak_mask_feeds():
    fv = pl.read_parquet(FLAT / "val/feeds.parquet").select("fidx", "user_id", "timestamp")
    ft = pl.read_parquet(FLAT / "test/feeds.parquet").select("user_id", "timestamp")
    allf = pl.concat([fv.select("user_id", "timestamp"), ft]).rename({"timestamp": "ts2"})
    j = fv.filter(pl.col("user_id") != 0).join(allf, on="user_id").filter(pl.col("ts2") > pl.col("timestamp"))
    leak = np.zeros(len(fv), bool)
    leak[j["fidx"].unique().to_numpy()] = True
    return leak


def main():
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    z_oof = zf(off, M, "nn_final2_val_oof")
    z_stk = zf(off, M, "stk2_res2_r3_val")
    z_clean = zf(off, M, "nn_final2_trainonly_val")
    best = 0.5 * z_oof + 0.5 * z_stk
    leakf = leak_mask_feeds()
    fid = np.repeat(np.arange(len(M)), M)
    leakc = leakf[fid]
    npos = np.bincount(fid, weights=y, minlength=len(M))
    for nm, s in [("nn_final2_val_oof", z_oof), ("stk2_res2_r3", z_stk), ("nn_final2_trainonly", z_clean), ("v5 blend", best)]:
        per = ndcg_per_feed(s, y, off, M)
        print(f"{nm:22s} all {per.mean():.4f}  leak-free {per[~leakf].mean():.4f}  leak feeds {per[leakf].mean():.4f}")
    print("leak feeds:", leakf.sum(), " feeds with positives:", (npos > 0).sum(), " leak-free with pos:", ((npos > 0) & ~leakf).sum())
    r = ranks(best, y, off, M)
    rc = ranks(z_clean, y, off, M)
    Mc = M[fid]
    q = r / np.maximum(Mc - 1, 1)
    # groups (leak-free feeds only)
    pos = y > 0
    MP = pos & (r >= Mc / 2) & ~leakc
    TP3 = pos & (r < 3) & ~leakc
    FA3 = (~pos) & (r < 3) & ~leakc & (npos[fid] > 0)
    print(f"positives (leak-free): {(pos & ~leakc).sum()}  MP(bottom half): {MP.sum()} ({MP.sum() / (pos & ~leakc).sum():.3f})"
          f"  TP3: {TP3.sum()} ({TP3.sum() / (pos & ~leakc).sum():.3f})  FA3 (in feeds with pos): {FA3.sum()}")
    # distribution of positive relative rank
    qp = q[pos & ~leakc]
    print("positive relative-rank deciles:", np.histogram(qp, bins=10, range=(0, 1))[0])
    # contribution of MP positives to the NDCG loss: per-feed NDCG for feeds whose only positive(s) are missed
    per = ndcg_per_feed(best, y, off, M)
    mp_feed = np.bincount(fid, weights=MP.astype(float), minlength=len(M)) > 0
    lf = ~leakf & (npos > 0)
    print(f"feeds with >=1 MP: {(mp_feed & lf).sum()} / {lf.sum()}, mean NDCG there {per[mp_feed & lf].mean():.4f} vs others {per[~mp_feed & lf].mean():.4f}")
    loss = 1 - per
    print(f"share of total (1-NDCG) mass (leak-free feeds with pos) in MP feeds: {loss[mp_feed & lf].sum() / loss[lf].sum():.3f}")
    np.savez(OUT / "frame.npz", y=y, best=best, z_oof=z_oof, z_stk=z_stk, z_clean=z_clean, r=r, rc=rc, q=q, fid=fid,
             M=M, off=off, leakf=leakf, MP=MP, TP3=TP3, FA3=FA3, npos=npos)


if __name__ == "__main__":
    main()
