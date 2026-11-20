"""Where is the NDCG gap located in rank space? Loss share by positive rank bucket, and 'move-up' oracles."""
from common_e import meta, ndcg_per_feed, ROOT, SCR
import numpy as np

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
s = np.load(f"{SCR}/out/v11_val.npy")
r = np.load(f"{SCR}/out/v11_rank.npy")
loss = np.load(f"{SCR}/out/v11_loss.npy")
nd = np.load(f"{SCR}/out/v11_nd.npy")
pos = y > 0
tot = loss.sum()
edges = [0, 1, 3, 10, 30, 100, 10 ** 6]
print("positive rank bucket: share of positives / share of gap / mean loss per positive")
for a, b in zip(edges[:-1], edges[1:]):
    m = pos & (r >= a) & (r < b)
    print(f"  rank [{a},{b}): {m.sum() / pos.sum():.3f} / {loss[m].sum() / tot:.3f} / {loss[m].mean() if m.any() else 0:.3f}")
# oracle: lift every positive ranked in [a,b) by d ranks (swap above negatives) -> approximate by adding to score
pp = np.where(pos)[0]
for k in (1, 3, 10):
    # oracle: positives at rank >= k gain exactly enough to move to rank max(r - k, 0)?  emulate: subtract k from rank of positives
    s2 = s.copy()
    # move each positive up k ranks: set its score to the score of the item k ranks above (+eps)
    order = np.lexsort((r, fid))
    idx_at = order  # row at (feed offset + rank)
    tgt = np.maximum(r[pp] - k, 0)
    s2[pp] = s[idx_at[off[fid[pp]] + tgt]] + 1e-6
    print(f"oracle: every positive moved up {k} ranks -> NDCG {ndcg_per_feed(s2, y, off, M).mean():.5f} (v11 {nd.mean():.5f})")
# oracle: fix only positives in the bottom half (move them to the median rank)
pct = r / np.maximum(M[fid] - 1, 1)
for thr in (0.5, 0.25):
    s2 = s.copy()
    sel = pp[pct[pp] >= thr]
    order = np.lexsort((r, fid))
    tgt = (thr * (M[fid[sel]] - 1)).astype(int) - 1
    tgt = np.maximum(tgt, 0)
    s2[sel] = s[order[off[fid[sel]] + tgt]] + 1e-6
    print(f"oracle: positives with pct >= {thr} moved to pct {thr} -> {ndcg_per_feed(s2, y, off, M).mean():.5f}")
# oracle: perfect top-1 for 1-positive feeds where the positive is already in top-3/top-10
npos = np.bincount(fid, y, n)
for kk in (3, 10):
    s2 = s.copy()
    sel = pp[(r[pp] < kk)]
    s2[sel] = 1e6
    print(f"oracle: positives already in top-{kk} moved to rank 0 -> {ndcg_per_feed(s2, y, off, M).mean():.5f}")
