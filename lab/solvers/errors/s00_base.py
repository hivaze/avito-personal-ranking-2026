"""Base arrays: v11 val score, ranks, per-feed NDCG, segments."""
from common_e import *  # noqa: F401,F403
from common_e import meta, v11, ranks, pos_loss, ndcg_per_feed, ROOT, SCR, fz
import numpy as np

f, M, off, fid = meta("val")
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
s = v11("val")
nd = ndcg_per_feed(s, y, off, M)
print("v11 val NDCG", nd.mean(), "n feeds", len(M))
loss, r = pos_loss(s, y, fid, off, M)
print("check loss sum", (loss.sum() / len(M)), 1 - nd.mean())
np.save(f"{SCR}/out/v11_val.npy", s)
np.save(f"{SCR}/out/v11_rank.npy", r)
np.save(f"{SCR}/out/v11_loss.npy", loss)
np.save(f"{SCR}/out/v11_nd.npy", nd)
