"""NDCG of z(v11) + delta * flag on val, by day: curvature of the cold/warm calibration optimum."""
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz
import numpy as np

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
z = fz(np.load(f"{SCR}/out/v11_val.npy"), fid, M)
cold = np.load(f"{ROOT}/eda/errors/synthesis/derived/val/cold_item.npy").astype(float)
luo0 = (np.load(f"{ROOT}/data/feats/val/it_hist_all_luo.npy") == 0).astype(float)
ts = f["timestamp"].to_numpy()
day = (ts + 3 * 3600) // 86400
day = np.where(day <= day.min() + 1, 0, 1)
base = ndcg_per_feed(z, y, off, M)
for nm, fl in (("cold", cold), ("luo0", luo0)):
    print(nm)
    for d in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3):
        dd = ndcg_per_feed(z + d * fl, y, off, M) - base
        print(f"  delta {d:+.2f}: all {dd.mean():+.5f}+-{dd.std() / np.sqrt(n):.5f}  Tue {dd[day == 0].mean():+.5f}  Thu {dd[day == 1].mean():+.5f}  Thu-Tue {dd[day == 1].mean() - dd[day == 0].mean():+.5f}")
