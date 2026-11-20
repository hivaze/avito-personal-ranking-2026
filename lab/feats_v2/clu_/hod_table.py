"""Label-free hour-of-day (MSK) cluster demand profile from ALL splits' histories: count of recent history events
(hpos < 5, i.e. events just before the request) per (feed hour, item cluster). hod_{K}.npz: dem [24, K]."""
import sys
sys.path.insert(0, "<repo>/feats_v2/clu_")
from clu_common import *

for K in (256, 1024):
    dem = np.zeros((24, K))
    for split in ["train", "val", "test"]:
        fe = feeds(split)
        hr = ((fe["ts"] + 3 * 3600) // 3600) % 24
        hcl = np.asarray(hist_cl(split, K)).astype(np.int64)
        hf = np.load(FLAT / split / "hist_fidx.npy")
        hp = np.load(FLAT / split / "hist_hpos.npy")
        ok = (hp < 5) & (hcl >= 0)
        dem += np.bincount(hr[hf[ok]] * K + hcl[ok], minlength=24 * K).reshape(24, K)
    np.savez(WORK / f"hod_{K}.npz", dem=dem)
    log("hod", K, dem.sum(1).astype(int))
