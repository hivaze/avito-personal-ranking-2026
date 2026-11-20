"""Label-free cluster event-type profile from ALL histories (train/val/test; identical for val and test).
etr_{K}.npz: cnt [K, 11] = number of history events of each event_type (1..10) per item cluster."""
import sys
sys.path.insert(0, "<repo>/feats_v2/clu_")
from clu_common import *

for K in IKS:
    cnt = np.zeros((K, 11))
    for split in ["train", "val", "test"]:
        hcl = np.asarray(hist_cl(split, K)).astype(np.int64)
        het = np.load(FLAT / split / "hist_event_type.npy").astype(np.int64)
        ok = hcl >= 0
        cnt += np.bincount(hcl[ok] * 11 + het[ok], minlength=K * 11).reshape(K, 11)
    np.savez(WORK / f"etr_{K}.npz", cnt=cnt)
    log("etr", K, (cnt.sum(0) / cnt.sum()).round(3))
