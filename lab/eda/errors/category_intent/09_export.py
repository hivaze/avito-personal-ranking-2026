"""Export per-candidate category-intent log-scores (aligned with data/flat/<split>/cand_*.npy) for optional stacker use."""
import numpy as np
from catlib import *
for split in ["val", "test"]:
    g = np.load(OUT / f"grp_{split}.npz", allow_pickle=True)
    cg = g["cand_g"]
    for tag in ["A", "S1"]:
        gp = np.load(OUT / f"cat_{tag}_{split}_grp.npy").astype(np.float64)
        lq = -np.log1p(np.exp(-gp)) if tag == "A" else gp
        np.save(OUT / f"catint_{tag}_{split}.npy", lq[cg].astype(np.float32))
        print(split, tag, len(cg))
