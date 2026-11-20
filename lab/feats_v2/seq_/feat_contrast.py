"""Explicit contrast features (differences that trees approximate poorly from separate columns).
  seq_drift_hl2_all  = seq_dcos_hl2 - cos_hall       recent interest vs whole-history interest (trajectory)
  seq_max_new_old    = seq_max_new8 - seq_max_old     best match in the last 8 events vs in events with hpos >= 32
  seq_dcos_s1_s2     = seq_dcos_s1 - seq_dcos_s2      feed-click (source 1) profile vs source-2 profile
  seq_u_minus_hist   = cos_u - seq_dcos_hl5           real-time user vector vs recent history profile
Usage: python feat_contrast.py <splits>"""
import sys
import numpy as np
from seq_lib import *

DEFS = {"seq_drift_hl2_all": ("seq_dcos_hl2", "cos_hall"), "seq_max_new_old": ("seq_max_new8", "seq_max_old"),
        "seq_dcos_s1_s2": ("seq_dcos_s1", "seq_dcos_s2"), "seq_u_minus_hist": ("cos_u", "seq_dcos_hl5")}

if __name__ == "__main__":
    for s in sys.argv[1].split(","):
        for nm, (a, b) in DEFS.items():
            x = np.load(FEATS / s / f"{a}.npy", mmap_mode="r")
            y = np.load(FEATS / s / f"{b}.npy", mmap_mode="r")
            out = np.empty(len(x), np.float32)
            step = 50_000_000
            for i in range(0, len(x), step):
                out[i:i + step] = x[i:i + step] - y[i:i + step]
            save(s, nm, out)
        log(s, "contrast done")
