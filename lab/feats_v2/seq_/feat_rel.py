"""Within-feed relative versions (pct-rank, z-score) of the strongest seq_ features (src/features.py::_rel_feats).
Usage: python feat_rel.py <splits> [threads]"""
import sys
import numba as nb
import numpy as np
from seq_lib import *

sys.path.insert(0, "<repo>/src")
from features import _rel_feats  # noqa: E402

# (exploration also had seq_icv_lift4096 and seq_tr_c4096_lift here; those groups were dropped)
COLS = ["seq_max_all", "seq_dcos_hl2", "seq_dcos_t1_hl5", "seq_cv_invd20"]
OUT = {"seq_max_all": ("rk", "z"), "seq_dcos_hl2": ("rk",), "seq_dcos_t1_hl5": ("z",), "seq_cv_invd20": ("rk",)}


def run(split):
    f = feeds(split)
    M = f["M"].to_numpy().astype(np.int64); co = f["cand_off"].to_numpy()
    x = np.stack([np.load(FEATS / split / f"{c}.npy") for c in COLS], axis=1)
    out = np.empty((len(x), 3 * len(COLS)), np.float32)
    _rel_feats(co, M, x, out)
    for i, c in enumerate(COLS):
        if "rk" in OUT[c]:
            save(split, f"{c}_rk", out[:, 3 * i])
        if "z" in OUT[c]:
            save(split, f"{c}_z", out[:, 3 * i + 1])
    log(split, "rel done")


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    for s in sys.argv[1].split(","):
        run(s)
