"""Gap simulation for item co-vis: val features with the train part of the source cut at day CUT (drops the last
train days), so val sees a train->feed gap like test-private. Writes TEMPORARY files seq_gapcv_* (val only),
deleted after the evaluation."""
import sys
import numba as nb
import numpy as np
from seq_lib import *
from sources import source_fidx, train_days
import feat_covis as FC

if __name__ == "__main__":
    nb.set_num_threads(16)
    days = train_days()
    cut = int(days[-3])  # Sep 17 -> source = Sep 6..14 (val gap 8-10 days, like private test)
    log("cut day", cut)
    parts = [("train", source_fidx(cut), False)] + [(s, np.arange(len(feeds(s))), True) for s in ("val", "test")]
    idx = FC.Index(parts)
    n = len(arr("val", "cand_fidx"))
    out = np.zeros((n, len(FC.NAMES)), np.float32)
    FC.run_queries(idx, "val", np.arange(len(feeds("val"))), out)
    for i, nm in enumerate(FC.NAMES):
        save("val", nm.replace("seq_cv_", "seq_gapcv_"), out[:, i])
    log("nonzero invd20 frac", (out[:, 0] > 0).mean(), "pop>0", (out[:, 5] > 0).mean())
