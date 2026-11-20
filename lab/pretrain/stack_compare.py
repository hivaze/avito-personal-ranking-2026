"""Paired comparison of two prestk_* stacker OOFs (and of the 0.5/0.5 blend with nn_final2_val_oof), overall and clean.
Usage: python pretrain/stack_compare.py res4_cpu res4_cpu_pre_v3pair"""
import sys
import numpy as np
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/pretrain")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from evaluate import clean_mask, meta, PREDS, FLAT

a, b = sys.argv[1], sys.argv[2]
f, M, off = meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
c = clean_mask()
z = lambda n: feed_rank_z(off, M, np.load(PREDS / f"{n}.npy").astype(np.float32))[:, 1]
nn = z("nn_final2_val_oof")
for name, xa, xb in [("stacker alone", z(f"prestk_{a}_val"), z(f"prestk_{b}_val")),
                     ("0.5 z(nn_final2) + 0.5 z(stacker)", 0.5 * nn + 0.5 * z(f"prestk_{a}_val"), 0.5 * nn + 0.5 * z(f"prestk_{b}_val"))]:
    pa, pb = ndcg_per_feed(xa, y, off, M), ndcg_per_feed(xb, y, off, M)
    d = pb - pa
    print(f"{name:36s} {a} {pa.mean():.4f} -> {b} {pb.mean():.4f}: delta {d.mean():+.5f} (SE {d.std()/np.sqrt(len(d)):.5f}); "
          f"clean {d[c].mean():+.5f} (SE {d[c].std()/np.sqrt(c.sum()):.5f})")
