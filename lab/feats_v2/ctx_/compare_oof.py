"""Paired comparison of two saved eval_ens OOF sets (same seeds/folds): per-seed and seed-ensemble delta B - A.
Usage: python compare_oof.py tagA tagB [n_seeds]
"""
import sys
sys.path.insert(0, "<repo>/src")
sys.path.insert(0, "<repo>/feats_v2/ctx_")
from common import *
from eval_ens import fz, OOF

a, b = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
fidx = np.load(FLAT / "val/cand_fidx.npy")
za, zb, ds = [], [], []
for s in range(n):
    oa, ob = np.load(OOF / f"{a}_s{s}.npy"), np.load(OOF / f"{b}_s{s}.npy")
    d = ndcg_per_feed(ob, y, off, M) - ndcg_per_feed(oa, y, off, M)
    ds.append(d)
    za.append(fz(oa, fidx, M)); zb.append(fz(ob, fidx, M))
    print(f"seed {s}: {b} - {a} = {d.mean():+.5f}")
d = np.mean(ds, 0)
print(f"mean single: {d.mean():+.5f} se {d.std()/np.sqrt(len(d)):.5f}")
de = ndcg_per_feed(np.mean(zb, 0), y, off, M) - ndcg_per_feed(np.mean(za, 0), y, off, M)
print(f"ensemble: {de.mean():+.5f} se {de.std()/np.sqrt(len(de)):.5f}")
