"""P2 read-out: paired per-feed val deltas between pilot arms (14,423 val feeds), plus rho (mean within-feed Pearson) to the W16 train-only pool.
usage: p2_eval.py [n_feeds=4000]"""
import os, sys, json, glob
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import meta
from metrics import ndcg_per_feed
n = sys.argv[1] if len(sys.argv) > 1 else "4000"
HERE = os.path.dirname(__file__); C = os.path.join(HERE, "..", "cache")
fv, Mv, ov, yv = meta("val")
P = {}
for f in sorted(glob.glob(f"{C}/p2_*_n{n}_val.npy")):
    k = os.path.basename(f)[3:].replace(f"_n{n}_val.npy", "")
    P[k] = ndcg_per_feed(np.load(f), yv, ov, Mv)
out = {"levels": {k: [float(v.mean()), float(v.std() / np.sqrt(len(v))), int(len(v))] for k, v in P.items()}}
print("arm            val NDCG   SE      n")
for k, v in out["levels"].items():
    print(f"{k:14s} {v[0]:.5f} {v[1]:.5f} {v[2]}")
def d(a, b):
    if a in P and b in P:
        x = P[a] - P[b]; m, s = float(x.mean()), float(x.std() / np.sqrt(len(x)))
        out.setdefault("paired", {})[f"{a} - {b}"] = [m, s, m / s]
        print(f"{a:14s} - {b:12s} {m:+.5f} +- {s:.5f} (z {m / s:+.2f}, n={len(x)})")
print("-- seed noise (same arm, seed 1 - seed 0) --")
for a in ["base_Y", "base_L", "base_S", "handlda_Y", "lda_L", "knn50_S", "emb_L"]:
    d(f"{a}_s1", f"{a}_s0")
print("-- arm - its baseline, per seed --")
for s in ["s0", "s1"]:
    for a, b in [("handlda_Y", "base_Y"), ("emb_Y", "base_Y"), ("knn50_Y", "base_Y"), ("lda_L", "base_L"), ("emb_L", "base_L"), ("knn50_L", "base_L"), ("knn50_S", "base_S"), ("emb_S", "base_S"),
                 ("base_S", "base_Y"), ("base_L", "base_Y")]:
        d(f"{a}_{s}", f"{b}_{s}")
print("-- 2-seed means (per-feed mean of the two seeds' NDCG) --")
for a, b in [("handlda_Y", "base_Y"), ("lda_L", "base_L"), ("emb_L", "base_L"), ("knn50_S", "base_S")]:
    if all(f"{x}_{s}" in P for x in (a, b) for s in ("s0", "s1")):
        x = 0.5 * (P[f"{a}_s0"] + P[f"{a}_s1"]) - 0.5 * (P[f"{b}_s0"] + P[f"{b}_s1"]); m, s = float(x.mean()), float(x.std() / np.sqrt(len(x)))
        out.setdefault("paired_2seed", {})[f"{a} - {b}"] = [m, s, m / s]
        print(f"{a:14s} - {b:12s} {m:+.5f} +- {s:.5f} (z {m / s:+.2f})  [mean over seeds 0,1]")
json.dump(out, open(os.path.join(HERE, "..", "logs", f"p2_eval_n{n}.json"), "w"), indent=1)
