"""Harness checks: noise column (null), f1 (already in the model), planted label signal (positive control)."""
import sys; sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import *
from dc_nest import Nested, fmt, lgbr
v = Split("val", hist=False); s = v.best()
log("base", v.ndcg(s).mean())
N = Nested(v, s)
P0, L0, B0 = N.control(); log("control done; mean beta", B0.mean(0), "ctrl ndcg", P0.mean(), "ll/click", L0.sum() / N.npos.sum())
rng = np.random.default_rng(7)
print(fmt(N.read("noise", [rng.standard_normal(v.N)])))
print(fmt(N.read("f1", [np.load(f"{FLAT}/val/cand_f1.npy")])))
# planted: a 0/1 flag that marks 2% of positives and 0.03% of negatives (a label echo on ~1% of feeds)
y = v.y; pl_ = ((y > 0) & (rng.random(v.N) < 0.02)) | ((y == 0) & (rng.random(v.N) < 0.0003))
print(fmt(N.read("planted 2% of positives", [pl_.astype(float)])))
log("lgbr f1", lgbr(N, [np.load(f"{FLAT}/val/cand_f1.npy")], "f1", seeds=(0,)))
log("done")
