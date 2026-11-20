"""cb26 diagnostic (not a candidate): does member disagreement (top CatBoost residual feature) carry a transferable linear
signal? S = z(pool) + lam * z_feed(sd). lam chosen in nested CV on val (n2cpv3 folds); also evaluated on trH (to-branch)."""
import sys
import numpy as np
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import meta, ens_feats, zf
from metrics import ndcg_per_feed
from cbrule import pse
from sklearn.model_selection import GroupKFold
fv, Mv, ov, yv = meta("val"); fh, Mh, oh, yh = meta("trH")
ev, eh = ens_feats("G3", "val"), ens_feats("G3", "trH")
zv = lambda x: zf(ov, Mv, x); zh = lambda x: zf(oh, Mh, x)
pool = zv(0.5 * zv(ev["ft"]) + 0.5 * zv(ev["to"])); sdv = 0.5 * ev["to_sd"] + 0.5 * ev["ft_sd"]
toh = zh(eh["to"]); sdh = eh["to_sd"]
b = ndcg_per_feed(pool, yv, ov, Mv); bh = ndcg_per_feed(toh, yh, oh, Mh)
grid = [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3]
P = {l: ndcg_per_feed(pool + l * sdv, yv, ov, Mv) - b for l in grid}
H = {l: ndcg_per_feed(toh + l * sdh, yh, oh, Mh) - bh for l in grid}
print("raw sd units: val sd mean %.3f, trH %.3f" % (sdv.mean(), sdh.mean()))
for l in grid:
    m, s = pse(P[l]); hm, hs = pse(H[l]); print(f"lam {l:+.2f}: val {m:+.5f} +- {s:.5f} | trH {hm:+.5f} +- {hs:.5f}")
u = fv["user_id"].to_numpy().astype(np.int64); g = np.where(u == 0, -1 - np.arange(len(u)), u)
nest = np.zeros(len(Mv))
for tr, te in GroupKFold(5).split(np.arange(len(Mv)), groups=g):
    best = max([0.0] + grid, key=lambda l: 0 if l == 0 else P[l][tr].mean())
    nest[te] = 0 if best == 0 else P[best][te]
    print("fold pick", best)
m, s = pse(nest); print(f"nested val delta {m:+.5f} +- {s:.5f}")
