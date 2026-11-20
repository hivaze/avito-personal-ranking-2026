"""How much item NDCG does a given category-NDCG gain buy? Partial category oracle on top of the NN:
score = s + k * y_c (y_c = category holds a positive) for small k. Within-category order unchanged.
Also: segment breakdown of NN-implied category ranking vs f3 and vs size prior.
"""
import json
import numpy as np
from catlib import *

v = load_view("val", with_hist=False)
v = build_groups(v)
leak = leak_mask_val()
keep = ~leak
mask = keep & (v["G"] >= 2)
s = feed_z(v["off"], v["M"], np.load(PREDS / "nn_final2_trainonly_val.npy").astype(np.float64))
nd0 = item_ndcg(v, s); c0 = cat_ndcg(v, glse(v, s))
yc = v["g_y"][v["cand_g"]]
rows = []
rng = np.random.default_rng(0)
for k in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    sk = s + k * yc
    ndi = item_ndcg(v, sk); ndc = cat_ndcg(v, glse(v, sk))
    di = paired(ndi, nd0, keep); dc = paired(ndc, c0, mask)
    rows.append(dict(k=k, d_item=di, d_cat=dc, ratio=di[0] / dc[0]))
    print(f"k={k}: d_catNDCG {dc[0]:+.4f}  d_itemNDCG {di[0]:+.4f}  ratio {di[0]/dc[0]:.3f}")
# noisy partial oracle: only a random fraction of feeds get the true category bonus (k=1)
for frac in (0.05, 0.1, 0.2):
    sel = rng.random(len(v["M"])) < frac
    sk = s + 1.0 * yc * sel[np.repeat(np.arange(len(v["M"])), v["M"])]
    ndi = item_ndcg(v, sk); ndc = cat_ndcg(v, glse(v, sk))
    di = paired(ndi, nd0, keep); dc = paired(ndc, c0, mask)
    rows.append(dict(frac=frac, d_item=di, d_cat=dc, ratio=di[0] / dc[0]))
    print(f"frac={frac}: d_catNDCG {dc[0]:+.4f}  d_itemNDCG {di[0]:+.4f}  ratio {di[0]/dc[0]:.3f}")
json.dump(rows, open(OUT / "05_sensitivity.json", "w"), indent=1)
