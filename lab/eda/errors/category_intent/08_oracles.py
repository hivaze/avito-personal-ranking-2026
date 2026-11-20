"""Where is the headroom: across categories or within? Two oracles on top of the NN (val, leak feeds excluded).
- category oracle: positives' categories first, NN order inside (changes only the category marginal).
- within-category oracle: inside each (feed, category) cell, the cell's own NN score values are re-assigned so that
  positives get the highest ones (category score multiset, hence the NN's category marginal, is unchanged).
Also: complementarity of NN-LSE and the category models at category level (per-feed max = oracle-of-two).
"""
import json
import numpy as np
from catlib import *

v = load_view("val", with_hist=False)
v = build_groups(v)
leak = leak_mask_val(); keep = ~leak; mask = keep & (v["G"] >= 2)
out = {}
for bname, fn in [("nn_trainonly", "nn_final2_trainonly_val"), ("nn_oof", "nn_final2_val_oof")]:
    s = feed_z(v["off"], v["M"], np.load(PREDS / f"{fn}.npy").astype(np.float64))
    y = v["c_target"]; g = v["cand_g"]
    nd0 = item_ndcg(v, s)
    ndc = item_ndcg(v, s + 100.0 * v["g_y"][g])
    o1 = np.lexsort((-s, g)); o2 = np.lexsort((-s, -y, g))
    sw = np.empty_like(s); sw[o2] = s[o1]
    ndw = item_ndcg(v, sw)
    assert np.allclose(glse(v, sw), glse(v, s))
    out[bname] = dict(base=mse(nd0, keep), cat_oracle=mse(ndc, keep), within_oracle=mse(ndw, keep),
                      both=mse(item_ndcg(v, sw + 100.0 * v["g_y"][g]), keep))
    print(bname, {k: round(x[0], 4) for k, x in out[bname].items()})
# complementarity at category level
ref = np.load(OUT / "01_val_catndcg_perfeed.npz")
nn = ref["nn_trainonly_LSE_T1.0"]
for tag in ["A", "B", "S1"]:
    gp = np.load(OUT / f"cat_{tag}_val_grp.npy").astype(np.float64)
    nd = cat_ndcg(v, gp)
    r = float(np.corrcoef(nd[mask], nn[mask])[0, 1])
    o2 = np.maximum(nd, nn)
    out[f"compl_{tag}"] = dict(catndcg=mse(nd, mask), corr_perfeed_with_nn=r, oracle_of_two=mse(o2, mask),
                               share_feeds_model_better=float((nd > nn + 1e-9)[mask].mean()),
                               share_feeds_nn_better=float((nn > nd + 1e-9)[mask].mean()))
    print(tag, out[f"compl_{tag}"])
json.dump(out, open(OUT / "08_oracles.json", "w"), indent=1)
