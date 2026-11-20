"""Guest-only (user_id 0) affinity tilts: s = g + w * u0 * z_feed(x); w per outer fold (user-grouped 5-fold), holdout at full-val w.
Deltas are reported on guest feeds and on all feeds."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, fz, fpct, folds5, ndcg
dv = load("val"); dh = load("hold"); fold = folds5(dv, 0)
W = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
se = lambda x: x.std() / np.sqrt(len(x))
bv = ndcg(dv, dv["g"]); bh = ndcg(dh, dh["g"])
def comp(d):
    return np.mean([fpct(d, d[c]) for c in ["f3", "d_fcs", "d_cat_last32", "xt_ctx_nd_knn_f1", "d_cos_u"]], 0)
for n in ["f3", "d_fcs", "d_cat_last32", "xt_ctx_nd_knn_f1", "d_cos_u", "COMP"]:
    xv = fz(dv, comp(dv) if n == "COMP" else dv[n]) * dv["u0"][dv["fid"]]
    xh = fz(dh, comp(dh) if n == "COMP" else dh[n]) * dh["u0"][dh["fid"]]
    per = np.stack([ndcg(dv, dv["g"] + w * xv) for w in W])
    out = np.empty(dv["nf"]); ws = []
    for k in range(5):
        sel = fold == k; j = int(np.argmax(per[:, ~sel].mean(1))); ws.append(W[j]); out[sel] = per[j, sel]
    dl = out - bv; g0 = dv["u0"]
    wf = W[int(np.argmax(per.mean(1)))]
    hd = ndcg(dh, dh["g"] + wf * xh) - bh; h0 = dh["u0"]
    print(f"{n:18s} OOF all {dl.mean():+.5f}+-{se(dl):.5f} | guests {dl[g0].mean():+.5f}+-{se(dl[g0]):.5f} (n {g0.sum()}) | fold w {ws} | "
          f"HOLD (w {wf}) all {hd.mean():+.5f}+-{se(hd):.5f} guests {hd[h0].mean():+.5f}+-{se(hd[h0]):.5f}")
