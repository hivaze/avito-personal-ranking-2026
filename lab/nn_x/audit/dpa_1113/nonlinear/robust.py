"""Robustness of the (b) survivors and near-misses: 5 outer-fold seeds, pct vs z form, TO-only / FT-only halves of G3,
joint 2-feature grid, public- and private-mix weights, holdout at the full-val weight."""
import sys, itertools
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, fz, fpct, folds5, ndcg, wmean_se, pmix_w, D  # noqa

dv = load("val"); dh = load("hold")
W = [-0.1, -0.07, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
PWm = pmix_w(); PW, PUB = PWm["private"], PWm["public"]
se = lambda x: x.std() / np.sqrt(len(x))
cands = {"rel_ctx_nd_knn_h": "xt_ctx_nd_knn_h", "rel_ctx_nd_knn_u": "xt_ctx_nd_knn_u", "rel_it_hist_et1_luo": "xt_it_hist_et1_luo",
         "rel_seq_cv_pop": "xt_seq_cv_pop", "rel_ctx_nd_hcent": "xt_ctx_nd_hcent"}


def nested(gname, xv, folds_seed):
    g = dv[gname]; base = ndcg(dv, g)
    per = np.stack([ndcg(dv, g + w * xv) for w in W])
    fold = folds5(dv, seed=folds_seed)
    out = np.empty(dv["nf"]); ws = []
    for k in range(5):
        sel = fold == k
        wb = int(np.argmax(per[:, ~sel].mean(1))); ws.append(W[wb]); out[sel] = per[wb, sel]
    return out - base, ws, W[int(np.argmax(per.mean(1)))]


for name, col in cands.items():
    for form in ["z", "pct"]:
        xv = fz(dv, dv[col]) if form == "z" else fz(dv, fpct(dv, dv[col]))
        xh = fz(dh, dh[col]) if form == "z" else fz(dh, fpct(dh, dh[col]))
        res = [nested("g", xv, s) for s in range(5)]
        dls = [r[0].mean() for r in res]
        dl0, ws0, wf = res[0]
        pm, ps = wmean_se(dl0, PW); pu, pus = wmean_se(dl0, PUB)
        hd = ndcg(dh, dh["g"] + wf * xh) - ndcg(dh, dh["g"])
        to = nested("gto", xv, 0)[0]; ft = nested("gft", xv, 0)[0]
        print(f"{name:22s} {form:3s} | OOF over 5 fold seeds {np.round(dls, 5).tolist()} mean {np.mean(dls):+.5f} (SE~{se(dl0):.5f}) | "
              f"w_full {wf} | pmix priv {pm:+.5f}+-{ps:.5f} pub {pu:+.5f}+-{pus:.5f} | TO-half {to.mean():+.5f}+-{se(to):.5f} "
              f"FT-half {ft.mean():+.5f}+-{se(ft):.5f} | HOLD {hd.mean():+.5f}+-{se(hd):.5f}", flush=True)
# joint knn_h + knn_u (2-D grid, nested)
xa = fz(dv, dv["xt_ctx_nd_knn_h"]); xb = fz(dv, dv["xt_ctx_nd_knn_u"])
ha = fz(dh, dh["xt_ctx_nd_knn_h"]); hb = fz(dh, dh["xt_ctx_nd_knn_u"])
G = [-0.03, 0, 0.02, 0.03, 0.05, 0.07]
pairs = list(itertools.product(G, G))
base = ndcg(dv, dv["g"])
per = np.stack([ndcg(dv, dv["g"] + a * xa + b * xb) for a, b in pairs])
fold = folds5(dv, seed=0); out = np.empty(dv["nf"]); ws = []
for k in range(5):
    sel = fold == k; j = int(np.argmax(per[:, ~sel].mean(1))); ws.append(pairs[j]); out[sel] = per[j, sel]
dl = out - base; jf = pairs[int(np.argmax(per.mean(1)))]
hd = ndcg(dh, dh["g"] + jf[0] * ha + jf[1] * hb) - ndcg(dh, dh["g"])
print(f"JOINT knn_h+knn_u: OOF {dl.mean():+.5f}+-{se(dl):.5f} (z {dl.mean()/se(dl):+.2f}) fold w {ws} | full w {jf} | HOLD {hd.mean():+.5f}+-{se(hd):.5f}")
print(f"corr(z knn_h, z knn_u) val {np.corrcoef(xa, xb)[0,1]:.3f}; corr with G3 z: knn_h {np.corrcoef(xa, dv['g'])[0,1]:.3f}, knn_u {np.corrcoef(xb, dv['g'])[0,1]:.3f}")
