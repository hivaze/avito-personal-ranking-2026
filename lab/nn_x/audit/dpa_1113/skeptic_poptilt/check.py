"""Skeptic re-run of P4 POP-TILT: s = g + w * z_feed(xt_seq_cv_pop). Fixed-w deltas on val (g, gto, gft), holdout (TO pool),
user-grouped nested 5-fold over 10 fold seeds, cold/warm split (LUO-zero share proxy via seq_cv_pop==min), pmix."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, fz, folds5, ndcg, wmean_se, pmix_w
dv = load("val"); dh = load("hold")
se = lambda x: x.std() / np.sqrt(len(x))
xv = fz(dv, dv["xt_seq_cv_pop"]); xh = fz(dh, dh["xt_seq_cv_pop"])
PW = pmix_w(); PRIV, PUB = PW["private"], PW["public"]
print("corr(g, z pop) val", np.corrcoef(dv["g"], xv)[0, 1], "hold", np.corrcoef(dh["g"], xh)[0, 1])
bases = {k: ndcg(dv, dv[k]) for k in ["g", "gto", "gft"]}
bh = ndcg(dh, dh["g"])
print("base val g %.5f gto %.5f gft %.5f | hold %.5f" % (bases["g"].mean(), bases["gto"].mean(), bases["gft"].mean(), bh.mean()))
for w in [-0.01, -0.02, -0.03, -0.05]:
    row = []
    for k in ["g", "gto", "gft"]:
        d = ndcg(dv, dv[k] + w * xv) - bases[k]; row.append(f"{k} {d.mean():+.5f}+-{se(d):.5f}")
    d = ndcg(dv, dv["g"] + w * xv) - bases["g"]
    pm, ps = wmean_se(d, PRIV); pu, pus = wmean_se(d, PUB)
    hd = ndcg(dh, dh["g"] + w * xh) - bh
    # holdout user-grouped cluster SE
    uid = dh["uid"]; uu, inv = np.unique(np.where(uid == 0, -np.arange(len(uid)) - 1, uid), return_inverse=True)
    cs = np.bincount(inv, hd); cn = np.bincount(inv)
    cse = np.sqrt(((cs - cn * hd.mean()) ** 2).sum()) / len(hd)
    print(f"w {w:+.2f} | VAL " + " | ".join(row) + f" | pmix priv {pm:+.5f}+-{ps:.5f} pub {pu:+.5f}+-{pus:.5f} | HOLD {hd.mean():+.5f}+-{se(hd):.5f} (user-cluster SE {cse:.5f}, z {hd.mean()/cse:+.2f})")
W = [-0.1, -0.07, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
per = np.stack([ndcg(dv, dv["g"] + w * xv) for w in W])
res = []
for s in range(10):
    fold = folds5(dv, seed=s); out = np.empty(dv["nf"])
    for k in range(5):
        sel = fold == k; j = int(np.argmax(per[:, ~sel].mean(1))); out[sel] = per[j, sel]
    res.append((out - bases["g"]).mean())
print("nested OOF over 10 fold seeds", np.round(res, 5).tolist(), "mean %+.5f" % np.mean(res))
print("full-val argmax w", W[int(np.argmax(per.mean(1)))], "full-val in-sample delta at argmax %+.5f" % (per.max(1).mean() if False else per[int(np.argmax(per.mean(1)))].mean() - bases["g"].mean()))
