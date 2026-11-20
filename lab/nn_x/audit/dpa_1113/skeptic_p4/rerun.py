"""Skeptic re-run of P4 POP-TILT: g - w*z_feed(seq_cv_pop) on G3 (CPU only; reads nonlinear/*.npz read-only)."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, fz, ndcg, folds5, wmean_se, pmix_w
dv = load("val"); dh = load("hold")
se = lambda x: x.std() / np.sqrt(len(x))
W = [-0.1, -0.07, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
xv = fz(dv, dv["xt_seq_cv_pop"]); xh = fz(dh, dh["xt_seq_cv_pop"])
PW = pmix_w(); PRIV, PUB = PW["private"], PW["public"]
print("val feeds", dv["nf"], "hold feeds", dh["nf"])
print("corr(g, x) val", np.corrcoef(dv["g"], xv)[0, 1], "hold", np.corrcoef(dh["g"], xh)[0, 1])
for gname in ["g", "gto", "gft"]:
    g = dv[gname]; b = ndcg(dv, g)
    per = np.stack([ndcg(dv, g + w * xv) for w in W]) - b
    line = " ".join(f"{w:+.2f}:{per[i].mean():+.5f}" for i, w in enumerate(W))
    print(f"VAL in-sample {gname:3s} base {b.mean():.5f} | {line}")
    # nested over 20 fold seeds
    res = []; wsall = []
    for s in range(20):
        fold = folds5(dv, s); out = np.empty(dv["nf"]); ws = []
        for k in range(5):
            sel = fold == k; j = int(np.argmax(per[:, ~sel].mean(1))); ws.append(W[j]); out[sel] = per[j, sel]
        res.append(out.mean()); wsall.append(ws)
        if s == 0:
            o0 = out
    print(f"   nested 20 seeds mean {np.mean(res):+.5f} min {np.min(res):+.5f} max {np.max(res):+.5f}; seed0 {res[0]:+.5f}+-{se(o0):.5f}; seed0 ws {wsall[0]}")
    if gname == "g":
        i = W.index(-0.02)
        d = per[i]
        print(f"   g at w=-0.02 (in-sample): all {d.mean():+.5f}+-{se(d):.5f} z {d.mean()/se(d):+.2f}")
        for nm, wt in [("priv", PRIV), ("pub", PUB)]:
            m, s_ = wmean_se(d, wt); print(f"   pmix {nm} w=-0.02 in-sample {m:+.5f}+-{s_:.5f}")
        M = dv["M"]
        for lab, msk in [("M<50", M < 50), ("M>=50", M >= 50)]:
            print(f"   {lab} n {msk.sum()} w=-0.02 {d[msk].mean():+.5f}+-{se(d[msk]):.5f}")
        # seed-0 nested by M slice
        for lab, msk in [("M<50", M < 50), ("M>=50", M >= 50)]:
            print(f"   nested seed0 {lab} {o0[msk].mean():+.5f}+-{se(o0[msk]):.5f}")
bh = ndcg(dh, dh["g"])
for w in [-0.01, -0.02, -0.03, -0.05]:
    d = ndcg(dh, dh["g"] + w * xh) - bh
    print(f"HOLD (train-only pool) w {w:+.2f}: {d.mean():+.5f}+-{se(d):.5f} z {d.mean()/se(d):+.2f}")
