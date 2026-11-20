"""Context for the only rule-passing (exploratory) candidate P11-W1-TOonly: paired comparison vs PF7x and G3 with the
decide5 proxies V (final val), Vto (train-only val), H (holdout; the fitted pool uses its 2-fold user-grouped OOS
per-feed NDCG), D = Vto - H, S = min(V, V - D); slices on val."""
import os, sys, json
H_ = "<repo>/nn_x/stack/hfit"
sys.argv = ["x", "P11", "none"]
src = open(f"{H_}/hfit.py").read().split("t0 = time.time()")[0]
exec(compile(src, f"{H_}/hfit.py", "exec"))
idx = {f: i for i, f in enumerate(FL)}
def wv(d):
    w = np.zeros(F)
    for f, n in d.items(): w[idx[f]] = n
    return w / w.sum()
PF7x = wv({"F5cd": 6, "F6": 6, "F7x": 7, "F9": 6, "F10": 6, "F12": 7}); G3 = wv({"F10": 6, "F7x": 7, "F7x70": 7})
R = json.load(open(f"{H_}/logs/P11_W1-W2-W3-W4-W5.json"))
wto = np.array(R["variants"]["W2"]["w"])
# 2-fold OOS holdout per-feed NDCG of the unshrunk (W2 == W1 here, lam 0) fit
fo2 = group_folds(uh, 2, 11); allf = np.arange(nh); oos = np.zeros(nh)
for k in range(2):
    wk = fit(allf[fo2 != k], "simplex", 0.0, W_ME); oos[fo2 == k] = hold_ndcg_pf(wk)[fo2 == k]
    print("fold", k, np.round(wk, 3))
C = {"PF7x": (PF7x, PF7x, hold_ndcg_pf(PF7x)), "G3": (G3, G3, hold_ndcg_pf(G3)), "P11ME": (W_ME, W_ME, hold_ndcg_pf(W_ME)),
     "P11W1to": (W_ME, wto, oos), "P11W1both": (wto, wto, oos)}
E = {}
for n, (wf, wt, h) in C.items():
    v = ndcg_pf(0.5 * fz(offv, Mv, Xv_ft @ wf) + 0.5 * fz(offv, Mv, Xv_to @ wt), yv, offv, Mv)
    vto = ndcg_pf(fz(offv, Mv, Xv_to @ wt), yv, offv, Mv)
    E[n] = (v, vto, h); print(f"{n:10s} val {v.mean():.5f} val_to {vto.mean():.5f} hold {h.mean():.5f}")
for ref in ["PF7x", "G3"]:
    for n in ["P11ME", "P11W1to", "P11W1both", "G3"]:
        if n == ref: continue
        (v, vto, h), (rv, rvto, rh) = E[n], E[ref]
        dV, sV = paired(v, rv); dT, sT = paired(vto, rvto); dH, sH = paired(h, rh); D = dT - dH
        sl = " ".join(f"{k} {paired(v, rv, m)[0]:+.5f}+-{paired(v, rv, m)[1]:.5f}" for k, m in SL.items())
        print(f"{n} vs {ref}: V {dV:+.5f}+-{sV:.5f} Vto {dT:+.5f}+-{sT:.5f} H {dH:+.5f}+-{sH:.5f} D {D:+.5f} S {min(dV, dV - D):+.5f} | {sl}")
