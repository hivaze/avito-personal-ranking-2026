"""(b)+(c) Per-feed FT/TO weight and pop-light weight as functions of label-free feed descriptors, cross-fitted by user halves.
Per-feed NDCG table over a grid (wft x wpl); gates fitted on half A choose a grid point (tree) or a smooth weight
(regularised linear-sigmoid, bilinear interpolation of the table), applied on half B; and the reverse."""
import ph, numpy as np, json
from feedfeat import feed_features
from scipy.optimize import minimize
D = ph.Data("val"); C = np.load("../out/comp_val.npz")
z = {k: ph.zf(C[k], D.off, D.M) for k in C.files}
WFT = np.round(np.arange(0, 1.01, 0.1), 2); WPL = np.round(np.arange(0, 0.61, 0.1), 2)
Mr = np.repeat(D.M, D.M)
try:
    T = np.load("../out/expB_table.npy")
except FileNotFoundError:
    T = np.zeros((D.n, len(WFT), len(WPL)))
    for j, wp in enumerate(WPL):
        ftm = ph.zf((1 - wp) * z["ft_base"] + wp * z["ft_pl"], D.off, D.M); tom = ph.zf((1 - wp) * z["to_base"] + wp * z["to_pl"], D.off, D.M)
        for i, wf in enumerate(WFT):
            T[:, i, j] = D.ev(wf * ftm + (1 - wf) * tom)
    np.save("../out/expB_table.npy", T)
# reference = the current file's rule (wft .5, wpl .4 if M<50 else .2) evaluated directly
base = D.base
X = feed_features("val", D); names = list(X); Xm = np.column_stack([X[k] for k in names])
Xs = (Xm - Xm.mean(0)) / Xm.std(0)
print("grid mean NDCG, rows wft, cols wpl:\n", np.round(T.mean(0), 5))
res = {}
def bilinear(Tf, a, b):   # Tf: (n, 11, 7); a in [0,1] ; b in [0,0.6]
    ia = np.clip(a * 10, 0, 9.9999); ib = np.clip(b * 10, 0, 5.9999)
    i0 = ia.astype(int); j0 = ib.astype(int); fa = ia - i0; fb = ib - j0; r = np.arange(len(a))
    return ((1 - fa) * (1 - fb) * Tf[r, i0, j0] + fa * (1 - fb) * Tf[r, i0 + 1, j0] + (1 - fa) * fb * Tf[r, i0, j0 + 1] + fa * fb * Tf[r, i0 + 1, j0 + 1])
newv = {}
for form in ("const", "lin_ft", "lin_pl", "lin_both", "tree2"):
    out = np.zeros(D.n)
    for h in (0, 1):
        A = D.half == h; B = ~A
        if form == "const":
            m = T[A].mean(0); i, j = np.unravel_index(m.argmax(), m.shape); out[B] = T[B, i, j]; print(form, "half", h, "chosen", WFT[i], WPL[j])
        elif form.startswith("lin"):
            k = Xs.shape[1]; lam = 2e-4
            def unpack(p):
                a = 1 / (1 + np.exp(-(p[0] + Xs @ p[1:k + 1]))) if form in ("lin_ft", "lin_both") else np.full(D.n, 0.5)
                b = 0.6 / (1 + np.exp(-(p[k + 1] + Xs @ p[k + 2:]))) if form in ("lin_pl", "lin_both") else np.where(D.M < 50, 0.4, 0.2)
                return a, b
            p0 = np.zeros(2 * k + 2); p0[k + 1] = np.log(0.3 / 0.3)  # b=0.3 start
            obj = lambda p: -bilinear(T[A], *[u[A] for u in unpack(p)]).mean() + lam * (np.sum(p[1:k + 1] ** 2) + np.sum(p[k + 2:] ** 2))
            r = minimize(obj, p0, method="Powell", options=dict(maxiter=20000, xtol=1e-3, ftol=1e-8))
            a, b = unpack(r.x); out[B] = bilinear(T[B], a[B], b[B])
            print(form, "half", h, "in-sample A gain vs const .5/.3:", -r.fun - bilinear(T[A], np.full(A.sum(), .5), np.full(A.sum(), .3)).mean())
        elif form == "tree2":
            # depth-2 tree: best split (feature, median/tercile thresholds) at root, then each child picks its best grid point
            best = (-1, None)
            def leafval(msk):
                m = T[msk].mean(0); return m.max() * msk.sum(), np.unravel_index(m.argmax(), m.shape)
            for f in range(len(names)):
                for qv in (0.25, 0.5, 0.75):
                    thr = np.quantile(Xm[A, f], qv); L = A & (Xm[:, f] <= thr); R = A & (Xm[:, f] > thr)
                    if L.sum() < 1500 or R.sum() < 1500: continue
                    v = leafval(L)[0] + leafval(R)[0]
                    if v > best[0]: best = (v, (f, thr))
            f, thr = best[1]
            for side in (Xm[:, f] <= thr, Xm[:, f] > thr):
                _, (i, j) = leafval(A & side); out[B & side] = T[B & side, i, j]
                print(form, "half", h, f"split {names[f]}<={thr:.3f} leaf chooses wft {WFT[i]} wpl {WPL[j]}")
    res[form] = D.report(out, f"B gate {form}")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'line'} for k, v in res.items()}, open("../out/expB.json", "w"), indent=1)
