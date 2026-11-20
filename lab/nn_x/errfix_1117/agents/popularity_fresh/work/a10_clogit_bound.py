"""Regularised conditional-logit residual model on lens features (+cat / segment interactions), cross-fitted.
Bound for what linear corrections in the lens can achieve."""
import sys
import clogit
from lib import *

D = Data()
F = len(D.sz)
xp = D.col("xp_imp_c50"); f4 = D.col("f4"); f5 = D.col("f5"); cat = D.col("cat"); H = D.col("H"); M = D.col("M")
cold = (xp == 0).astype(float)
base = {
    "cold": cold, "xp": xp, "xp2": (xp - 1.6) ** 2, "f4": f4, "f4sq": (f4 - .5) ** 2, "fresh1": (f4 < .1).astype(float),
    "f5": f5, "f5sq": (f5 - .45) ** 2, "f5xp": (f5 - .45) * (xp - 1.6), "f5zero": (f5 < .02).astype(float),
    "lh": np.log1p(D.col("it_hist_all_luo")), "lh1": np.log1p(D.col("it_hist_et1_luo")), "lh2": np.log1p(D.col("it_hist_et2_luo")),
    "lh3": np.log1p(D.col("it_hist_et3_luo")), "lh4": np.log1p(D.col("it_hist_et4_luo")), "lcv": np.log1p(D.col("seq_cv_invd20")),
    "pfe": D.col("pre2_fe_z"), "pnew": D.col("pre2_new_z"), "f4xp": (f4 - .5) * (xp - 1.6),
}
names = list(base)
cols = [base[k] for k in names]
mode = sys.argv[1]
if "cat" in mode:
    vals, cnt = np.unique(cat, return_counts=True)
    topc = vals[np.argsort(-cnt)[:20]]
    for c in topc:
        ind = (cat == c).astype(float)
        for k in ["cold", "xp", "f4", "fresh1", "f5", "f5xp", "lh"]:
            names.append(f"{k}@c{int(c)}"); cols.append(base[k] * ind)
if "seg" in mode:
    segs = {"guest": D.col("guest"), "H<30": (H < 30).astype(float), "H256": (H >= 256).astype(float), "M<50": (M < 50).astype(float),
            "M>=150": (M >= 150).astype(float), "cu1": (D.col("cat_user_rank") == 1).astype(float), "cu0": (D.col("cat_user_rank") == 0).astype(float),
            "src1": (D.col("newest_src") == 1).astype(float), "src2": (D.col("newest_src") == 2).astype(float)}
    for sn, sv in segs.items():
        for k in ["cold", "xp", "f4", "fresh1", "f5", "f5xp", "lh", "pfe", "pnew"]:
            names.append(f"{k}@{sn}"); cols.append(base[k] * sv)
X = np.column_stack(cols)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
print(mode, "features", X.shape[1])
lams = [3e2, 1e3, 3e3, 1e4, 3e4, 1e5]
D_oof = {l: np.empty(len(D.s)) for l in lams}
for k in range(5):
    tr = D.fold_row != k
    trf = D.fold_feed != k
    off_tr = np.r_[0, np.cumsum(D.sz[trf])[:-1]]
    fr_tr = np.repeat(np.arange(trf.sum()), D.sz[trf])
    b = None
    for l in sorted(lams):
        b = clogit.fit(X[tr], D.y[tr], off_tr, D.s[tr], l, fr_tr, trf.sum(), b0=b)
        te = ~tr
        D_oof[l][te] = D.s[te] + X[te] @ b
for l in lams:
    m, se = D.delta(D_oof[l])
    print(f"lam {l:8.0f}: OOF delta {m:+.5f} se {se:.5f} z {m / se:.2f}")
# full fit at a middle lam for coefficients
b = clogit.fit(X, D.y, D.off, D.s, 3e3, D.feed_of_row, F)
order = np.argsort(-np.abs(b))
print("largest coefs (lam 3e3, standardised):")
for i in order[:25]:
    print("  %-14s %+.4f" % (names[i], b[i]))
m, se = D.delta(D.s + X @ b); print("in-sample delta %+.5f" % m)
