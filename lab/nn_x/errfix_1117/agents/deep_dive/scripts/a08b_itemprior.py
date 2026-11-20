"""Item prior variants (out-of-fold counts from other users' feeds)."""
from util import *
df = load(); c = Ctx(df)
item = df["item_id"].to_numpy(); y = c.y; fold = c.folds_row
u, inv = np.unique(item, return_inverse=True)
oof_pos = np.zeros(len(y)); oof_imp = np.zeros(len(y))
for f in range(5):
    tr, te = fold != f, fold == f
    oof_pos[te] = np.bincount(inv[tr], weights=y[tr], minlength=len(u))[inv[te]]
    oof_imp[te] = np.bincount(inv[tr], minlength=len(u))[inv[te]]
b = y.mean()
V = {}
for a in [5, 20, 50]:
    rate = (oof_pos + a * b) / (oof_imp + a)
    V[f"posonly_logratio_a{a}"] = np.where(oof_pos >= 1, np.log(rate / b), 0.0)
V["log1p_imp"] = np.log1p(oof_imp)
V["imp_ge1"] = (oof_imp >= 1).astype(float)
V["pos_frac"] = oof_pos / (oof_imp + 1)
W = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
for k, g in V.items():
    r = c.crossfit(lambda w: c.base + w * g, W)
    print(f"{k:24s} cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins={r['ins_delta']:+.5f}")
# combined: logratio + imp
g1 = V["posonly_logratio_a20"]; g2 = V["log1p_imp"]
grid = [(w1, w2) for w1 in [0.3, 0.5, 0.8] for w2 in [0, 0.02, 0.05, 0.1]]
r = c.crossfit(lambda p: c.base + p[0] * g1 + p[1] * g2, grid)
print(f"combo cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins={r['ins_delta']:+.5f}")
