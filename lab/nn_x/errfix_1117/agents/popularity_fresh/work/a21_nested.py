"""Honest procedure-level estimate: pick the best (family, weight) among ALL single-indicator/tilt families tried,
on 4 folds; evaluate on the 5th. Compares with the best family's own cross-fit (which ignores family selection)."""
from lib import *
D = Data()
xp = D.col("xp_imp_c50"); f4 = D.col("f4"); f5 = D.col("f5"); H = D.col("H"); cat = D.col("cat")
cur = D.col("cat_user_rank"); cv = D.col("cat_views"); sh = D.col("cat_hist_share"); rank = D.col("rank")
guest = D.col("guest") > 0; uvm = D.col("uvec_missing") > 0
cold = (xp == 0).astype(float); s = D.s
f5m = D.feedwise(f5, np.median)
cs = D.feedwise(cold, np.mean)
imp = np.round(np.expm1(xp))
fam = {
    "fresh.1 guest": (((f4 < .1) & guest), [-.1, -.3, -.5, -.8, -1.2]),
    "fresh.1 guest|uvm|H<30": (((f4 < .1) & (guest | uvm | (H < 30))), [-.1, -.3, -.5, -.8, -1.2]),
    "fresh.1 cat48|50": (((f4 < .1) & np.isin(cat, [48, 50])), [-.1, -.3, -.5, -.8, -1.2]),
    "fresh.2 cold-heavy feed": (((f4 < .2) & (cs > .5)), [-.05, -.1, -.3, -.5, -.8]),
    "f5<.2&xp>3": (((f5 < .2) & (xp > 3)), [-.05, -.1, -.2, -.3, -.5]),
    "f5<.2&xp>3 cat10|11": (((f5 < .2) & (xp > 3) & np.isin(cat, [10, 11])), [-.1, -.3, -.5, -.8, -1.2]),
    "cold&cur1": (cold * (cur == 1), [.03, .05, .08, .12, .2]),
    "cold&cv>=20": (cold * (cv >= 20), [.03, .05, .08, .12, .2]),
    "cold&cv>=50": (cold * (cv >= 50), [.03, .05, .08, .12, .2]),
    "cold*share": (cold * sh, [.1, .2, .3, .5, .8]),
    "cold*(log1p(cv)-2)": (cold * (np.log1p(cv) - 2), [.01, .02, .04, .06, .1]),
    "cold&cur0": (cold * (cur == 0), [-.03, -.05, -.1, -.2, -.3]),
    "(f5-.45)*sat": ((f5 - .45) * (xp >= 3.9), [.05, .1, .2, .3, .5]),
    "(f5-.45)*(xp/3.93)^2": ((f5 - .45) * (xp / 3.93) ** 2, [.05, .1, .2, .3, .5]),
    "(f5-.45)*(xp/3.93)^4": ((f5 - .45) * (xp / 3.93) ** 4, [.05, .1, .2, .3, .5]),
    "(f5-f5med)*sat": ((f5 - f5m) * (xp >= 3.9), [.05, .1, .2, .3, .5]),
    "(f5-f5med)*(xp/3.93)^2": ((f5 - f5m) * (xp / 3.93) ** 2, [.05, .1, .2, .3, .5]),
    "(f5-.45)*(xp>=3)": ((f5 - .45) * (xp >= 3), [.05, .1, .2, .3, .5]),
    "f5": (f5 - .45, [-.1, -.05, .05, .1, .2]),
    "imp1-4 top1": (((imp >= 1) & (imp <= 4) & (rank == 1)), [-.02, -.05, -.1, -.2, -.3]),
    "imp1-9 top3": (((imp >= 1) & (imp <= 9) & (rank <= 3)), [-.02, -.05, -.1, -.2, -.3]),
    "young&cur0": (((f4 < .2) & (cur == 0)), [-.03, -.05, -.1, -.15, -.3]),
    "pre2_fe*cold": (cold * D.col("pre2_fe_z"), [-.1, -.05, -.02, .02, .05, .1]),
    "(sto-sft)*sat": ((D.col("s_to") - D.col("s_ft")) * (xp >= 3.9), [-.3, -.2, -.1, -.05, .05, .1]),
    "(sto-sft)*xp(0,1.7)": ((D.col("s_to") - D.col("s_ft")) * ((xp > 0) & (xp < 1.7)), [-.3, -.2, -.1, -.05, .05, .1]),
}
rows = []
labels = []
for k, (g, ws) in fam.items():
    g = g.astype(float)
    for w in ws:
        rows.append(D.ndcg(s + w * g + 1e-7 * s) - D.base)
        labels.append((k, w))
Dm = np.array(rows)
print("candidates", len(labels))
oof = np.empty(Dm.shape[1]); picks = []
for f in range(5):
    tr = D.fold_feed != f
    j = int(np.argmax(Dm[:, tr].mean(1)))
    picks.append(labels[j]); oof[~tr] = Dm[j, ~tr]
se = oof.std(ddof=1) / np.sqrt(len(oof))
print("NESTED (family+weight chosen on 4 folds): CF %+.5f se %.5f z %.2f" % (oof.mean(), se, oof.mean() / se))
print("picks", picks)
j = int(np.argmax(Dm.mean(1)))
print("best in-sample overall", labels[j], "%+.5f" % Dm[j].mean())
# repeat with other fold seeds for stability
for seed in [1, 2, 3]:
    fr = kit.folds(D.df, seed=seed)[D.off]
    oof = np.empty(Dm.shape[1]); picks = []
    for f in range(5):
        tr = fr != f
        j = int(np.argmax(Dm[:, tr].mean(1)))
        picks.append(labels[j][0]); oof[~tr] = Dm[j, ~tr]
    print("seed %d nested CF %+.5f z %.2f picks %s" % (seed, oof.mean(), oof.mean() / (oof.std(ddof=1) / np.sqrt(len(oof))), picks))
