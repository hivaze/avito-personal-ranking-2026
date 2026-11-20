"""Robustness of the best in-sample candidate (cold & cat_views>=50 boost): fine weight grid, fold seeds, segments."""
from lib import *
D = Data()
xp = D.col("xp_imp_c50"); cv = D.col("cat_views"); f4 = D.col("f4"); s = D.s
g = ((xp == 0) & (cv >= 50)).astype(float)
ws = [0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]
curve = {w: D.ndcg(s + w * g + 1e-7 * s) - D.base for w in ws}
print("fine curve:", " ".join("%.2f:%+.5f" % (w, d.mean()) for w, d in curve.items()))
for seed in range(6):
    fr = kit.folds(D.df, seed=seed)[D.off]
    oof = np.empty(len(D.sz)); ch = []
    for f in range(5):
        tr = fr != f
        w = max(ws, key=lambda w: curve[w][tr].mean()); ch.append(w)
        oof[~tr] = curve[w][~tr]
    print("seed %d CF %+.5f z %.2f chosen %s" % (seed, oof.mean(), oof.mean() / (oof.std(ddof=1) / np.sqrt(len(oof))), ch))
d = curve[0.08]
M = D.col("M")[D.off]
for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 301)]:
    m = (M >= lo) & (M < hi)
    print("M [%d,%d) feeds %d delta %+.5f" % (lo, hi, m.sum(), d[m].mean()))
ch = d != 0
print("feeds changed", ch.sum(), "improved", (d > 0).sum(), "worsened", (d < 0).sum(), "sum gain", d[d > 0].sum(), "sum loss", d[d < 0].sum())
# contribution of top feeds
srt = np.sort(d)[::-1]
print("top 10 gains sum %.4f, total %.4f" % (srt[:10].sum(), d.sum()))
# young vs old cold
for nm, gg in [("cold young f4<.2 & cv>=50", ((xp == 0) & (cv >= 50) & (f4 < .2))), ("cold old f4>=.2 & cv>=50", ((xp == 0) & (cv >= 50) & (f4 >= .2)))]:
    dd = D.ndcg(s + 0.08 * gg + 1e-7 * s) - D.base
    print(nm, "%+.5f (%.5f)" % (dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd))))
