"""Branch mix: NDCG of a*s_ft + (1-a)*s_to overall and by feed segments (feed cold share, M, H, guest)."""
from lib import *
D = Data()
sft = D.col("s_ft"); sto = D.col("s_to"); xp = D.col("xp_imp_c50"); f4 = D.col("f4")
F = len(D.sz)
for a in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
    m, se = D.delta(a * sft + (1 - a) * sto)
    print("a=%.1f delta vs score %+.5f (%.5f)" % (a, m, se))
print("corr(sft, sto) =", np.corrcoef(sft, sto)[0, 1])
nd = {a: D.ndcg(a * sft + (1 - a) * sto) - D.base for a in [0.0, 0.3, 0.7, 1.0]}
def seg(name, fm):
    print("%-26s n %5d  " % (name, fm.sum()) + "  ".join("a=%.1f %+.5f" % (a, d[fm].mean()) for a, d in nd.items()))
cs = np.add.reduceat((xp == 0).astype(float), D.off) / D.sz
fr = np.add.reduceat((f4 < 0.2).astype(float), D.off) / D.sz
M = D.col("M")[D.off]; H = D.col("H")[D.off]; g = D.col("guest")[D.off] > 0
for lo, hi in [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)]:
    seg(f"cold share [{lo},{hi})", (cs >= lo) & (cs < hi))
for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, 1.01)]:
    seg(f"fresh<.2 share [{lo},{hi})", (fr >= lo) & (fr < hi))
for lo, hi in [(0, 30), (30, 100), (100, 200), (200, 301)]:
    seg(f"M [{lo},{hi})", (M >= lo) & (M < hi))
for lo, hi in [(0, 30), (30, 150), (150, 256), (256, 257)]:
    seg(f"H [{lo},{hi})", (H >= lo) & (H < hi))
seg("guest", g)
