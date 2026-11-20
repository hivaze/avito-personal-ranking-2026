"""Active-shopping novelty: candidates in the category of the user's most recent events. A/E and lambda-z by age/exposure."""
import lam as LM
from lib import *
D = Data(); F = len(D.sz)
L = LM.get(D, 1.0)
npos = np.add.reduceat(D.y, D.off)
p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
xp = D.col("xp_imp_c50"); f4 = D.col("f4"); cni = D.col("cat_newest_idx"); rank = D.col("rank")
def cell(m):
    if m.sum() < 300: return "%26s" % "-"
    a, e = D.y[m].sum(), p[m].sum()
    pf = np.bincount(D.feed_of_row[m], weights=L[m], minlength=F)
    zl = pf.mean() / (pf.std(ddof=1) / np.sqrt(F))
    return "%26s" % f"{a / e:4.2f}/{(a - e) / np.sqrt(e):+4.1f}/{zl:+4.1f} n{m.sum():6d}"
fb = [("cold f4<.2", (xp == 0) & (f4 < .2)), ("cold f4>=.2", (xp == 0) & (f4 >= .2)), ("warm f4<.5", (xp > 0) & (f4 < .5)),
      ("warm f4.5-.8", (xp > 0) & (f4 >= .5) & (f4 < .8)), ("warm f4>=.8", (xp > 0) & (f4 >= .8))]
print("%-16s" % "" + "".join("%26s" % f for f, _ in fb))
for sn, sm in [("cni 0", cni == 0), ("cni 1-4", (cni >= 1) & (cni <= 4)), ("cni 5-19", (cni >= 5) & (cni <= 19)), ("cni 20-99", (cni >= 20) & (cni <= 99)),
               ("cni 100+", cni >= 100), ("cni -1 (absent)", cni == -1)]:
    print("%-16s" % sn + "".join(cell(sm & fm) for _, fm in fb))
print("\nhead (rank<=20) only")
for sn, sm in [("cni 0", cni == 0), ("cni 1-4", (cni >= 1) & (cni <= 4)), ("cni 5-19", (cni >= 5) & (cni <= 19)), ("cni 20-99", (cni >= 20) & (cni <= 99)),
               ("cni 100+", cni >= 100), ("cni -1 (absent)", cni == -1)]:
    print("%-16s" % sn + "".join(cell(sm & fm & (rank <= 20)) for _, fm in fb))
