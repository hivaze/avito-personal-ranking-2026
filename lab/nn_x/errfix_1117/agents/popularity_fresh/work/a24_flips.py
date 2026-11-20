"""Which pairs does the cold&cv>=50 boost flip, and who is positive? Counts over all flipped pairs (any rank),
weighted by head position."""
from lib import *
D = Data()
xp = D.col("xp_imp_c50"); cv = D.col("cat_views"); s = D.s
g = ((xp == 0) & (cv >= 50))
w = 0.09
up_pos = up_neg = 0; tab = np.zeros((2, 2)); tab_head = np.zeros((2, 2))
for o, m in zip(D.off, D.sz):
    ss = s[o:o + m]; gg = g[o:o + m]; yy = D.y[o:o + m]
    if not gg.any(): continue
    i_idx = np.nonzero(gg)[0]; j_idx = np.nonzero(~gg)[0]
    # pairs flipped: j above i before (s_j > s_i) and s_i + w > s_j
    for i in i_idx:
        jj = j_idx[(ss[j_idx] > ss[i]) & (ss[j_idx] < ss[i] + w)]
        for j in jj:
            tab[yy[i], yy[j]] += 1
            r = (ss > ss[j]).sum() + 1
            if r <= 5: tab_head[yy[i], yy[j]] += 1
print("flipped pairs (rows: boosted item label 0/1, cols: passed item label 0/1)\n", tab)
print("boosted pos & passed neg: %d   boosted neg & passed pos: %d" % (tab[1, 0], tab[0, 1]))
print("in head (passed item at rank<=5):\n", tab_head)
n = tab[1, 0] + tab[0, 1]
print("binomial z for boosted-positive share: %.2f (share %.3f)" % ((tab[1, 0] - n / 2) / np.sqrt(n / 4), tab[1, 0] / n))
n = tab_head[1, 0] + tab_head[0, 1]
print("head binomial z: %.2f (share %.3f, n %d)" % ((tab_head[1, 0] - n / 2) / np.sqrt(n / 4), tab_head[1, 0] / n, n))
