"""Residuals (actual / expected positives) by lens columns; expected = npos * softmax(score) within feed."""
from lib import *

D = Data()
df = D.df
npos = np.add.reduceat(D.y, D.off)
p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
rank = D.col("rank")


def table(name, x, bins, mask=None, label=None):
    m = np.ones(len(x), bool) if mask is None else mask
    print(f"\n== {name} {label or ''}")
    print("%-22s %8s %7s %7s %6s %7s" % ("bin", "rows", "pos", "exp", "A/E", "z"))
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = m & (x >= lo) & (x < hi)
        if b.sum() < 200:
            continue
        a, e = D.y[b].sum(), p[b].sum()
        z = (a - e) / np.sqrt(max(e, 1e-9))
        print("%-22s %8d %7d %7.1f %6.3f %7.2f" % (f"[{lo:.3g},{hi:.3g})", b.sum(), a, e, a / e, z))


# calibration by score rank
table("rank", rank, [1, 2, 4, 6, 11, 21, 51, 101, 400])
xp = D.col("xp_imp_c50")
table("xp_imp_c50", xp, [0, 0.01, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.6, 3.9, 4])
table("xp_imp_c50 head rank<=20", xp, [0, 0.01, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.6, 3.9, 4], rank <= 20)
table("xp_imp_c50 tail rank>20", xp, [0, 0.01, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.6, 3.9, 4], rank > 20)
f4 = D.col("f4")
qs = np.r_[0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
table("f4 age", f4, qs)
table("f4 age head<=20", f4, qs, rank <= 20)
table("f4 age tail>20", f4, qs, rank > 20)
f5 = D.col("f5")
table("f5", f5, [0, 0.001, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
for lo, hi in [(0, 0.01), (0.01, 1.5), (1.5, 2.9), (2.9, 4)]:
    table("f5", f5, [0, 0.001, 0.02, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0], (xp >= lo) & (xp < hi), f"xp in [{lo},{hi})")
for c in ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]:
    table(c, D.col(c), [0, 1, 2, 3, 5, 10, 20, 50, 100, 1e9])
cv = D.col("seq_cv_invd20")
table("seq_cv_invd20", cv, [0, 1e-9, 1e-3, 0.01, 0.05, 0.2, 1, 1e9])
for c in ["pre2_fe_z", "pre2_new_z"]:
    table(c, D.col(c), [-9, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 9])
    table(c + " head", D.col(c), [-9, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 9], rank <= 20)
