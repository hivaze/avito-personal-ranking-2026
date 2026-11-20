"""Head calibration by lens bins: A/E for rank 1, 2-3, 4-10 (expected = npos*softmax(score))."""
from lib import *
D = Data()
npos = np.add.reduceat(D.y, D.off)
p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
rank = D.col("rank")
xp = D.col("xp_imp_c50"); f4 = D.col("f4"); f5 = D.col("f5")
gap = D.col("gap_next")
bins = {
    "cold": xp == 0, "xp(0,1.7)": (xp > 0) & (xp < 1.7), "xp[1.7,3.6)": (xp >= 1.7) & (xp < 3.6), "xp>=3.6": xp >= 3.6,
    "f4<.1": f4 < 0.1, "f4[.1,.3)": (f4 >= .1) & (f4 < .3), "f4[.3,.7)": (f4 >= .3) & (f4 < .7), "f4>=.7": f4 >= .7,
    "f5<.1": f5 < 0.1, "f5[.1,.4)": (f5 >= .1) & (f5 < .4), "f5[.4,.7)": (f5 >= .4) & (f5 < .7), "f5>=.7": f5 >= .7,
    "f5>=.8&xp<1.5": (f5 >= .8) & (xp < 1.5), "f5>=.8&xp>=1.5": (f5 >= .8) & (xp >= 1.5),
    "hist0": D.col("it_hist_all_luo") == 0, "cv>0": D.col("seq_cv_invd20") > 0,
    "pre2fe<0": D.col("pre2_fe_z") < 0, "pre2fe>1.5": D.col("pre2_fe_z") > 1.5,
}
for rname, rm in [("rank1", rank == 1), ("rank2-3", (rank >= 2) & (rank <= 3)), ("rank4-10", (rank >= 4) & (rank <= 10)), ("rank11-30", (rank >= 11) & (rank <= 30))]:
    print(f"\n== {rname}: overall A/E {D.y[rm].sum() / p[rm].sum():.3f}")
    for bn, bm in bins.items():
        m = rm & bm
        a, e = D.y[m].sum(), p[m].sum()
        print("  %-16s n %6d pos %5d A/E %.3f z %5.2f" % (bn, m.sum(), a, a / e, (a - e) / np.sqrt(e)))
