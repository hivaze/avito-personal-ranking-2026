"""Score-test screen: does adding g(x) to the score improve the within-feed conditional logit?
For each candidate g, z = sum_pos (g_i - E_softmax[g]) / sqrt(sum_pos Var_softmax[g]); also lambda zL (NDCG gradient).
Also reports fold-consistency: sign of the per-fold U."""
import itertools
import lam as LM
from lib import *

D = Data(); df = D.df
F = len(D.sz)
L = LM.get(D, 1.0)
w1 = softmax_p(D.s, D.off, D.sz, np.ones(F), 1.0)  # softmax weights
npos = np.add.reduceat(D.y, D.off)
fr = D.feed_of_row


def screen(g):
    Eg = np.add.reduceat(w1 * g, D.off)
    Eg2 = np.add.reduceat(w1 * g * g, D.off)
    var = np.maximum(Eg2 - Eg ** 2, 0)
    Upf = np.add.reduceat(D.y * g, D.off) - npos * Eg  # per feed
    V = (npos * var).sum()
    z = Upf.sum() / np.sqrt(V) if V > 0 else 0.0
    Lpf = np.bincount(fr, weights=L * g, minlength=F)
    zl = Lpf.mean() / (Lpf.std(ddof=1) / np.sqrt(F)) if Lpf.std() > 0 else 0.0
    folds = [np.sign(Upf[D.fold_feed == k].sum()) for k in range(5)]
    return z, zl, Lpf.mean() * 1e4, int(sum(folds))


xp = D.col("xp_imp_c50"); f4 = D.col("f4"); f5 = D.col("f5"); M = D.col("M"); H = D.col("H")
rank = D.col("rank")
cold = (xp == 0).astype(float)
guest = D.col("guest"); uvm = D.col("uvec_missing")
items = {
    "cold": cold,
    "xp": xp,
    "xp_low(0<xp<1.2)": ((xp > 0) & (xp < 1.2)).astype(float),
    "xp_sat(>=3.9)": (xp >= 3.9).astype(float),
    "fresh<.05": (f4 < 0.05).astype(float),
    "fresh<.1": (f4 < 0.1).astype(float),
    "fresh<.2": (f4 < 0.2).astype(float),
    "old>.9": (f4 > 0.9).astype(float),
    "f4": f4,
    "f5": f5,
    "f5_c": f5 - 0.43,
    "f5_hi&xp_low": ((f5 > 0.8) & (xp < 1.5)).astype(float),
    "f5_hi&xp_low_warm": ((f5 > 0.8) & (xp < 1.5) & (xp > 0)).astype(float),
    "f5_lo&xp_hi": ((f5 < 0.2) & (xp > 3)).astype(float),
    "f5*xp_c": (f5 - 0.43) * (xp - 1.57),
    "f5_zero": (f5 < 0.001).astype(float),
    "f5<.02": (f5 < 0.02).astype(float),
    "cold&fresh": ((xp == 0) & (f4 < 0.1)).astype(float),
    "cold&old": ((xp == 0) & (f4 > 0.5)).astype(float),
    "warm&fresh": ((xp > 0) & (f4 < 0.2)).astype(float),
    "hist0": (D.col("it_hist_all_luo") == 0).astype(float),
    "loghist": np.log1p(D.col("it_hist_all_luo")),
    "logh_et2": np.log1p(D.col("it_hist_et2_luo")),
    "logh_et4": np.log1p(D.col("it_hist_et4_luo")),
    "fav_ratio": np.log1p(D.col("it_hist_et4_luo")) - np.log1p(D.col("it_hist_et1_luo")),
    "contact_ratio": np.log1p(D.col("it_hist_et2_luo") + D.col("it_hist_et3_luo")) - np.log1p(D.col("it_hist_et1_luo")),
    "cv>0": (D.col("seq_cv_invd20") > 0).astype(float),
    "logcv": np.log1p(D.col("seq_cv_invd20")),
    "pre2_fe": D.col("pre2_fe_z"),
    "pre2_new": D.col("pre2_new_z"),
    "pre2_diff": D.col("pre2_new_z") - D.col("pre2_fe_z"),
    "sto-sft": D.col("s_to") - D.col("s_ft"),
    "cold*(sto-sft)": cold * (D.col("s_to") - D.col("s_ft")),
    "fresh.2*(sto-sft)": (f4 < 0.2) * (D.col("s_to") - D.col("s_ft")),
}
segs = {
    "all": np.ones(len(xp)),
    "guest": guest,
    "uvec_miss": uvm,
    "H<30": (H < 30).astype(float),
    "H>=200": (H >= 200).astype(float),
    "M<50": (M < 50).astype(float),
    "M>=150": (M >= 150).astype(float),
    "head<=10": (rank <= 10).astype(float),
    "head<=30": (rank <= 30).astype(float),
    "feedcold>.5": (D.feedwise(cold, np.mean) > 0.5).astype(float),
    "feedcold<.2": (D.feedwise(cold, np.mean) < 0.2).astype(float),
    "newest_src1": (D.col("newest_src") == 1).astype(float),
    "newest_src2": (D.col("newest_src") == 2).astype(float),
    "cat48|50": np.isin(D.col("cat"), [48, 50]).astype(float),
    "cat10|11": np.isin(D.col("cat"), [10, 11]).astype(float),
    "cat2|3": np.isin(D.col("cat"), [2, 3]).astype(float),
    "cat_new_user(rank0)": (D.col("cat_user_rank") == 0).astype(float),
    "cat_top(rank1)": (D.col("cat_user_rank") == 1).astype(float),
}
res = []
for (iname, g), (sname, s) in itertools.product(items.items(), segs.items()):
    gg = g * s
    if (gg != 0).sum() < 500:
        continue
    z, zl, l, fs = screen(gg)
    res.append((iname, sname, z, zl, l, fs))
print("tested", len(res))
res.sort(key=lambda r: -abs(r[3]))
print("%-22s %-20s %7s %7s %8s %5s" % ("g", "segment", "z_ll", "z_L", "L*1e4", "folds"))
for r in res[:60]:
    print("%-22s %-20s %7.2f %7.2f %8.2f %5d" % r)
print("\n-- sorted by z_ll")
res.sort(key=lambda r: -abs(r[2]))
for r in res[:40]:
    print("%-22s %-20s %7.2f %7.2f %8.2f %5d" % r)
zs = np.array([r[3] for r in res])
print("\nzL distribution: sd %.2f, |z|>2: %d, |z|>3: %d of %d" % (zs.std(), (abs(zs) > 2).sum(), (abs(zs) > 3).sum(), len(zs)))
zs = np.array([r[2] for r in res])
print("zll distribution: sd %.2f, |z|>2: %d, |z|>3: %d of %d" % (zs.std(), (abs(zs) > 2).sum(), (abs(zs) > 3).sum(), len(zs)))
