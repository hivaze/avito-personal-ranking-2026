"""Near-duplicate pairs inside a feed (mutual nearest: equal cand_maxcos >= .95): which copy gets clicked
relative to the model -- fresher vs older, more vs less exposed?"""
from lib import *
D = Data(); df = D.df
npos = np.add.reduceat(D.y, D.off)
p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
cm = D.col("cand_maxcos"); f4 = D.col("f4"); xp = D.col("xp_imp_c50"); f5 = D.col("f5"); rank = D.col("rank")
fid = D.feed_of_row
m = cm >= 0.95
key = pl.DataFrame({"f": fid[m], "c": cm[m], "i": np.nonzero(m)[0]})
grp = key.group_by(["f", "c"]).agg(pl.col("i")).filter(pl.col("i").list.len() == 2)
pairs = np.array(grp["i"].to_list())
print("pairs", len(pairs))
a, b = pairs[:, 0], pairs[:, 1]
for name, x in [("f4 (older)", f4), ("xp (more exposed)", xp), ("f5", f5), ("score", D.s)]:
    hi = np.where(x[a] > x[b], a, b); lo = np.where(x[a] > x[b], b, a)
    ok = x[a] != x[b]
    hi, lo = hi[ok], lo[ok]
    print("%-18s n %5d | higher: pos %4d exp %6.1f A/E %.3f | lower: pos %4d exp %6.1f A/E %.3f" % (
        name, ok.sum(), D.y[hi].sum(), p[hi].sum(), D.y[hi].sum() / p[hi].sum(), D.y[lo].sum(), p[lo].sum(), D.y[lo].sum() / p[lo].sum()))
# both clicked?
both = D.y[a] & D.y[b]
print("pairs with any pos", (D.y[a] | D.y[b]).sum(), "both", both.sum())
# all items with dup vs none
for nm, mm in [("in dup pair", np.isin(np.arange(len(cm)), pairs.ravel())), ("cand_maxcos<.9", cm < .9)]:
    print(nm, "A/E %.3f" % (D.y[mm].sum() / p[mm].sum()), "n", mm.sum())
