"""Head-only conditional logit: softmax over the model's top-K of each feed, offset = score, lens features.
Re-rank only inside the top-K (items below stay below). Cross-fitted over L2 strength."""
import sys
import clogit
from lib import *
D = Data()
K = int(sys.argv[1])
rank = D.col("rank")
xp = D.col("xp_imp_c50"); f4 = D.col("f4"); f5 = D.col("f5")
feat = {
    "cold": (xp == 0).astype(float), "xp": xp, "f4": f4, "fresh": (f4 < .2).astype(float), "f5": f5, "f5xp": (f5 - .45) * (xp - 1.6),
    "lh": np.log1p(D.col("it_hist_all_luo")), "lh4": np.log1p(D.col("it_hist_et4_luo")), "lh23": np.log1p(D.col("it_hist_et2_luo") + D.col("it_hist_et3_luo")),
    "lcv": np.log1p(D.col("seq_cv_invd20")), "pfe": D.col("pre2_fe_z"), "pnew": D.col("pre2_new_z"), "dd": D.col("s_to") - D.col("s_ft"),
}
names = list(feat)
X = np.column_stack([feat[k] for k in names])
X = (X - X.mean(0)) / X.std(0)
head = rank <= K
# head sub-problem: rows in head, feeds contiguous (rows of a feed are contiguous; head subset preserves order)
hi = np.nonzero(head)[0]
fh = D.feed_of_row[hi]
has_pos = np.bincount(fh, weights=D.y[hi], minlength=len(D.sz)) > 0
lams = [1e2, 3e2, 1e3, 3e3, 1e4]
oof = {l: D.s.copy() for l in lams}
for k in range(5):
    sel = hi[(D.fold_feed[fh] != k) & has_pos[fh]]
    f_sel = D.feed_of_row[sel]
    starts = np.r_[0, np.nonzero(f_sel[1:] != f_sel[:-1])[0] + 1]
    fr = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(sel)]))
    b = None
    te = hi[D.fold_feed[fh] == k]
    for l in sorted(lams, reverse=True):
        b = clogit.fit(X[sel], D.y[sel], starts, D.s[sel], l, fr, len(starts), b0=b)
        # re-rank inside the head only: keep head above the rest by adding a big constant
        new = D.s[te] + X[te] @ b
        oof[l][te] = new + 100.0
    # tail rows of fold k keep score (below the head because head gets +100)
for l in lams:
    m, se = D.delta(oof[l] + 0.0)  # tail unchanged; head shifted +100 so order of head vs tail unchanged
    print(f"K {K} lam {l:6.0f}: OOF delta {m:+.5f} se {se:.5f} z {m / se:.2f}")
