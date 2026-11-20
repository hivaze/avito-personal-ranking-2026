"""Lambda (NDCG gradient) aggregated by lens bins, and 2-D interactions with segments.

Number reported per bin: L = sum(lambda in bin) / n_feeds * 1e4  (approx dNDCG/dw *1e4 for score + w*1[bin]),
plus z of the per-feed sums (feed-clustered), and A/E of positives for the head (rank<=20)."""
import sys

import lam as LM
from lib import *

D = Data()
L = LM.get(D, 1.0)
F = len(D.sz)
npos = np.add.reduceat(D.y, D.off)
p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
rank = D.col("rank")


def lstat(mask):
    per_feed = np.bincount(D.feed_of_row[mask], weights=L[mask], minlength=F)
    m = per_feed.mean()
    se = per_feed.std(ddof=1) / np.sqrt(F)
    return m * 1e4, (m / se if se > 0 else 0.0)


def table(name, x, bins, mask=None):
    m = np.ones(len(x), bool) if mask is None else mask
    print(f"\n== {name}")
    print("%-22s %8s %7s %6s %8s %6s" % ("bin", "rows", "pos", "A/E", "L*1e4", "zL"))
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = m & (x >= lo) & (x < hi)
        if b.sum() < 300:
            continue
        a, e = D.y[b].sum(), p[b].sum()
        l, z = lstat(b)
        print("%-22s %8d %7d %6.3f %8.2f %6.2f" % (f"[{lo:.3g},{hi:.3g})", b.sum(), a, a / e, l, z))


def cat_table(name, x, bins, catcol="cat", top=14):
    c = D.col(catcol)
    vals, cnt = np.unique(c, return_counts=True)
    order = np.argsort(-cnt)[:top]
    print(f"\n#### {name} x {catcol}: L*1e4 (zL) per bin")
    hdr = "%-8s" % catcol + "".join("%16s" % f"[{lo:.2g},{hi:.2g})" for lo, hi in zip(bins[:-1], bins[1:]))
    print(hdr)
    for v in vals[order]:
        row = "%-8d" % v
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = (c == v) & (x >= lo) & (x < hi)
            if b.sum() < 300:
                row += "%16s" % "-"
                continue
            l, z = lstat(b)
            a, e = D.y[b].sum(), p[b].sum()
            row += "%16s" % f"{l:5.1f}({z:4.1f}){a / e:4.2f}"
        print(row)


if __name__ == "__main__":
    xp = D.col("xp_imp_c50")
    f4 = D.col("f4")
    f5 = D.col("f5")
    table("rank", rank, [1, 2, 4, 6, 11, 21, 51, 101, 400])
    table("xp", xp, [0, 0.01, 0.7, 1.7, 2.7, 3.6, 4])
    table("f4", f4, [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    table("f5", f5, [0, 0.02, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1])
    cat_table("f4", f4, [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    cat_table("xp", xp, [0, 0.01, 1.7, 3.0, 4])
    cat_table("f5", f5, [0, 0.02, 0.2, 0.5, 0.8, 1.0])
    # vs M bucket
    Mb = np.digitize(D.col("M"), [30, 60, 100, 150, 200, 250])
    D.df = D.df.with_columns(pl.Series("Mb", Mb))
    cat_table("f4", f4, [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], "Mb")
    cat_table("xp", xp, [0, 0.01, 1.7, 3.0, 4], "Mb")
    Hb = np.digitize(D.col("H"), [1, 10, 50, 150, 256])
    D.df = D.df.with_columns(pl.Series("Hb", Hb))
    cat_table("f4", f4, [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], "Hb")
    cat_table("xp", xp, [0, 0.01, 1.7, 3.0, 4], "Hb")
    cat_table("f5", f5, [0, 0.02, 0.2, 0.5, 0.8, 1.0], "Hb")
    rb = np.digitize(rank, [2, 4, 11, 21, 51, 101])
    D.df = D.df.with_columns(pl.Series("rb", rb))
    cat_table("f4", f4, [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], "rb")
    cat_table("xp", xp, [0, 0.01, 1.7, 3.0, 4], "rb")
    cat_table("f5", f5, [0, 0.02, 0.2, 0.5, 0.8, 1.0], "rb")
    for c in ["hour", "weekday", "newest_src", "newest_type", "guest", "uvec_missing"]:
        cat_table("f4", f4, [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], c, top=10)
        cat_table("xp", xp, [0, 0.01, 1.7, 3.0, 4], c, top=10)
