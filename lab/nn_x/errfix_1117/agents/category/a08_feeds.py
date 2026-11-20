"""Print individual good and bad feeds with category context."""
import sys
import numpy as np
import polars as pl
import cu

df = cu.load()
fe = (df.group_by("fidx", maintain_order=True)
      .agg(best=pl.col("rank").filter(pl.col("label") == 1).min(), npos=pl.col("label").sum(),
           M=pl.col("M").first(), H=pl.col("H").first(), nt=pl.col("newest_type").first(),
           ns=pl.col("newest_src").first(), nc=pl.col("newest_cat").first(), nhc=pl.col("n_hist_cats32").first(),
           guest=pl.col("guest").first()))
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
mode = sys.argv[1] if len(sys.argv) > 1 else "bad"
if mode == "bad":
    pick = fe.filter((pl.col("best") >= 30) & (pl.col("npos") <= 2))
else:
    pick = fe.filter((pl.col("best") <= 3) & (pl.col("M") >= 60))
ids = rng.choice(pick["fidx"].to_numpy(), 12, replace=False)
cols = ["rank", "cat", "cat_views", "cat_actions", "cat_newest_idx", "cat_user_rank", "cat_n_in_feed",
        "cat_rank_in_feed", "score", "s_ft", "s_to", "ucos", "hmax", "hmax_idx", "hmax_type", "h5", "f1", "f3", "f4", "f5",
        "xp_imp_c50", "it_hist_all_luo", "label"]
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(260)
pl.Config.set_float_precision(2)
for f in ids:
    d = df.filter(pl.col("fidx") == f).sort("rank")
    r = fe.filter(pl.col("fidx") == f).row(0, named=True)
    # category distribution of the feed and history summary
    cats = (d.group_by("cat").agg(n=pl.len(), v=pl.col("cat_views").first(), a=pl.col("cat_actions").first(),
                                   ni=pl.col("cat_newest_idx").first(), ur=pl.col("cat_user_rank").first(),
                                   pos=pl.col("label").sum(), top=pl.col("rank").min()).sort("n", descending=True))
    print("=" * 200)
    print(f"feed {f}: M={r['M']} H={r['H']} npos={r['npos']} best={r['best']} newest type={r['nt']} src={r['ns']} cat={cu.CATNAME.get(r['nc'], r['nc'])} nhc32={r['nhc']} guest={r['guest']}")
    print("  cats in feed:", ", ".join(f"{cu.CATNAME[c]}(n{n} v{v} a{a} ni{ni} ur{ur} top{t}{' POS' + str(p) if p else ''})" for c, n, v, a, ni, ur, p, t in cats.select("cat", "n", "v", "a", "ni", "ur", "pos", "top").iter_rows()))
    show = d.filter((pl.col("rank") <= 6) | (pl.col("label") == 1)).select(cols).with_columns(
        pl.col("cat").map_elements(lambda c: cu.CATNAME[c], return_dtype=pl.Utf8))
    print(show)
