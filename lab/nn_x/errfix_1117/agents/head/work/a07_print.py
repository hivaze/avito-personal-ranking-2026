import hk, numpy as np, polars as pl, sys
c = hk.Ctx(); df = c.df
kind = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3])
bp = df.filter(pl.col("label") == 1).group_by("fidx").agg(pl.col("rank").min().alias("bp"))
df = df.join(bp, on="fidx", how="left")
fe = df.group_by("fidx").agg(pl.col("bp").first(), pl.col("M").first())
if kind == "good":
    sel = fe.filter((pl.col("bp") <= 3) & (pl.col("M") >= 60))
elif kind == "head":
    sel = fe.filter((pl.col("bp") >= 2) & (pl.col("bp") <= 6) & (pl.col("M") >= 60))
else:
    sel = fe.filter((pl.col("bp") >= 30) & (pl.col("M") >= 60))
rng = np.random.default_rng(seed)
ids = rng.choice(np.sort(sel["fidx"].to_numpy()), n, replace=False)
cols = [("rank","rk",0),("label","y",0),("score","sc",2),("s_ft","ft",2),("s_to","to",2),("cat","cat",0),("f1","f1",2),("f2","f2",2),("f3","f3",2),("f4","f4",2),("f5","f5",3),
        ("xp_imp_c50","xp",1),("it_hist_all_luo","pop",0),("pre2_fe_z","pfe",1),("seq_cv_invd20","cv",2),("ucos","ucos",2),("hmax","hmax",2),("hmax_idx","hidx",0),("hmax_type","ht",0),("hmax_src","hs",0),
        ("h5","h5",2),("c_contact","ccon",2),("c_fav","cfav",2),("c_vsearch","cvs",2),("cat_views","cv#",0),("cat_actions","ca#",0),("cat_newest_idx","cni",0),("cand_maxcos","cmax",2),("n_cand_dup95","cdup",0)]
hdr = " ".join(f"{h:>6s}" for _, h, _ in cols)
for f in ids:
    d = df.filter(pl.col("fidx") == f).sort("rank")
    r0 = d.row(0, named=True)
    print(f"\n### fidx={f} M={r0['M']} H={r0['H']} guest={r0['guest']} newest type/src/cat={r0['newest_type']}/{r0['newest_src']}/{r0['newest_cat']} nhc32={r0['n_hist_cats32']} nfc={r0['n_feed_cats']} hour={r0['hour']} bestpos={r0['bp']}")
    print(hdr)
    sub = pl.concat([d.filter(pl.col("rank") <= 8), d.filter((pl.col("label") == 1) & (pl.col("rank") > 8))])
    for row in sub.iter_rows(named=True):
        out = []
        for col, h, dec in cols:
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append(f"{'.':>6s}")
            else:
                out.append(f"{v:>6.{dec}f}" if dec else f"{int(v):>6d}")
        print(" ".join(out) + ("  <==" if row["label"] == 1 else ""))
