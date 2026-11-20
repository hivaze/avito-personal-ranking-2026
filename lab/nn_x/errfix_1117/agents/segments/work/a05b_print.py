"""Compact feed printer."""
import fx, kit, numpy as np, polars as pl, sys
df = kit.load()
c = fx.Ctx(df)
F = pl.read_parquet("feeds.parquet")
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
kind = sys.argv[1]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
best = F["best"].to_numpy(); M = F["M"].to_numpy()
sel = {"bad": (best > 30) & (M >= 60), "good": (best <= 3) & (M >= 60), "smallbad": (best > 5) & (M < 30),
       "smallgood": (best == 1) & (M < 30)}[kind]
pick = rng.choice(np.nonzero(sel)[0], size=n, replace=False)
cols = [("rank","d"),("score",".2f"),("s_ft",".2f"),("s_to",".2f"),("cat","d"),("f1",".2f"),("f2",".2f"),("f3",".2f"),("f4",".2f"),("f5",".2f"),
        ("ucos",".2f"),("h1",".2f"),("h5",".2f"),("hmax",".2f"),("hmax_idx","d"),("hmax_type","d"),("cat_user_rank","d"),("cat_newest_idx","d"),
        ("cat_views","d"),("cat_actions","d"),("cat_rank_in_feed","d"),("xp_imp_c50",".1f"),("it_hist_all_luo",".0f"),("seq_cv_invd20",".2f"),
        ("pre2_fe_z",".1f"),("cand_maxcos",".2f"),("label","d")]
short = ["rk","sc","sft","sto","cat","f1","f2","f3","f4","f5","ucos","h1","h5","hmax","hix","hty","cur","cni","cv","ca","crf","xp","ith","scv","pfe","cmx","y"]
for i in pick:
    o, m = c.off[i], c.sz[i]
    sub = df[o:o + m]
    r = sub.row(0, named=True)
    print(f"\n#### feed {r['fidx']} M={m} H={r['H']} guest={r['guest']} uvm={r['uvec_missing']} hr={r['hour']} "
          f"newest t/s/c={r['newest_type']}/{r['newest_src']}/{r['newest_cat']} nhc32={r['n_hist_cats32']} nfc={r['n_feed_cats']} "
          f"npos={int(c.npos[i])} best={best[i]} ndcg={c.base_nd[i]:.3f}")
    cs = sub.group_by("cat").agg(pl.len().alias("n"), pl.col("label").sum().alias("pos"), pl.col("rank").min().alias("minrank"),
                                 pl.col("cat_user_rank").first().alias("ucr"), pl.col("cat_newest_idx").first().alias("cni"),
                                 pl.col("f3").first().alias("f3")).sort("minrank")
    print("cats(cat:n/pos/minrank/userrank/newestidx/f3):", " ".join(f"{a}:{n_}/{p}/r{mr}/u{u}/i{ci}/{f:.2f}" for a, n_, p, mr, u, ci, f in cs.iter_rows()))
    show = sub.filter((pl.col("rank") <= 6) | (pl.col("label") == 1)).sort("rank")
    print(" ".join(f"{s:>5s}" for s in short))
    for row in show.select([cc for cc, _ in cols]).iter_rows():
        out = []
        for v, (cc, fm) in zip(row, cols):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append(f"{'nan':>5s}")
            else:
                out.append(f"{format(v, fm):>5s}")
        print(" ".join(out))
