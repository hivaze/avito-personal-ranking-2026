"""Within-feed pairwise test near the head: among discordant-label pairs (i above j, ranks <=R, rank gap<=G),
does dx = x_j - x_i predict that the lower one is the positive, beyond the score gap? Per segment."""
import fx, kit, numpy as np, polars as pl, sys
import logit
df = kit.load()
c = fx.Ctx(df)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 30
G = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rk = df["rank"].to_numpy().astype(np.int64)
y = c.y; s = c.base
# build index: position by (feed, rank)
order = np.lexsort((rk, c.feed))
pos_of = order  # rows sorted by feed, rank
I, J = [], []
for f in range(c.nf):
    o, m = c.off[f], c.sz[f]
    rows = order[o:o + min(m, R)]
    yy = y[rows]
    if yy.sum() == 0 or yy.sum() == len(rows):
        continue
    n = len(rows)
    for a in range(n):
        for g in range(1, G + 1):
            b = a + g
            if b >= n: break
            if yy[a] != yy[b]:
                I.append(rows[a]); J.append(rows[b])
I = np.array(I); J = np.array(J)
out = y[J]
gap = s[I] - s[J]
print("pairs", len(I), "lower-is-pos rate", out.mean())
F = pl.read_parquet("feeds.parquet")
Mf = F["M"].to_numpy()[c.feed[I]]
Hf = F["H"].to_numpy()[c.feed[I]]
feats = {}
for col in ["ucos", "h1", "h5", "h10", "h32", "hmax", "hmean5", "f1", "f2", "f3", "f4", "f5", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z",
            "pre2_new_z", "cand_maxcos", "cat_hist_share", "c_contact", "c_vfeed", "c_vsearch", "c_fav", "c_order", "c_lead"]:
    x = df[col].to_numpy().astype(float)
    fill = np.nanmin(x) - 0.1 if col.startswith("c_") else np.nanmean(x)
    feats[col] = np.where(np.isfinite(x), x, fill)
for col in ["hmax_idx", "cat_user_rank", "cat_newest_idx", "cat_rank_in_feed", "cat_n_in_feed", "n_hdup95", "n_cand_dup95", "cat_views", "cat_actions"]:
    feats["log_" + col] = np.log(3 + df[col].to_numpy().astype(float))
for col in ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]:
    feats["log_" + col] = np.log1p(df[col].to_numpy().astype(float))
feats["sft_minus_sto"] = (df["s_ft"].to_numpy() - df["s_to"].to_numpy()).astype(float)
feats["same_cat_newest"] = df["same_cat_newest"].to_numpy().astype(float)
feats["f2_home"] = df["f2_home"].to_numpy().astype(float)
feats["hmax_type_contact"] = np.isin(df["hmax_type"].to_numpy(), [2, 3, 9]).astype(float)
feats["hmax_src_search"] = (df["hmax_src"].to_numpy() == 2).astype(float)
segs = {"all": np.ones(len(I), bool), "M<30": Mf < 30, "M30-100": (Mf >= 30) & (Mf < 100), "M100+": Mf >= 100, "H<50": Hf < 50}
rows = []
for nm, x in feats.items():
    dx = x[J] - x[I]
    sd = dx.std()
    if sd == 0: continue
    dx = dx / sd
    for sn, m in segs.items():
        X = logit.add_constant(np.c_[gap[m], dx[m]])
        b, t = logit.fit(out[m], X)
        rows.append(dict(f=nm, seg=sn, n=int(m.sum()), b_gap=b[1], b_dx=b[2], z_dx=t[2]))
T = pl.DataFrame(rows)
pl.Config.set_tbl_rows(500); pl.Config.set_tbl_width_chars(200)
print(T.filter(pl.col("seg") == "all").sort("z_dx"))
print(T.filter((pl.col("seg") != "all") & (pl.col("z_dx").abs() > 2.5)).sort("z_dx"))
T.write_parquet(f"pairs_R{R}_G{G}.parquet")
