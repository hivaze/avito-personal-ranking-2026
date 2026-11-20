"""Anatomy of the RHO-LOSS per-feed losses on the 2,987,090 production train feeds (CPU, read-only).
L = listwise CE / log M; snap = F7 @77500 snapshots r101-104 (training domain token), ref = pool-only reference.
excess = L_snap - L_ref (rho70 keeps the top 70% by excess per root).
Outputs out/04_*.csv and prints tables.
"""
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path("<repo>")
OUT = ROOT / "nn_x/insights/data_usage/out"
RHO = ROOT / "nn_x/r6/rho"
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_float_precision(4)

t = pl.read_parquet(OUT / "train_feeds.parquet")
prod = np.load(RHO / "prod_fidx.npy")
L_ref = np.load(RHO / "L_ref_r6_rhoref_pool_s1.npy")
Ls = {r: np.load(RHO / f"L_snap_r{r}.npy") for r in (101, 102, 103, 104)}
L_snap = np.mean([Ls[r] for r in Ls], 0)
exc = {r: Ls[r] - L_ref for r in Ls}
kept = {}
for r in Ls:
    n_keep = int(round(0.7 * len(prod)))
    order = np.lexsort((prod, -exc[r]))
    k = np.zeros(len(prod), bool); k[order[:n_keep]] = True
    kept[r] = k
d = t[prod.tolist()] if False else t.filter(pl.col("fidx").is_in(prod))
assert (d["fidx"].to_numpy() == prod).all()
d = d.with_columns(pl.Series("L_snap", L_snap), pl.Series("L_ref", L_ref), pl.Series("excess", L_snap - L_ref),
                   pl.Series("kept101", kept[101]), pl.Series("kept_all4", kept[101] & kept[102] & kept[103] & kept[104]),
                   pl.Series("dropped_all4", ~(kept[101] | kept[102] | kept[103] | kept[104])))
print("cross-root Pearson of excess:", " ".join(f"{a}-{b}:{np.corrcoef(exc[a], exc[b])[0,1]:.3f}" for a, b in [(101, 102), (101, 103), (102, 104), (103, 104)]))
print("cross-root Pearson of L_snap:", f"{np.corrcoef(Ls[101], Ls[102])[0,1]:.3f}", "corr(L_snap, L_ref):", f"{np.corrcoef(L_snap, L_ref)[0,1]:.3f}")
print(f"kept in all 4 roots {d['kept_all4'].mean():.3f}, dropped in all 4 {d['dropped_all4'].mean():.3f}")

# segment columns
d = d.with_columns(
    pl.when(pl.col("M") < 20).then(pl.lit("a[5,20)")).when(pl.col("M") < 50).then(pl.lit("b[20,50)"))
      .when(pl.col("M") < 100).then(pl.lit("c[50,100)")).when(pl.col("M") < 200).then(pl.lit("d[100,200)")).otherwise(pl.lit("e[200,300]")).alias("Mb"),
    pl.when(pl.col("r0") == 0).then(pl.lit("a0")).when(pl.col("r0") <= 2).then(pl.lit("b1-2")).when(pl.col("r0") <= 9).then(pl.lit("c3-9"))
      .when(pl.col("r0") <= 49).then(pl.lit("d10-49")).otherwise(pl.lit("e>=50")).alias("r0b"),
    pl.when(pl.col("npos") == 1).then(pl.lit("1")).when(pl.col("npos") == 2).then(pl.lit("2")).otherwise(pl.lit(">=3")).alias("nposb"),
    pl.when(pl.col("H") == 0).then(pl.lit("a0")).when(pl.col("H") < 64).then(pl.lit("b1-63")).when(pl.col("H") < 256).then(pl.lit("c64-255")).otherwise(pl.lit("d256")).alias("Hb"),
    pl.when(pl.col("coldpos") == 0).then(pl.lit("warm")).when(pl.col("coldpos") >= 1).then(pl.lit("allcold")).otherwise(pl.lit("mixed")).alias("coldb"),
    (pl.col("user_id") == 0).alias("user0"),
    pl.when(pl.col("user_n_train") == 1).then(pl.lit("1")).when(pl.col("user_n_train") <= 3).then(pl.lit("2-3")).otherwise(pl.lit(">=4")).alias("unb"),
    pl.when(pl.col("gap_share") < 0.02).then(pl.lit("a<.02")).when(pl.col("gap_share") < 0.06).then(pl.lit("b.02-.06")).when(pl.col("gap_share") < 0.12).then(pl.lit("c.06-.12")).otherwise(pl.lit("d>=.12")).alias("gapb"),
    pl.when(pl.col("prand_raw").is_nan()).then(pl.lit("nan")).when(pl.col("prand_raw") < 0.05).then(pl.lit("a<.05")).when(pl.col("prand_raw") < 0.5).then(pl.lit("b.05-.5")).otherwise(pl.lit("c>.5(M<50)")).alias("prb"),
    pl.when(pl.col("cat_share") < 0.05).then(pl.lit("a<.05")).when(pl.col("cat_share") < 0.15).then(pl.lit("b.05-.15")).when(pl.col("cat_share") < 0.4).then(pl.lit("c.15-.4")).otherwise(pl.lit("d>=.4")).alias("catsb"),
)


def seg(col, extra=None):
    g = [col] + ([extra] if extra else [])
    s = (d.group_by(g).agg(n=pl.len(), L_snap=pl.col("L_snap").mean(), L_ref=pl.col("L_ref").mean(), excess=pl.col("excess").mean(),
                           kept=pl.col("kept101").mean(), top3=(pl.col("n_top3") > 0).mean(), last=pl.col("pos_last").mean(),
                           Mmean=pl.col("M").mean())
         .with_columns((pl.col("n") / pl.col("n").sum()).alias("share"), (pl.col("L_ref") - pl.col("L_snap")).alias("adv")).sort(g))
    return s


tabs = {}
for col in ["Mb", "r0b", "pos_last", "nposb", "day", "hour", "Hb", "user0", "unb", "user_in_valtest", "coldb", "gapb", "prb", "catsb"]:
    s = seg(col)
    tabs[col] = s
    print(f"\n--- by {col}"); print(s)
    s.write_csv(OUT / f"04_seg_{col}.csv")
# position of first positive within M bins (single-positive feeds)
s = (d.filter(pl.col("npos") == 1).group_by(["Mb", "r0b"]).agg(n=pl.len(), L_snap=pl.col("L_snap").mean(), L_ref=pl.col("L_ref").mean(),
     excess=pl.col("excess").mean(), kept=pl.col("kept101").mean()).with_columns((pl.col("L_ref") - pl.col("L_snap")).alias("adv")).sort(["Mb", "r0b"]))
print("\n--- single-positive feeds: by M bin x rank of the positive"); print(s); s.write_csv(OUT / "04_Mb_x_r0b_single.csv")
s = (d.filter(pl.col("npos") == 1).group_by(["Mb", "pos_last"]).agg(n=pl.len(), L_snap=pl.col("L_snap").mean(), L_ref=pl.col("L_ref").mean(),
     excess=pl.col("excess").mean(), kept=pl.col("kept101").mean()).with_columns((pl.col("L_ref") - pl.col("L_snap")).alias("adv")).sort(["Mb", "pos_last"]))
print("\n--- single-positive feeds: by M bin x last-item positive"); print(s); s.write_csv(OUT / "04_Mb_x_last_single.csv")
# top categories of the positive
top = d.group_by("pcat").agg(n=pl.len()).sort("n", descending=True).head(15)["pcat"].to_list()
s = seg("pcat").filter(pl.col("pcat").is_in(top)).sort("n", descending=True)
print("\n--- by category of the (first) positive, top 15"); print(s); s.write_csv(OUT / "04_seg_pcat_top15.csv")

# imitation slope: within M bins, single-positive feeds, regress L on log(1+r0)/log(M)
print("\n--- imitation slope: d L / d [log(1+r0)/log M] within M bins (single-positive, not last item)")
rows = []
for mb in sorted(d["Mb"].unique().to_list()):
    q = d.filter((pl.col("Mb") == mb) & (pl.col("npos") == 1) & (~pl.col("pos_last")))
    x = np.log1p(q["r0"].to_numpy()) / np.log(q["M"].to_numpy())
    for nm in ["L_snap", "L_ref"]:
        yv = q[nm].to_numpy()
        b = np.polyfit(x, yv, 1)[0]
        rows.append(dict(Mb=mb, model=nm, slope=b, n=len(x), corr=np.corrcoef(x, yv)[0, 1]))
S = pl.DataFrame(rows).pivot(on="model", index=["Mb"], values=["slope", "corr"])
S = S.with_columns((pl.col("slope_L_snap") / pl.col("slope_L_ref")).alias("snap/ref slope"))
print(S); S.write_csv(OUT / "04_imitation_slope.csv")

# M-matched advantage (L_ref - L_snap) on production train vs val (val: random order, relevance only)
v = pl.read_parquet(OUT / "val_feed_losses.parquet")
v = v.with_columns(((pl.col("L_snap101") + pl.col("L_snap102") + pl.col("L_snap103") + pl.col("L_snap104")) / 4).alias("L_snap"))
v = v.with_columns(pl.when(pl.col("M") < 20).then(pl.lit("a[5,20)")).when(pl.col("M") < 50).then(pl.lit("b[20,50)"))
      .when(pl.col("M") < 100).then(pl.lit("c[50,100)")).when(pl.col("M") < 200).then(pl.lit("d[100,200)")).otherwise(pl.lit("e[200,300]")).alias("Mb"))
vs = v.group_by("Mb").agg(n_val=pl.len(), Lsnap_val=pl.col("L_snap").mean(), Lref_val=pl.col("L_ref").mean(),
                          adv_val=(pl.col("L_ref") - pl.col("L_snap")).mean(), adv_val_se=(pl.col("L_ref") - pl.col("L_snap")).std() / pl.len().sqrt())
ps = d.group_by("Mb").agg(n_prod=pl.len(), Lsnap_prod=pl.col("L_snap").mean(), Lref_prod=pl.col("L_ref").mean(), adv_prod=(pl.col("L_ref") - pl.col("L_snap")).mean())
J = ps.join(vs, on="Mb").sort("Mb").with_columns((pl.col("adv_prod") - pl.col("adv_val")).alias("prod_only_adv"),
                                                 (1 - pl.col("adv_val") / pl.col("adv_prod")).alias("prod_only_share"))
print("\n--- advantage of the snapshot over the pool-only reference (L_ref - L_snap), production train vs val, by M"); print(J)
J.write_csv(OUT / "04_adv_prod_vs_val.csv")
w = J["n_val"].to_numpy() / J["n_val"].sum()
print(f"val-M-weighted: adv_prod {np.sum(w * J['adv_prod'].to_numpy()):.4f}  adv_val {np.sum(w * J['adv_val'].to_numpy()):.4f}  "
      f"-> production-only share {1 - np.sum(w * J['adv_val'].to_numpy()) / np.sum(w * J['adv_prod'].to_numpy()):.3f}")
# also with the cooled members (F7 / F7x) on val
for k in ["f7", "f7x"]:
    v2 = v.with_columns(((pl.col(f"L_{k}_101") + pl.col(f"L_{k}_102") + pl.col(f"L_{k}_103") + pl.col(f"L_{k}_104")) / 4).alias("Lk"))
    print(f"val mean L: ref {v2['L_ref'].mean():.4f}  snap {v2['L_snap'].mean():.4f}  {k} {v2['Lk'].mean():.4f}")
d.select(["fidx", "L_snap", "L_ref", "excess", "kept101", "kept_all4", "dropped_all4"]).write_parquet(OUT / "04_prod_losses.parquet")

# rho70 composition: kept vs dropped (root 101)
print("\n--- rho70 (root 101): kept vs dropped composition")
for nm, ex in [("share M<50", pl.col("M") < 50), ("share M>=200", pl.col("M") >= 200), ("positive at rank 0", pl.col("r0") == 0),
               ("positive at rank<=2", pl.col("r0") <= 2), ("last-item positive", pl.col("pos_last")), ("npos>=2", pl.col("npos") >= 2),
               ("all-cold positives", pl.col("coldpos") >= 1), ("user0", pl.col("user_id") == 0), ("H=256", pl.col("H") == 256),
               ("user in val/test", pl.col("user_in_valtest")), ("day>=11 (Sep 17-19)", pl.col("day") >= 11)]:
    a = d.group_by("kept101").agg(ex.mean().alias("v")).sort("kept101")["v"].to_list()
    print(f"  {nm:24s} dropped {a[0]:.3f}  kept {a[1]:.3f}")
