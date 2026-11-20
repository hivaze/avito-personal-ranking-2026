"""(d) distribution diagnostics val vs test (and public / private, holdout) of r6 members vs nn/ F7 members.
Per member and split: NaN/inf, mean/median per-feed raw std, within-feed tie fraction, mean per-feed Pearson and Spearman
correlation with the same-root F7 member, plus seed-to-seed baselines inside each family."""
import sys

import numpy as np

sys.path.insert(0, "<repo>/src")
from common import feed_meta  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402

R = "<repo>"
P = f"{R}/preds"
PUB_TS = 1790497920
roots = list(range(101, 108))
FAM = {"F7": "v17_pre2s_s{r}", "F7x": "r6_cd_m02_r{r}", "F7x70": "r6_cd_m02_p70k_r{r}", "F7b": "r6_cd_c0_r{r}"}

meta = {}
for sp in ("val", "test"):
    f, M, off = feed_meta(sp)
    meta[sp] = (M, off, f)
hm = np.load(f"{P}/holdout/holdout_meta.npy", allow_pickle=True)
meta["holdout"] = (np.asarray(hm[0], np.int64), np.asarray(hm[1], np.int64), None)


def subsets(sp):
    M, off, f = meta[sp]
    out = {sp: np.ones(len(M), bool)}
    if sp == "test":
        ts = f["timestamp"].to_numpy()
        out["public"] = ts < PUB_TS
        out["private"] = ts >= PUB_TS
    if sp == "val":
        day = (f["timestamp"].to_numpy() + 10800) // 86400
        out["val_tue"] = day < 20720
        out["val_thu"] = day >= 20720
    return out


def load(sp, v, kind):
    if sp == "holdout":
        return np.load(f"{P}/holdout/nn_{v}_holdout.npy").astype(np.float32) if kind == "to" else None
    fn = {"to": {"val": "trainonly_val", "test": "trainonly_test"},
          "ft": {"val": "n2cpv3_val_oof", "test": "n2cpv3_test"}}[kind][sp]
    return np.load(f"{P}/nn_{v}_{fn}.npy").astype(np.float32)


def feed_stats(x, M, off):
    fid = np.repeat(np.arange(len(M)), M)
    o = np.lexsort((x, fid))
    xs, fs = x[o], fid[o]
    dup = (xs[1:] == xs[:-1]) & (fs[1:] == fs[:-1])
    tied = np.zeros(len(x), bool)
    tied[1:] |= dup
    tied[:-1] |= dup
    tie_frac = np.bincount(fid[o], weights=tied, minlength=len(M)) / M
    s1 = np.add.reduceat(x.astype(np.float64), off)
    s2 = np.add.reduceat(x.astype(np.float64) ** 2, off)
    sd = np.sqrt(np.maximum(s2 / M - (s1 / M) ** 2, 0))
    return sd, tie_frac


def feed_corr(za, zb, M, off):
    return np.add.reduceat((za * zb).astype(np.float64), off) / M


def rank_std(rk, M, off):  # standardise the per-feed rank column of feed_rank_z (Spearman via Pearson of ranks)
    s1 = np.add.reduceat(rk.astype(np.float64), off) / M
    s2 = np.add.reduceat(rk.astype(np.float64) ** 2, off) / M
    mu = np.repeat(s1, M)
    sd = np.repeat(np.sqrt(np.maximum(s2 - s1 ** 2, 1e-18)), M)
    return (rk - mu) / sd


rows = []
for kind in ("to", "ft"):
    for sp in ("val", "test", "holdout"):
        if sp == "holdout" and kind == "ft":
            continue
        M, off, _ = meta[sp]
        subs = subsets(sp)
        Z, Rk = {}, {}
        for fam, pat in FAM.items():
            for r in roots:
                v = pat.format(r=r)
                try:
                    x = load(sp, v, kind)
                except FileNotFoundError:
                    print("missing", sp, kind, v)
                    continue
                assert len(x) == M.sum(), (v, sp, len(x), M.sum())
                nbad = int((~np.isfinite(x)).sum())
                sd, tf = feed_stats(x, M, off)
                rz = feed_rank_z(off, M, x)
                Z[(fam, r)] = rz[:, 1]
                Rk[(fam, r)] = rank_std(rz[:, 0], M, off)
                for sn, msk in subs.items():
                    rows.append(dict(kind=kind, sub=sn, fam=fam, root=r, stat="nonfinite", val=nbad if sn == sp else np.nan))
                    rows.append(dict(kind=kind, sub=sn, fam=fam, root=r, stat="feed_std_mean", val=sd[msk].mean()))
                    rows.append(dict(kind=kind, sub=sn, fam=fam, root=r, stat="feed_std_median", val=np.median(sd[msk])))
                    rows.append(dict(kind=kind, sub=sn, fam=fam, root=r, stat="tie_frac", val=tf[msk].mean()))
                    rows.append(dict(kind=kind, sub=sn, fam=fam, root=r, stat="level_mean", val=x[np.repeat(msk, M)].mean()))
        for fam in FAM:
            for r in roots:
                if (fam, r) not in Z or ("F7", r) not in Z:
                    continue
                if fam != "F7":
                    pc = feed_corr(Z[(fam, r)], Z[("F7", r)], M, off)
                    sc = feed_corr(Rk[(fam, r)], Rk[("F7", r)], M, off)
                    for sn, msk in subs.items():
                        rows.append(dict(kind=kind, sub=sn, fam=fam, root=r, stat="pearson_vsF7root", val=pc[msk].mean()))
                        rows.append(dict(kind=kind, sub=sn, fam=fam, root=r, stat="spearman_vsF7root", val=sc[msk].mean()))
                # seed-to-seed baseline inside the family (root r vs r+1)
                r2 = r + 1 if r < 107 else 101
                if (fam, r2) in Z:
                    pc = feed_corr(Z[(fam, r)], Z[(fam, r2)], M, off)
                    for sn, msk in subs.items():
                        rows.append(dict(kind=kind, sub=sn, fam=fam, root=r, stat="pearson_intra_nextroot", val=pc[msk].mean()))
                # cross: r6 member vs F7 member of ANOTHER root (to separate root-sharing from family similarity)
                if fam != "F7" and ("F7", r2) in Z:
                    pc = feed_corr(Z[(fam, r)], Z[("F7", r2)], M, off)
                    for sn, msk in subs.items():
                        rows.append(dict(kind=kind, sub=sn, fam=fam, root=r, stat="pearson_vsF7other", val=pc[msk].mean()))

import polars as pl  # noqa: E402

df = pl.DataFrame(rows)
df.write_parquet(f"{R}/nn_x/pubdiag/bughunt/d_dist_rows.parquet")
agg = (df.group_by(["kind", "stat", "fam", "sub"]).agg(pl.col("val").mean().alias("mean"), pl.col("val").std().alias("sd_roots"),
                                                       pl.len().alias("n"))
       .sort(["kind", "stat", "fam", "sub"]))
pl.Config.set_tbl_rows(400)
pl.Config.set_tbl_width_chars(200)
for kind in ("to", "ft"):
    for stat in ("nonfinite", "level_mean", "feed_std_mean", "feed_std_median", "tie_frac", "pearson_vsF7root",
                 "spearman_vsF7root", "pearson_vsF7other", "pearson_intra_nextroot"):
        t = agg.filter((pl.col("kind") == kind) & (pl.col("stat") == stat))
        if len(t) == 0:
            continue
        w = t.pivot(on="sub", index="fam", values="mean")
        print(f"\n## kind={kind} stat={stat} (mean over 7 roots)")
        print(w)
