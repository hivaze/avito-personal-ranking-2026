"""staleft_review: extra checks on the pilot predictions (CPU only).
 1 completeness of the OOF files (every feed scored; no all-zero feed) and identity run re-read
 2 where the pooled-W8 stale-val gain sits: by feed cold share (pmix segment), W8 form and W8P20 form
 3 member-set sensitivity of the pooled statistic (halves, by architecture, leave-one-seed-out)
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft_review")
import numpy as np
import polars as pl

import rederive as R

V, P, PR, W5, AB = R.V, R.P, R.PR, R.W5, R.AB
R.lines.clear()
A = ("kd05w192", "kd05t25w192")
W8 = R.W8

P("=== 1a. completeness: feeds whose prediction is constant (all equal, e.g. all zero) per file ===")
bad = 0
nfiles = 0
for m in W8:
    for t in ("p0", "p05", "p10"):
        for k in ("val_oof", "val_oof_stale", "val_e0", "val_e0_stale"):
            x = np.load(PR / f"nn_{m}_sft_{t}_{k}.npy").astype(np.float64)
            assert x.shape[0] == V.M.sum() and np.isfinite(x).all()
            mu = np.bincount(V.fid, x, V.n) / V.M
            var = np.bincount(V.fid, x * x, V.n) / V.M - mu ** 2
            c = int(((var <= 1e-12) & (V.M > 1)).sum())
            bad += c; nfiles += 1
P(f"{nfiles} val files of the 8 W8 members checked: {bad} constant feeds in total; all finite, all of length {int(V.M.sum())}")
x = np.load(PR / "nn_w5_e10_kd05w192_s951_sft_p0_val_e0.npy"); y = np.load(PR / "nn_w5_e10_kd05w192_s951_sft_p10_val_e0.npy"); z = np.load(PR / "nn_w5_e10_kd05w192_s951_sft_p05_val_e0.npy")
P(f"epoch-0 (untrained) fresh predictions p0 vs p10 vs p05, member s951 kd05w192: max|d| {np.abs(x - y).max():.2e} / {np.abs(x - z).max():.2e} (same start, same folds)")
xs = np.load(PR / "nn_w5_e10_kd05w192_s951_sft_p0_val_e0_stale.npy"); ys = np.load(PR / "nn_w5_e10_kd05w192_s951_sft_p10_val_e0_stale.npy")
P(f"epoch-0 stale predictions p0 vs p10: max|d| {np.abs(xs - ys).max():.2e}; fresh vs stale differ on {np.mean(x != xs):.3f} of rows")

P("\n=== 1b. identity run (--stale_train 0 --seeds 3 --only_folds 0) against the stored 3-seed file, fold-0 feeds ===")
for nm, stored in (("w5_e10_kd05w192_s951", W5 / "nn_w5_e10_kd05w192_s951_n2cpv3s3_val_oof.npy"), ("abl_e10_pop1kd05t25_s951", AB / "nn_abl_e10_pop1kd05t25_s951_n2cpv3s3_val_oof.npy")):
    z_ = np.load(PR / f"nn_ident_{nm}_s3_partial_folds0.npz")
    mine = z_["oof_fin"].astype(np.float64); st = np.load(stored).astype(np.float64)
    nz = np.bincount(V.fid, mine != 0, V.n) > 0
    c, t1 = V.corr_top1(np.where(nz[V.fid], mine, st), st)
    d = (V.ndcg(np.where(nz[V.fid], mine, st)) - V.ndcg(st))[nz]
    P(f"{nm}: fold-0 feeds {int(nz.sum())}; per-feed corr {c[nz].mean():.6f}; NDCG mine - stored {d.mean():+.6f} +- {d.std(ddof=1) / np.sqrt(len(d)):.6f}; top-1 differs {t1[nz].mean():.4f}")

P("\n=== 2. where the P=1.0 delta sits: by feed cold share (eda/errors/segments/feed_table.parquet), final file, paired vs the P=0 control ===")
d = pl.read_parquet("<repo>/eda/errors/segments/feed_table.parquet")
cs = d["cold_share"].to_numpy()
pf = np.load(R.OUT / "perfeed.npz")
bins = [(0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 1.01)]
tof8, tos8 = R.to_pools(W8)
key = {"fresh": "val_oof", "stale": "val_oof_stale"}
P1 = R.P1
tof = {"fresh": (tof8, R.zm([np.load(AB / f"nn_{m}_trainonly_val.npy") for m in P1])), "stale": (tos8, R.zm([R.ld(m, "p0", "val_e0_stale") for m in P1]))}
ftp = {v: R.zm([R.ld(m, "p0", key[v]) for m in P1]) for v in key}


def f20(tag, v):
    a = R.zm([R.ld(m, tag, key[v]) for m in W8])
    return V.ndcg(0.5 * V.z(0.8 * V.z(a) + 0.2 * V.z(ftp[v])) + 0.5 * V.z(0.8 * V.z(tof[v][0]) + 0.2 * V.z(tof[v][1])))


c20 = {v: f20("p0", v) for v in key}
for t in ("p10", "p05"):
    d20 = {v: f20(t, v) - c20[v] for v in key}
    for form, dd in (("W8 form  ", {"fresh": pf[f"w8_{t}_fin_fresh"] - pf["w8_p0_fin_fresh"], "stale": pf[f"w8_{t}_fin_stale"] - pf["w8_p0_fin_stale"]}), ("W8P20 form", d20)):
        for v in ("fresh", "stale"):
            s = " | ".join(f"cold {int(lo * 100)}-{int(min(hi, 1) * 100)}% (n {int(((cs >= lo) & (cs < hi)).sum())}): {R.fm(dd[v][(cs >= lo) & (cs < hi)])}" for lo, hi in bins)
            P(f"{t} {form} {v:5s}: {s}")

P("\n=== 3. member-set sensitivity of the P=1.0 pool statistic (final file; fresh | stale-val | stale at pmix-private) ===")


def stat(members, tag="p10"):
    tf, ts = R.to_pools(members)
    c = R.levels(members, "p0", tf, ts); v = R.levels(members, tag, tf, ts)
    return v["fin_fresh"] - c["fin_fresh"], v["fin_stale"] - c["fin_stale"]


sets = {"pilot s951-952 (4)": R.PILOT, "replication s953-954 (4)": R.REP, "arch kd05w192 (4)": [m for m in W8 if "_kd05w192_" in m], "arch kd05t25w192 (4)": [m for m in W8 if "_kd05t25w192_" in m],
        "odd seeds s951,s953 (4)": [m for m in W8 if m.endswith(("s951", "s953"))], "even seeds s952,s954 (4)": [m for m in W8 if m.endswith(("s952", "s954"))],
        "s951,s954 (4)": [m for m in W8 if m.endswith(("s951", "s954"))], "s952,s953 (4)": [m for m in W8 if m.endswith(("s952", "s953"))]}
for s in (951, 952, 953, 954):
    sets[f"leave out s{s} (6)"] = [m for m in W8 if not m.endswith(f"s{s}")]
sets["W8 (8)"] = W8
vals = []
for nm, mem in sets.items():
    a, b = stat(mem)
    vals.append((nm, a.mean(), b.mean()))
    P(f"{nm:28s}: fresh {R.fm(a)} | stale {R.fm(b)} | stale pmix-priv {R.fm(b, R.wpriv)} | stale pmix-pub {R.fm(b, R.wpub)}")
four = [v for v in vals if "(4)" in v[0]]
P(f"spread over the 8 four-member pools: fresh min {min(v[1] for v in four):+.5f} max {max(v[1] for v in four):+.5f} SD {np.std([v[1] for v in four], ddof=1):.5f}; "
  f"stale min {min(v[2] for v in four):+.5f} max {max(v[2] for v in four):+.5f} SD {np.std([v[2] for v in four], ddof=1):.5f}")
(R.OUT / "extra_checks.txt").write_text("\n".join(R.lines) + "\n")
