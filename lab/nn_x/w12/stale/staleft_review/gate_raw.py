"""staleft_review: (a) own rebuild from raw of two counterfactual columns (both_it_hist_all_luo = existing r6 file; both_it_hist_et5_luo = built by
the staleft agent); (b) own label-free gate (zero share by listing-age bucket, fresh val / stale val / private rows / public rows) for three columns;
(c) label-free facts on the extras that were KEPT FRESH inside the stale view (do they reveal the fresh count?); (d) prediction-level read of that.
CPU only; (d) opens no label file either.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FC = ROOT / "nn/feat_cache"
SF = ROOT / "nn_x/w12/stale/staleft"
R6 = ROOT / "nn_x/r6/stale_cf"
OUT = ROOT / "nn_x/w12/stale/staleft_review/out"
OUT.mkdir(exist_ok=True)
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


fv = pl.read_parquet(FLAT / "val/feeds.parquet")
ft = pl.read_parquet(FLAT / "test/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64)
offv = fv["cand_off"].to_numpy().astype(np.int64)
fidv = np.repeat(np.arange(len(Mv)), Mv)
item = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64)

# ---------------------------------------------------------------- (a) rebuild from raw
SEP17 = 1789851600 - 3 * 86400  # Sep 17 00:00 MSK (dense data end Sep 20 00:00 MSK = 1789851600)
ftr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["timestamp"])
late_feed = ftr["timestamp"].to_numpy().astype(np.int64) >= SEP17
hf = np.load(FLAT / "train/hist_fidx.npy", mmap_mode="r")
sel = late_feed[np.asarray(hf)]
hi = np.asarray(np.load(FLAT / "train/hist_item_id.npy", mmap_mode="r"))[sel].astype(np.int64)
he = np.asarray(np.load(FLAT / "train/hist_event_type.npy", mmap_mode="r"))[sel]
del sel, hf
thi = np.load(FLAT / "test/hist_item_id.npy").astype(np.int64)
the = np.load(FLAT / "test/hist_event_type.npy")
thf = np.load(FLAT / "test/hist_fidx.npy")
uq, inv = np.unique(np.concatenate([item, hi, thi]), return_inverse=True)  # compact ids, my own route (no 111M-bin count)
iv, il, it_ = inv[:len(item)], inv[len(item):len(item) + len(hi)], inv[len(item) + len(hi):]
nU = len(uq)
u_val = fv["user_id"].to_numpy()[fidv]
t_val = fv["timestamp"].to_numpy().astype(np.int64)[fidv]
H = pl.DataFrame({"user": ft["user_id"].to_numpy()[thf], "item": thi, "th": ft["timestamp"].to_numpy().astype(np.int64)[thf], "et": the.astype(np.int64)})
C = pl.DataFrame({"row": np.arange(len(item)), "user": u_val, "item": item, "tc": t_val})
J = C.filter(pl.col("user") != 0).join(H.filter(pl.col("user") != 0), on=["user", "item"], how="inner").filter(pl.col("th") > pl.col("tc"))
P("=== (a) rebuild from raw: stale = max(max(fresh - test_split_count + same_user_later_overlap, 0) - train_Sep17-19_count, 0) ===")
mine = {}
for nm, e in (("it_hist_all_luo", None), ("it_hist_et5_luo", 5)):
    lc = np.bincount(il if e is None else il[he == e], minlength=nU)[iv].astype(np.float64)
    tc = np.bincount(it_ if e is None else it_[the == e], minlength=nU)[iv].astype(np.float64)
    Je = J if e is None else J.filter(pl.col("et") == e)
    g = Je.group_by("row").agg(pl.len().alias("n"))
    ovl = np.zeros(len(item)); ovl[g["row"].to_numpy()] = g["n"].to_numpy()
    x = np.load(FC / "val" / f"{nm}.npy").astype(np.float64)
    st = np.maximum(np.maximum(x - tc + ovl, 0) - lc, 0)
    mine[nm] = st
    ref = (R6 if e is None else SF / "cf") / f"both_{nm}.npy"
    r = np.load(ref).astype(np.float64)
    P(f"{nm}: my rebuild vs {ref.relative_to(ROOT)}: max|d| {np.abs(st - r).max():.6f}, rows differing {int((st != r).sum())} of {len(r)}; "
      f"zero share fresh {np.mean(x == 0):.4f} stale {np.mean(st == 0):.4f}")
del hi, he, il

# ---------------------------------------------------------------- (b) gate, own code
kn = np.load(ROOT / "data/feats/life_f4_age_knots.npy")
EDG = np.array([0, 1, 3, 7, 10, 24, 76, np.inf])
NM = ["<1d", "1-3d", "3-7d", "7-10d", "10-24d", "24-76d", ">76d"]


def bucket(f4):
    x = f4.astype(np.float64); u = x * 64
    k = np.clip(np.floor(u).astype(int), 1, 62); w = u - k
    a = np.where(x < 63 / 64, kn[k] * (1 - w) + kn[k + 1] * w, np.nan)
    d = np.where(np.isnan(a), 1e9, a / 24.0)
    return np.clip(np.searchsorted(EDG, d, side="right") - 1, 0, 6)


bv = bucket(np.load(FLAT / "val/cand_f4.npy"))
bt = bucket(np.load(FLAT / "test/cand_f4.npy"))
pubrow = (ft["timestamp"].to_numpy().astype(np.int64) < 1790497920)[np.load(FLAT / "test/cand_fidx.npy")]
wv = np.bincount(bv, minlength=7) / len(bv)


def by(x, b):
    return np.bincount(b, weights=x.astype(np.float64), minlength=7) / np.maximum(np.bincount(b, minlength=7), 1)


P("\n=== (b) own gate: zero share by listing-age bucket; dZ = sum_b w_b |share_view - share_part|; dM on log1p scale / SD of the test part ===")
for nm, stale_path in (("it_hist_all_luo", R6 / "both_it_hist_all_luo.npy"), ("it_hist_et5_luo", SF / "cf/both_it_hist_et5_luo.npy"),
                       ("seq_cv_pop", R6 / "both_seq_cv_pop.npy"), ("pre_id_contrib_v2", SF / "cf/both_pre_id_contrib_v2.npy")):
    xv = np.load(FC / "val" / f"{nm}.npy").astype(np.float64); xs = np.load(stale_path).astype(np.float64); xt = np.load(FC / "test" / f"{nm}.npy").astype(np.float64)
    zf, zs = by(xv == 0, bv), by(xs == 0, bv)
    P(f"{nm}: pooled zero share fresh val {np.mean(xv == 0):.4f} | stale val {np.mean(xs == 0):.4f} | private {np.mean(xt[~pubrow] == 0):.4f} | public {np.mean(xt[pubrow] == 0):.4f}")
    cnt = nm.startswith(("it_hist", "seq_cv"))
    tf = (lambda a: np.log1p(np.maximum(a, 0))) if cnt else (lambda a: np.nan_to_num(a, nan=0.0))
    for pn, pm in (("private", ~pubrow), ("public", pubrow)):
        zt = by(xt[pm] == 0, bt[pm])
        sd = tf(xt[pm]).std()
        mf, msn, mt = by(tf(xv), bv), by(tf(xs), bv), by(tf(xt[pm]), bt[pm])
        P(f"   vs {pn:7s}: dZ fresh {np.sum(wv * np.abs(zf - zt)):.4f} -> stale {np.sum(wv * np.abs(zs - zt)):.4f} | dM fresh {np.sum(wv * np.abs(mf - mt)) / sd:.4f} -> stale {np.sum(wv * np.abs(msn - mt)) / sd:.4f}"
          f" | closer on dZ: {np.sum(wv * np.abs(zs - zt)) <= np.sum(wv * np.abs(zf - zt))}")
        if nm == "it_hist_all_luo":
            P("      zero share by bucket (fresh / stale / " + pn + "): " + "; ".join(f"{NM[b]} {zf[b]:.3f}/{zs[b]:.3f}/{zt[b]:.3f}" for b in range(7)))

# ---------------------------------------------------------------- (c) do the fresh-kept extras reveal the fresh count inside the stale view?
P("\n=== (c) stale view: 4 extras stay FRESH (gate_exclude: it_hist_all_luo_gap, et7, et9, et10) while their parent is stale ===")
x0 = np.load(FC / "val/it_hist_all_luo.npy").astype(np.float64)
xb = np.load(R6 / "both_it_hist_all_luo.npy").astype(np.float64)
gap = np.load(FC / "val/it_hist_all_luo_gap.npy").astype(np.float64)
mx = np.maximum.reduceat(x0, offv)[fidv]
P(f"gap == fresh count - fresh feed max exactly: max|d| {np.abs(gap - (x0 - mx)).max():.6f}")
z0 = xb == 0
nc = z0 & (x0 > 0)
ac = z0 & (x0 == 0)
P(f"rows with STALE count 0: {z0.mean():.4f} of val rows; of them newly cold (fresh > 0) {nc.sum() / z0.sum():.4f}, always cold {ac.sum() / z0.sum():.4f}")
# within a feed: is the fresh-kept gap of a stale-zero row above the feed's minimum gap (= the always-cold level -max)? -> flags a newly cold row exactly
flag = gap > (-mx)
P(f"flag 'kept-fresh gap > -feed max' among stale-zero rows: P(flag | newly cold) {flag[nc].mean():.4f}, P(flag | always cold) {flag[ac].mean():.4f}")
# the model does not see -max directly; it sees the gap of the other candidates. Feed-level: gap minimum over the feed equals -max iff a cold row exists
gmin = np.minimum.reduceat(gap, offv)[fidv]
f2 = gap > gmin
P(f"flag 'kept-fresh gap > feed min gap' among stale-zero rows: P(flag | newly cold) {f2[nc].mean():.4f}, P(flag | always cold) {f2[ac].mean():.4f}")
for e in (7, 9, 10):
    xe = np.load(FC / "val" / f"it_hist_et{e}_luo.npy").astype(np.float64)
    P(f"kept-fresh it_hist_et{e}_luo > 0 on a row whose stale all-count is 0 (a pattern no real row has): {np.mean(xe[z0] > 0):.4f} of stale-zero rows; "
      f"{np.mean(xe[nc] > 0):.4f} of newly cold rows")
ok = z0
cc = np.corrcoef(gap[ok] + mx[ok], np.log1p(x0[ok]))[0, 1]
P(f"among stale-zero rows, corr(log1p fresh count, kept-fresh gap) = {np.corrcoef(gap[ok], np.log1p(x0[ok]))[0, 1]:.3f} (gap + max recovers the count: {cc:.3f})")

# ---------------------------------------------------------------- (d) do the stale-trained models rank newly cold rows above always cold rows more than the control?
P("\n=== (d) label-free: mean per-feed z of the prediction on NEWLY COLD rows minus ALWAYS COLD rows (feeds that hold both), stale view unless said ===")
PR = SF / "preds"
W8 = [f"w5_e10_{a}_s{s}" for s in range(951, 955) for a in ("kd05w192", "kd05t25w192")]


def zfeed(x):
    x = np.asarray(x, np.float64)
    mu = np.bincount(fidv, x, len(Mv)) / Mv
    sd = np.sqrt(np.maximum(np.bincount(fidv, x * x, len(Mv)) / Mv - mu ** 2, 0)) + 1e-9
    return (x - mu[fidv]) / sd[fidv]


n_nc = np.bincount(fidv, nc, len(Mv)); n_ac = np.bincount(fidv, ac, len(Mv))
both = (n_nc > 0) & (n_ac > 0)


def gapstat(pred):
    z = zfeed(pred)
    d = np.bincount(fidv, z * nc, len(Mv))[both] / n_nc[both] - np.bincount(fidv, z * ac, len(Mv))[both] / n_ac[both]
    return d


res = {}
for lab, tag, k in (("train-only (no extras), stale view", "p0", "val_e0_stale"), ("P=0 fine-tune, stale view", "p0", "val_oof_stale"),
                    ("P=0.5 fine-tune, stale view", "p05", "val_oof_stale"), ("P=1.0 fine-tune, stale view", "p10", "val_oof_stale"),
                    ("P=0 fine-tune, FRESH view", "p0", "val_oof"), ("P=1.0 fine-tune, FRESH view", "p10", "val_oof")):
    d = np.mean([gapstat(np.load(PR / f"nn_{m}_sft_{tag}_{k}.npy")) for m in W8], 0)
    res[lab] = d
    P(f"{lab:38s}: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} ({int(both.sum())} feeds)")
for a, b in (("P=1.0 fine-tune, stale view", "P=0 fine-tune, stale view"), ("P=0.5 fine-tune, stale view", "P=0 fine-tune, stale view"),
             ("P=0 fine-tune, stale view", "train-only (no extras), stale view"), ("P=0 fine-tune, FRESH view", "P=0 fine-tune, stale view")):
    d = res[a] - res[b]
    P(f"paired: [{a}] - [{b}] = {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}")
(OUT / "gate_raw.txt").write_text("\n".join(lines) + "\n")
