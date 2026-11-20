"""EXTRA to step 5 (descriptive, no model, nothing trained): within-feed AUC of the density-matched fresh counts for the
TRAIN click label on the train pseudo-period L rows (population RM = the population whose counts match val / test),
next to the train-exposure yardstick built the same way as xp_imp_c50 (impressions in all train feeds of days <= 20708).
Purpose: the same descriptive read as step 2 on 2.6x more feeds, from rows whose count distribution equals val / test.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 NUMBA_NUM_THREADS=6 .venv/bin/python nn_x/w12/fresh/pregate/s5b_train_auc.py
"""
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import FLAT, OUT, WIN, count, day_of, forms

ftr = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
ts = ftr["timestamp"].to_numpy().astype(np.int64); uid = ftr["user_id"].to_numpy().astype(np.int64)
M = ftr["M"].to_numpy().astype(np.int64); off = ftr["cand_off"].to_numpy().astype(np.int64); day = day_of(ts)
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
p_random = np.zeros(len(ftr)); p_random[pr["fidx"]] = np.nan_to_num(pr["p_random"], nan=0.0)
items = np.load(f"{FLAT}/train/cand_item_id.npy", mmap_mode="r"); tgt = np.load(f"{FLAT}/train/cand_target.npy", mmap_mode="r")
fv = pl.read_parquet(f"{FLAT}/val/feeds.parquet"); fte = pl.read_parquet(f"{FLAT}/test/feeds.parquet")
M_vt = np.concatenate([fv["M"].to_numpy(), fte["M"].to_numpy()]).astype(np.int64)
mq = np.unique(np.quantile(M_vt, np.linspace(0, 1, 21)[1:-1])); mshare = np.bincount(np.digitize(M_vt, mq), minlength=len(mq) + 1) / len(M_vt)
DL = [20709, 20710, 20713, 20714, 20715]; NL = [7542, 6878, 7367, 7578, 7600]
rng = np.random.default_rng(7); fe = []
for d, n in zip(DL, NL):  # identical to s5_density.sample(..., "RM", seed 7)
    idx = np.flatnonzero((day == d) & (p_random > 0.9)); b = np.digitize(M[idx], mq)
    t = np.floor(mshare * n).astype(int); t[np.argsort(-(mshare * n - t))[:n - t.sum()]] += 1
    fe.append(np.sort(np.concatenate([rng.choice(idx[b == k], t[k], replace=False) for k in range(len(mshare))])))
fe = np.concatenate(fe)
rows = np.concatenate([np.arange(off[f], off[f] + M[f]) for f in fe])
it = np.asarray(items[rows]).astype(np.int64); y = np.asarray(tgt[rows]).astype(np.float64)
Ms = M[fe]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64); gid = np.repeat(np.arange(len(fe)), Ms)
F = dict(ts=ts[fe], day=day[fe], eff_user=np.where(uid[fe] == 0, -1 - np.arange(len(fe)), uid[fe]).astype(np.int64))
c = count(it, gid, F, it, gid)
cut = off[np.searchsorted(day, 20709)]
imp = np.bincount(np.asarray(items[:cut]), minlength=111_400_000)[it].astype(np.float64)  # train impressions, days <= 20708
cold = imp == 0
print(f"RM period L: {len(fe)} feeds, {len(it):,} rows, click rate {y.mean():.5f}, feeds with >= 1 click {(np.bincount(gid, y) > 0).mean():.3f}, cold share {cold.mean():.3f}")


def feed_auc(x, sub=None):
    r_ = np.arange(len(x)) if sub is None else np.flatnonzero(sub)
    g = gid[r_]; yy = y[r_]
    m = np.bincount(g, minlength=len(fe)).astype(np.int64); keep = m > 0
    Mk = m[keep]; ok_ = np.concatenate([[0], np.cumsum(Mk)[:-1]]).astype(np.int64)
    r, _ = forms(ok_, Mk, x[r_]); fk = np.repeat(np.arange(len(Mk)), Mk)
    P = np.bincount(fk, yy, len(Mk)); Ng = Mk - P; ok = (P > 0) & (Ng > 0)
    sr = np.bincount(fk, r * (Mk[fk] - 1) * yy, len(Mk))
    a = (sr[ok] - P[ok] * (P[ok] - 1) / 2) / (P[ok] * Ng[ok])
    return float(a.mean()), float(a.std() / np.sqrt(ok.sum())), int(ok.sum())


res = {}
print(f"{'feature':38s} {'ALL':>24s} {'COLD':>24s} {'WARM':>24s}")
for name, x in [("YARD train impressions <= 20708 (c50)", np.minimum(imp, 50))] + [(f"fresh n_{wn}", c[:, w].astype(np.float64)) for w, wn in enumerate(WIN)]:
    o = [feed_auc(x), feed_auc(x, cold), feed_auc(x, ~cold)]
    res[name] = o
    print(f"{name:38s} " + " ".join(f"{a:.4f}+-{s:.4f} n={n:5d}" for a, s, n in o))
bk = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 10 ** 9, "5+")]
print("click rate (percent) by count bucket, cold / warm [share of rows in percent]")
for w, wn in enumerate(WIN):
    for sub, sm in (("cold", cold), ("warm", ~cold)):
        print(f"  {wn:4s} {sub:5s} " + " | ".join(f"n={lab}: {100 * y[sm & (c[:, w] >= lo) & (c[:, w] <= hi)].mean():5.2f} [{100 * (sm & (c[:, w] >= lo) & (c[:, w] <= hi)).sum() / sm.sum():5.1f}]" for lo, hi, lab in bk))
# by position in the pseudo-period: val-like days (20709, 20710) vs test-like days (20713-20715), and the last day alone
d_r = day[fe][gid]
print("as-of count by position in the pseudo-period: share of rows with count >= 1, within-feed AUC (all / cold candidates)")
res["by_position"] = {}
for nm, m in (("val-like days 20709-20710", d_r <= 20710), ("test-like days 20713-20715", d_r >= 20713), ("last day 20715", d_r == 20715)):
    x = c[:, 0].astype(np.float64)
    o = [feed_auc(x, m), feed_auc(x, m & cold)]
    res["by_position"][nm] = dict(cov=float((c[m, 0] >= 1).mean()), cov_cold=float((c[m & cold, 0] >= 1).mean()), auc_all=o[0], auc_cold=o[1])
    print(f"  {nm:28s} coverage {100 * (c[m, 0] >= 1).mean():5.1f}% (cold {100 * (c[m & cold, 0] >= 1).mean():5.1f}%) | AUC all {o[0][0]:.4f}+-{o[0][1]:.4f} n={o[0][2]} | cold {o[1][0]:.4f}+-{o[1][1]:.4f} n={o[1][2]}")
json.dump(res, open(f"{OUT}/s5b_train_auc.json", "w"), indent=1)
print("DONE")
