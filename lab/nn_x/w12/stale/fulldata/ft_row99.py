"""Item 4: what COVERAGE_MAP row 99 ("refit on full val: -0.00025 +- 0.00018", ROUND5 section 2b) measured, reproduced from the
stored nested predictions, + the pool-level re-read declared in PREDECLARED.txt R3. CPU only, about 1 minute.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/fulldata/ft_row99.py

Inputs (read-only): nn_x/solvers/preds/refit/nested_{v15_cd_all_e10,v14_s93,v17_pre2s_s102}.npz with
  P_i  = raw-score mean of 5 INNER fold models, each fine-tuned on 4/5 of the outer fit set (64% of val),
  P_ii = ONE model fine-tuned on the whole outer fit set (80% of val), outer = the production u0-singleton GroupKFold(5),
  and preds/nn_<v>_n2cpv3_val_oof.npy = the production OOF (one model per fold on the same 80%, another seed).
Every val feed is scored once, by models that never saw it.
"""
import json
import os
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/src")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

R = "<repo>"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(f"{HERE}/logs/ft_row99.txt", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


f = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)
fid = np.repeat(np.arange(len(M)), M)
n = len(M)


def fz(s):
    s = np.asarray(s, np.float64)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


def ev(s):
    return ndcg_per_feed(np.asarray(s, np.float64), y, off, M)


def ms(d, m=None):
    d = d if m is None else d[m]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def fmt(t):
    return f"{t[0]:+.5f} +- {t[1]:.5f} (z {t[0] / t[1]:+.2f})"


V = ["v15_cd_all_e10", "v14_s93", "v17_pre2s_s102"]
Pi, Pii, Pp = {}, {}, {}
for v in V:
    z = np.load(f"{R}/nn_x/solvers/preds/refit/nested_{v}.npz")
    Pi[v], Pii[v] = z["P_i"].astype(np.float64), z["P_ii"].astype(np.float64)
    Pp[v] = np.load(f"{R}/preds/nn_{v}_n2cpv3_val_oof.npy").astype(np.float64)
    outer = z["outer"]
res = {"n_feeds": int(n)}
P(f"val feeds {n}; outer folds {np.bincount(outer).tolist()} (fit set of an outer fold = {1 - np.bincount(outer).mean() / n:.1%} of val)")
P("== 1. Row 99 reproduced (member level, the pre-declared statistic: per-feed mean of the 3 member deltas)")
d_ii = [ev(Pii[v]) - ev(Pi[v]) for v in V]
d_pr = [ev(Pp[v]) - ev(Pi[v]) for v in V]
d_2s = [ev(0.5 * Pii[v] + 0.5 * Pp[v]) - ev(Pi[v]) for v in V]
d_mix = [ev(0.5 * Pii[v] + 0.5 * Pi[v]) - ev(Pi[v]) for v in V]
for v, a, b in zip(V, d_ii, d_pr):
    P(f"   {v:16s} NDCG (i) 5x64% {ev(Pi[v]).mean():.5f} | (ii) 1x80% {ev(Pii[v]).mean():.5f} | prod OOF 1x80% {ev(Pp[v]).mean():.5f} | (ii)-(i) {fmt(ms(a))} | prod-(i) {fmt(ms(b))}")
res["row99_ii_minus_i"] = ms(np.mean(d_ii, 0))
res["prod_minus_i"] = ms(np.mean(d_pr, 0))
res["two_seed_refit_minus_i"] = ms(np.mean(d_2s, 0))
res["mix_minus_i"] = ms(np.mean(d_mix, 0))
P(f"   pooled (ii) - (i)  [ONE model on 80% of val  vs  mean of FIVE models on 64% each]: {fmt(res['row99_ii_minus_i'])}   <- row 99")
P(f"   pooled prod - (i)  [another seed of one 80% model vs the same five]:              {fmt(res['prod_minus_i'])}")
P(f"   pooled 2 x 80% - (i) [raw mean of the two 80% seeds vs the five]:                  {fmt(res['two_seed_refit_minus_i'])}")
P(f"   pooled 0.5 (i) + 0.5 (ii) - (i):                                                  {fmt(res['mix_minus_i'])}")
# 1/k law from the two points (1 model, 2 models at 80%): N(k) = a - b / k, relative to (i)
n1 = 0.5 * (res["row99_ii_minus_i"][0] + res["prod_minus_i"][0])
n2 = res["two_seed_refit_minus_i"][0]
b = 2 * (n2 - n1)
a = n1 + b
res["one_over_k"] = dict(n1=n1, n2=n2, b=b, a=a, five_models_80=a - b / 5, fifteen_models_80=a - b / 15)
P(f"   1/k extrapolation at 80% data per model (relative to 5 x 64%): 1 model {n1:+.5f}, 2 models {n2:+.5f} -> b {b:.5f}, "
  f"5 models {a - b / 5:+.5f}, infinite {a:+.5f}  (two-point fit: indicative only, SE about 0.0002)")
P("== 2. Pool-level re-read (PREDECLARED R3): members pooled by per-feed z BEFORE scoring")
zi = np.mean([fz(Pi[v]) for v in V], 0)
zii = np.mean([fz(Pii[v]) for v in V], 0)
zii2 = np.mean([0.5 * fz(Pii[v]) + 0.5 * fz(Pp[v]) for v in V], 0)
zp = np.mean([fz(Pp[v]) for v in V], 0)
ni, nii, nii2, npd = ev(zi), ev(zii), ev(zii2), ev(zp)
res["pool"] = dict(ndcg_pool_i=float(ni.mean()), ndcg_pool_ii=float(nii.mean()), ndcg_pool_ii2=float(nii2.mean()), ndcg_pool_prod=float(npd.mean()),
                   D_pool=ms(nii - ni), D_pool_prod=ms(npd - ni), D_pool2=ms(nii2 - ni))
P(f"   pool of 3 members: (i) 15 models x 64% {ni.mean():.5f} | (ii) 3 models x 80% {nii.mean():.5f} | prod 3 x 80% {npd.mean():.5f} | 6 x 80% {nii2.mean():.5f}")
P(f"   D_pool  = pool(ii) - pool(i)            : {fmt(res['pool']['D_pool'])}")
P(f"   (second seed) pool(prod) - pool(i)      : {fmt(res['pool']['D_pool_prod'])}")
P(f"   D_pool2 = pool(ii + prod) - pool(i)     : {fmt(res['pool']['D_pool2'])}")
for nm, m in (("M<50", M < 50), ("M>=50", M >= 50)):
    P(f"     {nm}: D_pool {fmt(ms(nii - ni, m))} | D_pool2 {fmt(ms(nii2 - ni, m))}")
d, se = res["pool"]["D_pool"]
verdict = ("more val per model wins once members are pooled" if d >= 2 * se else "the full-val refit loses at pool level too"
           if d <= -2 * se else "no evidence either way")
res["R3_reading"] = verdict
P(f"   R3 reading: {verdict}")
json.dump(res, open(f"{HERE}/logs/ft_row99.json", "w"), indent=1)
