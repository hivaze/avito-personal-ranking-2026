"""ap05 (mechanism diagnostic): is the production fresh-listing surplus a PLACEMENT effect? Full train, production cls 0
M>=50 vs exploration-like cls 1. (1) production positive-rate lift by age within shown-rank buckets; (2) the share of
fresh listings by rank bucket (does production front-load fresh listings?); (3) examination-adjusted production lift:
rows weighted by 1/theta_rand(rank) (data_usage out/theta_rand_rank.npy, clipped [0.3, 5] as 07_gbdt_arms.py);
(4) production lift at deep ranks (>= 20) only. L_* = log(prod lift / expl lift). Writes out/ap05.log."""
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("<repo>")
DU = ROOT / "nn_x/insights/data_usage/out"
AP = ROOT / "nn_x/audit/ageprod/out"
T0 = time.time()
lf = open(AP / "ap05.log", "w")


def log(*a):
    s = f"[{time.time() - T0:6.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); lf.write(s + "\n"); lf.flush()


KNOTS = np.load(ROOT / "data/feats/life_f4_age_knots.npy"); K = 64; CAP = 63 / 64 - 1e-6
EDGES = [0, 3, 12, 24, 72, 336, 2160, 8760, 1e9]
LAB = ["<3h", "3-12h", "12-24h", "1-3d", "3-14d", "2w-3m", "3m-1y", ">1y"]


def age_bin(x):
    x = np.asarray(x, dtype=np.float64); u = x * K; k = np.clip(np.floor(u).astype(int), 1, K - 2); w = u - k
    a = np.where(x < CAP, KNOTS[k] * (1 - w) + KNOTS[k + 1] * w, np.nan)
    return np.digitize(np.nan_to_num(a, nan=1e8), EDGES) - 1


t = pl.read_parquet(DU / "train_feeds.parquet", columns=["M", "cls"])
M = t["M"].to_numpy().astype(np.int64); cls = t["cls"].to_numpy()
N = int(M.sum())
f4 = np.load(ROOT / "data/feats/train/f4.npy", mmap_mode="r")
b = np.empty(N, np.int8)
for s in range(0, N, 20_000_000):
    b[s:s + 20_000_000] = age_bin(np.asarray(f4[s:s + 20_000_000]))
y = np.asarray(np.load(ROOT / "data/flat/train/cand_target.npy", mmap_mode="r")).astype(np.float32)
rank = np.asarray(np.load(DU / "nn_artifacts/train_rank_true.npy", mmap_mode="r")).astype(np.int64)
cr = np.repeat(cls, M); Mr = np.repeat(M, M)
THETA = np.clip(np.load(DU / "theta_rand_rank.npy"), 0.3, 5.0)
th = THETA[np.minimum(rank, 299)]
del f4
log("loaded", N, "rows")


def lift(mask, w=None):
    bb = b[mask]; yy = y[mask] if w is None else y[mask] * w[mask]
    n = np.bincount(bb, minlength=8).astype(float); p = np.bincount(bb, weights=yy, minlength=8)
    base = yy.sum() / len(yy)
    return (p / np.maximum(n, 1)) / base, n, p


expl = cr == 1
le, ne, pe = lift(expl)
log(f"exploration cls1 lift by age {np.round(le, 3)} rows {ne.astype(int)} pos {pe.astype(int)}")
prod = (cr == 0) & (Mr >= 50)
lp, npr, pp = lift(prod)
log(f"production M>=50 lift by age {np.round(lp, 3)}; L = {np.round(np.log(lp / le), 3)}")
log("== (1) production lift by age WITHIN shown-rank buckets, and (2) age share by rank bucket (prod vs expl) ==")
for lo, hi in [(0, 1), (1, 5), (5, 20), (20, 60), (60, 10 ** 6)]:
    mp = prod & (rank >= lo) & (rank < hi); me = expl & (rank >= lo) & (rank < hi)
    l1, n1, p1 = lift(mp); l2, n2, p2 = lift(me)
    log(f"  rank [{lo},{hi}): prod lift {np.round(l1, 2)} (rows {int(n1.sum())}, pos {int(p1.sum())}) | expl lift {np.round(l2, 2)} (rows {int(n2.sum())}) | L_within = {np.round(np.log(l1 / l2), 2)}")
    log(f"               age share prod {np.round(n1 / n1.sum(), 3)} | expl {np.round(n2 / n2.sum(), 3)}")
log("== (3) examination-adjusted production lift (y / theta_rand(rank)) ==")
la, _, _ = lift(prod, 1.0 / th)
log(f"  prod M>=50 exam-adjusted lift {np.round(la, 3)}; L_adj = log(adj lift / expl lift) = {np.round(np.log(la / le), 3)}")
log("== (4) deep-rank-only production lift (rank >= 20) ==")
ld, nd, pd_ = lift(prod & (rank >= 20))
log(f"  prod M>=50 rank>=20 lift {np.round(ld, 3)}; L_deep = {np.round(np.log(ld / le), 3)} (rows {int(nd.sum())}, pos {int(pd_.sum())})")
# where do fresh positives sit in production? share of positives at rank 0-4 by age bin
log("== share of production positives at rank 0-4, by age bin ==")
top = prod & (rank < 5)
sh = np.bincount(b[top], weights=y[top], minlength=8) / np.maximum(np.bincount(b[prod], weights=y[prod], minlength=8), 1)
log(f"  {np.round(sh, 3)} (exploration: {np.round(np.bincount(b[expl & (rank < 5)], weights=y[expl & (rank < 5)], minlength=8) / np.maximum(np.bincount(b[expl], weights=y[expl], minlength=8), 1), 3)})")
log("done")
