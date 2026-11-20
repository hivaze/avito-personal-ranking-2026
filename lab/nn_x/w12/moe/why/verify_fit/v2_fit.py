"""verify_fit v2 (CPU, read-only): independent checks of why/fit F1b / F2 / F3.
 (a) pure-numpy re-derivation (no fitlib / numba) of the in-sample (RIN) and holdout loss terms per feed for arms A/B/C vs the same-seed reference,
     @last checkpoints, from the analyst's forward-pass scores (scratchpad why/fit/scores) and meta.npz (y, teacher logits).
 (b) pool-level transfer: 4-pool (rank-z mean) delta on RIN (from scores), HOLD and VAL (stored @last preds) for several variants.
 (c) robustness of the F2 cross-variant correlations (leave-one-out, without the extreme variants, Spearman) from f12_summary@last.npy.
 (d) seed-variance vs systematic split of KL(teacher || student) for ref / B / C: KL of log-linear pools of N = 1, 2, 4 seeds, fit a + b / N.
Output: why/verify_fit/v2_fit.txt"""
import os

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/recheck/pool_composition")
from pc_common import Meta, ndcg_fast, se, feed_rank_z  # noqa: E402

R = Path("<repo>")
FIT = "/tmp/scratchpad/why/fit"
mt = np.load(f"{FIT}/meta.npz")
lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))


def terms(s, t, y, M):
    """numpy: per-feed click CE, KL(q||p), main = 0.5 cc + 0.5 (H(q) + KL). Feeds without clicks -> NaN for cc / main."""
    s = s.astype(np.float64); t = t.astype(np.float64); y = y.astype(np.float64)
    off = np.concatenate([[0], np.cumsum(M)[:-1]])
    fid = np.repeat(np.arange(len(M)), M)
    ms = np.maximum.reduceat(s, off); mt_ = np.maximum.reduceat(t, off)
    lps = s - (ms + np.log(np.add.reduceat(np.exp(s - ms[fid]), off)))[fid]
    lqt = t - (mt_ + np.log(np.add.reduceat(np.exp(t - mt_[fid]), off)))[fid]
    q = np.exp(lqt)
    ct = -np.add.reduceat(q * lps, off)
    hq = -np.add.reduceat(q * lqt, off)
    k = np.add.reduceat(y, off)
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(k > 0, -np.add.reduceat(y * lps, off) / k, np.nan)
    return dict(cc=cc, kl=ct - hq, main=0.5 * cc + 0.5 * ct)


def pse(x):
    x = x[np.isfinite(x)]
    return f"{x.mean():+.5f}+-{x.std(ddof=1) / np.sqrt(len(x)):.5f}"


S4 = range(951, 955)
ARMS = {"A": "kd05w192moeffn", "B": "kd05w192moevo", "C": "kd05w192b3ff1"}
P("== (a) pure-numpy re-derivation, @last, 4 seeds paired per feed with the same-seed reference (seed-averaged per feed, feed SE)")
for sp in ("rin", "hold"):
    y, t, M = mt[f"{sp}_y"], mt[f"{sp}_t"], mt[f"{sp}_M"]
    off = mt[f"{sp}_off"]
    ref = {s: np.load(f"{FIT}/scores/w5_e10_kd05w192_s{s}@last.npz")[sp] for s in S4}
    rt = {s: terms(ref[s], t, y, M) for s in S4}
    rn = {s: ndcg_fast(ref[s], y.astype(np.float32), off, M) for s in S4}
    for a, arm in ARMS.items():
        D = {c: [] for c in ("main", "cc", "kl", "nd")}
        for s in S4:
            x = np.load(f"{FIT}/scores/w5_e10_{arm}_s{s}@last.npz")[sp]
            tt = terms(x, t, y, M)
            for c in ("main", "cc", "kl"):
                D[c].append(tt[c] - rt[s][c])
            D["nd"].append(ndcg_fast(x, y.astype(np.float32), off, M) - rn[s])
        P(f"   {sp:4s} {a}: " + "  ".join(f"d{c} {pse(np.nanmean(D[c], 0))}" for c in ("main", "cc", "kl", "nd")))

# ---------------------------------------------------------------- (b) pool-level transfer
m = Meta()


def base(v):
    return R / "lab/nn_x/w12/moe/out" if ("moe" in v or "b3ff1" in v) else R / "nn_x/w5"


def stored(v, sp):
    p = base(v) / "preds" / f"nn_{v}@last_trainonly_val.npy" if sp == "val" else base(v) / "preds/holdout" / f"nn_{v}@last_holdout.npy"
    return np.load(p).astype(np.float32)


def pool_nd(fam, sp):
    if sp == "rin":
        zs = [feed_rank_z(mt["rin_off"], mt["rin_M"], np.load(f"{FIT}/scores/w5_e10_{fam}_s{s}@last.npz")["rin"])[:, 1] for s in S4]
        return ndcg_fast(np.mean(zs, 0), mt["rin_y"].astype(np.float32), mt["rin_off"], mt["rin_M"])
    zs = [m.z(sp, stored(f"w5_e10_{fam}_s{s}", sp)) for s in S4]
    return ndcg_fast(np.mean(zs, 0), m.y[sp], m.off[sp], m.M[sp])


P("")
P("== (b) 4-pool (rank-z mean, seeds 951-954, @last) delta vs the reference 4-pool: in-sample RIN -> HOLD -> VAL; and member deltas for contrast")
refp = {sp: pool_nd("kd05w192", sp) for sp in ("rin", "hold", "val")}
for fam in ("kd05w192moeffn", "kd05w192moevo", "kd05w192b3ff1", "kd05w224", "kd05", "kwh6", "kwd2", "kw256d2", "kd08w192"):
    d = {sp: pool_nd(fam, sp) - refp[sp] for sp in ("rin", "hold", "val")}
    P(f"   {fam:16s} pool: RIN {pse(d['rin'])} HOLD {pse(d['hold'])} VAL {pse(d['val'])} | pool transfer hold/RIN {d['hold'].mean() / d['rin'].mean():+.2f} "
      f"val/RIN {d['val'].mean() / d['rin'].mean():+.2f}")

# ---------------------------------------------------------------- (c) F2 correlation robustness
P("")
P("== (c) F2 cross-variant correlations (member deltas, @last; from the analyst's f12_summary@last.npy)")
S = np.load(f"{FIT}/f12_summary@last.npy", allow_pickle=True).item()
names, X = [], []
for k, v in S.items():
    if k in ("base", "kd08w192"):
        continue
    names.append(k)
    X.append((v[("rin", "nd")][0], v[("hold_st", "nd")][0], v[("val_st", "nd")][0], v[("hold_st", "kl")][0], v[("val_st", "kl")][0],
              v[("rin", "cc")][0], v[("rin", "kl")][0]))
X = np.array(X)
rk = lambda a: np.argsort(np.argsort(a))
for nm, i, j in (("in-sample dcc vs HOLD dKL", 5, 3), ("in-sample dKL vs HOLD dKL", 6, 3), ("RIN dNDCG vs HOLD dNDCG", 0, 1),
                 ("RIN dNDCG vs VAL dNDCG", 0, 2), ("HOLD dKL vs VAL dNDCG", 3, 2)):
    r = np.corrcoef(X[:, i], X[:, j])[0, 1]
    loo = [np.corrcoef(np.delete(X[:, i], q), np.delete(X[:, j], q))[0, 1] for q in range(len(X))]
    keep = [q for q, n in enumerate(names) if n not in ("kw_drop0", "kw_e20", "kd05", "kwd2")]
    r_mid = np.corrcoef(X[keep, i], X[keep, j])[0, 1]
    rs = np.corrcoef(rk(X[:, i]), rk(X[:, j]))[0, 1]
    P(f"   {nm:28s} n={len(X)} r {r:+.2f} | leave-one-out range [{min(loo):+.2f}, {max(loo):+.2f}] (dropping {names[int(np.argmax(np.abs(np.array(loo) - r)))]} moves it most) "
      f"| without drop0 / e20 / d128 / dropout0.2 (n={len(keep)}) r {r_mid:+.2f} | Spearman {rs:+.2f}")
P("   note: in-sample KL vs out-of-sample KL per variant: " + " ".join(f"{n.replace('kd05w192', '')}:{X[q, 6]:+.4f}/{X[q, 3]:+.4f}" for q, n in enumerate(names)))

# ---------------------------------------------------------------- (d) variance vs systematic KL
P("")
P("== (d) KL(teacher || log-linear pool of N seeds), N = 1, 2, 4 (all subsets of 951-954), fit a + b/N: a = systematic, b = seed variance")
tvh = np.load(f"{FIT}/vh/teacher.npz")
T = {"val": tvh["val"], "hold": tvh["hold"]}


def kl_all(t, s, M):
    return np.nanmean(terms(s, t, np.zeros_like(t), M)["kl"])


for sp in ("hold", "val"):
    for fam in ("kd05w192", "kd05w192moeffn", "kd05w192moevo", "kd05w192b3ff1", "kd05w224"):
        raw = [stored(f"w5_e10_{fam}_s{s}", sp) for s in S4]
        Ns, Ks = [], []
        for N in (1, 2, 4):
            Ks.append(np.mean([kl_all(T[sp], np.mean([raw[i] for i in c], 0), m.M[sp]) for c in combinations(range(4), N)]))
            Ns.append(N)
        b, a0 = np.polyfit(1 / np.array(Ns, float), Ks, 1)
        P(f"   {sp:4s} {fam:16s} KL N=1 {Ks[0]:.5f} N=2 {Ks[1]:.5f} N=4 {Ks[2]:.5f} | systematic a {a0:.5f} seed-variance b {b:.5f} (share of member KL: "
          f"systematic {a0 / Ks[0]:.2f})")
open("<repo>/lab/nn_x/w12/moe/why/verify_fit/v2_fit.txt", "w").write("\n".join(lines) + "\n")
