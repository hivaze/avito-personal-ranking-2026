"""why/fit F1 / F2 / F3 read (CPU, read-only): train-objective fit on in-sample train feeds vs transfer to holdout / val.
Inputs: <OUT>/meta.npz + <OUT>/scores/<v>.npz (fwd_train.py, selected checkpoints scored on HOLD / RIN / PIN) and <OUT>/vh/<v>.npz
(f3_valhold.py, stored val / holdout preds). Every delta is paired per feed against the SAME-SEED reference w5_e10_kd05w192_s<seed>:
d_f = mean over seeds of (model_s,f - ref_s,f); SE = sd(d_f)/sqrt(n_feeds) (feed-level, seeds averaged) and, in brackets, the SE over
the seed-level deltas (between-seed). Pool rows: 4-member mean of per-feed rank-z vs the same-seed reference pool.
Output: why/fit/f12_train.txt"""
import os
import sys

sys.path.insert(0, "<repo>/lab/nn_x/w12/moe/why/fit")
from fitlib import stats_dict, ndcg_fast, feed_rank_z  # noqa: E402
from fwd_list import FAMS  # noqa: E402
import numpy as np  # noqa: E402

OUT = "/tmp/scratchpad/why/fit"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""  # "@last": every run's last checkpoint (step 98,350) instead of the holdout-selected one
TXT = "<repo>/lab/nn_x/w12/moe/why/fit/f12_train" + ("_last" if SUF else "") + ".txt"
mt = np.load(f"{OUT}/meta.npz")
SETS = ("pin", "rin", "hold")
lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))
cache = {}


def train_stats(v):
    if v not in cache:
        cp = f"{OUT}/tstats/{v}.npz"
        if os.path.exists(cp):
            z = np.load(cp)
            cache[v] = {k: {c: z[f"{k}_{c}"] for c in ("nd", "cc", "ct", "kl", "main", "exc", "r", "cct", "k")} for k in SETS}
        else:
            sc = np.load(f"{OUT}/scores/{v}.npz")
            d = {k: stats_dict(sc[k], mt[f"{k}_t"], mt[f"{k}_y"], mt[f"{k}_off"], mt[f"{k}_M"]) for k in SETS}
            os.makedirs(f"{OUT}/tstats", exist_ok=True)
            np.savez(cp, **{f"{k}_{c}": x for k, dd in d.items() for c, x in dd.items()})
            cache[v] = d
    return cache[v]


def vh_stats(v):
    z = np.load(f"{OUT}/vh/{v}.npz")
    return {sp: {c: z[f"{sp}_{c}"] for c in ("nd", "cc", "kl", "main", "exc", "r")} for sp in ("val", "hold")}


def z_scores(v, k):
    sc = np.load(f"{OUT}/scores/{v}.npz")[k]
    return feed_rank_z(mt[f"{k}_off"], mt[f"{k}_M"], sc.astype(np.float32))[:, 1]


ready = lambda v: os.path.exists(f"{OUT}/scores/{v}.npz")
# ---- absolute levels of the reference (8 seeds) and the teacher
P(f"== LEVELS ({'@last' if SUF else 'holdout-selected'} checkpoints; per-feed means; loss terms over feeds with >= 1 click)")
P("   sets: PIN = 60,000 production-order TRAIN feeds (in sample) | RIN = 59,794 random-order TRAIN feeds, p_random > 0.9, M >= 50 (in sample)"
  " | HOLD = the 30,000 holdout feeds (same selection, out of sample) | VAL = later feeds (from the stored preds)")
tch = {k: stats_dict(mt[f"{k}_t"], mt[f"{k}_t"], mt[f"{k}_y"], mt[f"{k}_off"], mt[f"{k}_M"]) for k in SETS}
tvh = np.load(f"{OUT}/vh/teacher.npz")
P(f"   teacher_mean11: NDCG pin {tch['pin']['nd'].mean():.5f} rin {tch['rin']['nd'].mean():.5f} hold {tch['hold']['nd'].mean():.5f} "
  f"val {tvh['val_nd'].mean():.5f} | own click CE pin {np.nanmean(tch['pin']['cct']):.4f} rin {np.nanmean(tch['rin']['cct']):.4f} "
  f"hold {np.nanmean(tch['hold']['cct']):.4f} val {np.nanmean(tvh['val_cct']):.4f} | H(q) pin {np.nanmean(tch['pin']['ct']):.4f} "
  f"rin {np.nanmean(tch['rin']['ct']):.4f} hold {np.nanmean(tch['hold']['ct']):.4f}")
fam_seeds = dict(FAMS)
for fam, seeds in FAMS:
    vs = [f"w5_e10_{fam}_s{s}{SUF}" for s in seeds if ready(f"w5_e10_{fam}_s{s}{SUF}")]
    if not vs:
        continue
    T = [train_stats(v) for v in vs]
    V = [vh_stats(v) for v in vs]
    row = []
    for k in SETS:
        row.append(f"{k} nd {np.mean([t[k]['nd'].mean() for t in T]):.5f} cc {np.mean([np.nanmean(t[k]['cc']) for t in T]):.4f} "
                   f"kl {np.mean([np.nanmean(t[k]['kl']) for t in T]):.4f} r {np.mean([np.nanmean(t[k]['r']) for t in T]):.4f} "
                   f"main {np.mean([np.nanmean(t[k]['main']) for t in T]):.4f} exc {np.mean([np.nanmean(t[k]['exc']) for t in T]):.4f}")
    row.append(f"val nd {np.mean([x['val']['nd'].mean() for x in V]):.5f} cc {np.mean([np.nanmean(x['val']['cc']) for x in V]):.4f} "
               f"kl {np.mean([np.nanmean(x['val']['kl']) for x in V]):.4f} r {np.mean([np.nanmean(x['val']['r']) for x in V]):.4f}")
    P(f"   {fam:15s} n={len(vs)}: " + " | ".join(row))
    # sanity: holdout NDCG from the fresh forward pass vs the stored holdout preds of the same checkpoint
    dchk = np.mean([T[i]["hold"]["nd"].mean() - V[i]["hold"]["nd"].mean() for i in range(len(vs))])
    P(f"   {'':15s} check: hold NDCG fresh forward - stored preds = {dchk:+.6f}")

# ---- paired deltas vs the same-seed reference
P("")
P("== PAIRED DELTAS vs same-seed kd05w192 (member level: per-feed difference averaged over the seeds; +- feed-level SE [seed-level SE])")
P("   columns: train objective (main = 0.5 click CE + 0.5 teacher CE, lower = better fit), its click part cc and teacher part KL(q||p),"
  " NDCG; then holdout and val NDCG and KL")
summary = {}


def pdelta(arrs_a, arrs_r):
    """arrs: list over seeds of per-feed vectors; returns mean, feed SE, seed SE."""
    D = np.array([a - r for a, r in zip(arrs_a, arrs_r)])  # seeds x feeds
    d = np.nanmean(D, 0)
    d = d[np.isfinite(d)]
    seed_means = np.nanmean(D, 1)
    sse = seed_means.std(ddof=1) / np.sqrt(len(seed_means)) if len(seed_means) > 1 else np.nan
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), sse


for fam, seeds in FAMS:
    if fam == "kd05w192":
        continue
    pairs = [(f"w5_e10_{fam}_s{s}{SUF}", f"w5_e10_kd05w192_s{s}{SUF}") for s in seeds
             if ready(f"w5_e10_{fam}_s{s}{SUF}") and ready(f"w5_e10_kd05w192_s{s}{SUF}")]
    if not pairs:
        continue
    TA = [train_stats(a) for a, _ in pairs]
    TR = [train_stats(r) for _, r in pairs]
    VA = [vh_stats(a) for a, _ in pairs]
    VR = [vh_stats(r) for _, r in pairs]
    P(f"-- {fam}{SUF} (seeds {[a.split('_s')[-1][:3] for a, _ in pairs]})")
    res = {}
    for k in SETS:
        parts = []
        for c, fmt in (("main", ".5f"), ("cc", ".5f"), ("kl", ".5f"), ("nd", ".5f")):
            m_, s1, s2 = pdelta([t[k][c] for t in TA], [t[k][c] for t in TR])
            res[(k, c)] = (m_, s1, s2)
            parts.append(f"d{c} {m_:+{fmt}}+-{s1:{fmt[1:]}}[{s2:{fmt[1:]}}]")
        P(f"   {k:4s}: " + "  ".join(parts))
    for sp in ("hold", "val"):
        parts = []
        for c in ("nd", "kl", "cc"):
            m_, s1, s2 = pdelta([x[sp][c] for x in VA], [x[sp][c] for x in VR])
            res[(sp + "_st", c)] = (m_, s1, s2)
            parts.append(f"d{c} {m_:+.5f}+-{s1:.5f}[{s2:.5f}]")
        P(f"   {sp + '(stored)':14s}: " + "  ".join(parts))
    # in-sample lift: (RIN - HOLD) change; RIN and HOLD are disjoint feed sets of the same selection, SEs add
    a, sa, _ = res[("rin", "nd")]
    b, sb, _ = res[("hold", "nd")]
    P(f"   in-sample NDCG lift change d(RIN) - d(HOLD) = {a - b:+.5f} +- {np.hypot(sa, sb):.5f}; "
      f"d(RIN) - d(VAL) = {a - res[('val_st', 'nd')][0]:+.5f} +- {np.hypot(sa, res[('val_st', 'nd')][1]):.5f}")
    summary[fam] = res
    # pool level (4 seeds) on the train feed sets: mean rank-z
    if len(pairs) >= 2:
        prow = []
        for k in SETS:
            za = np.mean([z_scores(a, k) for a, _ in pairs], 0)
            zr = np.mean([z_scores(r, k) for _, r in pairs], 0)
            na = ndcg_fast(za, mt[f"{k}_y"], mt[f"{k}_off"], mt[f"{k}_M"])
            nr = ndcg_fast(zr, mt[f"{k}_y"], mt[f"{k}_off"], mt[f"{k}_M"])
            d = na - nr
            prow.append(f"{k} {d.mean():+.5f}+-{d.std(ddof=1) / np.sqrt(len(d)):.5f}")
        P(f"   {len(pairs)}-pool NDCG delta (rank-z mean): " + "  ".join(prow))
np.save(f"{OUT}/f12_summary{SUF}.npy", summary, allow_pickle=True)
open(TXT, "w").write("\n".join(lines) + "\n")
