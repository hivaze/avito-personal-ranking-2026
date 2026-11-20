# Calibration read-out behind the popularity-light view. It splits validation rows by the engagement-rate
# feature f5 and by how much exposure the item has, and prints actual over expected clicks per cell for
# the base, the popularity-free arm and the popularity-light arm of the 2-epoch screen.
# Run as: cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python
# nn_x/abl/pop1/pop1_actexp.py
# Produced pop1/out/pop1_actexp.{log,json}: the exposure input did not repair the cell it was designed
# for (0.938 against 0.938 without it, base 0.969).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/abl/pop1/pop1_actexp.py
#
"""pop1 information readout (ledger nn_x/abl/ledger/ledger.md section 7; pre-declared there, not a gate): NOISY-HIGH /
RELIABLE-HIGH / MID-LOW / NO-DATA act/exp of the 2-epoch screen arms base, xdrop_pop and pop1 (LAST checkpoints = the
frozen primary read, seeds 23/24, 2-member rz-mean pools), exactly the cell / act-exp / SE definitions of
nn_x/codes/followup/f04c_2ep_ref.py and f04_popfree.py (imported read-only through fu_common):
  f5 q90 over val rows; n = tte_it_imp + it_hist_all_luo (val); NOISY-HIGH = f5 >= q90 & n < 5; RELIABLE-HIGH = f5 >= q90 &
  n >= 5; MID/LOW = 0 < f5 < q90; NO-DATA = f5 == 0. act/exp = sum y / sum expected in the cell, expected = within-feed decile
  click rate of the score, rescaled per feed to the feed's clicks. SE: user-clustered ratio SE (ratio_cl); differences:
  paired user-cluster bootstrap (200 multinomial draws, seed 0).
Also per seed (single members) and the pool NDCG. CPU only, read-only on predictions.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/pop1/pop1_actexp.py
        -> nn_x/abl/pop1/out/pop1_actexp.{log,json}
"""
import json
import os
import sys

sys.path.insert(0, "<repo>/nn_x/codes/followup")
from fu_common import *  # noqa: E402,F401,F403  (Split, ratio_cl, FLAT, FEATS, ROOT, np)

OUTD = ROOT / "nn_x/abl/pop1/out"
os.makedirs(OUTD, exist_ok=True)
TAG = "" if len(sys.argv) < 2 else "_check"
LOG = open(OUTD / f"pop1_actexp{TAG}.log", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")


V = Split("val")
y, fid, n = V.y, V.fid, V.n
PA = ROOT / "nn_x/abl/preds"
f5 = np.load(FLAT / "val/cand_f5.npy").astype(np.float64)
nexp = (np.nan_to_num(np.load(FEATS / "val/tte_it_imp.npy").astype(np.float64))
        + np.nan_to_num(np.load(FEATS / "val/it_hist_all_luo.npy").astype(np.float64)))
q90 = np.quantile(f5, 0.9)
C = {"NOISY-HIGH": (f5 >= q90) & (nexp < 5), "RELIABLE-HIGH": (f5 >= q90) & (nexp >= 5),
     "MID/LOW": (f5 > 0) & (f5 < q90), "NO-DATA": f5 == 0}


def expected(s):
    o = np.lexsort((s, fid))
    fs_ = np.r_[True, fid[o][1:] != fid[o][:-1]]
    rk = np.arange(len(o)) - np.flatnonzero(fs_)[np.cumsum(fs_) - 1]
    pct = np.empty(len(o))
    pct[o] = rk / np.maximum(V.M[fid[o]] - 1, 1)
    dec = np.clip((pct * 10).astype(int), 0, 9)
    rate = np.bincount(dec, y, minlength=10) / np.bincount(dec, minlength=10)
    ex = rate[dec]
    return ex * (np.bincount(fid, y, minlength=n) / np.maximum(np.bincount(fid, ex, minlength=n), 1e-12))[fid]


G = V.cl.max() + 1
BW = np.random.default_rng(0).multinomial(G, np.full(G, 1.0 / G), size=200).astype(np.float64)
ARMS = tuple(a for a in sys.argv[1:] if not a.startswith("-")) or ("base", "xdrop_pop", "pop1")  # argv: plumbing check
raw = {a: {s: np.load(PA / f"nn_abl_{a}_s{s}@last_trainonly_val.npy") for s in (23, 24)} for a in ARMS}
S = {a: V.rz(np.mean([V.rz(raw[a][s]) for s in (23, 24)], 0)) for a in ARMS}
S.update({f"{a}_s{s}": raw[a][s] for a in ARMS for s in (23, 24)})
EX = {k: expected(s) for k, s in S.items()}
P(f"== pop1 information readout (2-epoch screen, LAST checkpoints, seeds 23/24 rz-mean pools; f5 q90 {q90:.4f})")
P("  cell shares (val rows / positives): " + "; ".join(f"{cn} {m.mean():.4f} / {int(y[m].sum())}" for cn, m in C.items()))
RES = {"q90": float(q90), "ndcg": {k: float(V.nd(s).mean()) for k, s in S.items()}, "cells": {}}
P("  pool NDCG: " + "  ".join(f"{a} {RES['ndcg'][a]:.5f}" for a in ARMS) + " | members: "
  + "  ".join(f"{a}_s{s} {RES['ndcg'][f'{a}_s{s}']:.5f}" for a in ARMS for s in (23, 24)))
for cn, m in C.items():
    ycl = np.bincount(V.cl[fid], y * m, G)
    out = {}
    for k in S:
        ecl = np.bincount(V.cl[fid], EX[k] * m, G)
        _, se = ratio_cl(np.bincount(fid, y * m, n), np.bincount(fid, EX[k] * m, n), V.cl)
        out[k] = dict(ae=float(ycl.sum() / ecl.sum()), se=float(se), boot=(BW @ ycl) / (BW @ ecl))
    rec = {k: dict(ae=v["ae"], se=v["se"]) for k, v in out.items()}
    P(f"\n  {cn}:")
    for a in ARMS:
        P(f"    {a:10s} pool act/exp {out[a]['ae']:.3f} +- {out[a]['se']:.3f}  (s23 {out[f'{a}_s23']['ae']:.3f}, "
          f"s24 {out[f'{a}_s24']['ae']:.3f})")
    for a, b in (("pop1", "xdrop_pop"), ("pop1", "base"), ("xdrop_pop", "base")):
        if a not in ARMS or b not in ARMS:
            continue
        d = out[a]["boot"] - out[b]["boot"]
        rec[f"{a}-{b}"] = dict(d=out[a]["ae"] - out[b]["ae"], se=float(d.std()))
        P(f"    {a} - {b}: {out[a]['ae'] - out[b]['ae']:+.3f} +- {d.std():.3f} (paired cluster bootstrap)")
    gap = out["base"]["ae"] - out["xdrop_pop"]["ae"] if {"base", "xdrop_pop"} <= set(ARMS) else 0.0
    if abs(gap) > 1e-9 and "pop1" in ARMS:
        rec["closed_share"] = float((out["pop1"]["ae"] - out["xdrop_pop"]["ae"]) / gap)
        P(f"    share of the xdrop_pop -> base gap closed by pop1: {rec['closed_share']:+.2f}")
    RES["cells"][cn] = rec
json.dump(RES, open(OUTD / f"pop1_actexp{TAG}.json", "w"), indent=1)
P(f"\nwrote {OUTD}/pop1_actexp{TAG}.{{log,json}}")
