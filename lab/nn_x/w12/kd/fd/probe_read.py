"""Read R1 of the probe files (ledger 2026-11-17T15:07Z): FDX16P20:fsw50t - FDX16P20:ref on validation, paired per feed.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/kd/fd/probe_read.py

Final score = 0.5 ens_<tag>_ft_val + 0.5 ens_<tag>_to_val (the gate's expression for these files). Reported: all feeds,
pmix-public, pmix-private (pmix.weights(), as build_r4.py), M<50, M>=50, and each branch alone; also each file - FD16P20.
"Replicates" iff val delta >= 0 AND pmix-private delta >= 0 (fixed in the ledger before the runs).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "<repo>/src")
from common import feed_meta  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
import pmix  # noqa: E402

P = "<repo>/preds"
fv, Mv, ov = feed_meta("val")
yv = np.load("<repo>/data/flat/val/cand_target.npy").astype(np.float32)
w = pmix.weights(); big = Mv >= 50
ld = lambda tag, br: np.load(f"{P}/ens_{tag}_{br}_val.npy").astype(np.float64)
nd = lambda s: ndcg_per_feed(s, yv, ov, Mv)


def scores(tag):
    ft, to = ld(tag, "ft"), ld(tag, "to")
    return {"final": nd(0.5 * ft + 0.5 * to), "ft": nd(ft), "to": nd(to)}


def fmt(d, wt=None):
    m, se = (pmix.wmean_se(d, wt) if wt is not None else (d.mean(), d.std(ddof=1) / np.sqrt(len(d))))
    return float(m), float(se)


A, B, C = sys.argv[1:4] if len(sys.argv) > 3 else ("FDX16P20_fsw50tkd", "FDX16P20_refkd", "FD16P20kd")   # override only for a plumbing test
S = {t: scores(t) for t in (A, B, C)}
out = {}
print(f"levels (final): " + " | ".join(f"{t} {S[t]['final'].mean():.5f}" for t in S))
for a, b in [(A, B), (A, C), (B, C)]:
    print(f"\n{a} - {b}")
    for br in ("final", "ft", "to"):
        d = S[a][br] - S[b][br]
        r = {"val": fmt(d), "pmix_public": fmt(d, w["public"]), "pmix_private": fmt(d, w["private"]), "M<50": fmt(d[~big]), "M>=50": fmt(d[big])}
        out[f"{a}-{b}:{br}"] = r
        print(f"  {br:5s} " + " | ".join(f"{k} {m:+.5f} +- {se:.5f}" for k, (m, se) in r.items()))
r = out[f"{A}-{B}:final"]
rep = r["val"][0] >= 0 and r["pmix_private"][0] >= 0
print(f"\nR1 (val >= 0 AND pmix-private >= 0): {'REPLICATES' if rep else 'DOES NOT REPLICATE'}")
out["R1_replicates"] = bool(rep)
if len(sys.argv) <= 3:
    json.dump(out, open("<repo>/nn_x/w12/kd/fd/logs/probe_read_R1.json", "w"), indent=1)
