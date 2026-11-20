"""why/fit F3c (CPU, read-only): where does the reference student sit relative to the TEACHER on the student's own training objective?
On each feed set the teacher is scored as if it were a student (p = q): main_T = 0.5 click-CE(q) + 0.5 H(q). Paired per feed (clicked
feeds only): student - teacher for main, click CE (cc) and NDCG. @last checkpoints; reference = 8 seeds, arms = 4 seeds (seed-averaged per
feed; feed-level SE). Output: why/fit/f3_objective.txt"""
import sys

sys.path.insert(0, "<repo>/lab/nn_x/w12/moe/why/fit")
from fitlib import stats_dict  # noqa: E402
import numpy as np  # noqa: E402

OUT = "/tmp/scratchpad/why/fit"
mt = np.load(f"{OUT}/meta.npz")
lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))


def ts(v):
    z = np.load(f"{OUT}/tstats/{v}.npz")
    return {k: {c: z[f"{k}_{c}"] for c in ("nd", "cc", "main", "kl")} for k in ("pin", "rin", "hold")}


fams = {"reference kd05w192 (8 seeds)": [f"w5_e10_kd05w192_s{s}@last" for s in range(951, 959)],
        "A routed FFN": [f"w5_e10_kd05w192moeffn_s{s}@last" for s in range(951, 955)],
        "B routed V/O": [f"w5_e10_kd05w192moevo_s{s}@last" for s in range(951, 955)],
        "C depth 7 ff1": [f"w5_e10_kd05w192b3ff1_s{s}@last" for s in range(951, 955)],
        "d224": [f"w5_e10_kd05w224_s{s}@last" for s in range(951, 955)],
        "20 epochs (2 seeds)": [f"w5_e10_kw_e20_s{s}@last" for s in (951, 952)]}
for k in ("pin", "rin", "hold"):
    T = stats_dict(mt[f"{k}_t"], mt[f"{k}_t"], mt[f"{k}_y"], mt[f"{k}_off"], mt[f"{k}_M"])
    clk = mt[f"{k}_y"]
    P(f"== {k.upper()}: teacher main {np.nanmean(T['main']):.4f} (click CE {np.nanmean(T['cc']):.4f}, H(q) part "
      f"{np.nanmean(2 * T['main'] - T['cc']):.4f}), NDCG {T['nd'].mean():.5f}; feeds with a click "
      f"{np.isfinite(T['main']).mean():.3f}")
    for nm, vs in fams.items():
        S = [ts(v)[k] for v in vs]
        row = []
        for c in ("main", "cc", "nd"):
            d = np.nanmean([s[c] - T[c] for s in S], 0)
            d = d[np.isfinite(d)]
            row.append(f"{c} {d.mean():+.5f}+-{d.std(ddof=1) / np.sqrt(len(d)):.5f}")
        kl = np.mean([np.nanmean(s["kl"]) for s in S])
        P(f"   {nm:30s} student - teacher: " + "  ".join(row) + f"   (student KL(q||p) {kl:.4f}, i.e. 0.5 KL = {kl / 2:.4f} of main)")
open("<repo>/lab/nn_x/w12/moe/why/fit/f3_objective.txt", "w").write("\n".join(lines) + "\n")
