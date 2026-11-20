"""why/fit F3b (CPU, read-only on stored @last predictions): how much of a student's distance to the teacher is seed noise and how much
is systematic, and how close do the arms' seeds sit to each other?
  member KL     = mean over seeds of KL(q || softmax(s_seed))                      (q = softmax(teacher_mean11 / val mean of its 11 members))
  pool KL       = KL(q || softmax(mean over the 4 seeds of the logits))            (log-linear pool: seed noise averaged out)
  seed-pair KL  = mean over the 6 seed pairs of the symmetric KL(p_a, p_b)/2 ;  seed-pair r = within-feed Pearson r of the logits
Feeds: holdout (30,000) and val; @last checkpoints (all at step 98,350, no selection) of seeds 951-954. Output: why/fit/f3_pool.txt"""
import sys
from itertools import combinations

sys.path.insert(0, "<repo>/lab/nn_x/w12/moe/why/fit")
from fitlib import Meta, feed_stats  # noqa: E402
import numba as nb  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

R = Path("<repo>")
OUT = "/tmp/scratchpad/why/fit"
m = Meta()
tvh = np.load(f"{OUT}/vh/teacher.npz")
T = {"val": tvh["val"], "hold": tvh["hold"]}
OFF = {"val": m.ov, "hold": m.oh}
MM = {"val": m.Mv, "hold": m.Mh}
Y = {"val": m.yv, "hold": m.yh}


@nb.njit(parallel=True, cache=True)
def sym_kl(a, b, off, M):
    n = len(M)
    out = np.empty(n)
    for f in nb.prange(n):
        o, mm = off[f], M[f]
        x = a[o:o + mm].astype(np.float64)
        y = b[o:o + mm].astype(np.float64)
        lx = x - (x.max() + np.log(np.exp(x - x.max()).sum()))
        ly = y - (y.max() + np.log(np.exp(y - y.max()).sum()))
        out[f] = 0.5 * ((np.exp(lx) * (lx - ly)).sum() + (np.exp(ly) * (ly - lx)).sum())
    return out


def load(v, sp):
    B = R / "lab/nn_x/w12/moe/out" if ("moe" in v or "b3ff1" in v) else R / "nn_x/w5"
    p = B / "preds" / f"nn_{v}@last_trainonly_val.npy" if sp == "val" else B / "preds/holdout" / f"nn_{v}@last_holdout.npy"
    return np.load(p).astype(np.float32)


lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))
P("F3b @last checkpoints, seeds 951-954; KL in nats per feed (mean over feeds); r = within-feed Pearson r of logits")
res = {}
for fam in ("kd05w192", "kd05w192moeffn", "kd05w192moevo", "kd05w192b3ff1", "kd05", "kd05w224", "kd08w192", "kwd2", "base"):
    row = []
    for sp in ("hold", "val"):
        S = [load(f"w5_e10_{fam}_s{s}", sp) for s in range(951, 955)]
        mem = np.mean([np.nanmean(feed_stats(s_, T[sp], Y[sp], OFF[sp], MM[sp])[:, 2]) for s_ in S])
        pool = np.mean(S, 0)
        pst = feed_stats(pool.astype(np.float32), T[sp], Y[sp], OFF[sp], MM[sp])
        pkl, pr = np.nanmean(pst[:, 2]), np.nanmean(pst[:, 5])
        pairs = list(combinations(range(4), 2))
        skl = np.mean([sym_kl(S[i], S[j], OFF[sp], MM[sp]).mean() for i, j in pairs])
        sr = np.mean([np.nanmean(feed_stats(S[i], S[j], Y[sp], OFF[sp], MM[sp])[:, 5]) for i, j in pairs])
        res[(fam, sp)] = (mem, pkl, skl, sr, pr)
        row.append(f"{sp}: member KL {mem:.4f} | 4-pool KL {pkl:.4f} (r {pr:.4f}) | systematic share {pkl / mem:.2f} | seed-pair symKL {skl:.4f} r {sr:.4f}")
    P(f"{fam:15s} " + " || ".join(row))
open("<repo>/lab/nn_x/w12/moe/why/fit/f3_pool.txt", "w").write("\n".join(lines) + "\n")
