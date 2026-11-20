"""verify_fit v3 (CPU, read-only): F3 re-check.
 (a) pure-numpy student - teacher on the student's objective (main = 0.5 click CE + 0.5 teacher CE), RIN / PIN / HOLD, reference 8 seeds @last.
 (b) the teacher's own in-sample advantage: RIN and HOLD are random draws from the same feed population (p_random > 0.9, M >= 50), so any model
     that never trained on either scores them equally up to sampling noise. RIN - HOLD gap of NDCG and click CE for the teacher, a click-only
     teacher member (base), and the reference student, with the unpaired sampling SE of the gap.
Output: why/verify_fit/v3_teacher.txt"""
import os

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys

import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/recheck/pool_composition")
from pc_common import ndcg_fast  # noqa: E402

FIT = "/tmp/scratchpad/why/fit"
mt = np.load(f"{FIT}/meta.npz")
lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))


def terms(s, t, y, M):
    s = s.astype(np.float64); t = t.astype(np.float64); y = y.astype(np.float64)
    off = np.concatenate([[0], np.cumsum(M)[:-1]])
    fid = np.repeat(np.arange(len(M)), M)
    ms = np.maximum.reduceat(s, off); mt_ = np.maximum.reduceat(t, off)
    lps = s - (ms + np.log(np.add.reduceat(np.exp(s - ms[fid]), off)))[fid]
    lqt = t - (mt_ + np.log(np.add.reduceat(np.exp(t - mt_[fid]), off)))[fid]
    q = np.exp(lqt)
    ct = -np.add.reduceat(q * lps, off)
    k = np.add.reduceat(y, off)
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(k > 0, -np.add.reduceat(y * lps, off) / k, np.nan)
    return dict(cc=cc, main=0.5 * cc + 0.5 * ct)


pse = lambda x: f"{np.nanmean(x):+.5f}+-{np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum()):.5f}"
P("== (a) reference student (8 seeds @last, seed-averaged per feed) - teacher on the student's objective (pure numpy)")
NDS = {}
for sp in ("pin", "rin", "hold"):
    y, t, M, off = mt[f"{sp}_y"], mt[f"{sp}_t"], mt[f"{sp}_M"], mt[f"{sp}_off"]
    T = terms(t, t, y, M)
    nt = ndcg_fast(t, y.astype(np.float32), off, M)
    S = [np.load(f"{FIT}/scores/w5_e10_kd05w192_s{s}@last.npz")[sp] for s in range(951, 959)]
    St = [terms(x, t, y, M) for x in S]
    ns = np.mean([ndcg_fast(x, y.astype(np.float32), off, M) for x in S], 0)
    B = [np.load(f"{FIT}/scores/w5_e10_base_s{s}@last.npz")[sp] for s in range(951, 955)]
    Bt = [terms(x, t, y, M) for x in B]
    nb_ = np.mean([ndcg_fast(x, y.astype(np.float32), off, M) for x in B], 0)
    P(f"   {sp:4s}: dmain {pse(np.mean([s['main'] for s in St], 0) - T['main'])}  dcc {pse(np.mean([s['cc'] for s in St], 0) - T['cc'])}  "
      f"dNDCG {pse(ns - nt)}")
    NDS[sp] = dict(teacher=(nt, T["cc"]), student=(ns, np.mean([s["cc"] for s in St], 0)), base_member=(nb_, np.mean([b["cc"] for b in Bt], 0)))
P("")
P("== (b) RIN - HOLD gap (both random draws of the same feed population; unpaired SE of the difference of means)")
for who in ("teacher", "base_member", "student"):
    out = []
    for i, nm in enumerate(("NDCG", "click CE")):
        a, b = NDS["rin"][who][i], NDS["hold"][who][i]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        g = a.mean() - b.mean()
        s_ = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        out.append(f"{nm} {g:+.5f}+-{s_:.5f}")
    P(f"   {who:12s}: " + "  ".join(out))
open("<repo>/lab/nn_x/w12/moe/why/verify_fit/v3_teacher.txt", "w").write("\n".join(lines) + "\n")
