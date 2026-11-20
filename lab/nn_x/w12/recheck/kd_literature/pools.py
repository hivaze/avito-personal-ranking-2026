"""CPU read-outs from stored train-only predictions (kd_literature). Nothing here builds a model input.
A. Fidelity: do students reproduce the teacher pool on unseen feeds (val, holdout)?  (Beyer et al. / Stanton et al.)
B. Does a DIFFERENT teacher give more diverse students than a different seed?  (BAN ensembles, CSRec multi-teacher)
C. Does distillation amplify popularity bias? KD delta on cold-positive vs warm-positive val feeds.  (UnKD)
"""
import sys

sys.path.insert(0, "<repo>/src")
from pathlib import Path

import numpy as np
from common import FLAT, feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

W = Path("<repo>/nn_x/w5/preds")
R6 = Path("<repo>/nn_x/r6/preds")
P = Path("<repo>/preds")
fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Mh, oh = np.load(W / "holdout/holdout_meta.npy")
Mh, oh = Mh.astype(np.int64), oh.astype(np.int64)
yh = np.load(W / "holdout/holdout_y.npy")
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]
se = lambda d: d.std() / np.sqrt(len(d))
f = lambda d: f"{d.mean():+.5f}+-{se(d):.5f}"
S = [951, 952, 953, 954]


def ld(v, D=W):
    return zv(np.load(D / f"nn_{v}_trainonly_val.npy")), zh(np.load(D / f"holdout/nn_{v}_holdout.npy"))


arm = lambda a, seeds=S: [ld(f"w5_e10_{a}_s{s}") for s in seeds]
pool = lambda ms: (np.mean([m[0] for m in ms], 0), np.mean([m[1] for m in ms], 0))
A = {a: arm(a) for a in ["base", "kd03", "kd05", "kd08", "kd10", "kd05w192", "kd05t25", "kd05t25w192", "kd05g2w192",
                         "kx05", "kx08"]}
tf11 = [ld(f"r6_cd_m02_r{r}", R6) for r in range(101, 108)] + A["base"]
T = pool(tf11)
eT = (ev(T[0]), eh(T[1]))
print("== A. fidelity to the teacher pool tf11 (7 r6_cd_m02 + 4 w5_e10_base; the members behind teacher_mean11)")
print(f" teacher pool: val {eT[0].mean():.5f} hold {eT[1].mean():.5f}")
for a in ["base", "kd03", "kd05", "kd08", "kd10", "kd05w192"]:
    p = pool(A[a])
    cm = np.mean([np.corrcoef(m[0], T[0])[0, 1] for m in A[a]])
    cmh = np.mean([np.corrcoef(m[1], T[1])[0, 1] for m in A[a]])
    iu = np.triu_indices(4, 1)
    cc = np.corrcoef([m[0] for m in A[a]])[iu].mean()
    mem = np.mean([ev(m[0]) for m in A[a]], 0)
    memh = np.mean([eh(m[1]) for m in A[a]], 0)
    print(f" {a:9s} member val {mem.mean():.5f} hold {memh.mean():.5f} | 4-pool val {ev(p[0]).mean():.5f} hold {eh(p[1]).mean():.5f}"
          f" | pool - teacher: val {f(ev(p[0]) - eT[0])} hold {f(eh(p[1]) - eT[1])}"
          f" | corr(member, teacher) val {cm:.4f} hold {cmh:.4f}; corr(pool, teacher) val {np.corrcoef(p[0], T[0])[0, 1]:.4f}"
          f" hold {np.corrcoef(p[1], T[1])[0, 1]:.4f}; member-member val {cc:.4f}")
print(" (base members are IN the teacher pool: their corr to it is inflated by 1/11 self-weight)")

print("\n== B. teacher identity vs seed as a diversity source (val rank-z correlations between single students)")


def xcorr(a, b, same_seed):
    va, vb = [m[0] for m in A[a]], [m[0] for m in A[b]]
    c = [np.corrcoef(va[i], vb[j])[0, 1] for i in range(4) for j in range(4) if (i == j) == same_seed and (a != b or i < j or same_seed)]
    return np.mean(c) if c else float("nan")


prs = [("kd05w192", "kd05w192"), ("kd05t25w192", "kd05t25w192"), ("kd05g2w192", "kd05g2w192"), ("kx05", "kx05"),
       ("kd05w192", "kd05t25w192"), ("kd05w192", "kd05g2w192"), ("kd05w192", "kx05"), ("kd05t25w192", "kd05g2w192"),
       ("kd05t25w192", "kx05"), ("kd05", "kd05"), ("kd05", "kd05t25"), ("kd05", "kd05w192"), ("base", "base")]
for a, b in prs:
    print(f" {a:12s} x {b:12s}: different seeds {xcorr(a, b, False):.4f}" + ("" if a == b else f" | same seed {xcorr(a, b, True):.4f}"))
print(" equal-size pools, one teacher vs two teachers (2 + 2 students, seeds disjoint so no seed is shared):")
ref8 = pool(A["kd05w192"] + A["kd05t25w192"])
for nm, ms in [("kd05w192 x4 (mean11)", A["kd05w192"]), ("kd05t25w192 x4 (mean25)", A["kd05t25w192"]),
               ("kd05w192 s951,952 + kd05t25w192 s953,954", A["kd05w192"][:2] + A["kd05t25w192"][2:]),
               ("kd05w192 s953,954 + kd05t25w192 s951,952", A["kd05w192"][2:] + A["kd05t25w192"][:2]),
               ("kd05w192 x4 vs kx05 mix: kd05w192 s951,952 + kx05 s953,954", A["kd05w192"][:2] + A["kx05"][2:]),
               ("kd05w192 s951,952 + kd05g2w192 s953,954", A["kd05w192"][:2] + A["kd05g2w192"][2:]),
               ("W8 = kd05w192 x4 + kd05t25w192 x4", A["kd05w192"] + A["kd05t25w192"])]:
    p = pool(ms)
    print(f"  {nm:58s} val {ev(p[0]).mean():.5f} hold {eh(p[1]).mean():.5f} | vs W8: val {f(ev(p[0]) - ev(ref8[0]))} hold {f(eh(p[1]) - eh(ref8[1]))}")
p1, p2 = pool(A["kd05w192"]), pool(A["kd05t25w192"])
mixes = [pool(A["kd05w192"][:2] + A["kd05t25w192"][2:]), pool(A["kd05w192"][2:] + A["kd05t25w192"][:2])]
dv = np.mean([ev(m[0]) for m in mixes], 0) - 0.5 * (ev(p1[0]) + ev(p2[0]))
dh = np.mean([eh(m[1]) for m in mixes], 0) - 0.5 * (eh(p1[1]) + eh(p2[1]))
print(f"  two-teacher 4-pools minus one-teacher 4-pools (mean of both splits): val {f(dv)} hold {f(dh)}")

print("\n== C. popularity slices of the KD gain on val (cold = it_hist_all_luo == 0, the project's CPU2 definition)")
luo = np.load("<repo>/nn_x/r6/feat_cache/val/it_hist_all_luo.npy", mmap_mode="r")
cold = np.nan_to_num(np.asarray(luo, dtype=np.float64), nan=0.0) == 0
fid = np.repeat(np.arange(len(Mv)), Mv)
npos = np.bincount(fid, yv, len(Mv))
ncold = np.bincount(fid, yv * cold, len(Mv))
shc = np.bincount(fid, cold, len(Mv)) / Mv
sl = {"all": np.ones(len(Mv), bool), "coldpos": (npos > 0) & (ncold == npos), "warmpos": (npos > 0) & (ncold == 0),
      "M<50": Mv < 50, "M>=50": Mv >= 50}
print(f" feeds: coldpos {sl['coldpos'].mean():.3f}, warmpos {sl['warmpos'].mean():.3f}; cold share of candidates {cold.mean():.3f}")
c3 = zv(np.load(P / "ens_Q2a8s3_to_val.npy"))
w8 = zv(np.load(P / "ens_W8kd_to_val.npy"))
rows = [("kd05 member - base member (4 seeds)", np.mean([ev(a[0]) - ev(b[0]) for a, b in zip(A["kd05"], A["base"])], 0)),
        ("kd10 member - base member", np.mean([ev(a[0]) - ev(b[0]) for a, b in zip(A["kd10"], A["base"])], 0)),
        ("kd05w192 member - base member", np.mean([ev(a[0]) - ev(b[0]) for a, b in zip(A["kd05w192"], A["base"])], 0)),
        ("kd05 4-pool - base 4-pool", ev(pool(A["kd05"])[0]) - ev(pool(A["base"])[0])),
        ("W8 TO pool - C3 TO pool", ev(w8) - ev(c3)),
        ("W8 TO pool - teacher tf11 pool", ev(ref8[0]) - eT[0])]
for nm, d in rows:
    print(f" {nm:38s} " + "  ".join(f"{k} {d[m].mean():+.5f}+-{se(d[m]):.5f}" for k, m in sl.items()))
# mean rank-z given to cold candidates (label-free): does the student push cold items down relative to the base / teacher?
for nm, s in [("base 4-pool", pool(A["base"])[0]), ("teacher tf11", T[0]), ("kd05 4-pool", pool(A["kd05"])[0]),
              ("kd10 4-pool", pool(A["kd10"])[0]), ("W8 TO pool", ref8[0]), ("C3 TO pool", c3)]:
    z = zv(s)
    print(f" mean rank-z of cold candidates, {nm:14s}: {z[cold].mean():+.4f} (cold positives {z[cold & (yv > 0)].mean():+.4f}, warm positives {z[~cold & (yv > 0)].mean():+.4f})")
