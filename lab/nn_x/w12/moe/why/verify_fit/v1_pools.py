"""verify_fit v1 (CPU, read-only on stored preds): independent re-derivation of the why/fit F5 / F3 / F1a numbers.
  F5: 4-pool (rank-z mean) val / holdout deltas vs the same-seed reference pool, selected vs @last; per-seed member deltas; leave-one-seed-out
      jackknife SE of the 4-pool delta (seed-aware); which single seed carries the selected -> @last change.
  F3: absolute NDCG of the teacher (raw mean of 11 member logits) vs reference member / 4-pool / 8-pool and kd08w192 / kd05 (d128) 4-pools;
      KL(teacher || pool) for 1 / 4 / 8 reference seeds (does pool KL follow sys + var/N?).
  F1a: is arm A a near-copy of its same-seed reference? symmetric KL and within-feed r of A_s vs ref_s against ref_s vs ref_s'.
Output: why/verify_fit/v1_pools.txt"""
import os

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
from itertools import combinations
from pathlib import Path

import numba as nb
import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/recheck/pool_composition")
from pc_common import Meta, ndcg_fast, se  # noqa: E402

R = Path("<repo>")
m = Meta()
lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))
Y = {"val": m.yv, "hold": m.yh}
OFF = {"val": m.ov, "hold": m.oh}
MM = {"val": m.Mv, "hold": m.Mh}


def base(v):
    return R / "lab/nn_x/w12/moe/out" if ("moe" in v or "b3ff1" in v) else R / "nn_x/w5"


def load(v, suf, sp):
    p = base(v) / "preds" / f"nn_{v}{suf}_trainonly_val.npy" if sp == "val" else base(v) / "preds/holdout" / f"nn_{v}{suf}_holdout.npy"
    return np.load(p).astype(np.float32)


def nd(x, sp):
    return ndcg_fast(x, Y[sp], OFF[sp], MM[sp])


@nb.njit(parallel=True, cache=True)
def kl_qp(t, s, off, M):
    """per-feed KL(softmax t || softmax s) and Pearson r."""
    n = len(M)
    out = np.empty((n, 2))
    for f in nb.prange(n):
        o, mm = off[f], M[f]
        a = t[o:o + mm].astype(np.float64)
        b = s[o:o + mm].astype(np.float64)
        la = a - (a.max() + np.log(np.exp(a - a.max()).sum()))
        lb = b - (b.max() + np.log(np.exp(b - b.max()).sum()))
        out[f, 0] = (np.exp(la) * (la - lb)).sum()
        am, bm = a.mean(), b.mean()
        den = np.sqrt(((a - am) ** 2).sum() * ((b - bm) ** 2).sum())
        out[f, 1] = ((a - am) * (b - bm)).sum() / den if den > 0 else np.nan
    return out


def skl(a, b, sp):
    x = kl_qp(a, b, OFF[sp], MM[sp])
    y = kl_qp(b, a, OFF[sp], MM[sp])
    return 0.5 * (x[:, 0] + y[:, 0]).mean(), np.nanmean(x[:, 1])


S4 = list(range(951, 955))
ARMS = {"A": "kd05w192moeffn", "B": "kd05w192moevo", "C": "kd05w192b3ff1"}
Z = {}


def zz(v, suf, sp):
    k = (v, suf, sp)
    if k not in Z:
        Z[k] = m.z(sp, load(v, suf, sp))
    return Z[k]


# ------------------------------------------------------------------ F5
P("== F5 re-derivation: 4-pool (rank-z mean) delta arm - same-seed reference pool; feed SE | leave-one-seed-out jackknife SE (3-pools)")
for a, arm in ARMS.items():
    for sp in ("val", "hold"):
        res = {}
        for suf in ("", "@last"):
            za = [zz(f"w5_e10_{arm}_s{s}", suf, sp) for s in S4]
            zr = [zz(f"w5_e10_kd05w192_s{s}", suf, sp) for s in S4]
            d = nd(np.mean(za, 0), sp) - nd(np.mean(zr, 0), sp)
            jk = []
            for i in range(4):
                keep = [j for j in range(4) if j != i]
                jk.append((nd(np.mean([za[j] for j in keep], 0), sp) - nd(np.mean([zr[j] for j in keep], 0), sp)).mean())
            jk = np.array(jk)
            jse = np.sqrt(3 / 4 * ((jk - jk.mean()) ** 2).sum())
            res[suf] = d
            mem = [nd(load(f"w5_e10_{arm}_s{s}", suf, sp), sp).mean() - nd(load(f"w5_e10_kd05w192_s{s}", suf, sp), sp).mean() for s in S4]
            P(f"   {a} {sp:4s} {'selected' if suf == '' else '@last   '}: 4-pool {d.mean():+.5f}+-{se(d):.5f} (jackknife SE {jse:.5f}; "
              f"leave-out deltas {' '.join(f'{x:+.5f}' for x in jk)}) | member per seed {' '.join(f'{x:+.5f}' for x in mem)}")
        dd = res["@last"] - res[""]
        P(f"   {a} {sp:4s} @last - selected: {dd.mean():+.5f}+-{se(dd):.5f}")
    # which seed carries the change: swap one seed at a time from selected to @last (arm side only; ref kept selected)
    for sp in ("val",):
        zr = np.mean([zz(f"w5_e10_kd05w192_s{s}", "", sp) for s in S4], 0)
        nr = nd(zr, sp)
        base_d = (nd(np.mean([zz(f"w5_e10_{arm}_s{s}", "", sp) for s in S4], 0), sp) - nr).mean()
        sw = []
        for s0 in S4:
            za = np.mean([zz(f"w5_e10_{arm}_s{s}", "@last" if s == s0 else "", sp) for s in S4], 0)
            sw.append((nd(za, sp) - nr).mean() - base_d)
        P(f"   {a} val: change of the 4-pool delta when only seed s is switched to @last (ref selected): "
          + " ".join(f"s{s}:{x:+.5f}" for s, x in zip(S4, sw)))
    # reference side: selected vs @last
    for sp in ("val", "hold"):
        d = nd(np.mean([zz(f"w5_e10_kd05w192_s{s}", "@last", sp) for s in S4], 0), sp) - nd(np.mean([zz(f"w5_e10_kd05w192_s{s}", "", sp) for s in S4], 0), sp)
    P("")
P("   reference pool @last - selected (refs selected at 97.5k / 98.35k): " + " | ".join(
    f"{sp} {(nd(np.mean([zz(f'w5_e10_kd05w192_s{s}', '@last', sp) for s in S4], 0), sp) - nd(np.mean([zz(f'w5_e10_kd05w192_s{s}', '', sp) for s in S4], 0), sp)).mean():+.5f}"
    for sp in ("val", "hold")))

# ------------------------------------------------------------------ F3
P("")
P("== F3: absolute NDCG and KL to the teacher (teacher = raw mean of the 11 member logits; @last students)")
TM = [("preds", f"r6_cd_m02_r{r}") for r in range(101, 108)] + [("nn_x/w5/preds", f"w5_e10_base_s{s}") for s in range(951, 955)]
T = {}
for sp in ("val", "hold"):
    acc = np.zeros(m.n[sp], np.float64)
    for d, v in TM:
        p = R / d / f"nn_{v}_trainonly_val.npy" if sp == "val" else R / d / "holdout" / f"nn_{v}_holdout.npy"
        acc += np.load(p).astype(np.float64)
    T[sp] = (acc / len(TM)).astype(np.float32)
mt = np.load("/tmp/scratchpad/why/fit/meta.npz")
th_file = mt["hold_t"]
fid = np.repeat(np.arange(len(m.Mh)), m.Mh)
cen = lambda x: x - (np.bincount(fid, x.astype(np.float64)) / m.Mh)[fid]
P(f"   teacher holdout rebuilt from 11 stored holdout preds vs teacher_mean11 rows: max |centred diff| {np.abs(cen(T['hold']) - cen(th_file)).max():.2e}")
T["hold"] = th_file
for sp in ("val", "hold"):
    nt = nd(T[sp], sp)
    zt = nd(np.mean([m.z(sp, load(v, '', sp)) if False else m.z(sp, np.load(R / d / (f'nn_{v}_trainonly_val.npy' if sp == 'val' else f'holdout/nn_{v}_holdout.npy')).astype(np.float32)) for d, v in TM], 0), sp)
    ref8 = [f"w5_e10_kd05w192_s{s}" for s in range(951, 959)]
    mem = np.mean([nd(load(v, "@last", sp), sp) for v in ref8], 0)
    p4 = nd(np.mean([zz(v, "@last", sp) for v in ref8[:4]], 0), sp)
    p4b = nd(np.mean([zz(v, "@last", sp) for v in ref8[4:]], 0), sp)
    p8 = nd(np.mean([zz(v, "@last", sp) for v in ref8], 0), sp)
    k8 = nd(np.mean([zz(f"w5_e10_kd08w192_s{s}", "@last", sp) for s in S4], 0), sp)
    d128 = nd(np.mean([zz(f"w5_e10_kd05_s{s}", "@last", sp) for s in S4], 0), sp)
    P(f"   {sp}: teacher raw-mean {nt.mean():.5f} (z-mean of the 11: {zt.mean():.5f}) | ref member (8-seed avg) {mem.mean():.5f} | ref 4-pool s951-954 "
      f"{p4.mean():.5f} | ref 4-pool s955-958 {p4b.mean():.5f} | ref 8-pool {p8.mean():.5f} | kd08 4-pool {k8.mean():.5f} | d128 4-pool {d128.mean():.5f}")
    for nm, x in (("ref member - teacher", mem - nt), ("ref 4-pool(951-954) - teacher", p4 - nt), ("ref 4-pool(955-958) - teacher", p4b - nt),
                  ("ref 8-pool - teacher", p8 - nt), ("kd08 4-pool - ref 4-pool", k8 - p4), ("d128 4-pool - ref 4-pool", d128 - p4),
                  ("ref 8-pool - ref 4-pool(951-954)", p8 - p4)):
        P(f"      {nm:34s} {x.mean():+.5f}+-{se(x):.5f}")
    # KL decomposition: pool of N seeds (log-linear = mean raw logits) for N = 1, 2, 4, 8
    raw = [load(v, "@last", sp) for v in ref8]
    row = []
    for N in (1, 2, 4, 8):
        ks = []
        for c in list(combinations(range(8), N))[:8]:
            ks.append(kl_qp(T[sp], np.mean([raw[i] for i in c], 0).astype(np.float32), OFF[sp], MM[sp])[:, 0].mean())
        row.append((N, np.mean(ks)))
    Ns = np.array([r[0] for r in row], float)
    Ks = np.array([r[1] for r in row])
    b, a0 = np.polyfit(1 / Ns, Ks, 1)
    P(f"      KL(teacher || log-linear pool of N ref seeds): " + " ".join(f"N={int(n)}:{k:.5f}" for n, k in zip(Ns, Ks))
      + f" | fit KL = {a0:.5f} + {b:.5f}/N -> systematic share of member KL {a0 / Ks[0]:.2f}")

# ------------------------------------------------------------------ F1a
P("")
P("== F1a: same-seed closeness (val, @last): symmetric KL / within-feed r of arm_s vs ref_s, against ref_s vs ref_s' (different seeds)")
for sp in ("val",):
    R4 = {s: load(f"w5_e10_kd05w192_s{s}", "@last", sp) for s in S4}
    cross = [skl(R4[i], R4[j], sp) for i, j in combinations(S4, 2)]
    P(f"   ref_s vs ref_s' (6 pairs): symKL {np.mean([c[0] for c in cross]):.5f} r {np.mean([c[1] for c in cross]):.5f}")
    for a, arm in list(ARMS.items()) + [("d224", "kd05w224")]:
        same = [skl(load(f"w5_e10_{arm}_s{s}", "@last", sp), R4[s], sp) for s in S4]
        P(f"   {a:4s}_s vs ref_s (same seed): symKL {np.mean([c[0] for c in same]):.5f} r {np.mean([c[1] for c in same]):.5f} "
          f"(per seed symKL {' '.join(f'{c[0]:.4f}' for c in same)})")
open("<repo>/lab/nn_x/w12/moe/why/verify_fit/v1_pools.txt", "w").write("\n".join(lines) + "\n")
