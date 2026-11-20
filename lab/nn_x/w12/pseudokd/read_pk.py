"""Rule read of the PSEUDO-KD 2-seed screen (rule: RULE_predeclared.txt in this folder, fixed before any run of the arm existed).
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/pseudokd/read_pk.py [arm=kd05w192ps] [seeds=951,952] [ref=kd05w192]
Members = train-only val predictions of the SELECTED checkpoints in nn_x/w5/preds (w5_e10_<arm>_s<seed>), paired per feed and seed with the
stored reference runs w5_e10_<ref>_s<seed>. Decision on the MEMBER view (RULE): KILL unless member val delta >= +0.0008 AND pmix-private
(member d_f) >= 0 AND both val M halves (M < 50, M >= 50) >= -1 SE. The 2-seed POOL view, pmix-public and holdout are information only.
CPU only, read-only. --selftest: runs the read with the reference against itself (all deltas exactly 0) to check the plumbing."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "<repo>/nn_x/w12/recheck/pool_composition")
from pc_common import Meta, ndcg_fast, paths  # noqa: E402
import numpy as np  # noqa: E402
import pmix  # noqa: E402  (lab/src/pmix.py; src is on sys.path through pc_common)

argv = [x for x in sys.argv[1:] if not x.startswith("--")]
SELFTEST = "--selftest" in sys.argv
arm = argv[0] if len(argv) > 0 else "kd05w192ps"
seeds = [int(s) for s in (argv[1] if len(argv) > 1 else "951,952").split(",")]
ref = argv[2] if len(argv) > 2 else "kd05w192"
if SELFTEST:
    arm = ref
A = [f"w5_e10_{arm}_s{s}" for s in seeds]
Rf = [f"w5_e10_{ref}_s{s}" for s in seeds]
miss = [v for v in A + Rf for p in paths(v)[:2] if not p.exists()]
assert not miss, f"missing prediction files (run not finished?): {sorted(set(miss))}"
m = Meta()
W = pmix.weights()
big = m.Mv >= 50


def nd(sp, z):
    return ndcg_fast(z, m.y[sp], m.off[sp], m.M[sp])


def member_pool(names, sp, i, suf=""):
    zs = [m.z(sp, np.load(paths(v, suf)[i]).astype(np.float32)) for v in names]
    return np.mean([nd(sp, z) for z in zs], 0), nd(sp, np.mean(zs, 0)), [nd(sp, z) for z in zs]


def f(d, w=None):
    mu, s = pmix.wmean_se(d, w)
    return f"{mu:+.5f} +- {s:.5f} (z {mu / max(s, 1e-12):+.2f})"


def verdict(d):
    mu = d.mean()
    pv, _ = pmix.wmean_se(d, W["private"])
    halves = [(d[h].mean(), d[h].std() / np.sqrt(h.sum())) for h in (~big, big)]
    c1, c2, c3 = mu >= 0.0008, pv >= 0, all(hm >= -hs for hm, hs in halves)
    return (c1 and c2 and c3), (c1, c2, c3), mu, pv, halves


print(f"== PSEUDO-KD screen: {arm} vs {ref}, seeds {seeds} (paired per feed and seed; rule RULE_predeclared.txt)"
      + ("  [SELFTEST: reference vs itself]" if SELFTEST else ""))
mv_a, pv_a, per_a = member_pool(A, "val", 0)
mv_r, pv_r, per_r = member_pool(Rf, "val", 0)
d_mem, d_pool = mv_a - mv_r, pv_a - pv_r
for s, xa, xr in zip(seeds, per_a, per_r):
    print(f"  seed {s}: val member {xa.mean():.5f} vs ref {xr.mean():.5f}  delta {f(xa - xr)}")
out = {}
for tag, d in (("MEMBER (decision)", d_mem), ("POOL (information)", d_pool)):
    ok, cs, mu, pv, halves = verdict(d)
    out[tag] = ok
    print(f"  {tag:19s} val {f(d)} | pmix-public {f(d, W['public'])} | pmix-private {f(d, W['private'])} | "
          f"M<50 {f(d[~big])} | M>=50 {f(d[big])}")
    print(f"  {'':19s} checks: (1) val >= +0.0008 {'yes' if cs[0] else 'NO'} ({mu:+.5f}); (2) pmix-private >= 0 {'yes' if cs[1] else 'NO'} "
          f"({pv:+.5f}); (3) no M half < -1 SE {'yes' if cs[2] else 'NO'} (M<50 {halves[0][0]:+.5f} vs -{halves[0][1]:.5f}, "
          f"M>=50 {halves[1][0]:+.5f} vs -{halves[1][1]:.5f}) -> {'PASS (screen survives)' if ok else 'KILL'}")
for tag, suf in (("holdout (selected)", ""), ("holdout (@last)", "@last")):
    ha, hpa, _ = member_pool(A, "hold", 1, suf)
    hr, hpr, _ = member_pool(Rf, "hold", 1, suf)
    print(f"  {tag:19s} member {f(ha - hr)} | pool {f(hpa - hpr)}  (information)")
print(f"  absolute: val member {mv_a.mean():.5f} (ref {mv_r.mean():.5f}), {len(seeds)}-pool {pv_a.mean():.5f} (ref {pv_r.mean():.5f})")
if out["MEMBER (decision)"] != out["POOL (information)"]:
    print("  NOTE: the member view (decision) and the pool view (information) give DIFFERENT verdicts; the decision is the member view.")
print(f"RULE VERDICT: {'PASS (screen survives; next step per the lead)' if out['MEMBER (decision)'] else 'KILL'}")
