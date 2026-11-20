# The read-out used for every later change of the student recipe: an 8-seed pool of the changed recipe against
# an 8-seed pool of the adopted recipe, paired per feed on val and on the holdout. PASS needs both differences
# non-negative and (z_val + z_holdout) / sqrt(2) >= 2. It also prints the member-level difference and a second
# reference, the mean of 200 random same-size subsets of the 16 wide students.
# Run as (docstring): CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/kd/arm_read8.py
#   <arm> <seeds csv> [ref arm = kd05w192] [ref seeds csv]
# It imports helpers from lab/nn_x/w12/recheck/pool_composition/pc_common.py.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/kd/arm_read8.py
"""One-stage N-seed pooled-z read-out of a student arm (rule: nn_x/w5/logs/E_kd_predeclared.txt, 2026-11-14 21:2xZ WAVE 1 AMENDMENT).
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/kd/arm_read8.py <arm> <seeds csv> [ref arm = kd05w192] [ref seeds csv = same]
Train-only pools (member-equal per-feed rank-z), selected checkpoints, paired per feed against the reference arm on the SAME seeds (matched N):
  DECISION: PASS iff N-pool val delta >= 0 AND N-pool holdout delta >= 0 AND (z_val + z_holdout) / sqrt(2) >= 2.
  Secondary (no decision): member delta, holdout @last, asymptote A_n = (n pool - mean member) / (n - 1), val M<50 / M>=50,
  HOLD-SHORT = stored holdout predictions restricted to random sub-feeds (sizes from val M<50, uniform subset conditioned on >= 1 positive
  by forcing one random positive; 4 replicates, fixed seed), re-z-scored inside the sub-feed.
CPU only, read-only."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
arm = sys.argv[1]
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
def raw(v, suf=''):
    pv, ph, _ = paths(v, suf); return np.load(pv).astype(np.float32), np.load(ph).astype(np.float32)
if arm.startswith('rc:'):
    # RE-COOL HARNESS: arm_read8.py rc:<arm> [ref arm = ref | orig] [seeds csv = 951..958]; members = K and L snapshots of every seed whose arm AND reference runs are complete
    arm = arm[3:]; refarm = sys.argv[2] if len(sys.argv) > 2 else 'ref'; sd = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else list(range(951, 959))
    rc = lambda a, st, x: (f'w5_e10_{"kd05w192" if st == "k" else "kd05t25w192"}_s{x}' if a == 'orig' else f'w5_{st}cd_{a}_s{x}')
    done = lambda v: (W / f'holdout/nn_{v}@last_holdout.npy').exists() and (W / f'nn_{v}_trainonly_val.npy').exists() and (ROOT / f'nn_x/w5/logs/{v}.json').exists()
    pairs = [(rc(arm, st, x), rc(refarm, st, x)) for st in 'kl' for x in sd]; pairs = [(a_, r_) for a_, r_ in pairs if done(a_) and done(r_)]
    assert pairs, 'no completed (arm, reference) snapshot pair yet'
    An, Rn = [a_ for a_, _ in pairs], [r_ for _, r_ in pairs]; seeds = rseeds = [a_[-7:] for a_ in An]; arm, refarm = f're-cool {arm}', f're-cool {refarm}' if refarm != 'orig' else 'the ORIGINAL runs'
else:
    seeds = [int(s) for s in sys.argv[2].split(',')]; refarm = sys.argv[3] if len(sys.argv) > 3 else 'kd05w192'
    rseeds = [int(s) for s in sys.argv[4].split(',')] if len(sys.argv) > 4 else seeds
    name = lambda a, s: f'w5_e10_{a}_s{s}'; An = [name(arm, s) for s in seeds]; Rn = [name(refarm, s) for s in rseeds]
A = [raw(v) for v in An]; R = [raw(v) for v in Rn]; AL = [raw(v, '@last') for v in An]; RL = [raw(v, '@last') for v in Rn]
n, nr = len(seeds), len(rseeds)
def stats(X, sp, e, i):
    zs = [m.z(sp, x[i]) for x in X]; mem = np.mean([e(z) for z in zs], 0); p = e(np.mean(zs, 0)); k = len(X)
    return mem, p, ((k * p - mem) / (k - 1) if k > 1 else p)
print(f'== {arm} ({n} members: {seeds}) vs {refarm} ({nr} members)  (train-only pools, paired per feed)')
Z = {}
for tag, sp, e, i, X, Y in (('val', 'val', ev, 0, A, R), ('holdout', 'hold', eh, 1, A, R), ('holdout @last', 'hold', eh, 1, AL, RL)):
    ma, pa, aa = stats(X, sp, e, i); mr, pr, ar = stats(Y, sp, e, i); d = pa - pr; Z[tag] = (d.mean(), se(d))
    x = ''
    if tag == 'val':
        big = m.Mv >= 50; x = f' | M>=50 {fmt(d[big])} M<50 {fmt(d[~big])}'
    print(f'  {tag:14s}: member {ma.mean():.5f} ({fmt(ma - mr)}) | {n}-pool {pa.mean():.5f} ({fmt(d)}; z {d.mean() / se(d):+.2f}) | asymptote {aa.mean():.5f} ({fmt(aa - ar)}){x}')
# HOLD-SHORT (free form): random sub-feeds of the stored holdout predictions
mshort = m.Mv[m.Mv < 50]; rng = np.random.default_rng(20261114); fid = np.repeat(np.arange(len(m.Mh)), m.Mh); REP = 4
ds = np.zeros(len(m.Mh))
for r in range(REP):
    ms = rng.choice(mshort, len(m.Mh)); key = rng.random(len(m.yh))
    pk = np.where(m.yh > 0, key, 2.0); o = np.lexsort((pk, fid)); first = o[m.oh]; key[first] = -1.0
    o = np.lexsort((key, fid)); rank = np.empty(len(key), np.int64); rank[o] = np.arange(len(key)) - np.repeat(m.oh, m.Mh)
    sel = np.flatnonzero(rank < np.repeat(ms, m.Mh)); Ms = ms.astype(np.int64); os_ = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64); ys = m.yh[sel]
    pz = lambda X: np.mean([feed_rank_z(os_, Ms, x[1][sel])[:, 1] for x in X], 0)
    ds += (ndcg_fast(pz(A), ys, os_, Ms) - ndcg_fast(pz(R), ys, os_, Ms)) / REP
print(f'  HOLD-SHORT    : {n}-pool delta {fmt(ds)} (z {ds.mean() / se(ds):+.2f}; {REP} replicates; secondary)')
BAR = 2.5 if sys.argv[1].startswith('rc:') else 2.0
if not sys.argv[1].startswith('rc:'):
    # SECOND REFERENCE (E_kd_predeclared.txt 22:1xZ item 4): matched-N mean over 200 random N-subsets of the 16 wide students (cached per N)
    cf = ROOT / f'nn_x/w12/kd/out/matchedN_ref_N{n}.npz'
    if cf.exists(): c_ = np.load(cf); rv, rh = c_['val'], c_['hold']
    else:
        W16 = [f'w5_e10_{a_}_s{x_}' for a_ in ('kd05w192', 'kd05t25w192') for x_ in range(951, 959)]; Zw = [(m.z('val', raw(v)[0]), m.z('hold', raw(v)[1])) for v in W16]
        g = np.random.default_rng(20261115); rv = np.zeros(len(m.Mv)); rh = np.zeros(len(m.Mh)); K = 200
        for _ in range(K):
            ix = g.choice(16, n, replace=False); rv += ev(np.mean([Zw[i][0] for i in ix], 0)) / K; rh += eh(np.mean([Zw[i][1] for i in ix], 0)) / K
        np.savez(cf, val=rv, hold=rh)
    pv_ = ev(np.mean([m.z('val', x[0]) for x in A], 0)); ph_ = eh(np.mean([m.z('hold', x[1]) for x in A], 0)); d1, d2 = pv_ - rv, ph_ - rh
    zb = (d1.mean() / se(d1) + d2.mean() / se(d2)) / np.sqrt(2)
    print(f'  vs matched-N  : reference {rv.mean():.5f} / {rh.mean():.5f} (mean of 200 random {n}-subsets of the 16 wide students) | val {fmt(d1)} holdout {fmt(d2)} pooled z {zb:+.2f}'
          f' -> {"holds" if d1.mean() >= 0 and d2.mean() >= 0 and zb >= BAR else "does not hold"} at the PASS bar (SE here ignores the reference\'s own subset noise)')
(dv, sv), (dh, sh) = Z['val'], Z['holdout']; zp = (dv / sv + dh / sh) / np.sqrt(2)
ok = dv >= 0 and dh >= 0 and zp >= BAR
print(f'  RULE: val {dv:+.5f} (z {dv / sv:+.2f}), holdout {dh:+.5f} (z {dh / sh:+.2f}), pooled z {zp:+.2f} (bar {BAR}) -> {"PASS" if ok else "no pass"}'
      + (f'  [REPLICATE band: both >= 0 and 1.6 <= pooled z < 2 -> 4 fresh seeds; bundle entry: both >= 0 and pooled z >= 1.0]' if n >= 8 else f'  [{n} seeds: both >= 0 and pooled z >= 1.6 -> extend to 8 and re-read; pooled z < 1 = closed]'))
