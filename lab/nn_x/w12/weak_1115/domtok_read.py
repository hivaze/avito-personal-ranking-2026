"""Read-only CPU re-read (auditor, 2026-11-15) of COVERAGE_MAP row 157 (PRODDIR / DOMAIN-DIAG), which the veto audit could not re-read
("side-observation files not located"; reports/STALE_1115/veto.md 1.2).  The files exist: solvers/out/auxexp/<member>_val_main_{rl,prod}.npy
(12 hard-label members of 11-12/11-13; main head scored on val with the random-like token [= what every file uses] and with the production token).
DECLARED BEFORE THE RUN (3 forms, no scan):
  F1  production token on every val feed                      (the form on record: +0.00011 +- 0.00017, cold-positive feeds +0.0026)
  F2  production token on feeds with M < 50 only, random-like on M >= 50  (= the token x feed-size combinations that exist in training:
      nn_x/w5/train.py:291 sets p_random = 0 for M < 50; reports/review_nnmath.md NNMATH-8)
  F3  token TTA: per-feed z-mean of the two passes
Statistic: 12-member pool (member-equal per-feed z-mean), paired per feed against the pool under the random-like token; columns val, M<50, M>=50,
pmix-public, pmix-private (src/pmix.py), cold-positive feeds (all positives have it_hist_all_luo == 0), member-level sign count.
Not a decision rule: old members, same val feeds as the two earlier looks -> information only.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/weak_1115/domtok_read.py"""
import os, sys, glob
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition'); sys.path.insert(0, '<repo>/src')
from pc_common import *
import pmix
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); wts = pmix.weights(); big = m.Mv >= 50
D = ROOT / 'solvers/out/auxexp'
mem = sorted({os.path.basename(f)[:-len('_val_main_rl.npy')] for f in glob.glob(str(D / '*_val_main_rl.npy'))})
RL = [m.z('val', np.load(D / f'{v}_val_main_rl.npy').astype(np.float32)) for v in mem]
PR = [m.z('val', np.load(D / f'{v}_val_main_prod.npy').astype(np.float32)) for v in mem]
bigrow = np.repeat(big, m.Mv)
# cold-positive feeds: every positive of the feed has it_hist_all_luo == 0
luo = np.load(ROOT / 'data/feats/val/it_hist_all_luo.npy'); fid = np.repeat(np.arange(len(m.Mv)), m.Mv)
npos = np.bincount(fid, weights=(m.yv > 0), minlength=len(m.Mv)); ncold = np.bincount(fid, weights=((m.yv > 0) & (luo == 0)), minlength=len(m.Mv))
coldpos = (npos > 0) & (ncold == npos); warmpos = (npos > 0) & (ncold == 0)
f2 = lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
def line(tag, A):
    ea = [ev(a) for a in A]; er = [ev(r) for r in RL]
    d = ev(np.mean(A, 0)) - ev(np.mean(RL, 0)); md = np.array([(a - r).mean() for a, r in zip(ea, er)])
    pu, su = pmix.wmean_se(d, wts['public']); pr, sr = pmix.wmean_se(d, wts['private'])
    print(f'{tag}: 12-pool val {f2(d)} (z {d.mean() / se(d):+.2f}) | M<50 {f2(d[~big])} | M>=50 {f2(d[big])} | pmix-public {pu:+.5f}+-{su:.5f} | pmix-private {pr:+.5f}+-{sr:.5f}'
          f' | cold-positive feeds ({coldpos.mean():.3f}) {f2(d[coldpos])} | warm-positive ({warmpos.mean():.3f}) {f2(d[warmpos])} | member mean {md.mean():+.5f}, positive in {(md > 0).sum()} of {len(md)}')
print('members:', ' '.join(mem)); print(f'pool under the random-like token (reference): {ev(np.mean(RL, 0)).mean():.5f}')
line('F1 production token, all feeds   ', PR)
line('F2 production token on M<50 only ', [np.where(bigrow, r, p) for r, p in zip(RL, PR)])
line('F3 token TTA (z-mean of 2 passes)', [m.z('val', 0.5 * r + 0.5 * p) for r, p in zip(RL, PR)])
