# Acceptance check for the later all-data students (seeds 959 and up), which have no held-out twin. Structure
# as in accept_twins.py; the in-sample holdout lift must fall inside the band of the 16 accepted students;
# pools of the new students are compared on val with pools of the old ones of equal size.
# Run on CPU as in the script's docstring: accept_wide_twinfree.py <first seed> <last seed>, for 959 966 and
# 967 974. Its printed output is quoted in the ledger blocks of 2026-11-16 03:26 and 05:59 UTC.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/stale/fulldata/accept_wide_twinfree.py
#
"""Twin-free acceptance of the NEW full-data WIDE students (w5_fd10_kd05w192 / kd05t25w192, seeds >= 959: rule G clause (1), ledger 2026-11-15 21:07Z). They have no held-out twin,
so the checks are made against the accepted full-data students of seeds 951-958 (same lane and code) and the held-out wide students (w5_e10_*, seeds 951-958).
  A1 structure: train feeds 3,151,208 and total steps 99,840 in the log; holdout_in_train; all four prediction files with the right row counts, finite.
  A2 positive control: IN-SAMPLE holdout NDCG of each new student minus the mean OUT-OF-SAMPLE holdout NDCG of the held-out students of its arm; accepted band = the lifts of the accepted students s951-958.
  A3 level on val (information): train-only val per member; pools of the new students vs pools of s951-954 / s955-958 of equal size.
  accept_wide_twinfree.py <first seed> <last seed>      CPU, read-only."""
import os, sys, json
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
H = ROOT / 'nn_x/w5'; a, b = int(sys.argv[1]), int(sys.argv[2]); ARMS = ('kd05w192', 'kd05t25w192'); bad = 0
def ld(v, s):
    for d in (ROOT / 'preds', H / 'preds'):
        if (d / f'nn_{v}_{s}.npy').exists(): return np.load(d / f'nn_{v}_{s}.npy')
    return None
def one(v):
    log = (H / f'logs/{v}.log').read_text(); js = json.dumps(json.load(open(H / f'logs/{v}.json')))
    okf = 'train feeds 3151208' in log and 'total steps 99840' in log; okh = 'holdout_in_train' in log or 'holdout_in_train' in js
    P = {s: ld(v, s) for s in ('trainonly_val', 'trainonly_test', 'n2cpv3s3_val_oof', 'n2cpv3s3_test')}
    oks = all(x is not None for x in P.values()) and P['trainonly_val'].shape[0] == m.n['val'] == P['n2cpv3s3_val_oof'].shape[0] and P['trainonly_test'].shape[0] == m.n['test'] == P['n2cpv3s3_test'].shape[0] \
        and all(np.isfinite(x).all() for x in P.values())
    hi = np.load(H / f'preds/holdout/nn_{v}_holdoutINSAMPLE.npy').astype(np.float32)
    return (okf and okh and oks and np.isfinite(hi).all()), (okf, okh, oks), ev(m.z('val', P['trainonly_val'].astype(np.float32))).mean() if P['trainonly_val'] is not None else float('nan'), \
        ev(m.z('val', P['n2cpv3s3_val_oof'].astype(np.float32))).mean() if P['n2cpv3s3_val_oof'] is not None else float('nan'), eh(m.z('hold', hi)).mean()
for arm in ARMS:
    ho = np.mean([eh(m.z('hold', np.load(H / f'preds/holdout/nn_w5_e10_{arm}_s{s}_holdout.npy').astype(np.float32))).mean() for s in range(951, 959)])
    acc = [one(f'w5_fd10_{arm}_s{s}') for s in range(951, 959)]; lifts = [x[4] - ho for x in acc]
    print(f'{arm}: held-out students s951-958 mean out-of-sample holdout {ho:.5f}; accepted full-data students s951-958: in-sample lift {min(lifts):+.5f}..{max(lifts):+.5f}, train-only val {np.mean([x[2] for x in acc]):.5f} (min {min(x[2] for x in acc):.5f}, max {max(x[2] for x in acc):.5f}), FT val {np.mean([x[3] for x in acc]):.5f}')
    for s in range(a, b + 1):
        v = f'w5_fd10_{arm}_s{s}'
        if not (H / f'logs/{v}.json').exists(): print(f'  {v}: not finished'); bad += 1; continue
        a1, d, vl, fl, hl = one(v); lift = hl - ho; a2 = min(lifts) - 0.002 <= lift <= max(lifts) + 0.002; bad += (not a1) + (not a2)
        print(f'  {v}: A1 {"ok" if a1 else "DEFECT " + str(d)} | A2 in-sample holdout {hl:.5f} lift {lift:+.5f} {"ok" if a2 else "OUT OF BAND"} | A3 train-only val {vl:.5f}, FT val {fl:.5f}')
f = lambda d: f'{d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f}'
def pool(seeds, suf='trainonly_val'): return ev(np.mean([m.z('val', ld(f'w5_fd10_{arm}_s{s}', suf).astype(np.float32)) for arm in ARMS for s in seeds], 0))
new = list(range(a, b + 1)); k = len(new)
try:
    for suf in ('trainonly_val', 'n2cpv3s3_val_oof'):
        pn = pool(new, suf); refs = [list(range(951, 959))[i:i + k] for i in range(0, 8, k)] if k <= 8 else [list(range(951, 959))]
        print(f'A3 {suf}: pool of the {2 * k} new students {pn.mean():.5f} | ' + ' | '.join(f'minus pool s{r[0]}-{r[-1]} ({pool(r, suf).mean():.5f}): {f(pn - pool(r, suf))}' for r in refs))
except Exception as e: print('A3 pool read skipped:', e)
print('RESULT:', 'no structural defect' if bad == 0 else f'{bad} DEFECT(S) / unfinished')
