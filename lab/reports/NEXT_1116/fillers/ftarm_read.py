"""ftarm_read.py (2026-11-16, reports/NEXT_1116/fillers, lane B): read-out of the PAIRED FINE-TUNE ARMS, extended copy of reports/TRICKS_1115/followup/mixft_read_v2.py
(the original is not edited). CPU, read-only.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python reports/NEXT_1116/fillers/ftarm_read.py                 # the declared arms
  ... ftarm_read.py --selftest mixft3 [suffix ...]   # plumbing check on STORED suffixes (verdicts meaningless; mixft3 must reproduce logs/mixft_read_out.txt of Q4)

Arms = the kinds listed in reports/NEXT_1116/fillers/ftarm_DECLARED.txt (written by ftarm_launch.sh BEFORE the first run; the Holm family is that list, whatever is read).
Suffix of an arm = ft<kind> (ftbs05, ftbs2, ftxm30, ...); control suffix = mixctl3 (the stored controls of Q4 MIXFT, admitted by ftarm_pathcheck.py); production = n2cpv3s3.
Members: the 8 all-data wide students of FD8P20 (w5_fd10_{kd05w192,kd05t25w192}_s951-954), every run 3 fine-tune seeds, production flags.

FILE(x) = 0.5 z(0.8 z(FT pool of the 8 members, suffix x) + 0.2 z(stored 3-seed FT pool of the 4 popularity-light students))
        + 0.5 z(stored train-only branch preds/ens_FD8P20kd_to_val.npy)          (= the FD8P20 form; identical to mixft_read_v2.py)
All deltas are arm minus control at FILE level, paired per val feed (14,423 feeds); SE = paired per-feed SE (private mix: src/pmix.py weights, pmix.wmean_se).

RULE (task text of lane B, "rules on record to reuse unchanged"; operational form written 2026-11-16 before any run):
  PASS(arm) iff  (1) val delta >= +2 SE with Holm over the declared arms: one-sided p of z = delta / SE, family alpha = P(z > 2) = 0.02275, step-down:
                     the smallest p must be <= alpha / k, the next <= alpha / (k - 1), ...; the first failure fails that arm and every arm after it
                     (z thresholds: k = 3: 2.43 / 2.28 / 2.00; k = 2: 2.28 / 2.00; k = 1: 2.00)
            AND  (2) private-mix delta >= 0
            AND  (3) val M<50 delta >= -1 SE.
  A PASS only buys a 3-seed replication of the arm on the 16 all-data members and then a candidate for PICK 2, never pick 1. If several arms pass, the one with the
  largest val z goes to the replication. Anything else: CLOSED. INCOMPLETE (no verdict, no number printed) while any member of any declared arm is missing.
  One look, after all members of all declared arms are complete and the path check of the day reads PASS.
Reported without a decision: val M>=50, pmix-public, the fine-tuned branch and the wide fine-tuned pool alone, per-member deltas, control minus production (the size of a
no-effect difference), the measured paired SE of the design on record (Q4: val 0.00022, private mix 0.00035)."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
from math import erfc, sqrt
from pathlib import Path
R = Path('<repo>'); P = R / 'preds'; W5 = R / 'nn_x/w5/preds'; FD = R / 'reports/NEXT_1116/fillers'
C = 'mixctl3'; S = 'n2cpv3s3'; ALPHA = 0.5 * erfc(2 / sqrt(2))   # P(z > 2) = 0.02275
args = sys.argv[1:]; selftest = bool(args) and args[0] == '--selftest'
if selftest:
    SUFS = args[1:] or ['mixft3']; print(f'SELFTEST on stored suffixes {SUFS}: plumbing only, the verdicts below are meaningless')
else:
    dec = FD / 'ftarm_DECLARED.txt'; assert dec.exists(), f'{dec} missing: the arms are declared by ftarm_launch.sh before the first run'
    kinds = [l.split(':', 1)[1].split() for l in dec.read_text().splitlines() if l.startswith('arms:')][0]
    SUFS = [f'ft{k}' for k in kinds]; print(f'declared arms ({dec.name}): {kinds}')
    pc = FD / 'logs/ftarm_pathcheck_out.txt'
    assert pc.exists() and 'PATH CHECK: PASS' in pc.read_text(), f'{pc}: the path check of the day is missing or did not PASS -> no read'
MEM = [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 955)]
miss = {suf: [v for v in MEM if not (W5 / f'nn_{v}_{suf}_val_oof.npy').exists()] for suf in SUFS + [C, S]}
if any(miss.values()):
    print('INCOMPLETE: missing val_oof files: ' + '; '.join(f'{k}: {len(v)} of 8 ({", ".join(x[8:] for x in v)})' for k, v in miss.items() if v)); sys.exit(2)
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv).astype(np.float64)
WT = pmix.weights()
zpop = zv(np.mean([zv(np.load(P / f'nn_abl_fd10_pop1kd05w192_s{s}_n2cpv3s3_val_oof.npy')) for s in range(951, 955)], 0))
zto = zv(np.load(P / 'ens_FD8P20kd_to_val.npy'))
mem = {suf: {v: np.load(W5 / f'nn_{v}_{suf}_val_oof.npy') for v in MEM} for suf in SUFS + [C, S]}
def file_(suf):
    ft = zv(np.mean([zv(mem[suf][v]) for v in MEM], 0))
    return ev(0.5 * zv(0.8 * ft + 0.2 * zpop) + 0.5 * zto), ev(ft), ev(0.8 * ft + 0.2 * zpop)
F_ = {suf: file_(suf) for suf in SUFS + [C, S]}
def f(d, w=None, m=None):
    if m is not None: d = d[m]; w = None if w is None else w[m]
    mu, s = (d.mean(), d.std() / np.sqrt(len(d))) if w is None else pmix.wmean_se(d, w)
    return mu, s, f'{mu:+.5f} +- {s:.5f} (z {mu / s:+.2f})'
WIT = {'val': (None, None), 'val M<50': (None, Mv < 50), 'val M>=50': (None, Mv >= 50), 'pmix-public': (WT['public'], None), 'pmix-private': (WT['private'], None)}
def table(d): return {k: f(d, w, m) for k, (w, m) in WIT.items()}
print(f'control suffix {C}, production suffix {S}; val feeds {len(Mv)}, M<50 {int((Mv < 50).sum())}, M>=50 {int((Mv >= 50).sum())}')
print('LEVELS (val mean NDCG): ' + ' | '.join(f'{s}: FILE {F_[s][0].mean():.5f}, wide FT pool {F_[s][1].mean():.5f}' for s in [C, S] + SUFS))
print('CONTROL minus PRODUCTION (same recipe, two realisations of every member = the size of a no-effect difference):')
for name, i in (('FILE', 0), ('WIDE FT POOL alone', 1)):
    r = table(F_[C][i] - F_[S][i]); print(f'   {name}: ' + ' | '.join(f'{k} {v[2]}' for k, v in r.items()))
RES = {}
for suf in SUFS:
    rF = table(F_[suf][0] - F_[C][0]); rP = table(F_[suf][1] - F_[C][1]); rB = table(F_[suf][2] - F_[C][2]); RES[suf] = rF
    print(f'== arm {suf} minus control {C}')
    for name, r in (('FILE', rF), ('FT BRANCH (0.8 wide + 0.2 pop)', rB), ('WIDE FT POOL alone', rP)):
        print(f'   {name}: ' + ' | '.join(f'{k} {v[2]}' for k, v in r.items()))
    pm = []
    for v in MEM:
        ea, ec = ev(mem[suf][v]), ev(mem[C][v]); r = table(ea - ec); pm.append([r[k][0] for k in WIT])
    pm = np.array(pm)
    print('   fine-tuned member alone, mean over the 8 members: ' + ' | '.join(f'{k} {pm[:, i].mean():+.5f} (sd {pm[:, i].std(ddof=1):.5f}, > 0: {int((pm[:, i] > 0).sum())}/8)' for i, k in enumerate(WIT)))
# Holm step-down on the val clause
k = len(SUFS); zval = {s: RES[s]['val'][0] / RES[s]['val'][1] for s in SUFS}; pval = {s: 0.5 * erfc(zval[s] / sqrt(2)) for s in SUFS}
order = sorted(SUFS, key=lambda s: pval[s]); holm = {}; alive = True
print(f'== Holm step-down over {k} arm(s), family alpha {ALPHA:.5f} (= P(z > 2)); order by p: {order}')
for i, s in enumerate(order):
    thr = ALPHA / (k - i); ok = alive and pval[s] <= thr; alive = ok; holm[s] = ok
    print(f'   {s}: val z {zval[s]:+.3f}, one-sided p {pval[s]:.4f}, threshold p <= {thr:.5f} -> val clause {"holds" if ok else "MISSED"}')
print('== RULE per arm (FILE level): (1) val >= +2 SE, Holm; (2) pmix-private >= 0; (3) val M<50 >= -1 SE')
passed = []
for s in SUFS:
    r = RES[s]; c2 = r['pmix-private'][0] >= 0; c3 = r['val M<50'][0] >= -r['val M<50'][1]; ok = holm[s] and c2 and c3
    if ok: passed.append(s)
    print(f"   {s}: val {r['val'][2]} -> (1) {'holds' if holm[s] else 'MISSED'} | pmix-private {r['pmix-private'][2]} -> (2) {'holds' if c2 else 'MISSED'} | "
          f"val M<50 {r['val M<50'][2]} vs -1 SE = {-r['val M<50'][1]:+.5f} -> (3) {'holds' if c3 else 'MISSED'} ==> "
          f"{'PASS -> 3-seed replication on the 16 all-data members, then candidate for PICK 2 only' if ok else 'NOT PASSED -> CLOSED'}")
if len(passed) > 1:
    print(f'   several arms pass: {max(passed, key=lambda s: zval[s])} (largest val z) goes to the replication')
if selftest: print('SELFTEST: verdicts above are meaningless (stored suffixes, not declared arms)')
