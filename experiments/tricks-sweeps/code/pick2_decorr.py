# How different is a candidate second final from the first one? For pairs of files it reports the
# sampling part of the standard deviation of their private difference (from per-feed val deltas),
# the val delta and the correlation of their test scores.
# How it was run: CPU, 2026-11-15 22:11 UTC, one look; the command line is not on record.
# What it produced: pick2_decorr.txt (results/pick2_decorr.txt): two all-data files built from
# disjoint members correlate 0.9973 on test, the same as the held-out file against the all-data
# file (0.9975), so a second final from disjoint all-data members needs no held-out member.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/reports/TRICKS_1115/synth_checks/pick2_decorr.py
"""SYNTHESIS check (CPU, read-only): how different are candidate pick-2 files from pick 1 FD16P20? Descriptive, nothing selected; one look.
For a pair of files of equal expected level the option value of holding both (if the better of two finals counts) is 0.40 x s, s = sd of their private difference.
Lower bound of s = sampling part = sd of the per-feed val delta / sqrt(20033 private feeds). Also: val delta, test correlation of the final scores.
Disjoint all-data halves on disk today: A8 = FD8(s951-954) + pop1 s951-954, B8 = FD8(s955-958) + pop1 s955-958 (stand-in for the 16-vs-16 pair that exists once seeds 959-966 are fine-tuned)."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
R = Path('<repo>'); P = R / 'preds'
fv, Mv, ov = feed_meta('val'); ft_, Mt, ot = feed_meta('test'); yv = np.load(FLAT / 'val/cand_target.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
zt = lambda x: feed_rank_z(ot, Mt, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv).astype(np.float64)
def pool(ms, kind, sp):
    z = zv if sp == 'val' else zt
    suf = {('ft', 'val'): 'n2cpv3s3_val_oof', ('ft', 'test'): 'n2cpv3s3_test', ('to', 'val'): 'trainonly_val', ('to', 'test'): 'trainonly_test'}[(kind, sp)]
    return np.mean([z(np.load(P / f'nn_{m}_{suf}.npy')) for m in ms], 0)
def build(base, pop, sp):
    z = zv if sp == 'val' else zt
    br = [0.8 * z(pool(base, k, sp)) + 0.2 * z(pool(pop, k, sp)) for k in ('ft', 'to')]
    return 0.5 * z(br[0]) + 0.5 * z(br[1])
def stored(tag, sp):
    z = zv if sp == 'val' else zt
    return 0.5 * z(np.load(P / f'ens_{tag}_ft_{sp}.npy')) + 0.5 * z(np.load(P / f'ens_{tag}_to_{sp}.npy'))
W = lambda ss: [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in ss]
PO = lambda ss: [f'abl_fd10_pop1kd05w192_s{s}' for s in ss]
F = {'FD16P20 (pick 1, stored)': (stored('FD16P20kd', 'val'), stored('FD16P20kd', 'test')),
     'FD16P20vf (stored)': (stored('FD16P20vfkd', 'val'), stored('FD16P20vfkd', 'test')),
     'W8P20 (stored, held-out members)': (stored('W8P20kd', 'val'), stored('W8P20kd', 'test')),
     'FD8P20 (stored) = A8': (stored('FD8P20kd', 'val'), stored('FD8P20kd', 'test')),
     'B8 = FD8(s955-958) + pop1 s955-958': (build(W(range(955, 959)), PO(range(955, 959)), 'val'), build(W(range(955, 959)), PO(range(955, 959)), 'test')),
     'FD16 + pop1 s955-958 (same base, other 4 pop students)': (build(W(range(951, 959)), PO(range(955, 959)), 'val'), build(W(range(951, 959)), PO(range(955, 959)), 'test'))}
E = {k: ev(v[0]) for k, v in F.items()}
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
w_('pair | val level A | val level B | val delta A - B +- paired SE | sampling SE of the delta on 8,562 public feeds | on 20,033 private feeds | test corr | option value 0.40 x private sampling SE (lower bound)')
def pair(a, b):
    d = E[a] - E[b]; sd = d.std()
    c = float(np.corrcoef(F[a][1], F[b][1])[0, 1])
    w_(f'{a} vs {b} | {E[a].mean():.5f} | {E[b].mean():.5f} | {d.mean():+.5f} +- {sd / np.sqrt(len(d)):.5f} | {sd / np.sqrt(8562):.5f} | {sd / np.sqrt(20033):.5f} | {c:.5f} | {0.40 * sd / np.sqrt(20033):+.5f}')
k = list(F)
pair(k[1], k[0]); pair(k[2], k[0]); pair(k[5], k[0]); pair(k[4], k[3]); pair(k[3], k[0]); pair(k[4], k[0])
open(R / 'reports/TRICKS_1115/synth_checks/pick2_decorr.txt', 'w').write('\n'.join(out) + '\n')
