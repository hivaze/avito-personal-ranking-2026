"""int_moe / a5_check.py: re-read of the one trained sparse-MoE experiment (A5, nnx_exp_attn.md) from the archived prediction files. Val only, CPU."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', ''); sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/src')
import numpy as np
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
A = '<repo>/archive/preds_x/attn'; O = '<repo>/archive/preds_x/optim'
f, M, off = feed_meta('val'); y = np.load(FLAT / 'val/cand_target.npy'); W = pmix.weights()
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), y, off, M)
z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]
out = open('a5_check.log', 'w')
def say(s):
    print(s); out.write(s + '\n')
def rd(d, w=None):
    m, s = pmix.wmean_se(np.asarray(d, np.float64), w); return f'{m:+.5f} +- {s:.5f}'
pairs = {'A5 alone (moe - base), train-only': ([f'{A}/nn_attn_moe_s{s}_trainonly_val.npy' for s in (23, 24)], [f'{O}/nn_base_s{s}_trainonly_val.npy' for s in (23, 24)]),
         'A5 on top of A1+A4, train-only': ([f'{A}/nn_attn_gate_retr_moe_s{s}_trainonly_val.npy' for s in (23, 24)], [f'{A}/nn_attn_gate_retr_s{s}_trainonly_val.npy' for s in (23, 24)]),
         'A5 on top of A1+A4, after stage 2 + fine-tune (OOF)': ([f'{A}/nn_ftb_attn_gate_retr_moe_s{s}_val_oof.npy' for s in (23, 24)], [f'{A}/nn_ftb_attn_gate_retr_s{s}_val_oof.npy' for s in (23, 24)])}
for name, (arm, ref) in pairs.items():
    if not all(os.path.exists(p) for p in arm + ref):
        say(f'{name}: files missing'); continue
    ea = [ev(np.load(p)) for p in arm]; er = [ev(np.load(p)) for p in ref]
    d = (ea[0] - er[0] + ea[1] - er[1]) / 2          # per-feed mean of the two seed pairs
    # pooled SE as in the source report: two independent seed pairs
    se2 = np.sqrt(((ea[0] - er[0]).var() + (ea[1] - er[1]).var()) / len(d)) / 2
    say(f'{name}: seed 23 {np.mean(ea[0] - er[0]):+.5f}, seed 24 {np.mean(ea[1] - er[1]):+.5f}; pooled {d.mean():+.5f} +- {se2:.5f} (independent-pair SE); '
        f'M<50 {rd(d[M < 50])}, M>=50 {rd(d[M >= 50])}; public mix {rd(d, W["public"])}, private mix {rd(d, W["private"])}')
    pa = ev((z(np.load(arm[0])) + z(np.load(arm[1]))) / 2); pr = ev((z(np.load(ref[0])) + z(np.load(ref[1]))) / 2)
    say(f'   2-member pools: arm pair {pa.mean():.5f}, ref pair {pr.mean():.5f}, arm pair - ref pair {rd(pa - pr)}')
