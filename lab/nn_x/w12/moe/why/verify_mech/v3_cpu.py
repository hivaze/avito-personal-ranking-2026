"""verify_mech v3 (CPU, read-only).
 (1) arms vs ALL 8 reference seeds 951-958 (unpaired by seed): member delta = per-feed mean of the 4 arm members - per-feed mean of the
     8 ref members; pool delta = arm 4-pool - per-feed mean over all 70 ref 4-pools. Selected and @last, val and holdout.
 (2) M2 (arm B): what does 'random routing' keep? From the saved val routing records: mean selected-gate sum under the model's top-2 vs
     the expected sum under a random 2-of-4 draw; mean cross-expert correlation of the sigmoid gates (per head, over tokens).
 (3) M2 / M1 expert weights: mean |cos| between the flattened weights of different experts of the same head (B: Wv, Wo) / block (A: W2).
"""
import itertools, os, sys
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
import numpy as np, torch
from pathlib import Path
from pc_common import Meta, _ndcg
torch.set_num_threads(12)
ROOT = Path('<repo>')
MOE, W5 = ROOT / 'lab/nn_x/w12/moe/out', ROOT / 'nn_x/w5'
ARMS = dict(ref='kd05w192', A='kd05w192moeffn', B='kd05w192moevo', C='kd05w192b3ff1')
se = lambda d: float(np.std(d) / np.sqrt(len(d)))

m = Meta()
nd = lambda sp, s: _ndcg(np.ascontiguousarray(s, np.float32), m.y[sp], m.off[sp], m.M[sp])


def path(g, s, sp, suf=''):
    v = f'w5_e10_{ARMS[g]}_s{s}'
    d = (W5 if g == 'ref' else MOE) / 'preds'
    return d / (f'nn_{v}{suf}_trainonly_val.npy' if sp == 'val' else f'holdout/nn_{v}{suf}_holdout.npy')


print('== (1) arms vs all 8 reference seeds (unpaired); +- = feed SE')
REF8 = range(951, 959)
for sp in ('val', 'hold'):
    for suf in ('', '@last'):
        zr = {s: m.z(sp, np.load(path('ref', s, sp, suf))) for s in REF8}
        mem_ref = np.mean([nd(sp, np.load(path('ref', s, sp, suf))) for s in REF8], 0)
        pools = np.array([nd(sp, np.mean([zr[s] for s in G], 0)) for G in itertools.combinations(REF8, 4)])
        pool_ref = pools.mean(0)
        line = f'  {sp:4s} {"selected" if suf == "" else "@last":8s} ref8 member {mem_ref.mean():.5f} mean ref 4-pool {pool_ref.mean():.5f} (sd over 70 pools {pools.mean(1).std():.5f}; ' \
               f'ref951-954 pool rank {int((pools.mean(1) > pools.mean(1)[0]).sum()) + 1}/70)'
        print(line)
        for g in 'ABC':
            zs = [m.z(sp, np.load(path(g, s, sp, suf))) for s in range(951, 955)]
            dm = np.mean([nd(sp, np.load(path(g, s, sp, suf))) for s in range(951, 955)], 0) - mem_ref
            dp = nd(sp, np.mean(zs, 0)) - pool_ref
            pct = float((pools.mean(1) < nd(sp, np.mean(zs, 0)).mean()).mean())
            print(f'     {g}: member {dm.mean():+.5f}+-{se(dm):.5f} | pool {dp.mean():+.5f}+-{se(dp):.5f} | arm pool beats {pct:.2f} of the 70 ref 4-pools')

print()
print('== (2) arm B routing records (val): what random routing keeps')
for b in ('cross.0', 'cross.1'):
    for R in 'VO':
        rows = []
        for s in (951, 952, 953, 954):
            z = np.load(MOE.parent / f'why/mech/out/B/route_w5_e10_kd05w192moevo_s{s}_val_{b}.npz')
            sel = z[f'sel{R}'].astype(np.int64); ok = sel[:, 0, 0] >= 0
            sv = z[f's{R}'][ok].astype(np.float32); sl = sel[ok]
            if len(sv) > 1_000_000:
                idx = np.random.default_rng(0).choice(len(sv), 1_000_000, replace=False); sv, sl = sv[idx], sl[idx]
            top = np.take_along_axis(sv, sl, -1).sum(-1).mean()          # model's top-2 (biased selection) gate sum
            rnd = sv.sum(-1).mean() / 2                                   # E[random 2-of-4 sum]
            best = np.sort(sv, -1)[..., -2:].sum(-1).mean()               # top-2 by raw gate
            cors = []
            for h in range(sv.shape[1]):
                c = np.corrcoef(sv[:, h, :].T)
                cors.append(c[np.triu_indices(4, 1)].mean())
            rows.append((top, rnd, best, np.mean(cors)))
        r = np.array(rows)
        print(f'  {b} {R}: selected-gate sum {r[:, 0].mean():.3f} | random-2 expected {r[:, 1].mean():.3f} (ratio {r[:, 1].mean() / r[:, 0].mean():.2f}) | '
              f'top-2 by raw gate {r[:, 2].mean():.3f} | mean cross-expert gate corr {r[:, 3].mean():+.3f} (per seed {" ".join(f"{x:+.2f}" for x in r[:, 3])})')

print()
print('== (3) expert weight similarity (selected checkpoints): mean |cos| between experts (same head / block); 1 = identical')
for s in (951, 952, 953, 954):
    ck = torch.load(MOE / 'ckpt' / f'w5_e10_kd05w192moevo_s{s}.pt', map_location='cpu', weights_only=False)
    sd = ck['state_dict']; h = ck['cfg'].get('heads', 4)
    line = f'  B s{s}:'
    for k in [k for k in sd if k.endswith('vo.wv') or k.endswith('vo.wo')]:
        W = sd[k].float(); E = W.shape[0] // h
        W = W.view(h, E, -1)
        Wn = W / W.norm(dim=-1, keepdim=True)
        cs = [float((Wn[i] @ Wn[i].T)[np.triu_indices(E, 1)].abs().mean()) for i in range(h)]
        line += f' {k.replace(".attn.vo", "")} {np.mean(cs):.3f}'
    print(line)
for s in (951, 952, 953, 954):
    ck = torch.load(MOE / 'ckpt' / f'w5_e10_kd05w192moeffn_s{s}.pt', map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    line = f'  A s{s}:'
    for k in [k for k in sd if k.endswith('moe.w2')]:
        W = sd[k].float(); E = W.shape[0]
        Wn = W.reshape(E, -1); Wn = Wn / Wn.norm(dim=-1, keepdim=True)
        line += f' {k.replace(".moe.w2", "")} {float((Wn @ Wn.T)[np.triu_indices(E, 1)].abs().mean()):.3f}'
    print(line)
