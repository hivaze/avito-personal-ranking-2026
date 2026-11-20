"""Step 3 / 4 driver. usage: step3_nested.py <RUN>  with RUN in FD16P4 | W16P4 | PL1..PL4 (pool placebos inside FD16) | PERM101..PERM105 (feature-permutation placebos on FD16P4)."""
from gate import *
import sys
run = sys.argv[1]; y = labels().astype(np.float32); f, M, off = meta('val'); g = groups()
PL = {'PL1': ('kd05w192', range(951, 955)), 'PL2': ('kd05w192', range(955, 959)), 'PL3': ('kd05t25w192', range(951, 955)), 'PL4': ('kd05t25w192', range(955, 959))}
plain = (0.2, 0.5)
if run in PL:
    pop = [f'w5_fd10_{PL[run][0]}_s{s}' for s in PL[run][1]]; base = [m for m in MEM['FD16'] if m not in pop]; assert len(base) == 12 and len(pop) == 4
    q = pools4('val', base, pop, tag=run); E = tensor(q, y, tag=run); X = features('val', q); plain = (0.25, 0.5)
elif run.startswith('PERM'):
    q = pools4('val', None, None, tag='FD16P4'); E = tensor(q, y, tag='FD16P4'); X = features('val', q); X = X[np.random.default_rng(int(run[4:])).permutation(len(X))]
else:
    q = pools4('val', MEM['FD16'] if run == 'FD16P4' else MEM['W16'], MEM['FDP'] if run == 'FD16P4' else MEM['P1w'], tag=run); E = tensor(q, y, tag=run); X = features('val', q)
ctx = Ctx(E, X, q, y); lf = open(HERE / f'log_step3_{run}.txt', 'a')
def log(*a):
    print(*a, flush=True); print(*a, file=lf, flush=True)
R = nested(ctx, g, M, plain=plain, tag=run, log=log)
W = pmixw(); import pmix
def cs(d): m, se = cluster_se(d, g); return [float(m), float(se)]
def pm(d, k): m, se = pmix.wmean_se(d, W[k]); return [float(m), float(se)]
res = {'run': run, 'plain_w': plain, 'plain': float(R['plain'].mean()), 'mrule': float(R['mrule'].mean()), 'mrule_minus_plain': cs(R['mrule'] - R['plain']), 'fam': {}}
for k in FAMS:
    ne, ins = R['nest'][k], R['ins'][k]
    res['fam'][k] = dict(nested=float(ne.mean()), insample=float(ins.mean()), n_vs_plain=cs(ne - R['plain']), i_vs_plain=cs(ins - R['plain']), n_vs_mrule=cs(ne - R['mrule']), i_vs_mrule=cs(ins - R['mrule']),
                         n_vs_const2=cs(ne - R['nest']['CONST2']), i_vs_const2=cs(ins - R['ins']['CONST2']), n_vs_constp=cs(ne - R['nest']['CONSTp']), n_vs_mstep=cs(ne - R['nest']['MSTEP']),
                         n_vs_mrule_pub=pm(ne - R['mrule'], 'public'), n_vs_mrule_priv=pm(ne - R['mrule'], 'private'), n_vs_plain_pub=pm(ne - R['plain'], 'public'), n_vs_plain_priv=pm(ne - R['plain'], 'private'),
                         full=R['full'][k], folds=R['desc'][k], wmean=[float(R['wmean'][k][0].mean()), float(R['wmean'][k][1].mean())])
json.dump(res, open(HERE / f'out_step3_{run}.json', 'w'), indent=1, default=float)
np.savez(HERE / f'cache/nest_{run}.npz', plain=R['plain'], mrule=R['mrule'], **{f'n_{k}': R['nest'][k] for k in FAMS}, **{f'i_{k}': R['ins'][k] for k in FAMS}, **{f'wp_{k}': R['fullw'][k][0] for k in FAMS}, **{f'wf_{k}': R['fullw'][k][1] for k in FAMS})
log(f'=== {run}: PLAIN{plain} {res["plain"]:.6f} | MRULE {res["mrule"]:.6f} ({res["mrule_minus_plain"][0]:+.5f}+-{res["mrule_minus_plain"][1]:.5f})')
log(f'{"family":8s} {"in-sample":>9s} {"nested":>9s} | {"nested - PLAIN":>18s} {"(in-sample)":>11s} | {"nested - MRULE":>18s} {"(in-sample)":>11s} | {"nested - nCONST2":>18s} | pmix pub / priv of nested - MRULE')
fm = lambda t: f'{t[0]:+.5f}+-{t[1]:.5f}'
for k in FAMS:
    r = res['fam'][k]; log(f'{k:8s} {r["insample"]:.6f} {r["nested"]:.6f} | {fm(r["n_vs_plain"])} ({r["i_vs_plain"][0]:+.5f}) | {fm(r["n_vs_mrule"])} ({r["i_vs_mrule"][0]:+.5f}) | {fm(r["n_vs_const2"])} | {fm(r["n_vs_mrule_pub"])} / {fm(r["n_vs_mrule_priv"])}')
from collections import Counter
log('AUTO picks over 25 outer fits:', dict(Counter(d['pick'] for d in res['fam']['AUTO']['folds'])), '| all-feed AUTO pick:', res['fam']['AUTO']['full']['pick'])
log('MSTEP picks (cut, lo, hi):', dict(Counter((d['cut'], d['lo'], d['hi']) for d in res['fam']['MSTEP']['folds'])), '| all-feed:', res['fam']['MSTEP']['full'])
log('STUMP1 picks (feature):', dict(Counter(d['feat'] for d in res['fam']['STUMP1']['folds'])), '| all-feed:', res['fam']['STUMP1']['full'])
log('PSTUMP picks (pop feat, ft feat):', dict(Counter((d['pop']['feat'], d['ft']['feat']) for d in res['fam']['PSTUMP']['folds'])), '| all-feed:', res['fam']['PSTUMP']['full'])
log('CONST2 picks:', dict(Counter((d['wp'], d['wf']) for d in res['fam']['CONST2']['folds'])), '| all-feed:', res['fam']['CONST2']['full'], '| CONSTp:', dict(Counter(d['wp'] for d in res['fam']['CONSTp']['folds'])))
for k in ('LINM', 'LINp', 'LIN2'): log(f'{k} ridge picks:', dict(Counter(d['lam'] for d in res['fam'][k]['folds'])), '| all-feed:', {kk: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 5)) for kk, v in res['fam'][k]['full'].items()})
