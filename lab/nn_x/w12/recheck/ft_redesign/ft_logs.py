"""Collect the per-epoch OOF curves of every n2cpv3s3 / n2cpv3 fine-tune log (no predictions needed).
epoch 0 = train-only start (mean of fold models = the TO model on held-out folds); epochs 1..4 = raw weights, 3-seed mean; ema = saved prediction."""
import json, glob, re, numpy as np
from pathlib import Path
R = Path('<repo>')
rows = {}
for d in ('nn/logs', 'nn_x/abl/logs', 'nn_x/r6/logs', 'nn_x/w5/logs'):
    for f in glob.glob(str(R / d / '*_n2cpv3s3_ft.json')) + glob.glob(str(R / d / 'w5_e10_base_s95*_n2cpv3_ft.json')):
        j = json.load(open(f)); v = Path(f).name.replace('_ft.json', '')
        if v in rows: continue
        a = j['args']
        rows[v] = dict(c=np.array(j['oof_by_epoch']), snap=j['oof_ndcg_snapshot'], ema=j['oof_ndcg_ema_final'], lr=a['lr'], ep=a['epochs'], seeds=a['seeds'],
                       decay=a['decay'], emab=a['ema'], extra=a['extra'], wd=a['wd'], mf=a['max_feeds'], cb=a['cand_budget'], ln=a['loss_norm'], u0=a['u0_singleton'])
def grp(v):
    for k, g in (('kd05t25w192', 'S:kd05t25w192'), ('kd05w192', 'S:kd05w192'), ('kd05t25', 'S:kd05t25'), ('kd05_', 'S:kd05'), ('kd08', 'S:kd08'),
                 ('base', 'H:w5base(1seed)'), ('pop1', 'H:pop1'), ('p70k', 'H:m02p70k'), ('m02', 'H:m02'), ('gru', 'H:gru')):
        if k in v: return g
    return 'other'
G = {}
for v, r in sorted(rows.items()):
    G.setdefault(grp(v), []).append((v, r))
print('args check (unique):', {(r['lr'], r['ep'], r['seeds'], r['decay'], r['emab'], r['extra'], r['wd'], r['mf'], r['cb'], r['ln'], r['u0']) for r in rows.values()})
print(f"{'group':18s} n  TO(ep0)  d_ep1    d_ep2    d_ep3    d_ep4    d_snap   d_EMA(saved)  [sd over members of d_EMA]  ep1-EMA")
for g, L in sorted(G.items()):
    C = np.array([r['c'] for _, r in L]); sn = np.array([r['snap'] for _, r in L]); em = np.array([r['ema'] for _, r in L])
    d = C[:, 1:] - C[:, :1]
    print(f"{g:18s} {len(L):2d} {C[:,0].mean():.5f} " + ' '.join(f'{x:+.5f}' for x in d.mean(0)) + f" {np.mean(sn-C[:,0]):+.5f} {np.mean(em-C[:,0]):+.5f}      [{np.std(em-C[:,0]):.5f}]   {np.mean(C[:,1]-em):+.5f}")
print('\nper member (students):')
for g, L in sorted(G.items()):
    if not g.startswith('S:'): continue
    for v, r in L:
        c = r['c']; print(f" {v:45s} TO {c[0]:.5f} ep1..4 " + ' '.join(f'{x-c[0]:+.5f}' for x in c[1:]) + f" snap {r['snap']-c[0]:+.5f} ema {r['ema']-c[0]:+.5f}")
