"""Recompute every famN run and every paired arm-vs-reference read with own code, from the saved prediction files."""
import sys, glob; sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review')
from rv_common import *
wv, wh = Wit('val'), Wit('hold'); zwv, zwh = w16(wv), w16(wh)
P = ROOT / 'nn_x/w12/cb/famN/preds'
tags = sorted(Path(p).name[:-8] for p in glob.glob(str(P / '*_val.npy')))
print(len(tags), 'runs', flush=True)
res = {}; pf = {}
_, lwv = wv.clogit(zwv); _, lwh = wh.clogit(zwh)
for t in tags:
    xv, xh = np.load(P / f'{t}_val.npy'), np.load(P / f'{t}_hold.npy')
    o = {'tag': t}
    for w, zw, x, k, lw in ((wv, zwv, xv, 'val', lwv), (wh, zwh, xh, 'hold', lwh)):
        zx = w.z(x); b2, l2 = w.clogit(np.column_stack([zw, zx])); d = lw - l2
        e0 = w.ndcg(zw); ex = w.ndcg(zx); dm = w.ndcg(0.9 * zw + 0.1 * zx) - e0
        o[k] = dict(level=float(ex.mean()), rho=float(np.corrcoef(zx, zw)[0, 1]), nats=float(d.mean()), nats_se=se(d), mix=float(dm.mean()), mix_se=se(dm))
        pf[(t, k)] = (ex, l2)
    v, h = o['val'], o['hold']; o['pooled_z'] = float((v['mix'] / v['mix_se'] + h['mix'] / h['mix_se']) / np.sqrt(2))
    o['promising'] = bool(v['nats'] >= 0.0016 and h['nats'] >= 0.0016); o['member_pass'] = bool(v['mix'] >= 0 and h['mix'] >= 0 and o['pooled_z'] >= 2)
    res[t] = o; print(line(o), flush=True)
print('--- ranges over runs: max NATS val', max((r['val']['nats'], t) for t, r in res.items()), 'max NATS hold', max((r['hold']['nats'], t) for t, r in res.items()),
      'pooled z min/max', min((r['pooled_z'], t) for t, r in res.items()), max((r['pooled_z'], t) for t, r in res.items()), 'any promising', any(r['promising'] for r in res.values()),
      'any member pass', any(r['member_pass'] for r in res.values()))
REF = dict(Y='Yref', R='Rref', L='Lref')
arms = {}
for t in tags:
    arm, nk, s = t.rsplit('_', 2); arms.setdefault((arm, nk), []).append(int(s[1:]))
print('--- paired arm - same-loss reference (seed-paired, per-feed differences averaged over seeds)')
cmp = {}
def paired(arm, ref, nk, seeds, name=None):
    o = {}
    for k in ('val', 'hold'):
        dl = np.mean([pf[(f'{arm}_{nk}_s{s}', k)][0] - pf[(f'{ref}_{nk}_s{s}', k)][0] for s in seeds], 0)
        dn = np.mean([pf[(f'{ref}_{nk}_s{s}', k)][1] - pf[(f'{arm}_{nk}_s{s}', k)][1] for s in seeds], 0)
        o[k] = dict(dlevel=float(dl.mean()), dlevel_se=se(dl), dnats=float(dn.mean()), dnats_se=se(dn))
    v, h = o['val'], o['hold']; o['winner'] = bool(v['dnats'] >= 2 * v['dnats_se'] and h['dnats'] >= 2 * h['dnats_se'])
    o['winner_either'] = bool(v['dnats'] >= 2 * v['dnats_se'] or h['dnats'] >= 2 * h['dnats_se'])
    o['level_2se_both'] = bool(v['dlevel'] >= 2 * v['dlevel_se'] and h['dlevel'] >= 2 * h['dlevel_se'])
    nm = name or f'{arm}_{nk} - {ref} seeds {seeds}'; cmp[nm] = o
    print(f"{nm}: d level {v['dlevel']:+.5f}+-{v['dlevel_se']:.5f} / {h['dlevel']:+.5f}+-{h['dlevel_se']:.5f} | d NATS {v['dnats']:+.5f}+-{v['dnats_se']:.5f} / {h['dnats']:+.5f}+-{h['dnats_se']:.5f}"
          f" | winner {o['winner']} (either witness {o['winner_either']}) | level >= 2 SE on both {o['level_2se_both']}", flush=True)
for (arm, nk), seeds in sorted(arms.items()):
    if arm in REF.values(): continue
    ref = 'Yref' if arm == 'Yref_nt' else REF[arm[0]]
    paired(arm, ref, nk, sorted(seeds))
    if len(seeds) > 1:
        for s in sorted(seeds): paired(arm, ref, nk, [s], f'   single seed {arm}_{nk}_s{s}')
paired('Rref', 'Yref', 'n30k', [0, 1]); paired('Lref', 'Yref', 'n30k', [0, 1]); paired('L_lda_all', 'Yref', 'n30k', [0, 1])
# seed noise of the reference: s1 - s0
for a_, nk in (('Yref', 'n30k'), ('Rref', 'n30k'), ('Lref', 'n30k'), ('Yref', 'n6k'), ('Rref', 'n6k')):
    o = {}
    for k in ('val', 'hold'):
        dl = pf[(f'{a_}_{nk}_s1', k)][0] - pf[(f'{a_}_{nk}_s0', k)][0]; dn = pf[(f'{a_}_{nk}_s0', k)][1] - pf[(f'{a_}_{nk}_s1', k)][1]
        o[k] = (float(dl.mean()), se(dl), float(dn.mean()), se(dn))
    print(f"seed noise {a_}_{nk} s1 - s0: d level {o['val'][0]:+.5f}+-{o['val'][1]:.5f} / {o['hold'][0]:+.5f}+-{o['hold'][1]:.5f} | d NATS {o['val'][2]:+.5f}+-{o['val'][3]:.5f} / {o['hold'][2]:+.5f}+-{o['hold'][3]:.5f}")
# arm means
print('--- arm means (level, NATS)')
for (arm, nk), seeds in sorted(arms.items()):
    r = [res[f'{arm}_{nk}_s{s}'] for s in sorted(seeds)]
    print(f"{arm}_{nk} {sorted(seeds)}: level {np.mean([x['val']['level'] for x in r]):.5f} / {np.mean([x['hold']['level'] for x in r]):.5f} NATS {np.mean([x['val']['nats'] for x in r]):+.5f} / {np.mean([x['hold']['nats'] for x in r]):+.5f}")
# bundle of the two Yref seeds
o = read(wv, wh, zwv, zwh, [np.load(P / 'Yref_n30k_s0_val.npy'), np.load(P / 'Yref_n30k_s1_val.npy')], [np.load(P / 'Yref_n30k_s0_hold.npy'), np.load(P / 'Yref_n30k_s1_hold.npy')], 'bundle Yref n30k 2 seeds'); print(line(o))
json.dump(dict(runs=res, paired=cmp), open(HERE / 'out/famN_own.json', 'w'), indent=1)
