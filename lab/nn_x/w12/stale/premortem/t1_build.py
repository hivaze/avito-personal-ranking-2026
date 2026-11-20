"""premortem t1_build: pools, candidate compositions, per-feed NDCG arrays (val FINAL / FT / TO / stale TO / SVF, holdout TO), test FINAL scores.
Outputs: cache/pool_<kind>_<pool>.npy, out/pf.npz (per-feed NDCG, one key per candidate x view), cache/testfin_<cand>.npy, logs/t1_build.txt.
Usage: t1_build.py            (CPU, 12 threads)"""
import sys, json, time
from pm_common import *
from metrics import ndcg_per_feed

t0 = time.time()
m = Meta()
L = open(LOGS / 't1_build.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.write(s + '\n'); L.flush()

# ---- check 1: my NDCG against src/metrics on a tied random score
rng = np.random.default_rng(0); s = np.round(rng.normal(size=m.n['val']), 1).astype(np.float32)
a = m.ndcg('val', s); b = ndcg_per_feed(s, m.y['val'], m.off['val'], m.M['val'])
log('check NDCG vs src/metrics (tied random score): max abs diff', float(np.abs(a - b).max()))
assert np.abs(a - b).max() < 1e-9

KINDS = ('val_ft', 'val_to', 'sv', 'hold', 'test_ft', 'test_to')
POOLS = ('W8', 'W8b', 'W16', 'N8', 'S16', 'Pd', 'Pw', 'Px8')
avail = {}
B = {}
for pn in POOLS:
    for kd in KINDS:
        names = [v for v in G[pn] if path(v, kd).exists()]
        avail[f'{pn}|{kd}'] = len(names)
        if len(names) != len(G[pn]): log(f'NOTE pool {pn} kind {kd}: {len(names)} of {len(G[pn])} members present: missing {[v for v in G[pn] if v not in names]}')
        B[(pn, kd)] = pool(m, names, kd, keep=(kd in ('val_ft', 'val_to', 'sv')))
    log(f'pool {pn} done {time.time() - t0:.0f}s')
# G3 / C3: FT and TO branches of the SCORED files from the stored ens files (G3 = one-seed FT n2cpv3; C3 = Q2a8s3); stale / holdout from member pools
for tag, pn in (('G3', 'G3'), ('Q2a8s3', 'C3')):
    for kd, f in (('val_ft', 'ft_val'), ('val_to', 'to_val'), ('test_ft', 'ft_test'), ('test_to', 'to_test')):
        B[(pn, kd)] = np.load(P / f'ens_{tag}_{f}.npy').astype(np.float32)
    for kd in ('sv', 'hold'):
        B[(pn, kd)] = pool(m, G[pn], kd, keep=False)
    # the stored TO branch is the member-equal train-only pool
    c = np.corrcoef(m.z('val', B[(pn, 'val_to')]), m.z('val', pool(m, G[pn], 'val_to', keep=False)))[0, 1]
    log(f'check {pn}: corr(stored ens TO val branch, member-equal TO pool of mine) = {c:.6f}')
# reproduction of the stored branches of the scored student files
for tag, pn in (('W8kd', 'W8'), ('W16kd', 'W16'), ('S16kd', 'S16')):
    for kd, f in (('val_ft', 'ft_val'), ('val_to', 'to_val'), ('test_ft', 'ft_test'), ('test_to', 'to_test')):
        c = np.corrcoef(B[(pn, kd)], np.load(P / f'ens_{tag}_{f}.npy'))[0, 1]
        assert c > 0.999999, (tag, kd, c)
    log(f'check {tag}: my four branch pools reproduce preds/ens_{tag}_* (corr > 0.999999)')
for (pn, kd), x in B.items():
    if pn in POOLS: np.save(CACHE / f'pool_{kd}_{pn}.npy', x)

Z = lambda kd, x: m.z(SPLIT[kd], x)
def branch(base, pop, w, kd):
    if pop is None: return B[(base, kd)]
    return (1 - w) * Z(kd, B[(base, kd)]) + w * Z(kd, B[(pop, kd)])

C = {'W8': ('W8', None, 0.0), 'W16': ('W16', None, 0.0), 'S16': ('S16', None, 0.0), 'W8b': ('W8b', None, 0.0), 'C3': ('C3', None, 0.0), 'G3': ('G3', None, 0.0),
     'Pd': ('Pd', None, 0.0), 'Pw': ('Pw', None, 0.0), 'Px8': ('Px8', None, 0.0)}
for w in (0.1, 0.2, 0.3, 0.4):
    k = int(round(w * 100))
    C[f'W8P{k}'] = ('W8', 'Pd', w); C[f'W8P{k}x8'] = ('W8', 'Px8', w); C[f'W16P{k}d'] = ('W16', 'Pd', w); C[f'W16P{k}x8'] = ('W16', 'Px8', w); C[f'W16P{k}w'] = ('W16', 'Pw', w)
R = {}
for name, (base, pop, w) in C.items():
    ft, to, sv, ho = (branch(base, pop, w, kd) for kd in ('val_ft', 'val_to', 'sv', 'hold'))
    zft, zto, zsv = m.z('val', ft), m.z('val', to), m.z('val', sv)
    R[f'{name}|fin'] = m.ndcg('val', 0.5 * zft + 0.5 * zto)
    R[f'{name}|ft'] = m.ndcg('val', zft); R[f'{name}|to'] = m.ndcg('val', zto); R[f'{name}|sv'] = m.ndcg('val', zsv)
    R[f'{name}|svf'] = m.ndcg('val', 0.5 * zft + 0.5 * zsv)
    R[f'{name}|hold'] = m.ndcg('hold', m.z('hold', ho))
    tf = 0.5 * m.z('test', branch(base, pop, w, 'test_ft')) + 0.5 * m.z('test', branch(base, pop, w, 'test_to'))
    np.save(CACHE / f'testfin_{name}.npy', tf.astype(np.float32))
    np.save(CACHE / f'valfin_{name}.npy', (0.5 * zft + 0.5 * zto).astype(np.float32))
    if name in ('W8', 'W16'):
        R[f'{name}TO|fin'] = R[f'{name}|to']; R[f'{name}TO|ft'] = R[f'{name}|to']; R[f'{name}TO|to'] = R[f'{name}|to']; R[f'{name}TO|sv'] = R[f'{name}|sv']
        R[f'{name}TO|svf'] = R[f'{name}|sv']; R[f'{name}TO|hold'] = R[f'{name}|hold']
        np.save(CACHE / f'testfin_{name}TO.npy', m.z('test', branch(base, pop, w, 'test_to')).astype(np.float32))
        np.save(CACHE / f'valfin_{name}TO.npy', zto.astype(np.float32))
    log(f'{name:10s} val FINAL {R[f"{name}|fin"].mean():.5f} FT {R[f"{name}|ft"].mean():.5f} TO {R[f"{name}|to"].mean():.5f} staleTO {R[f"{name}|sv"].mean():.5f} SVF {R[f"{name}|svf"].mean():.5f} hold {R[f"{name}|hold"].mean():.5f}')
np.savez(OUT / 'pf.npz', **R)
json.dump(avail, open(OUT / 'avail.json', 'w'), indent=1)
# checks against numbers on record
d = R['W8P20|fin'] - R['W8|fin']; log(f'check W8P20 - W8 FINAL val {d.mean():+.5f} +- {se(d):.5f} (on record +0.00049 +- 0.00031); W8P20 val {R["W8P20|fin"].mean():.5f} (0.40214)')
d = R['W16|fin'] - R['W8|fin']; log(f'check W16 - W8 FINAL val {d.mean():+.5f} +- {se(d):.5f} (on record -0.00007 +- 0.00034); W16 val {R["W16|fin"].mean():.5f} (0.40158)')
d = R['W8P20|hold'] - R['W8|hold']; log(f'check W8P20 - W8 holdout TO {d.mean():+.5f} +- {se(d):.5f} (on record -0.00063 +- 0.00022)')
d = R['W8P20|sv'] - R['W8|sv']; log(f'check W8P20 - W8 stale-val TO {d.mean():+.5f} +- {se(d):.5f} (witness +0.00088 +- 0.00036)')
log(f'done {time.time() - t0:.0f}s')
