"""Build the z cache for every member (sel and @last where available) on val / holdout / test. Verifies fast NDCG vs src/metrics."""
from pc_common import *
import time
m = Meta(); G = groups()
names = []
for g, ms in G.items():
    for v in ms:
        names.append(v)
        if v.startswith('w5_e10_') or v.startswith('abl_e10_pop1kd'): names.append(v + '@last')
cap = len(names) + 40
idx = {'cap': cap, 'members': {}, 'missing': {}}
mm = {sp: np.memmap(CACHE/f'z_{sp}.f32', np.float32, 'w+', shape=(cap, m.n[sp])) for sp in ('val', 'hold', 'test')}
t0 = time.time(); k = 0
for nme in names:
    v, suf = (nme[:-5], '@last') if nme.endswith('@last') else (nme, '')
    pv, ph, pt = paths(v, suf)
    miss = [str(p) for p in (pv, ph, pt) if not p.exists()]
    if not pv.exists() or not ph.exists():
        idx['missing'][nme] = miss; print('MISSING', nme, miss, flush=True); continue
    mm['val'][k] = m.z('val', np.load(pv)); mm['hold'][k] = m.z('hold', np.load(ph))
    if pt.exists(): mm['test'][k] = m.z('test', np.load(pt))
    else: mm['test'][k] = np.nan; idx['missing'][nme] = miss
    idx['members'][nme] = k; k += 1
    if k % 20 == 0: print(k, nme, f'{time.time()-t0:.0f}s', flush=True)
for sp in mm: mm[sp].flush()
json.dump(idx, open(CACHE/'index.json', 'w'), indent=1)
print('cached', k, 'members; missing', idx['missing'])
# parity of the fast NDCG
for nme in ('w5_e10_kd05w192_s951', 'w5_e10_base_s952'):
    i = idx['members'][nme]
    a = ndcg_per_feed(np.asarray(mm['val'][i]), m.yv, m.ov, m.Mv); b = ndcg_fast(mm['val'][i], m.yv, m.ov, m.Mv)
    c = ndcg_per_feed(np.asarray(mm['hold'][i]), m.yh, m.oh, m.Mh); d = ndcg_fast(mm['hold'][i], m.yh, m.oh, m.Mh)
    raw = ndcg_per_feed(np.load(paths(nme)[0]).astype(np.float32), m.yv, m.ov, m.Mv)
    print(nme, 'val', a.mean(), b.mean(), 'max|diff|', np.abs(a-b).max(), 'raw-vs-z', np.abs(raw-a).max(), '| hold', c.mean(), d.mean(), np.abs(c-d).max())
