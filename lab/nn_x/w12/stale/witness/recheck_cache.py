"""witness resume check (2026-11-15T03:35Z): integrity of every cached prediction file + re-read of the control A numbers from the cache.
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python recheck_cache.py
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import lagged_score as LS
import stale_val_cpu as SVC
import wt_read as WR
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
P(f'=== witness cache re-check {now()} ===')
bad = 0
for view in VIEWS:
    fs = sorted((PRED / view).glob('*.npy'))
    for f in fs:
        if f.name.startswith('tmp_'): continue
        x = np.load(f, mmap_mode='r')
        ok = x.shape == (N_SUB_ROWS,) and x.dtype == np.float32 and bool(np.isfinite(x).all()) and float(np.std(x)) > 0.1
        bad += not ok
        if not ok: P(f' BAD {f}: shape {x.shape} dtype {x.dtype}')
    P(f' pred/{view}: {len(fs)} files checked')
wt = WR.Witness(); V = wt.V; nval = int(V.M.sum())
ev = lambda x: ndcg_fast(feed_z(V.off, V.M, x), V.y, V.off, V.M)
rows = []
for f in sorted(SV.glob('nn_*.npy')):
    x = np.load(f)
    ok = x.shape == (nval,) and x.dtype == np.float32 and bool(np.isfinite(x).all()) and float(np.std(x)) > 0.1
    bad += not ok
    v = f.name[3:].replace('_staleval.npy', '').replace('_valbase.npy', '')
    kind = 'stale' if f.name.endswith('_staleval.npy') else 'base'
    g = ev(np.load(stored_val(v))) if stored_val(v).exists() else None
    a = ev(x)
    rows.append((v, kind, float(a.mean()), None if g is None else float(g.mean())))
    P(f' {f.name:48s} {"ok " if ok else "BAD"} NDCG {a.mean():.5f}' + ('' if g is None else f' | stored GPU val {g.mean():.5f} | paired {fmt(a - g)}'))
P(f' files failing the integrity check: {bad}')
# control A numbers again from the cache
L = wt.L
W8, P4 = [(1.0, G['W8'])], [(1.0, G['pop1kd'])]
c = L.compare([(0.8, G['W8']), (0.2, G['pop1kd'])], W8)
a0, a1 = L.level(W8, 'lag0').mean(), L.level(W8, 'priv').mean()
P(f' control A from the cache: W8 pool {a0:.5f} -> {a1:.5f}; S_0 {"%+.5f +- %.5f" % c["d_lag0"]}; S_lag {"%+.5f +- %.5f" % c["d_priv"]}; D {"%+.5f +- %.5f" % c["D_priv"]}')
ok = abs(a0 - 0.32334) <= 2e-5 and abs(a1 - 0.32002) <= 2e-5 and abs(c['d_priv'][0] - 0.00024) <= 2e-5 and bad == 0
P(f' RE-CHECK: {"VALID" if ok else "NOT VALID"}')
(LOGS / 'recheck_cache_0335Z.txt').write_text('\n'.join(OUT) + '\n')
