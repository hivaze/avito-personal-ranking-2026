"""CRITIC c02: family T (first-choice / Bradley-Terry click strength on exploration feeds) with the FIT OF RECORD (fit_bt, gather_rows imported
read-only from nn_x/insights/ccb_1113/scratch/tester3/c06_build.py; the data selection below is a line-by-line copy of its main()):
  (R) all-train fit -> val rows against cache/c06_val.npz; the fit on slots < 6 -> holdout rows of slot 6 against cache/c06_hold.npz
  (L) LAG VIEW of DESIGN.md: fits on source feeds of slots < cut' (cut' = 5, 6, 7) -> the 15,225 holdout feeds of slots >= 6
  (T) stale-val twin: fits on slots < slot - L (cut 7..12) -> val rows
  (N) natural test rows: the all-train fit (label-free on the test side: coverage reference)
CPU only, 12 threads, read-only on the project; writes only critic/out/."""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
CR = '<repo>/reports/INPUTS_1116/critic'
os.environ['NUMBA_CACHE_DIR'] = CR + '/out/numba_cache'
sys.dont_write_bytecode = True
import numba as nb
import numpy as np
import polars as pl
sys.path.insert(0, '<repo>/nn_x/insights/ccb_1113/scratch/tester3')
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import c06_build as c6
import hz2_common as hc
from pathlib import Path
FLAT = Path('<repo>/data/flat'); OUT = Path(CR) / 'out'; C6 = Path('<repo>/nn_x/insights/ccb_1113/scratch/tester3/cache')
T0 = time.time(); logf = open(OUT / 'c02_lagbt.log', 'w')
def P(*a):
    s = f'[{time.time() - T0:6.0f}s] ' + ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()
nb.set_num_threads(12)
ft = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['M', 'timestamp', 'cand_off', 'user_id', 'feed_id'])
Mt = ft['M'].to_numpy().astype(np.int64); ot = ft['cand_off'].to_numpy().astype(np.int64)
tday = (ft['timestamp'].to_numpy().astype(np.int64) + 10800) // 86400; tslot = tday - hc.DAY0
praw = np.nan_to_num(np.load('<repo>/eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
ph = praw.copy(); ph[Mt < 50] = 0.0
HOLD = np.sort(np.random.default_rng(12345).choice(np.where((ph > 0.9) & (Mt >= 50))[0], 30000, replace=False))
is_hold = np.zeros(len(Mt), bool); is_hold[HOLD] = True
cand = np.nonzero((praw > 0.5) & ~is_hold)[0]
yt = np.load(FLAT / 'train/cand_target.npy', mmap_mode='r'); it_t = np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')
Mc = Mt[cand]; oc = np.concatenate([[0], np.cumsum(Mc)[:-1]]).astype(np.int64)
y_c = np.empty(int(Mc.sum()), np.int8); i_c = np.empty(int(Mc.sum()), np.int32)
c6.gather_rows(cand, ot, Mt, oc, np.asarray(yt), y_c); c6.gather_rows(cand, ot, Mt, oc, np.asarray(it_t), i_c)
npos_c = np.add.reduceat(y_c.astype(np.int64), oc); keep = npos_c >= 1; src = cand[keep]
Ms = Mt[src]; rowsel = np.repeat(keep, Mc); y_s = y_c[rowsel].astype(np.float64); i_s = i_c[rowsel]; sslot = tslot[src]
P(f'source feeds (p_random > 0.5, not holdout, >= 1 positive): {len(src)}; by slot {dict(zip(*[x.tolist() for x in np.unique(sslot, return_counts=True)]))}')

def fit_subset(fmask, lam=3.0):
    rmask = np.repeat(fmask, Ms); items = i_s[rmask]; uit, inv = np.unique(items, return_inverse=True)
    Msub = Ms[fmask]; osub = np.concatenate([[0], np.cumsum(Msub)[:-1]]).astype(np.int64); yy = y_s[rmask]; npos = np.add.reduceat(yy, osub)
    th, r = c6.fit_bt(inv.astype(np.int64), yy, osub, Msub, npos, len(uit), lam, None)
    P(f'  fit: {int(fmask.sum())} feeds, {int(rmask.sum())} rows, {len(uit)} items, nit {r.nit}, conv {r.success}, theta sd {th.std():.4f}')
    return uit, th
def lookup(fitted, items):
    uit, th = fitted; k = np.minimum(np.searchsorted(uit, items), len(uit) - 1); seen = uit[k] == items
    return np.where(seen, th[k], 0.0), seen.astype(np.float64)
fits = {}
def get(c):
    if c not in fits:
        P(f'fit for cut {c}'); fits[c] = fit_subset(sslot < c if c is not None else np.ones(len(src), bool))
    return fits[c]
res = {}
# natural val / test
iv = np.load(FLAT / 'val/cand_item_id.npy'); ie = np.load(FLAT / 'test/cand_item_id.npy')
t, s = lookup(get(None), iv); res['val_nat_theta'] = t; res['val_nat_seen'] = s
c6v = np.load(C6 / 'c06_val.npz'); P(f'(R) val: theta max abs diff vs c06_val {np.abs(t - c6v["theta_l3"]).max():.2e}, seen mismatches {int((s != c6v["seen_l3"]).sum())}, seen share {s.mean():.4f}')
t, s = lookup(get(None), ie); res['test_nat_theta'] = t; res['test_nat_seen'] = s; P(f'(N) test: seen share {s.mean():.4f}')
# holdout
Mh = Mt[HOLD]; hoff = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64); ih = np.empty(int(Mh.sum()), np.int32)
c6.gather_rows(HOLD, ot, Mt, hoff, np.asarray(it_t), ih)
L = np.load(hc.HERE / 'feats_L/train/hz2_lag.npy').astype(np.int64); hslot = tslot[HOLD]; hcut = np.maximum(hslot - L[HOLD], 5); has = hslot >= 6
rslot = np.repeat(hslot, Mh); rcut = np.repeat(hcut, Mh); rhas = np.repeat(has, Mh)
th_l = np.full(len(ih), np.nan); se_l = np.full(len(ih), np.nan)
for c in (5, 6, 7):
    m = rhas & (rcut == c); t, s = lookup(get(c), ih[m]); th_l[m] = t; se_l[m] = s
    P(f'(L) cut {c}: rows {int(m.sum())}, seen share {s.mean():.4f}')
res['hold_lag_theta'] = th_l; res['hold_lag_seen'] = se_l
c6h = np.load(C6 / 'c06_hold.npz'); m = rslot == 6; t, s = lookup(get(6), ih[m])
P(f'(R) holdout slot 6, fit on slots < 6: theta max abs diff vs c06_hold {np.abs(t - c6h["theta_l3"][m]).max():.2e}, seen mismatches {int((s != c6h["seen_l3"][m]).sum())}')
for sl in (6, 7, 8, 11, 12, 13):
    m = rslot == sl; P(f'(L) holdout slot {sl}: seen share fresh (cut = own day) {c6h["seen_l3"][m].mean():.4f} -> lag view {se_l[m].mean():.4f}')
# stale-val twin
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); Mv = fv['M'].to_numpy().astype(np.int64)
vslot = (fv['timestamp'].to_numpy().astype(np.int64) + 10800) // 86400 - hc.DAY0; vL = np.asarray(hc.assign_lags(fv['feed_id'].to_numpy())[1]).astype(np.int64)
vcut = np.repeat(vslot - vL, Mv); th_t = np.full(len(iv), np.nan); se_t = np.full(len(iv), np.nan)
for c in np.unique(vcut):
    m = vcut == c; t, s = lookup(get(int(c)), iv[m]); th_t[m] = t; se_t[m] = s
    P(f'(T) val cut {c}: rows {int(m.sum())}, seen share {s.mean():.4f}')
res['val_twin_theta'] = th_t; res['val_twin_seen'] = se_t
P(f'(T) val twin: seen share {se_t.mean():.4f} (natural {res["val_nat_seen"].mean():.4f})')
np.savez(OUT / 'c02_bt.npz', **res); P('DONE')
