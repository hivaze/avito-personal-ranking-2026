"""witness: label-free parity of the STALE-VAL inputs against the test rows (public / private) and the lagged-holdout views.
Per input column: zero share and quantiles of the log1p-scaled value on warm rows; per feed: cold share (it_hist_all_luo == 0).
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python sv_parity.py
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import polars as pl
import stale_val_cpu as SVC
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
BOUND = 1790497920                                   # src/pmix.py: public = test feeds with timestamp < BOUND
ft = pl.read_parquet(FLAT / 'test/feeds.parquet'); Mt = ft['M'].to_numpy().astype(np.int64); pubf = ft['timestamp'].to_numpy() < BOUND
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); Mv = fv['M'].to_numpy().astype(np.int64)
rt = np.repeat(pubf, Mt); fidt = np.repeat(np.arange(len(Mt)), Mt); fidv = np.repeat(np.arange(len(Mv)), Mv)
Ms = np.load(SUB / 'M.npy').astype(np.int64); fids = np.repeat(np.arange(len(Ms)), Ms)
P(f'=== label-free parity of the stale-val inputs, {now()} ===')
P(f'test feeds: public {int(pubf.sum())}, private {int((~pubf).sum())}; val feeds {len(Mv)}; lagged subset feeds {len(Ms)}')
cols = ['it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'seq_cv_invd20', 'seq_cv_norm', 'seq_cv_pop', 'pre2_fe']
def desc(x):
    x = np.nan_to_num(np.asarray(x, np.float64), nan=0.0); w = x[x != 0]
    q = np.quantile(np.log1p(np.maximum(w, 0)), [0.5, 0.9]) if len(w) else [np.nan, np.nan]
    return f'{(x == 0).mean():.3f} / {q[0]:.2f} / {q[1]:.2f}'
P('\ncolumn | zero share / median / p90 of log1p(value) on non-zero rows: val as stored | STALE-VAL override | test public | test private | lagged subset lag0 | lagged subset private-like')
for c in cols:
    v = np.load(FC / 'val' / f'{c}.npy', mmap_mode='r'); o = np.load(SVC.OVR[c], mmap_mode='r'); t = np.load(FC / 'test' / f'{c}.npy', mmap_mode='r')
    row = [desc(v), desc(o), desc(np.asarray(t)[rt]), desc(np.asarray(t)[~rt])]
    for view in ('lag0', 'priv'):
        f = SUB / f'xt_{view}_{c}.npy'
        row.append(desc(np.load(f)) if f.exists() else 'n/a')
    P(f'{c:18s} | ' + ' | '.join(row))
cs = lambda x, fid, n: np.bincount(fid, (np.nan_to_num(np.asarray(x, np.float64)) == 0).astype(float), n) / np.bincount(fid, minlength=n)
a = 'it_hist_all_luo'
cv, co = cs(np.load(FC / 'val' / f'{a}.npy'), fidv, len(Mv)), cs(np.load(SVC.OVR[a]), fidv, len(Mv))
ct = cs(np.load(FC / 'test' / f'{a}.npy'), fidt, len(Mt))
c0, c1 = cs(np.load(SUB / f'xt_lag0_{a}.npy'), fids, len(Ms)), cs(np.load(SUB / f'xt_priv_{a}.npy'), fids, len(Ms))
P('\nfeed cold share (share of candidates with it_hist_all_luo == 0), mean over feeds [M>=50 feeds | M<50 feeds]:')
for name, c_, M_ in (('val as stored', cv, Mv), ('STALE-VAL', co, Mv), ('test public', ct[pubf], Mt[pubf]), ('test private', ct[~pubf], Mt[~pubf]),
                     ('lagged subset lag0', c0, Ms), ('lagged subset private-like', c1, Ms)):
    big = M_ >= 50
    P(f'   {name:28s} {c_.mean():.3f} [{c_[big].mean():.3f} | {c_[~big].mean() if (~big).any() else float("nan"):.3f}]  (share of feeds with M<50: {(~big).mean():.3f})')
(LOGS / 'sv_parity.txt').write_text('\n'.join(OUT) + '\n')
