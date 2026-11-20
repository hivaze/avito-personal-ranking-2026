"""hz2 ADDENDUM 4: model-free conditional check = within-feed AUC of single popularity inputs under each view (train labels; val labels for the stored val columns only)."""
from hz2_common import *
import numba as nb
nb.set_num_threads(2)
L_ = []
def P(s):
    print(s, flush=True); L_.append(s)

@nb.njit(parallel=True, cache=True)
def feed_auc(off, M, x, y, use, out):
    for q in nb.prange(len(M)):
        o = off[q]; m = M[q]
        npos = 0; nneg = 0
        for i in range(m):
            if use[o + i]:
                if y[o + i] > 0:
                    npos += 1
                else:
                    nneg += 1
        if npos == 0 or nneg == 0:
            out[q] = np.nan
            continue
        s = 0.0
        for i in range(m):
            if use[o + i] and y[o + i] > 0:
                xi = x[o + i]
                for j in range(m):
                    if use[o + j] and y[o + j] == 0:
                        xj = x[o + j]
                        if xi > xj:
                            s += 1.0
                        elif xi == xj:
                            s += 0.5
        out[q] = s / (npos * nneg)

def rows_of(off, m):
    return np.repeat(off - np.concatenate([[0], np.cumsum(m)[:-1]]), m) + np.arange(int(m.sum()))

def stat(a):
    a = a[~np.isnan(a)]
    return f'{a.mean():.4f} +- {a.std(ddof=1) / np.sqrt(len(a)):.4f} (n {len(a)})'

kn = knots()
ftr = feeds('train'); nF = len(ftr); M = ftr['M'].to_numpy().astype(np.int64); coff = ftr['cand_off'].to_numpy().astype(np.int64)
hold, p = holdout_feeds(ftr); pool = (p > 0.5) & (M >= 50); pool[hold] = False
part_pub = np.load(OUT / 'train' / 'hz2_part_pub.npy'); d = day_of(ftr['timestamp'].to_numpy()) - DAY0
COLS = ['it_hist_all_luo', 'seq_cv_invd20', 'xp_imp_c50']
def vpath(view, col):
    if view == 'hz2': return HERE / 'feats/train' / f'{HZ2_NAME[col]}.npy'
    if view == 'hz2L': return HERE / 'feats_L/train' / f'{HZ2_NAME[col]}.npy'
    if view == 'hz' and col in LUO: return FNEW / 'train' / f'{HZ2_NAME[col]}.npy'
    return FC / 'train' / f'{col}.npy'
for sub, dm in (('days >= Sep 13', d >= 7), ('Sep 17-19', d >= 11), ('Sep 13-14', (d >= 7) & (d <= 9))):
    sel = np.flatnonzero(pool & ~part_pub & dm)
    crow = rows_of(coff[sel], M[sel]); off = np.concatenate([[0], np.cumsum(M[sel])[:-1]]).astype(np.int64)
    y = np.asarray(np.load(FLAT / 'train/cand_target.npy', mmap_mode='r')[crow]).astype(np.int8)
    old = age_bucket(age_hours(np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[crow]), kn)) >= 5
    allr = np.ones(len(crow), np.bool_)
    P(f'\ntrPool private-like, {sub}: {len(sel)} feeds, {len(crow)} rows (TRAIN labels)')
    for col in COLS:
        for view in ('asbuilt', 'hz', 'hz2', 'hz2L'):
            if view == 'hz' and col not in LUO:
                continue
            x = np.asarray(np.load(vpath(view, col), mmap_mode='r')[crow]).astype(np.float32)
            a = np.empty(len(sel)); b = np.empty(len(sel))
            feed_auc(off, M[sel], x, y, allr, a); feed_auc(off, M[sel], x, y, old, b)
            P(f'  {col:16s} {view:8s} within-feed AUC all rows {stat(a)} | listings older than 24 d {stat(b)}')
fv = feeds('val'); Mv = fv['M'].to_numpy().astype(np.int64); sel = np.flatnonzero(Mv >= 50)
crow = rows_of(fv['cand_off'].to_numpy().astype(np.int64)[sel], Mv[sel]); off = np.concatenate([[0], np.cumsum(Mv[sel])[:-1]]).astype(np.int64)
y = np.load(FLAT / 'val/cand_target.npy')[crow].astype(np.int8)
old = age_bucket(age_hours(np.load(FLAT / 'val/cand_f4.npy')[crow], kn)) >= 5
P(f'\nval feeds with M >= 50: {len(sel)} feeds, {len(crow)} rows (VAL labels, stored val columns: lag 2-4 d, depth 12 d)')
for col in COLS:
    x = np.load(FC / 'val' / f'{col}.npy')[crow].astype(np.float32)
    a = np.empty(len(sel)); b = np.empty(len(sel))
    feed_auc(off, Mv[sel], x, y, np.ones(len(crow), np.bool_), a); feed_auc(off, Mv[sel], x, y, old, b)
    P(f'  {col:16s} stored   within-feed AUC all rows {stat(a)} | listings older than 24 d {stat(b)}')
open(LOGS / 'cond.txt', 'w').write('\n'.join(L_) + '\n')
