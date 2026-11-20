"""hz2_review: the reviewer's own NDCG read of the study's cached holdout predictions (REVIEW_RULES.txt R4). Own metric code, own pooling code,
labels = data/flat/train/cand_target.npy at the holdout rows (TRAIN-holdout labels only; no val / test label). No import from the study or from src/.
Usage: CUDA_VISIBLE_DEVICES="" python read_own.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
import sys
sys.dont_write_bytecode = True
import numpy as np
import polars as pl

R = '<repo>'
C = f'{R}/nn_x/w12/stale/hz2/cache'; PL = f'{R}/nn_x/w12/next/poplag/cache'
MSK, DAY, DAY0 = 3 * 3600, 86400, 20702
ftr = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet')
Mtr = ftr['M'].to_numpy().astype(np.int64); otr = ftr['cand_off'].to_numpy().astype(np.int64); tstr = ftr['timestamp'].to_numpy().astype(np.int64)
# own holdout (rule of nn_x/w5/train.py --holdout 30000)
pr = np.load(f'{R}/eda/labels_position/out/train_prandom.npz')
p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
cand = np.where((p > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
mh = np.load(f'{PL}/mini/hold.npy'); crow = np.load(f'{PL}/mini/crow.npy')
print('holdout: my 30,000 feeds == poplag mini hold.npy:', bool(np.array_equal(hold, mh)))
M = Mtr[hold]; off = np.concatenate([[0], np.cumsum(M)[:-1]])
rows = np.concatenate([np.arange(otr[f], otr[f] + Mtr[f]) for f in hold])
print('rows == mini crow:', bool(np.array_equal(rows, crow)), '| rows', len(rows))
y = np.asarray(np.load(f'{R}/data/flat/train/cand_target.npy', mmap_mode='r')[rows]).astype(np.float64)
d = (tstr[hold] + MSK) // DAY - DAY0
late = d >= 7
fid = np.repeat(np.arange(len(M)), M)
late_row = late[fid]
print('holdout feeds of days >= Sep 13:', int(late.sum()), 'rows', int(late_row.sum()), '| Sep 13-14', int(((d >= 7) & (d <= 8)).sum()), '| Sep 17-19', int((d >= 11).sum()))
LAG = np.load(f'{R}/nn_x/w12/stale/hz2/feats/train/hz2_lag.npy')[hold]; PP = np.load(f'{R}/nn_x/w12/stale/hz2/feats/train/hz2_part_pub.npy')[hold]
print('late feeds public-like', int((PP & late).sum()), 'private-like', int((~PP & late).sum()))
IDX = np.flatnonzero(late)


def ndcg(s):
    """Per-feed full-list NDCG, binary gain, pessimistic ties (a tied positive is ranked below every tied negative). Feeds without a positive -> 1."""
    out = np.ones(len(M))
    for f in IDX:
        a = off[f]; b = a + M[f]
        yy = y[a:b]; ss = np.asarray(s[a:b], np.float64)
        npos = int(yy.sum())
        if npos == 0:
            continue
        # pessimistic rank of each positive: 1 + number of items with a strictly higher score + number of NEGATIVES with an equal score + tied positives before it
        sp = np.sort(ss[yy > 0])[::-1]
        neg = np.sort(ss[yy == 0])
        higher_neg = len(neg) - np.searchsorted(neg, sp, side='right')          # negatives strictly above
        tie_neg = np.searchsorted(neg, sp, side='right') - np.searchsorted(neg, sp, side='left')
        rank = higher_neg + tie_neg + np.arange(npos) + 1                         # positives sorted descending occupy consecutive places after those
        dcg = (1.0 / np.log2(rank + 1.0)).sum()
        idcg = (1.0 / np.log2(np.arange(npos) + 2.0)).sum()
        out[f] = dcg / idcg
    return out


def z(x):
    x = np.nan_to_num(np.asarray(x, np.float64), nan=0.0)
    mu = np.bincount(fid, weights=x) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x * x) / M - mu * mu, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


def f(dv, m=late):
    dv = np.asarray(dv)[m]
    return f'{dv.mean():+.5f} +- {dv.std(ddof=1) / np.sqrt(len(dv)):.5f} (n {len(dv)})'


W8 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in (951, 952, 953, 954)]
PK = [f'abl_e10_pop1kd05t25_s{s}' for s in (951, 952, 953, 954)]
ld = lambda view, v: np.load(f'{PL}/pred_lag0_{v}.npy') if view == 'lag0' else np.load(f'{PL}/pred_priv_v3_{v}.npy') if view == 'v3' else np.load(f'{C}/hpred_{view}_late_{v}.npy')
x = ld('feats', W8[0])
print('NaN rows of a view prediction file == rows of feeds before Sep 13:', bool(np.array_equal(np.isnan(x), ~late_row)))
pool = lambda view, mem: np.mean([z(ld(view, v)) for v in mem], 0)
w0 = pool('lag0', W8); p0 = pool('lag0', PK); e_w0 = ndcg(w0); e_p0 = ndcg(p0)
S = lambda pw, pp, w=0.2: ndcg((1 - w) * z(pw) + w * z(pp)) - ndcg(pw)
s0 = S(w0, p0)
print(f'stored inputs: W8 pool NDCG {e_w0[late].mean():.5f} pop1kd pool {e_p0[late].mean():.5f}; S(0.2) stored {f(s0)}')
g17 = d >= 11; g13 = (d >= 7) & (d <= 8)
for view in ('feats', 'feats_L'):
    wv = pool(view, W8); pv = pool(view, PK); e_w = ndcg(wv); e_p = ndcg(pv); sv = S(wv, pv)
    print(f'[{view}] W8 pool view - stored: {f(e_w - e_w0)} | Sep 17-19 {f(e_w - e_w0, g17)} | Sep 13-14 {f(e_w - e_w0, g13)}')
    print(f'[{view}] pop1kd pool view - stored: {f(e_p - e_p0)} | S(0.2) view {f(sv)} | D {f(sv - s0)}')
    print(f'[{view}] private-like dW {f(e_w - e_w0, late & ~PP)} | public-like dW {f(e_w - e_w0, late & PP)}')
    if view == 'feats':
        EWP = e_w
    else:
        print(f'[E4] W8 pool feats_L - feats, Sep 17-19: {f(e_w - EWP, g17)}')
try:
    w3 = pool('v3', W8); e3 = ndcg(w3)
    print(f'[v3] W8 pool poplag v3 - stored, same feeds: {f(e3 - e_w0)} | Sep 17-19 {f(e3 - e_w0, g17)}')
except Exception as e:
    print('v3 not read:', repr(e)[:120])

# ---------------- HZ11
H = [f'w5_cd_hzcd_r{r}' for r in (101, 102, 103, 104)]; RF = [f'w5_cd_m02ref_r{r}' for r in (101, 102, 103, 104)]
for suf in ('', '@last'):
    for view in ('feats', 'feats_L'):
        hv = np.mean([z(ld(view, v + suf)) for v in H], 0); rv = np.mean([z(ld(view, v + suf)) for v in RF], 0)
        hs = np.mean([z(ld('stored', v + suf)) for v in H], 0); rs = np.mean([z(ld('stored', v + suf)) for v in RF], 0)
        qv = ndcg(hv) - ndcg(rv); qs = ndcg(hs) - ndcg(rs)
        single = [float((ndcg(ld(view, a + suf)) - ndcg(ld(view, b + suf)))[late].mean()) for a, b in zip(H, RF)]
        print(f'[HZ11 {"selected" if not suf else "@last"} {view}] Q_view {f(qv)} | Q_stored {f(qs)} | I {f(qv - qs)} | Sep 17-19 Q_view {f(qv, g17)} | by root ' + ' / '.join(f'{s_:+.5f}' for s_ in single))
        if view == 'feats':
            print(f'    staleness cost reference pool {f(ndcg(rv) - ndcg(rs))} | hz pool {f(ndcg(hv) - ndcg(hs))}')
# ---------------- path control: CPU "stored" replay vs the project's GPU holdout files
HD = f'{R}/nn_x/w5/preds/holdout'
for v in (H[0], RF[0], H[0] + '@last'):
    g = np.load(f'{HD}/nn_{v}_holdout.npy'); cpu = ld('stored', v)
    zz1 = z(np.where(late_row, g, 0.0))[late_row]; zz2 = z(cpu)[late_row]
    print(f'[path control {v}] rows GPU file {len(g)}; corr of z on late rows {np.corrcoef(zz1, zz2)[0, 1]:.6f}; NDCG CPU - GPU {f(ndcg(cpu) - ndcg(g))}')
# in-period reading of the 11-12 hz arm from the GPU files (30,000 feeds): member level @last
IDX = np.arange(len(M))
dm = np.mean([ndcg(np.load(f'{HD}/nn_{a}@last_holdout.npy')) - ndcg(np.load(f'{HD}/nn_{b}@last_holdout.npy')) for a, b in zip(H, RF)], 0)
print(f'[11-12 read-out, GPU files, 30,000 feeds, member level @last] hz - ref {f(dm, np.ones(len(M), bool))}')
