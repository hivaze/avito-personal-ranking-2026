"""hz2_review: (1) path control: study's CPU reader vs the trainer's GPU view output of the same checkpoint (holdout feeds of days >= Sep 13);
(2) label-free extras: lag vs M / p_random / part; share of rows where the view is WARMER than the stored column (depth-factor extrapolation)."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'; HZ2 = f'{R}/nn_x/w12/stale/hz2'; PL = f'{R}/nn_x/w12/next/poplag/cache'; HERE = f'{R}/nn_x/w12/stale/hz2_review'
MSK, DAY, DAY0 = 3 * 3600, 86400, 20702
ftr = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet')
Mtr = ftr['M'].to_numpy().astype(np.int64); otr = ftr['cand_off'].to_numpy().astype(np.int64); tstr = ftr['timestamp'].to_numpy().astype(np.int64)
dtr = (tstr + MSK) // DAY - DAY0
hold = np.load(f'{PL}/mini/hold.npy'); crow = np.load(f'{PL}/mini/crow.npy')
M = Mtr[hold]; off = np.concatenate([[0], np.cumsum(M)[:-1]]); fid = np.repeat(np.arange(len(M)), M)
late = dtr[hold] >= 7; lr = late[fid]
y = np.asarray(np.load(f'{R}/data/flat/train/cand_target.npy', mmap_mode='r')[crow]).astype(np.float64)
def ndcg(s):
    out = np.ones(len(M))
    for f in np.flatnonzero(late):
        a = off[f]; b = a + M[f]; yy = y[a:b]; ss = np.asarray(s[a:b], np.float64); npos = int(yy.sum())
        if npos == 0: continue
        sp = np.sort(ss[yy > 0])[::-1]; neg = np.sort(ss[yy == 0])
        rank = (len(neg) - np.searchsorted(neg, sp, side='left')) + np.arange(npos) + 1
        out[f] = (1 / np.log2(rank + 1.0)).sum() / (1 / np.log2(np.arange(npos) + 2.0)).sum()
    return out
def z(x):
    x = np.nan_to_num(np.asarray(x, np.float64)); mu = np.bincount(fid, weights=x) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x * x) / M - mu * mu, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]
for v in sys.argv[1:]:
    cpu = np.load(f'{HERE}/chk_hpred_feats_late_{v}.npy'); gpu = np.load(f'{R}/nn_x/w5/preds/holdout/nn_{v}_holdout_hz.npy'); gst = np.load(f'{R}/nn_x/w5/preds/holdout/nn_{v}_holdout.npy')
    assert np.array_equal(np.isnan(cpu), ~lr)
    c1 = np.corrcoef(z(cpu)[lr], z(np.where(lr, gpu, 0))[lr])[0, 1]; c0 = np.corrcoef(z(cpu)[lr], z(np.where(lr, gst, 0))[lr])[0, 1]
    dn = (ndcg(cpu) - ndcg(gpu))[late]
    print(f'[{v}] CPU reader (view) vs GPU trainer file _holdout_hz (view, selected ckpt), {int(late.sum())} feeds / {int(lr.sum())} rows: corr of per-feed z {c1:.6f} '
          f'(against the AS-BUILT GPU file of the same checkpoint: {c0:.6f}); NDCG CPU - GPU {dn.mean():+.5f} +- {dn.std(ddof=1) / np.sqrt(len(dn)):.5f}; max |z diff| {np.abs(z(cpu)[lr] - z(np.where(lr, gpu, 0))[lr]).max():.3f}', flush=True)
# ---- label-free extras
LAG = np.load(f'{HZ2}/feats/train/hz2_lag.npy').astype(np.float64); PP = np.load(f'{HZ2}/feats/train/hz2_part_pub.npy')
pr = np.nan_to_num(np.load(f'{R}/eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float64))
print(f'lag vs M corr {np.corrcoef(LAG, Mtr)[0, 1]:+.5f}; lag vs p_random corr {np.corrcoef(LAG, pr)[0, 1]:+.5f}; public-like share by day slot min {min(PP[dtr == s].mean() for s in (0,1,2,3,4,5,6,7,8,11,12,13)):.4f} max {max(PP[dtr == s].mean() for s in (0,1,2,3,4,5,6,7,8,11,12,13)):.4f}; holdout feeds public-like share {PP[hold].mean():.4f}')
rng = np.random.default_rng(9); fs = np.sort(rng.choice(np.flatnonzero(dtr >= 3), 60000, replace=False))
rw = np.repeat(otr[fs] - np.concatenate([[0], np.cumsum(Mtr[fs])[:-1]]), Mtr[fs]) + np.arange(int(Mtr[fs].sum()))
kn = np.load(f'{R}/data/feats/life_f4_age_knots.npy').astype(np.float64)
f4 = np.asarray(np.load(f'{R}/data/flat/train/cand_f4.npy', mmap_mode='r')[rw]).astype(np.float64); u = f4 * 64; k = np.clip(np.floor(u).astype(int), 1, 62); w = u - k
age = np.where(f4 < 63 / 64, (kn[k] * (1 - w) + kn[k + 1] * w) / 24, 1e9); b = np.clip(np.searchsorted(np.array([0, 1, 3, 7, 10, 24, 76, np.inf]), age, side='right') - 1, 0, 6)
d_row = np.repeat(dtr[fs], Mtr[fs])
for col, hzn in (('it_hist_all_luo', 'it_hist_hz_all'), ('seq_cv_pop', 'seq_cv_hz_pop')):
    a = np.asarray(np.load(f'{R}/nn/feat_cache/train/{col}.npy', mmap_mode='r')[rw]).astype(np.float64); h = np.asarray(np.load(f'{HZ2}/feats/train/{hzn}.npy', mmap_mode='r')[rw]).astype(np.float64)
    nonint = np.abs(h - np.round(h)) > 1e-3
    print(f'{col}: 60,000 random train feeds of days >= Sep 9 ({len(rw)} rows): view > stored on {float((h > a + 1e-6).mean()):.4f} of rows (by age bucket ' + ' '.join(f'{float((h > a + 1e-6)[b == q].mean()):.3f}' for q in range(7))
          + f'); non-integer view value on {float(nonint.mean()):.4f} of rows, {float(nonint[h > 0].mean()):.4f} of warm rows (test rows: 0 by construction); by day group Sep 9-12 {float(nonint[(d_row <= 6) & (h > 0)].mean()):.3f} Sep 13-14 {float(nonint[(d_row >= 7) & (d_row <= 8) & (h > 0)].mean()):.3f} Sep 17-19 {float(nonint[(d_row >= 11) & (h > 0)].mean()):.3f}')
