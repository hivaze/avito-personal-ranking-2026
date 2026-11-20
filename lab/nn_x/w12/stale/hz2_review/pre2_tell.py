"""hz2_review R3c / R3e: the joint of the swapped counters with the UNSWAPPED encoder scores (pre2_*). Label-free part + train-label part (non-holdout trPool feeds only)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
import sys, time
sys.dont_write_bytecode = True
import numpy as np
import polars as pl
R = '<repo>'; FLAT = f'{R}/data/flat'; FC = f'{R}/nn/feat_cache'; HZ2 = f'{R}/nn_x/w12/stale/hz2'
MSK, DAY, DAY0 = 3 * 3600, 86400, 20702
T0 = time.time()
def lg(*a): print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)
def rows_of(off, M): return np.repeat(off - np.concatenate([[0], np.cumsum(M)[:-1]]), M) + np.arange(int(M.sum()))
nz = lambda x: np.nan_to_num(np.asarray(x, np.float32), nan=0.0)
# ---- exact counts on val / test rows
for s in ('val', 'test'):
    ih = np.load(f'{FC}/{s}/it_hist_all_luo.npy'); idc = nz(np.load(f'{FC}/{s}/pre2_id_contrib.npy'))
    lg(f'{s}: rows {len(ih)}; cold (it_hist_all == 0) {int((ih == 0).sum())}; cold AND pre2_id_contrib != 0: {int(((ih == 0) & (idc != 0)).sum())}; warm AND id_contrib == 0: {int(((ih > 0) & (idc == 0)).sum())} ({float(((ih > 0) & (idc == 0)).mean()):.4f} of rows)')
ftr = pl.read_parquet(f'{FLAT}/train/feeds.parquet')
Mtr = ftr['M'].to_numpy().astype(np.int64); otr = ftr['cand_off'].to_numpy().astype(np.int64)
dtr = (ftr['timestamp'].to_numpy().astype(np.int64) + MSK) // DAY - DAY0
pr = np.load(f'{R}/eda/labels_position/out/train_prandom.npz')
p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
cand = np.where((p > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
pool = (p > 0.5) & (Mtr >= 50); pool[hold] = False
PP = np.load(f'{HZ2}/feats/train/hz2_part_pub.npy')
mm = lambda path: np.load(path, mmap_mode='r')
def pop(fs, tag):
    rw = rows_of(otr[fs], Mtr[fs])
    asb = np.asarray(mm(f'{FC}/train/it_hist_all_luo.npy')[rw]); h2 = np.asarray(mm(f'{HZ2}/feats/train/it_hist_hz_all.npy')[rw]); hz = np.asarray(mm(f'{R}/data/feats_new/train/it_hist_hz_all.npy')[rw])
    keep = np.asarray(mm(f'{HZ2}/feats/train/hz2_keep_rows.npy')[rw]); h2k = np.where(keep, asb, h2)
    idc = nz(mm(f'{FC}/train/pre2_id_contrib.npy')[rw]) != 0
    has = ~np.isnan(np.asarray(mm(f'{FC}/train/pre2_fe.npy')[rw]))
    out = {}
    for vn, v in (('as built', asb), ('hz half dose', hz), ('hz2', h2), ('hz2k (arm as trained)', h2k)):
        c = v == 0
        out[vn] = (float((c & idc).mean()), float(idc[c].mean()), float(c.mean()))
    lg(f'{tag}: feeds {len(fs)} rows {len(rw)}; has:pre2_fe {float(has.mean()):.3f}; id_contrib != 0 {float(idc.mean()):.3f} | share of ALL rows cold AND id_contrib != 0 / P(id != 0 | cold) / cold share: '
       + ' | '.join(f'{k}: {a:.4f} / {b:.4f} / {c_:.4f}' for k, (a, b, c_) in out.items()))
    return rw, asb, h2, idc
rng = np.random.default_rng(5)
allf = np.sort(rng.choice(len(Mtr), 150000, replace=False))
pop(allf, 'ALL train feeds (150,000 random)')
pop(allf[dtr[allf] >= 7], 'ALL train feeds, days >= Sep 13')
pop(allf[(dtr[allf] >= 3) & (dtr[allf] <= 6)], 'ALL train feeds, Sep 9-12')
pop(np.flatnonzero(pool), 'trPool all days')
fs = np.flatnonzero(pool & ~PP & (dtr >= 7))
rw, asb, h2, idc = pop(fs, 'trPool private-like, days >= Sep 13')
# ---- model-input level on view-cold rows (label-free)
fes = nz(mm(f'{FC}/train/pre2_fe_s.npy')[rw]); idv = nz(mm(f'{FC}/train/pre2_id_contrib.npy')[rw]); fe = nz(mm(f'{FC}/train/pre2_fe.npy')[rw])
cold = h2 == 0
fte = pl.read_parquet(f'{FLAT}/test/feeds.parquet'); tts = fte['timestamp'].to_numpy().astype(np.int64); Mte = fte['M'].to_numpy().astype(np.int64); ote = fte['cand_off'].to_numpy().astype(np.int64)
tf_ = np.flatnonzero((tts >= 1790497920) & (Mte >= 50)); trw = rows_of(ote[tf_], Mte[tf_])
t_ih = np.load(f'{FC}/test/it_hist_all_luo.npy')[trw]; t_fes = nz(np.load(f'{FC}/test/pre2_fe_s.npy')[trw]); t_fe = nz(np.load(f'{FC}/test/pre2_fe.npy')[trw]); t_id = nz(np.load(f'{FC}/test/pre2_id_contrib.npy')[trw])
sd = float(t_fes.std())
lg(f'pre2_fe_s (model input), SD on private rows {sd:.4f}. Mean on COLD rows: private {t_fes[t_ih == 0].mean():.4f} | hz2-cold & id_contrib == 0 {fes[cold & ~idc].mean():.4f} | hz2-cold & id_contrib != 0 {fes[cold & idc].mean():.4f} '
   f'(gap to private cold rows {(fes[cold & idc].mean() - t_fes[t_ih == 0].mean()) / sd:+.3f} SD) | all hz2-cold {fes[cold].mean():.4f}')
lg(f'pre2_id_contrib on hz2-cold rows with id != 0: mean {idv[cold & idc].mean():+.4f} sd {idv[cold & idc].std():.4f} mean |.| {np.abs(idv[cold & idc]).mean():.4f} | SD of pre2_fe on private rows {t_fe.std():.4f} | private warm rows: mean id_contrib {t_id[t_ih > 0].mean():+.4f} sd {t_id[t_ih > 0].std():.4f}')
lg(f'stored it_hist_all_luo on the hz2-cold rows with id_contrib != 0: median {np.median(asb[cold & idc]):.0f}, q90 {np.quantile(asb[cold & idc], .9):.0f}; share with stored count 0: {float((asb[cold & idc] == 0).mean()):.4f}')
# ---- R3e: within-feed AUC among hz2-cold rows (train labels, non-holdout pool feeds)
y = np.asarray(mm(f'{FLAT}/train/cand_target.npy')[rw]).astype(np.int8)
fo = np.repeat(np.arange(len(fs)), Mtr[fs])
def wauc(score, sel):
    from scipy.stats import rankdata
    out = []
    idx = np.flatnonzero(sel); f_ = fo[idx]; b = np.flatnonzero(np.r_[True, f_[1:] != f_[:-1], True])
    for a_, e_ in zip(b[:-1], b[1:]):
        ii = idx[a_:e_]; yy = y[ii]
        n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 == 0 or n0 == 0: continue
        r = rankdata(score[ii])
        out.append((r[yy > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    out = np.array(out)
    return out.mean(), out.std(ddof=1) / np.sqrt(len(out)), len(out)
for nm, sc in (('pre2_id_contrib', idv), ('pre2_fe_s (model input)', fes), ('stored it_hist_all_luo (removed by the view)', asb.astype(np.float64))):
    m_, s_, n_ = wauc(sc, cold)
    lg(f'R3e within-feed AUC among hz2-COLD rows: {nm}: {m_:.4f} +- {s_:.4f} (feeds {n_}; z {(m_ - 0.5) / s_:+.1f})')
m_, s_, n_ = wauc(fes, cold & ~idc); lg(f'   for scale: pre2_fe_s among hz2-cold rows with id_contrib == 0 (the test-like kind): {m_:.4f} +- {s_:.4f} (feeds {n_})')
m_, s_, n_ = wauc(fes, cold & idc); lg(f'   pre2_fe_s among hz2-cold rows with id_contrib != 0 (the kind that does not exist on test): {m_:.4f} +- {s_:.4f} (feeds {n_})')
lg(f'   positive rate: hz2-cold & id == 0 {y[cold & ~idc].mean():.5f} | hz2-cold & id != 0 {y[cold & idc].mean():.5f} | hz2-warm {y[~cold].mean():.5f}')
lg('PRE2_TELL DONE')
