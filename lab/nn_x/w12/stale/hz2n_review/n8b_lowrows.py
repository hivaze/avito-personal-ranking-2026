"""N8b: size of the remaining 'counter in (0,3) + ID-carrying encoder score' combination (impossible at test) under views a and c, and in the stale-val witness. Label-free."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'
FC, HZ2 = f'{R}/nn/feat_cache', f'{R}/nn_x/w12/stale/hz2'
mm = lambda p: np.load(p, mmap_mode='r')
out = {}
# targets
ft = pl.read_parquet(f'{R}/data/flat/test/feeds.parquet'); Mt = ft['M'].to_numpy().astype(np.int64); tts = ft['timestamp'].to_numpy().astype(np.int64)
fidt = np.repeat(np.arange(len(Mt)), Mt); priv = (tts >= 1790497920)[fidt]
luo_t = np.load(f'{FC}/test/it_hist_all_luo.npy'); fs_t = np.load(f'{FC}/test/pre2_fe_s.npy'); ns_t = np.load(f'{FC}/test/pre2_new_s.npy'); idc_t = np.nan_to_num(np.load(f'{FC}/test/pre2_id_contrib.npy'), nan=0.0)
SD = float(fs_t[priv].std()); SDN = float(ns_t[priv].std())
low_t = priv & (luo_t > 0) & (luo_t < 3)
out['private'] = dict(sd_fs=SD, share_low=float(low_t.sum() / priv.sum()), n_low=int(low_t.sum()), n_low_with_id=int((low_t & (idc_t != 0)).sum()), fs_low_mean=float(fs_t[low_t].mean()), fs_cold_mean=float(fs_t[priv & (luo_t == 0)].mean()),
                      ns_low_mean=float(ns_t[low_t].mean()), counter_noninteger=int((luo_t != np.round(luo_t)).sum()))
# witness (val, scenario both)
luo_v = np.load(f'{FC}/val/it_hist_all_luo.npy'); both = np.load(f'{R}/nn_x/r6/stale_cf/both_it_hist_all_luo.npy'); idc_v = np.nan_to_num(np.load(f'{FC}/val/pre2_id_contrib.npy'), nan=0.0)
fs_v = np.load(f'{FC}/val/pre2_fe_s.npy')
Lv = (both > 0) & (both < 3) & (idc_v != 0)
ncv = (both == 0) & (luo_v > 0)
out['witness_val_both'] = dict(rows=len(both), both_noninteger=int((both != np.round(both)).sum()), share_both_low=float(((both > 0) & (both < 3)).mean()), P_id_given_both_low=float((idc_v[(both > 0) & (both < 3)] != 0).mean()),
                               share_L_rows=float(Lv.mean()), n_L=int(Lv.sum()), share_nc=float(ncv.mean()), share_nc_with_id=float((ncv & (idc_v != 0)).mean()),
                               mean_idc_L=float(idc_v[Lv].mean()), mean_idc_ncB=float(idc_v[ncv & (idc_v != 0)].mean()),
                               live_val_low_with_id=int(((luo_v > 0) & (luo_v < 3) & (idc_v != 0)).sum()),
                               P_id_by_both_bin={k: float((idc_v[m] != 0).mean()) for k, m in (('0', both == 0), ('(0,1]', (both > 0) & (both <= 1)), ('(1,2]', (both > 1) & (both <= 2)), ('(2,3]', (both > 2) & (both <= 3)), ('>3', both > 3))})
ftr = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet')
M = ftr['M'].to_numpy().astype(np.int64); off = ftr['cand_off'].to_numpy().astype(np.int64)
slot = (ftr['timestamp'].to_numpy().astype(np.int64) + 3 * 3600) // 86400 - 20702
pr = np.load(f'{R}/eda/labels_position/out/train_prandom.npz')
p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
pool = (p > 0.5) & (M >= 50); pool[hold] = False
PP = np.load(f'{HZ2}/feats/train/hz2_part_pub.npy')
rng = np.random.default_rng(20261115)
rows_of = lambda fs: np.repeat(off[fs] - np.concatenate([[0], np.cumsum(M[fs])[:-1]]), M[fs]) + np.arange(int(M[fs].sum()))
for pn, fs in (('pool13', np.sort(rng.choice(np.flatnonzero(pool & ~PP & (slot >= 7)), 30000, replace=False))), ('ge13', np.sort(rng.choice(np.flatnonzero(slot >= 7), 30000, replace=False)))):
    rows = rows_of(fs)
    S = np.asarray(mm(f'{FC}/train/it_hist_all_luo.npy')[rows]); idc = np.nan_to_num(np.asarray(mm(f'{FC}/train/pre2_id_contrib.npy')[rows]), nan=0.0)
    fsx = np.nan_to_num(np.asarray(mm(f'{FC}/train/pre2_fe_s.npy')[rows]), nan=0.0); has = ~np.isnan(np.asarray(mm(f'{FC}/train/pre2_fe.npy')[rows]))
    out[f'{pn}_as_built'] = dict(n_low_with_id=int(((S > 0) & (S < 3) & (idc != 0)).sum()), share_low=float(((S > 0) & (S < 3)).mean()))
    for v, base, kp in (('a', f'{HZ2}/feats/train', f'{HZ2}/feats/train/hz2_keep_rows.npy'), ('c', f'{R}/data/feats_new/train', None)):
        V = np.asarray(mm(f'{base}/it_hist_hz_all.npy')[rows]); keep = np.asarray(mm(kp)[rows]) if kp else np.zeros(len(rows), bool)
        eff = np.where(keep, S, V); nc = (V == 0) & (S > 0) & ~keep
        low = (eff > 0) & (eff < 3)
        L = low & (idc != 0); B = nc & (idc != 0); Lno = low & (idc == 0) & has
        o = dict(rows=len(rows), share_low=float(low.mean()), P_id_low=float((idc[low] != 0).mean()), share_L=float(L.mean()), share_B=float(B.mean()), L_over_B=float(L.sum() / max(B.sum(), 1)),
                 mean_idc_L=float(idc[L].mean()), mean_idc_B=float(idc[B].mean()), shift_L_sd=float(idc[L].mean() / 6 / SD), shift_B_sd=float(idc[B].mean() / 6 / SD),
                 fs_L_mean=float(fsx[L].mean()), fs_low_noid_mean=float(fsx[Lno].mean()), smd_L_vs_private_low=float((fsx[L].mean() - out['private']['fs_low_mean']) / SD),
                 smd_low_noid_vs_private_low=float((fsx[Lno].mean() - out['private']['fs_low_mean']) / SD), median_stored_counter_L=float(np.median(S[L])), q10_q90_stored_counter_L=[float(q) for q in np.quantile(S[L], [0.1, 0.9])],
                 share_L_by_bin={'(0,1]': float((L & (eff <= 1)).mean()), '(1,2]': float((L & (eff > 1) & (eff <= 2)).mean()), '(2,3)': float((L & (eff > 2)).mean())})
        o['I_L'] = o['shift_L_sd'] * o['share_L']; o['I_B_idc'] = o['shift_B_sd'] * o['share_B']
        out[f'{pn}_view_{v}'] = o
json.dump(out, open(f'{R}/nn_x/w12/stale/hz2n_review/n8b_lowrows.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
