"""Target-side statistics (val, public test, private test), label-free: cold rows = it_hist_all_luo == 0."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'
PUB_TS = 1790497920
out = {}
for split in ('val', 'test'):
    FCs = f'{R}/nn/feat_cache/{split}'
    f = pl.read_parquet(f'{R}/data/flat/{split}/feeds.parquet')
    M = f['M'].to_numpy().astype(np.int64); ts = f['timestamp'].to_numpy().astype(np.int64)
    fid = np.repeat(np.arange(len(M)), M)
    L = lambda n: np.load(f'{FCs}/{n}.npy')
    luo, fe, fs, fz, idc, new, news, newz, hm, pop = (L(n) for n in ('it_hist_all_luo', 'pre2_fe', 'pre2_fe_s', 'pre2_fe_z', 'pre2_id_contrib', 'pre2_new', 'pre2_new_s', 'pre2_new_z', 'pre2_hmax32', 'seq_cv_pop'))
    noid = np.load(f'{R}/data/feats/{split}/pre2_fe_noid.npy')
    tags = ('v3At2', 'v4At2') if split == 'val' else ('v3t2', 'v4t2')
    rid_new = np.zeros(len(luo), bool); rid_fe = np.zeros(len(luo), bool)
    for t in tags:
        rid_new |= np.load(f'{R}/preds/pre_{t}_new_{split}.npy') != np.load(f'{R}/preds/pre_{t}_new_noid_{split}.npy')
        rid_fe |= np.load(f'{R}/preds/pre_{t}_fe_{split}.npy') != np.load(f'{R}/preds/pre_{t}_fe_noid_{split}.npy')
    parts = {'val': np.ones(len(M), bool)} if split == 'val' else {'public': ts < PUB_TS, 'private': ts >= PUB_TS, 'private_M50': (ts >= PUB_TS) & (M >= 50), 'public_M50': (ts < PUB_TS) & (M >= 50)}
    for pn, fm in parts.items():
        rm = fm[fid]; cold = rm & (luo == 0); warm = rm & (luo > 0)
        zc = np.clip(fz, -5, 5); nzc = np.clip(newz, -5, 5); hmc = np.clip(hm, -10, 10)
        o = dict(feeds=int(fm.sum()), rows=int(rm.sum()), cold_rows=int(cold.sum()), cold_share=float(cold.sum() / rm.sum()), nan_fe=int(np.isnan(fe[rm]).sum()),
                 P_idc_ne0_cold=float((np.nan_to_num(idc[cold]) != 0).mean()), n_idc_ne0_cold=int((np.nan_to_num(idc[cold]) != 0).sum()),
                 P_raw_fe_id_cold=float(rid_fe[cold].mean()), P_raw_new_id_cold=float(rid_new[cold].mean()), n_raw_new_id_cold=int(rid_new[cold].sum()),
                 P_idc_ne0_warm=float((np.nan_to_num(idc[warm]) != 0).mean()),
                 P_seq_cv_pop_pos_cold=float((pop[cold] > 0).mean()), n_seq_cv_pop_pos_cold=int((pop[cold] > 0).sum()),
                 offset_noid_minus_fe_noID_mean=float((noid - fe)[rm & (np.nan_to_num(idc) == 0)].mean()), offset_noid_minus_fe_noID_absmax=float(np.abs(noid - fe)[rm & (np.nan_to_num(idc) == 0)].max()),
                 fs_cold_mean=float(fs[cold].mean()), fs_all_sd=float(fs[rm].std()), fs_cold_sd=float(fs[cold].std()), fs_warm_mean=float(fs[warm].mean()),
                 z_cold_mean=float(zc[cold].mean()), z_all_sd=float(zc[rm].std()),
                 news_cold_mean=float(news[cold].mean()), news_all_sd=float(news[rm].std()), newz_cold_mean=float(nzc[cold].mean()), newz_all_sd=float(nzc[rm].std()),
                 hm_cold_mean=float(hmc[cold].mean()), hm_all_sd=float(hmc[rm].std()), fe_all_sd=float(fe[rm].std()))
        out[pn] = o
json.dump(out, open(f'{R}/nn_x/w12/stale/hz2n_review/targets.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
