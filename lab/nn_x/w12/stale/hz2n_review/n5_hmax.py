"""N5: does the untouched model input pre2_hmax32 carry the CANDIDATE's item-ID on rows made cold by the view (B rows)?
hmax32 = max over the newest 32 history items of <out_vec(rep(cand)), out_vec(rep(hist))> with ID-aware reps (pretrain/train.py infer()); hmax32_noid (nn_x/abl/encdose/score_hmax_noid.py)
zeroes the ID slot of the candidate AND of the history. d = raw hmax32 - raw hmax32_noid. E rows (made cold, candidate without ID): d = history-ID effect only (the test-like kind).
B rows (made cold, candidate WITH ID): d = history-ID effect + candidate-ID effect. Estimate of the candidate-ID effect on B rows: mean_B(d) - mean_E(d) (same feeds, same windows).
Cross-check on the targets: cold val / test rows have no candidate ID -> their d is the history-only effect. Label-free."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'
FC, HZ2, PT = f'{R}/nn/feat_cache/train', f'{R}/nn_x/w12/stale/hz2', f'{R}/data/pretrain'
WS = f'{PT}/winscores'
mm = lambda p: np.load(p, mmap_mode='r')
f = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet')
M_all = f['M'].to_numpy().astype(np.int64); slot_all = (f['timestamp'].to_numpy().astype(np.int64) + 3 * 3600) // 86400 - 20702
res = {}
for k in (1, 2, 3):
    feeds = np.load(f'{PT}/win{k}/window_feeds.npy'); rows = np.load(f'{PT}/win{k}/cand_rows.npy')
    r0, r1 = int(rows[0]), int(rows[-1]) + 1; assert r1 - r0 == len(rows)
    M = M_all[feeds]; fid = np.repeat(np.arange(len(feeds)), M); ge13 = (slot_all[feeds] >= 7)[fid]
    G = lambda a: np.asarray(a[r0:r1])
    S = G(mm(f'{FC}/it_hist_all_luo.npy')); idc = np.nan_to_num(G(mm(f'{FC}/pre2_id_contrib.npy')), nan=0.0); hm_st = G(mm(f'{FC}/pre2_hmax32.npy'))
    w = {}
    for c in ('v3', 'v4'):
        h = np.load(f'{WS}/pre_w{k}r_{c}_hmax32_win{k}.npy').astype(np.float64); h0 = np.load(f'{WS}/pre_w{k}r_{c}_hmax32_noid_win{k}.npy').astype(np.float64)
        ok = (h > -2) & (h0 > -2)
        d = h - h0
        sd = float(h[ok].std())
        r = {'sd_raw_hmax32': sd, 'rows_no_history': int((~ok).sum()), 'share_d_ne0': float((d[ok] != 0).mean())}
        for v, base, kp in (('a', f'{HZ2}/feats/train', f'{HZ2}/feats/train/hz2_keep_rows.npy'), ('c', f'{R}/data/feats_new/train', None)):
            V = G(mm(f'{base}/it_hist_hz_all.npy')); keep = G(mm(kp)) if kp else np.zeros(len(S), bool)
            nc = (V == 0) & (S > 0) & ~keep
            pops = {'B': nc & (idc != 0), 'E': nc & (idc == 0), 'D': S == 0, 'W_id': ~nc & (S > 0) & (idc != 0), 'W_noid': ~nc & (S > 0) & (idc == 0)}
            o = {}
            for g, gm in (('all', np.ones(len(S), bool)), ('ge13', ge13)):
                if not gm.any(): continue
                q = {}
                for pn, pm in pops.items():
                    m_ = pm & ok & gm
                    if m_.sum() == 0: continue
                    q[pn] = dict(n=int(m_.sum()), mean_d=float(d[m_].mean()), mean_h=float(h[m_].mean()), mean_h0=float(h0[m_].mean()), sd_d=float(d[m_].std()))
                if 'B' in q and 'E' in q:
                    q['cand_ID_effect_B_minus_E_in_SD'] = (q['B']['mean_d'] - q['E']['mean_d']) / sd
                    q['cand_ID_effect_B_minus_D_in_SD'] = (q['B']['mean_d'] - q['D']['mean_d']) / sd
                    q['warm_check_Wid_minus_Wnoid_in_SD'] = (q['W_id']['mean_d'] - q['W_noid']['mean_d']) / sd
                    q['level_B_minus_E_in_SD'] = (q['B']['mean_h'] - q['E']['mean_h']) / sd
                    q['level_noid_B_minus_E_in_SD'] = (q['B']['mean_h0'] - q['E']['mean_h0']) / sd
                    mB = pops['B'] & ok & gm
                    q['corr_d_idc_on_B'] = float(np.corrcoef(d[mB], idc[mB])[0, 1])
                    q['share_B_rows'] = float((pops['B'] & gm).sum() / gm.sum())
                o[g] = q
            r[f'view_{v}'] = o
        w[c] = r
    res[f'win{k}'] = w
    print(f'win{k}', json.dumps({c: {v: {g: {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in q.items() if not isinstance(x, dict)} for g, q in w[c][v].items()} for v in ('view_a', 'view_c')} for c in w}), flush=True)
# targets: d on cold / warm rows of val and test (candidate without ID on every cold row)
for split, tags in (('val', ('v3At2', 'v4At2')), ('test', ('v3t2', 'v4t2'))):
    luo = np.load(f'{R}/nn/feat_cache/{split}/it_hist_all_luo.npy'); idc = np.nan_to_num(np.load(f'{R}/nn/feat_cache/{split}/pre2_id_contrib.npy'), nan=0.0)
    o = {}
    for t in tags:
        p1, p0 = f'{R}/preds/pre_{t}_hmax32_{split}.npy', f'{R}/preds/pre_{t}_hmax32_noid_{split}.npy'
        if not (os.path.exists(p1) and os.path.exists(p0)):
            o[t] = 'files missing'; continue
        h, h0 = np.load(p1).astype(np.float64), np.load(p0).astype(np.float64); ok = (h > -2) & (h0 > -2); d = h - h0; sd = float(h[ok].std())
        o[t] = dict(sd=sd, cold_mean_d=float(d[ok & (luo == 0)].mean()), warm_id_mean_d=float(d[ok & (luo > 0) & (idc != 0)].mean()), warm_noid_mean_d=float(d[ok & (luo > 0) & (idc == 0)].mean()),
                    warm_id_minus_cold_in_SD=float((d[ok & (luo > 0) & (idc != 0)].mean() - d[ok & (luo == 0)].mean()) / sd),
                    warm_id_minus_warm_noid_in_SD=float((d[ok & (luo > 0) & (idc != 0)].mean() - d[ok & (luo > 0) & (idc == 0)].mean()) / sd))
    res[split] = o
    print(split, json.dumps(o), flush=True)
json.dump(res, open(f'{R}/nn_x/w12/stale/hz2n_review/n5_hmax.json', 'w'), indent=1)
