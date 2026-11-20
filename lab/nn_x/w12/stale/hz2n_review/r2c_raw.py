"""R2c: from the RAW window scores (data/pretrain/winscores), my own code, ALL rows of the three encoder windows (not a sample: it is cheap enough).
Rebuilds the per-config alignment, the mixed scores and the within-feed z (float64 moments) and compares with the builder's train files of every view. Label-free."""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'
FC, F, HZ2, N, PT = f'{R}/nn/feat_cache/train', f'{R}/data/feats/train', f'{R}/nn_x/w12/stale/hz2', f'{R}/nn_x/w12/stale/hz2n', f'{R}/data/pretrain'
WS = f'{PT}/winscores'
mm = lambda p: np.load(p, mmap_mode='r')
T0 = time.time()
lg = lambda *a: print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)
f = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet')
M_all = f['M'].to_numpy().astype(np.int64); off_all = f['cand_off'].to_numpy().astype(np.int64)
VIEWS = {'a': dict(base=f'{HZ2}/feats/train', keep=f'{HZ2}/feats/train/hz2_keep_rows.npy', d='feats_n', F='feats_nF'),
         'b': dict(base=f'{HZ2}/feats_L/train', keep=f'{HZ2}/feats_L/train/hz2_keep_rows.npy', d='feats_Ln', F='feats_LnF'),
         'c': dict(base=f'{R}/data/feats_new/train', keep=None, d='feats_hn', F='feats_hnF')}
bits = lambda a: np.ascontiguousarray(a, np.float32).view(np.int32)
res = {}

def mom(x):
    m = np.isfinite(x) & (x > -2)
    v = x[m].astype(np.float64)
    return v.mean(), v.std()

for k in (1, 2, 3):
    feeds = np.load(f'{PT}/win{k}/window_feeds.npy'); rows = np.load(f'{PT}/win{k}/cand_rows.npy'); s_rows = np.load(f'{PT}/win{k}/cand_rows_s60000.npy')
    M = M_all[feeds]; nf = len(feeds)
    # my own row list from the feed table; must equal cand_rows.npy
    r_first = off_all[feeds]
    assert len(rows) == int(M.sum())
    fid = np.repeat(np.arange(nf), M)
    myrows = np.repeat(r_first - np.concatenate([[0], np.cumsum(M)[:-1]]), M) + np.arange(len(rows))
    assert np.array_equal(myrows, rows)
    pos = np.searchsorted(rows, s_rows); assert (rows[pos] == s_rows).all()
    contiguous = bool((np.diff(rows) == 1).all())
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    G = (lambda a: np.asarray(a[r0:r1])) if contiguous else (lambda a: np.asarray(a[rows]))
    w = {'rows': len(rows), 'feeds': nf, 'contiguous': contiguous, 'sample_rows': len(s_rows), 'sample_feeds': int(len(np.unique(fid[pos])))}
    raw, al, cal = {}, {}, {}
    for c in ('v3', 'v4'):
        for key in ('fe', 'fe_noid', 'new', 'new_noid'):
            x = np.load(f'{WS}/pre_w{k}r_{c}_{key}_win{k}.npy').astype(np.float32); assert x.shape == rows.shape
            raw[c, key] = x
        for key in ('fe', 'fe_noid', 'new'):
            mx, sx = mom(raw[c, key][pos]); mr, sr = mom(np.load(f'{WS}/pre_{c}t2_{key}_win{k}_s60000.npy'))
            a = sr / sx; b = mr - a * mx
            cal[c, key] = (float(a), float(b))
        A = lambda key, mapkey: np.where(raw[c, key] > -2, cal[c, mapkey][0] * raw[c, key].astype(np.float64) + cal[c, mapkey][1], raw[c, key]).astype(np.float32)
        al[c, 'fe'] = A('fe', 'fe'); al[c, 'noid'] = A('fe_noid', 'fe_noid'); al[c, 'new'] = A('new', 'new'); al[c, 'new0'] = A('new_noid', 'new')
    w['calib'] = {f'{c}_{key}': cal[c, key] for c, key in cal}
    cm = lambda key: np.mean([al['v3', key], al['v4', key]], 0)
    st = {n: G(mm(p)) for n, p in dict(fe=f'{FC}/pre2_fe.npy', fs=f'{FC}/pre2_fe_s.npy', fz=f'{FC}/pre2_fe_z.npy', noid=f'{F}/pre2_fe_noid.npy', idc=f'{FC}/pre2_id_contrib.npy',
                                        new=f'{FC}/pre2_new.npy', news=f'{FC}/pre2_new_s.npy', newz=f'{FC}/pre2_new_z.npy', S=f'{FC}/it_hist_all_luo.npy').items()}
    w['nan_in_stored_fe'] = int(np.isnan(st['fe']).sum())
    for key, sk in (('fe', 'fe'), ('noid', 'noid'), ('new', 'new')):
        dd = np.abs(cm(key) - st[sk])
        w[f'repro_stored_{sk}_max_abs'] = float(dd.max()); w[f'repro_stored_{sk}_rows_bits_diff'] = int((bits(cm(key)) != bits(st[sk])).sum())

    def z64(x):
        x = x.astype(np.float64)
        mu = np.bincount(fid, x, nf) / M
        sd = np.sqrt(np.maximum(np.bincount(fid, (x - mu[fid]) ** 2, nf) / M, 0))
        return (x - mu[fid]) / (sd[fid] + 1e-9), sd
    z_fe = [z64(al[c, 'fe']) for c in ('v3', 'v4')]; z_new = [z64(al[c, 'new']) for c in ('v3', 'v4')]
    zfe_un = (z_fe[0][0] + z_fe[1][0]) / 2; znew_un = (z_new[0][0] + z_new[1][0]) / 2
    dg_feed = np.minimum(np.minimum(z_fe[0][1], z_fe[1][1]), np.minimum(z_new[0][1], z_new[1][1])) < 1e-3
    w['degenerate_feeds_unmixed'] = int(dg_feed.sum())
    w['repro_stored_fe_z_max_abs_nondeg'] = float(np.abs(zfe_un - st['fz'])[~dg_feed[fid]].max())
    w['repro_stored_new_z_max_abs_nondeg'] = float(np.abs(znew_un - st['newz'])[~dg_feed[fid]].max())
    idc = np.nan_to_num(st['idc'], nan=0.0)
    rid_fe = (raw['v3', 'fe'] != raw['v3', 'fe_noid']) | (raw['v4', 'fe'] != raw['v4', 'fe_noid'])
    rid_new = (raw['v3', 'new'] != raw['v3', 'new_noid']) | (raw['v4', 'new'] != raw['v4', 'new_noid'])
    w['rows_idc_ne0'] = int((idc != 0).sum()); w['rows_raw_fe_ne_noid'] = int(rid_fe.sum()); w['rows_raw_new_ne_new_noid'] = int(rid_new.sum())
    w['idc_ne0_xor_raw_fe_ne_noid'] = int(((idc != 0) ^ rid_fe).sum()); w['raw_fe_id_xor_raw_new_id'] = int((rid_fe ^ rid_new).sum())
    w['share_id_rows'] = float(rid_fe.mean())
    # calibration offset between the two keys on rows WITHOUT an ID contribution (train-only? val: +0.003, test: 0)
    no = ~rid_fe & (st['fe'] > -2)
    w['offset_noid_minus_fe_noID_rows_mean'] = float((st['noid'] - st['fe'])[no].mean()); w['offset_noid_minus_fe_noID_rows_sd'] = float((st['noid'] - st['fe'])[no].std())
    w['sd_fe_all_rows'] = float(st['fe'][st['fe'] > -2].std())
    w['within_feed_sd_fe_mean'] = float(np.mean([z_fe[0][1].mean(), z_fe[1][1].mean()]))
    w['mean_id_contrib_on_id_rows'] = float(idc[idc != 0].mean())
    w['views'] = {}
    for v, cfg in VIEWS.items():
        V = G(mm(f"{cfg['base']}/it_hist_hz_all.npy")); keep = G(mm(cfg['keep'])) if cfg['keep'] else np.zeros(len(rows), bool)
        nc = (V == 0) & (st['S'] > 0) & ~keep
        fnc = (np.bincount(fid, nc, nf) > 0); rf = fnc[fid]
        d = f"{N}/{cfg['d']}/train"; dF = f"{N}/{cfg['F']}/train"
        b = {n: G(mm(f'{d}/{n}.npy')) for n in ('pre2_hz_fe', 'pre2_hz_fe_s', 'pre2_hz_fe_z')}
        bF = {n: G(mm(f'{dF}/{n}.npy')) for n in ('pre2_hz_new', 'pre2_hz_new_s', 'pre2_hz_new_z')}
        r = {'keep_rows_in_window': int(keep.sum()), 'nc': int(nc.sum()), 'nc_share': float(nc.mean()), 'feeds_with_nc': int(fnc.sum()), 'nc_with_id_share': float(rid_fe[nc].mean()),
             'cold_eff_share': float(((V == 0) & ~keep | (keep & (st['S'] == 0))).mean())}
        # fe head
        mix = [np.where(nc, al[c, 'noid'], al[c, 'fe']) for c in ('v3', 'v4')]
        fe2 = np.mean(mix, 0)
        r['fe_file_vs_mine_max_abs'] = float(np.abs(b['pre2_hz_fe'] - fe2).max()); r['fe_file_vs_mine_rows_bits_diff'] = int((bits(b['pre2_hz_fe']) != bits(fe2)).sum())
        fs2 = ((fe2 + np.float32(20)) / np.float32(6)).astype(np.float32)
        r['fe_s_file_vs_mine_max_abs_nc'] = float(np.abs(b['pre2_hz_fe_s'] - fs2)[nc].max()) if nc.any() else 0.0
        zz = [z64(x) for x in mix]; z2 = (zz[0][0] + zz[1][0]) / 2
        dg = (np.minimum(zz[0][1], zz[1][1]) < 1e-3)[fid]
        dz = np.abs(b['pre2_hz_fe_z'] - z2)
        r['fe_z_file_vs_mine_max_abs_nc_feeds_nondeg'] = float(dz[rf & ~dg].max()); r['fe_z_rows_gt_1e-4_nc_feeds_nondeg'] = int((dz[rf & ~dg] > 1e-4).sum())
        r['fe_z_degenerate_rows_in_nc_feeds'] = int((rf & dg).sum()); r['fe_z_max_abs_degenerate'] = float(dz[rf & dg].max()) if (rf & dg).any() else None
        r['control_unmixed_z_vs_file_max_abs_nc_feeds'] = float(np.abs(b['pre2_hz_fe_z'] - zfe_un)[rf].max())
        r['control_unmixed_z_vs_file_mean_abs_nc_feeds'] = float(np.abs(b['pre2_hz_fe_z'] - zfe_un)[rf].mean())
        # control 2: z over a wrong row set (warm rows only) must not match
        r['fe_z_file_feed_mean_abs_max_nc_feeds'] = float(np.abs(np.bincount(fid, b['pre2_hz_fe_z'].astype(np.float64), nf) / M)[fnc].max())
        sdz = np.sqrt(np.bincount(fid, b['pre2_hz_fe_z'].astype(np.float64) ** 2, nf) / M)
        r['fe_z_file_feed_sd_q01_q50_q99_nc_feeds'] = [float(q) for q in np.quantile(sdz[fnc], [0.01, 0.5, 0.99])]
        # mean |dz| of non-nc rows in nc feeds (re-standardisation effect) and on nc rows
        r['mean_abs_dz_non_nc_rows_in_nc_feeds'] = float(np.abs(b['pre2_hz_fe_z'] - st['fz'])[rf & ~nc].mean())
        r['mean_dz_nc_rows_with_id'] = float((b['pre2_hz_fe_z'] - st['fz'])[nc & rid_fe].mean()); r['mean_dz_nc_rows_no_id'] = float((b['pre2_hz_fe_z'] - st['fz'])[nc & ~rid_fe].mean())
        r['mean_dfe_nc_rows_with_id'] = float((b['pre2_hz_fe'] - st['fe'])[nc & rid_fe].mean()); r['mean_dfe_nc_rows_no_id'] = float((b['pre2_hz_fe'] - st['fe'])[nc & ~rid_fe].mean())
        r['rows_nc_no_id'] = int((nc & ~rid_fe).sum()); r['rows_nc_with_id'] = int((nc & rid_fe).sum())
        # the P alternative (fe key's own map on the raw no-ID score) for comparison: by construction 0 shift on no-ID rows
        # new head, rule build_pre2full: the NEW key's own map on raw new_noid
        mixn = [np.where(nc, al[c, 'new0'], al[c, 'new']) for c in ('v3', 'v4')]
        new2 = np.mean(mixn, 0)
        r['new_file_vs_mine_max_abs'] = float(np.abs(bF['pre2_hz_new'] - new2).max()); r['new_file_vs_mine_rows_bits_diff'] = int((bits(bF['pre2_hz_new']) != bits(new2)).sum())
        r['new_s_file_vs_mine_max_abs'] = float(np.abs(bF['pre2_hz_new_s'] - ((new2 + np.float32(20)) / np.float32(6)).astype(np.float32)).max())
        zzn = [z64(x) for x in mixn]; zn2 = (zzn[0][0] + zzn[1][0]) / 2
        dgn = (np.minimum(zzn[0][1], zzn[1][1]) < 1e-3)[fid]
        dzn = np.abs(bF['pre2_hz_new_z'] - zn2)
        r['new_z_file_vs_mine_max_abs_nc_feeds_nondeg'] = float(dzn[rf & ~dgn].max()); r['new_z_rows_gt_1e-4'] = int((dzn[rf & ~dgn] > 1e-4).sum())
        r['control_unmixed_new_z_vs_file_max_abs_nc_feeds'] = float(np.abs(bF['pre2_hz_new_z'] - znew_un)[rf].max())
        r['mean_dnew_nc_rows_with_id'] = float((bF['pre2_hz_new'] - st['new'])[nc & rid_new].mean()); r['max_abs_dnew_nc_rows_no_id'] = float(np.abs(bF['pre2_hz_new'] - st['new'])[nc & ~rid_new].max())
        # N3: new head still carrying an ID contribution on effective-cold rows
        cold = np.where(keep, st['S'], V) == 0
        r['N3_P(new_head_ID|cold)_as_asked'] = float(rid_new[cold].mean()); r['N3_P(fe_head_ID|cold)_before'] = float(rid_fe[cold].mean())
        r['N3_after_F_rows_cold_with_new_ID_and_file_equals_stored_new'] = int((cold & rid_new & (bits(bF['pre2_hz_new']) == bits(st['new']))).sum())
        r['N3_after_F_rows_cold_with_new_ID_and_file_ne_mine'] = int((cold & rid_new & (np.abs(bF['pre2_hz_new'] - new2) > 2e-5)).sum())
        w['views'][v] = r
        lg(f'win{k} view {v}: fe |d| {r["fe_file_vs_mine_max_abs"]:.2e} z |d| {r["fe_z_file_vs_mine_max_abs_nc_feeds_nondeg"]:.2e} new |d| {r["new_file_vs_mine_max_abs"]:.2e} newz |d| {r["new_z_file_vs_mine_max_abs_nc_feeds_nondeg"]:.2e}')
    res[f'win{k}'] = w
    json.dump(res, open(f'{R}/nn_x/w12/stale/hz2n_review/r2c_raw.json', 'w'), indent=1)
    lg(f'win{k} done')
print(json.dumps(res, indent=1))
