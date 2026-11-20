"""hz2n builder. Usage (CPU, 12 threads, label-free):
  python build_enc.py val      # P2: witness parity on the VAL rows -> logs/p2_val_parity.log, p2_val_parity.json
  python build_enc.py train    # P0 + the new files of the three views + P3a / P3b over all 246,745,836 rows -> <view>/train/pre2_hz_*.npy, build_train.json
"""
import json
import os
import sys

from enc_common import (F, FC, HERE, N_TRAIN, NEW_FILES, ROOT, VIEWS, bits_eq, feed_any, lg, mix_rows, mm, val_configs, window_configs, FLAT)
import numpy as np
import polars as pl

CF = f'{ROOT}/solvers/out/stale/cf'
CF_BOTH = f'{ROOT}/nn_x/r6/stale_cf'  # the both_* files stale_pred.py reads (md5-identical durable copy of the review_shift files)


def run_val(family=''):
    f = pl.read_parquet(f'{FLAT}/val/feeds.parquet')
    M, off = f['M'].to_numpy().astype(np.int64), f['cand_off'].to_numpy().astype(np.int64)
    L = lambda n: np.load(f'{FC}/val/{n}.npy')  # noqa: E731  (the live files the models read; build_pre2both's L)
    fe, fs, fz = L('pre2_fe'), L('pre2_fe_s'), L('pre2_fe_z')
    noid = np.load(f'{F}/val/pre2_fe_noid.npy')
    luo = L('it_hist_all_luo')
    bluo = np.load(f'{CF_BOTH}/both_it_hist_all_luo.npy')
    nc = (bluo == 0) & (luo > 0)
    fes, noids = val_configs(family)
    if family == 'P':
        noid = np.mean(noids, 0).astype(np.float32)  # P family: config mean of the fe-map no-ID score replaces the stored pre2_fe_noid
    res = {'rows': int(len(fe)), 'nc_rows': int(nc.sum()), 'nc_share': float(nc.mean())}
    res['repro_stored_fe_max_abs'] = float(np.abs(np.mean(fes, 0) - fe).max())
    res['repro_stored_noid_max_abs'] = float(np.abs(np.mean(noids, 0) - noid).max())
    fe2, fs2, z2, has2, fnc = mix_rows(nc, fe, noid, fs, fz, fes, noids, off, M)
    w = {k: np.load(f'{CF}/pre2both_{k}.npy') for k in ('pre2_fe', 'pre2_fe_z', 'pre2_fe_s')}
    for k, x in (('pre2_fe', fe2), ('pre2_fe_z', z2), ('pre2_fe_s', fs2)):
        d = np.abs(x.astype(np.float64) - w[k].astype(np.float64))
        res[f'P2_max_abs_{k}'] = float(d.max())
        res[f'P2_rows_not_bitwise_{k}'] = int((~bits_eq(x, w[k])).sum())
        res[f'P2_same_nan_{k}'] = bool((np.isnan(x) == np.isnan(w[k])).all())
        lg(f'P2 {k}: max|d| vs solvers/out/stale/cf/pre2both_{k}.npy = {res[f"P2_max_abs_{k}"]:.3e}; rows not bitwise equal {res[f"P2_rows_not_bitwise_{k}"]} of {len(x)}')
    res['P2_has_rows_diff'] = int((has2 != (~np.isnan(w['pre2_fe'])).astype(np.float32)).sum())
    res['P2_has_rows_diff_vs_stored_has'] = int((has2 != (~np.isnan(fe)).astype(np.float32)).sum())
    res['P2_nc_rows_diff_vs_witness_newly_cold'] = int((nc != np.load(f'{CF}/newly_cold.npy')).sum())
    res['rows_changed_fe'] = int((~bits_eq(fe2, fe)).sum())
    res['rows_changed_fs'] = int((~bits_eq(fs2, fs)).sum())
    res['rows_changed_fz'] = int((~bits_eq(z2, fz)).sum())
    res['feeds_with_nc'] = int(fnc.sum()); res['feeds'] = int(len(M))
    inn = np.repeat(fnc, M)
    res['mean_abs_dz_non_nc_rows_in_nc_feeds'] = float(np.abs(z2 - fz)[inn & ~nc].mean())
    res['mean_abs_dz_non_nc_rows_all'] = float(np.abs(z2 - fz)[~nc].mean())
    res['family'] = family or 'as asked'
    res['P2_pass'] = bool(max(res[f'P2_max_abs_{k}'] for k in w) <= 1e-6 and res['P2_has_rows_diff'] == 0 and res['P2_nc_rows_diff_vs_witness_newly_cold'] == 0)
    lg(json.dumps(res, indent=1))
    json.dump(res, open(f'{HERE}/p2_val_parity{family}.json', 'w'), indent=1)
    lg('P2', 'PASS' if res['P2_pass'] else 'FAIL')


def run_train(family=''):
    f = pl.read_parquet(f'{FLAT}/train/feeds.parquet')
    M_all = f['M'].to_numpy().astype(np.int64); coff = f['cand_off'].to_numpy().astype(np.int64)
    nF = len(f)
    S = np.load(f'{FC}/train/it_hist_all_luo.npy')
    st = {k: np.load(p) for k, p in (('fe', f'{FC}/train/pre2_fe.npy'), ('noid', f'{F}/train/pre2_fe_noid.npy'), ('fs', f'{FC}/train/pre2_fe_s.npy'),
                                     ('fz', f'{FC}/train/pre2_fe_z.npy'))}
    assert all(v.shape == (N_TRAIN,) and v.dtype == np.float32 for v in st.values())
    lg('stored columns loaded')
    res = {'views': {}, 'P0': {}}
    ncs, outs = {}, {}
    for vn, (base, keepf) in VIEWS.items():
        V = np.load(f'{base}/train/it_hist_hz_all.npy')
        keep = np.load(keepf) if keepf else np.zeros(N_TRAIN, bool)
        nc = (V == 0) & (S > 0) & ~keep
        r = {'base': base, 'keep': keepf, 'V_min': float(np.nanmin(V)), 'V_nan': int(np.isnan(V).sum()), 'S_nan': int(np.isnan(S).sum()), 'S_min': float(np.nanmin(S)),
             'keep_rows': int(keep.sum()), 'view_cold_rows_raw_V': int((V == 0).sum()), 'view_cold_rows_effective': int((np.where(keep, S, V) == 0).sum()),
             'stored_cold_rows': int((S == 0).sum()), 'nc_rows': int(nc.sum()),
             # raw counter vs model scale: rows with V > 0 whose fp16(log1p(V)) is 0 (must be 0 for "cold" to mean the same on both scales)
             'rows_V_pos_but_fp16_log1p_zero': int(((V > 0) & (np.log1p(np.maximum(V, 0)).astype(np.float16) == 0)).sum()),
             'rows_V_zero_and_S_zero': int(((V == 0) & (S == 0)).sum()), 'rows_V_pos_and_S_zero': int(((V > 0) & (S == 0)).sum()),
             'keep_rows_with_finite_pre2_fe': int((keep & np.isfinite(st['fe'])).sum())}
        ncs[vn] = nc
        outs[vn] = {'fe': st['fe'].copy(), 'fs': st['fs'].copy(), 'fz': st['fz'].copy()}
        res['views'][vn] = r
        lg(vn, json.dumps(r))
        del V, keep
    in_win = np.zeros(N_TRAIN, bool)
    for k in (1, 2, 3):
        feeds, rows, off, M, fes, noids, calib = window_configs(k, family)
        assert not in_win[rows].any()
        in_win[rows] = True
        fe, noid, fs, fz = (st[x][rows] for x in ('fe', 'noid', 'fs', 'fz'))
        if family == 'P':
            noid = np.mean(noids, 0).astype(np.float32)  # P family: config mean of the fe-map no-ID score replaces the stored pre2_fe_noid
        # P0: the arithmetic without any override reproduces the stored columns
        z0 = np.mean([__import__('enc_common').feed_rank_z(off, M, x)[:, 1] for x in fes], 0).astype(np.float32)
        p0 = {'rows': int(len(rows)), 'feeds': int(len(feeds)), 'calib': calib,
              'fe_max_abs': float(np.nanmax(np.abs(np.mean(fes, 0) - fe))), 'noid_max_abs': float(np.nanmax(np.abs(np.mean(noids, 0) - noid))),
              'fz_max_abs': float(np.nanmax(np.abs(z0 - fz))),
              'fe_rows_not_bitwise': int((~bits_eq(np.mean(fes, 0).astype(np.float32), fe)).sum()),
              'noid_rows_not_bitwise': int((~bits_eq(np.mean(noids, 0).astype(np.float32), noid)).sum()),
              'fz_rows_not_bitwise': int((~bits_eq(z0, fz)).sum()),
              'fs_vs_(fe+20)/6_max_abs': float(np.nanmax(np.abs(fs - ((fe + 20) / 6).astype(np.float32)))),
              'fs_vs_(fe+20)/6_rows_not_bitwise': int((~bits_eq(fs, ((fe + 20) / 6).astype(np.float32))).sum()),
              'nan_rows_fe': int(np.isnan(fe).sum()), 'nan_rows_noid': int(np.isnan(noid).sum()), 'nan_rows_fs': int(np.isnan(fs).sum()), 'nan_rows_fz': int(np.isnan(fz).sum())}
        res['P0'][k] = p0
        lg(f'window {k} P0', json.dumps(p0))
        del z0
        for vn in VIEWS:
            nc = ncs[vn][rows]
            fe2, fs2, z2, has2, fnc = mix_rows(nc, fe, noid, fs, fz, fes, noids, off, M)
            o = outs[vn]
            o['fe'][rows] = fe2; o['fs'][rows] = fs2; o['fz'][rows] = z2
            inn = np.repeat(fnc, M)
            w = {'nc_rows': int(nc.sum()), 'feeds_with_nc': int(fnc.sum()), 'rows_in_feeds_with_nc': int(inn.sum()),
                 'has_rows_changed': int((has2 != (~np.isnan(fe)).astype(np.float32)).sum()),
                 'mean_abs_dz_non_nc_rows_in_nc_feeds': float(np.abs(z2 - fz)[inn & ~nc].mean()) if (inn & ~nc).any() else 0.0,
                 'mean_dz_nc_rows': float((z2 - fz)[nc].mean()) if nc.any() else 0.0, 'mean_dfe_nc_rows': float((fe2 - fe)[nc].mean()) if nc.any() else 0.0,
                 'z_new_vs_stored_max_abs_in_feeds_without_nc': 0.0}
            res['views'][vn][f'win{k}'] = w
            lg(f'window {k} {vn}', json.dumps(w))
            del fe2, fs2, z2, has2, fnc, inn, nc
        del fes, noids, fe, noid, fs, fz, rows
    res['rows_in_windows'] = int(in_win.sum())
    res['rows_outside_windows_with_finite_fe_or_noid'] = int((~in_win & (np.isfinite(st['fe']) | np.isfinite(st['noid']))).sum())
    res['rows_in_windows_nan_fe'] = int((in_win & np.isnan(st['fe'])).sum())
    res['rows_in_windows_nan_noid'] = int((in_win & np.isnan(st['noid'])).sum())
    lg(f"rows in windows {res['rows_in_windows']}; outside windows with a finite fe / noid {res['rows_outside_windows_with_finite_fe_or_noid']}")
    has_st = (~np.isnan(st['fe'])).astype(np.float32)
    row_feed_off = coff
    for vn in VIEWS:
        o, nc = outs[vn], ncs[vn]
        has2 = (~np.isnan(o['fe'])).astype(np.float32)
        fnc = feed_any(nc & in_win, coff, M_all)       # feeds whose z was re-standardised (an nc row with a score)
        fnc_any = feed_any(nc, coff, M_all)
        inn = np.repeat(fnc, M_all)
        ch = {'fe': ~bits_eq(o['fe'], st['fe']), 'fs': ~bits_eq(o['fs'], st['fs']), 'fz': ~bits_eq(o['fz'], st['fz']), 'has': has2 != has_st}
        r = res['views'][vn]
        r.update({'nc_rows_outside_windows(no score, untouched)': int((nc & ~in_win).sum()), 'nc_rows_in_windows': int((nc & in_win).sum()),
                  'feeds_with_nc_row': int(fnc_any.sum()), 'feeds_with_nc_row_with_score': int(fnc.sum()), 'feeds': int(nF),
                  'P3a_rows_changed_outside_nc': {k: int((ch[k] & ~nc).sum()) for k in ('fe', 'fs', 'has')},
                  'P3b_fz_rows_changed_in_feeds_without_nc': int((ch['fz'] & ~inn).sum()),
                  'rows_changed': {k: int(v.sum()) for k, v in ch.items()},
                  'nc_rows_in_windows_not_changed': {k: int((nc & in_win & ~ch[k]).sum()) for k in ('fe', 'fs', 'fz')},
                  'feeds_changed_any_file': int(feed_any(ch['fe'] | ch['fs'] | ch['fz'] | ch['has'], coff, M_all).sum()),
                  'fz_non_nc_rows_changed_in_nc_feeds': int((ch['fz'] & inn & ~nc).sum()),
                  'mean_abs_dz_non_nc_rows_in_nc_feeds': float(np.abs(o['fz'] - st['fz'])[inn & ~nc].mean())})
        r['P3a_pass'] = bool(sum(r['P3a_rows_changed_outside_nc'].values()) == 0)
        r['P3b_pass'] = bool(r['P3b_fz_rows_changed_in_feeds_without_nc'] == 0)
        d = f'{HERE}/{vn}{family}/train'
        os.makedirs(d, exist_ok=True)
        for nm, x in (('pre2_hz_fe', o['fe']), ('pre2_hz_fe_s', o['fs']), ('pre2_hz_fe_z', o['fz']), ('pre2_hz_fe_has', has2)):
            assert x.dtype == np.float32 and x.shape == (N_TRAIN,)
            tmp = f'{d}/.{nm}.tmp.npy'
            np.save(tmp, x)
            os.replace(tmp, f'{d}/{nm}.npy')
        np.save(f'{d}/hz2n_newly_cold.npy', nc)
        lg(f'{vn} written; P3a {"PASS" if r["P3a_pass"] else "FAIL"} P3b {"PASS" if r["P3b_pass"] else "FAIL"}', json.dumps({k: v for k, v in r.items() if not k.startswith('win')}))
        del o, has2, ch, inn, fnc
        outs[vn] = None
    res['P0_pass'] = bool(all(max(p['fe_max_abs'], 0.0 if family == 'P' else p['noid_max_abs'], p['fz_max_abs']) <= 1e-6 for p in res['P0'].values())
                          and res['rows_outside_windows_with_finite_fe_or_noid'] == 0)
    res['family'] = family or 'as asked'
    json.dump(res, open(f'{HERE}/build_train{family}.json', 'w'), indent=1)
    lg('P0', 'PASS' if res['P0_pass'] else 'FAIL', '| BUILD TRAIN DONE')


if __name__ == '__main__':
    {'val': run_val, 'train': run_train}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else '')
