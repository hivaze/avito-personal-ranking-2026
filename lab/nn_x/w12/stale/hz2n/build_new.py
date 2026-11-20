"""hz2n ADDENDUM 2: the NEW-head files of the F family (pre2_hz_new / pre2_hz_new_s / pre2_hz_new_z), built on the same newly-cold rows exactly as
solvers/CPU2_stale_replay_pre2s_and_arms.py::build_pre2full builds them on val (per config the NEW key's own affine map applied to the raw no-ID score).
  python build_new.py val      # P2n: parity with solvers/out/stale/cf/pre2full_{pre2_new,pre2_new_z,pre2_new_s}.npy
  python build_new.py train    # P0n + files of feats_nF / feats_LnF / feats_hnF (+ symlinks to the as-asked fe-head files and the base view) + P3n
CPU, 12 threads, label-free.
"""
import json
import os
import sys

from enc_common import (CONFIGS, F, FC, FLAT, HERE, N_TRAIN, P, PT, ROOT, RTAG, VAL_CALIB, VAL_TAGS, VIEWS, WS, WTAG, affine, bits_eq, feed_any, feed_rank_z, lg, mix_rows,
                        moments)
import numpy as np
import polars as pl

CF = f'{ROOT}/solvers/out/stale/cf'
CF_BOTH = f'{ROOT}/nn_x/r6/stale_cf'
HEAD = 'new'


def val_head():
    xs, ns = [], []
    for t, c in zip(VAL_TAGS, VAL_CALIB):
        L0 = lambda k: np.load(f'{P}/pre_{t}_{k}_val.npy').astype(np.float32)  # noqa: E731
        R = lambda k: np.load(f'{P}/pre_{c}_{k}_val.npy').astype(np.float32)  # noqa: E731
        fn, _ = affine(L0(HEAD), R(HEAD))  # build_pre2full: fn, _ = affine(L0("new"), R("new")); n1, n0 = fn(L0("new")), fn(L0("new_noid"))
        xs.append(fn(L0(HEAD))); ns.append(fn(L0(HEAD + '_noid')))
    return xs, ns


def window_head(k):
    """assemble_train.window_feats for the key 'new' (verbatim expression), and the same map applied to the raw 'new_noid'. Also the fe-head sentinel counts."""
    f = pl.read_parquet(f'{FLAT}/train/feeds.parquet')
    M_all = f['M'].to_numpy().astype(np.int64)
    feeds = np.load(f'{PT}/win{k}/window_feeds.npy')
    rows = np.load(f'{PT}/win{k}/cand_rows.npy')
    s_rows = np.load(f'{PT}/win{k}/cand_rows_s60000.npy')
    pos = np.searchsorted(rows, s_rows)
    assert (rows[pos] == s_rows).all()
    M = M_all[feeds]
    off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    xs, ns, calib = [], [], {}
    raw_id = np.zeros(len(rows), bool); sent = np.zeros(len(rows), bool)
    for c in CONFIGS:
        L0 = lambda key: np.load(f'{WS}/pre_{WTAG.format(k=k, c=c)}_{key}_win{k}.npy').astype(np.float32)  # noqa: E731
        fns = {}
        x = L0(HEAD)
        mx, sx = moments(x[pos])
        mr, sr = moments(np.load(f'{WS}/pre_{RTAG.format(c=c)}_{HEAD}_win{k}_s60000.npy'))
        aa = sr / sx; bb = mr - aa * mx
        fns[HEAD] = (aa, bb)
        calib[f'{c}_{HEAD}'] = (round(float(aa), 4), round(float(bb), 4))
        xs.append(np.where(L0(HEAD) > -2, fns[HEAD][0] * L0(HEAD) + fns[HEAD][1], L0(HEAD)).astype(np.float32))
        ns.append(np.where(L0(HEAD + '_noid') > -2, fns[HEAD][0] * L0(HEAD + '_noid') + fns[HEAD][1], L0(HEAD + '_noid')).astype(np.float32))
        raw_id |= L0(HEAD) != L0(HEAD + '_noid')
        sent |= (L0('fe') > -2) != (L0('fe_noid') > -2)   # fe head: the affine guard (x > -2) differs between the score and its no-ID twin
    return feeds, rows, off, M, xs, ns, calib, raw_id, sent


def run_val():
    f = pl.read_parquet(f'{FLAT}/val/feeds.parquet')
    M, off = f['M'].to_numpy().astype(np.int64), f['cand_off'].to_numpy().astype(np.int64)
    L = lambda n: np.load(f'{FC}/val/{n}.npy')  # noqa: E731
    x, xs_, xz = L('pre2_new'), L('pre2_new_s'), L('pre2_new_z')
    nc = (np.load(f'{CF_BOTH}/both_it_hist_all_luo.npy') == 0) & (L('it_hist_all_luo') > 0)
    assert np.array_equal(nc, np.load(f'{CF}/newly_cold.npy'))
    xs, ns = val_head()
    res = {'rows': int(len(x)), 'nc_rows': int(nc.sum()), 'repro_stored_new_max_abs': float(np.abs(np.mean(xs, 0) - x).max())}
    x2, s2, z2, has2, fnc = mix_rows(nc, x, np.mean(ns, 0).astype(np.float32), xs_, xz, xs, ns, off, M)
    w = {k: np.load(f'{CF}/pre2full_{k}.npy') for k in ('pre2_new', 'pre2_new_z', 'pre2_new_s')}
    for k, v in (('pre2_new', x2), ('pre2_new_z', z2), ('pre2_new_s', s2)):
        res[f'P2n_max_abs_{k}'] = float(np.abs(v.astype(np.float64) - w[k]).max())
        res[f'P2n_rows_not_bitwise_{k}'] = int((~bits_eq(v, w[k])).sum())
        lg(f'P2n {k}: max|d| vs solvers/out/stale/cf/pre2full_{k}.npy = {res[f"P2n_max_abs_{k}"]:.3e}; rows not bitwise equal {res[f"P2n_rows_not_bitwise_{k}"]} of {len(v)}')
    res['rows_changed'] = {'new': int((~bits_eq(x2, x)).sum()), 'new_s': int((~bits_eq(s2, xs_)).sum()), 'new_z': int((~bits_eq(z2, xz)).sum())}
    res['P2n_pass'] = bool(max(res[f'P2n_max_abs_{k}'] for k in w) <= 1e-6)
    lg(json.dumps(res)); lg('P2n', 'PASS' if res['P2n_pass'] else 'FAIL')
    json.dump(res, open(f'{HERE}/p2n_val_parity.json', 'w'), indent=1)


def run_train():
    f = pl.read_parquet(f'{FLAT}/train/feeds.parquet')
    M_all = f['M'].to_numpy().astype(np.int64); coff = f['cand_off'].to_numpy().astype(np.int64)
    st = {k: np.load(p) for k, p in (('x', f'{FC}/train/pre2_new.npy'), ('s', f'{FC}/train/pre2_new_s.npy'), ('z', f'{FC}/train/pre2_new_z.npy'))}
    idc = np.nan_to_num(np.load(f'{FC}/train/pre2_id_contrib.npy'), nan=0.0) != 0
    ncs = {vn: np.load(f'{HERE}/{vn}/train/hz2n_newly_cold.npy') for vn in VIEWS}
    outs = {vn: {k: v.copy() for k, v in st.items()} for vn in VIEWS}
    res = {'P0n': {}, 'views': {vn: {} for vn in VIEWS}, 'fe_guard': {}}
    in_win = np.zeros(N_TRAIN, bool); raw_id_all = np.zeros(N_TRAIN, bool)
    for k in (1, 2, 3):
        feeds, rows, off, M, xs, ns, calib, raw_id, sent = window_head(k)
        in_win[rows] = True; raw_id_all[rows] = raw_id
        x, s, z = (st[q][rows] for q in ('x', 's', 'z'))
        z0 = np.mean([feed_rank_z(off, M, v)[:, 1] for v in xs], 0).astype(np.float32)
        p0 = {'rows': int(len(rows)), 'calib': calib, 'new_max_abs': float(np.abs(np.mean(xs, 0) - x).max()), 'new_z_max_abs': float(np.abs(z0 - z).max()),
              'new_rows_not_bitwise': int((~bits_eq(np.mean(xs, 0).astype(np.float32), x)).sum()), 'new_z_rows_not_bitwise': int((~bits_eq(z0, z)).sum()),
              'new_s_vs_(new+20)/6_rows_not_bitwise': int((~bits_eq(s, ((x + 20) / 6).astype(np.float32))).sum()),
              'rows_new_head_raw_id': int(raw_id.sum()), 'rows_fe_id_contrib': int(idc[rows].sum()), 'rows_new_raw_id_xor_fe_id_contrib': int((raw_id != idc[rows]).sum()),
              'noid_map_equals_score_on_rows_without_raw_id_not_bitwise': int((~np.all([bits_eq(a_, b_) for a_, b_ in zip(xs, ns)], 0) & ~raw_id).sum())}
        res['P0n'][k] = p0
        lg(f'window {k} P0n', json.dumps(p0))
        res['fe_guard'][k] = {vn: int((sent & ncs[vn][rows] & idc[rows]).sum()) for vn in VIEWS}
        lg(f'window {k}: nc rows with fe ID contribution where the (x > -2) guard differs between raw fe and raw fe_noid in a config: {res["fe_guard"][k]}')
        noid = np.mean(ns, 0).astype(np.float32)
        for vn in VIEWS:
            nc = ncs[vn][rows]
            x2, s2, z2, has2, fnc = mix_rows(nc, x, noid, s, z, xs, ns, off, M)
            o = outs[vn]
            o['x'][rows] = x2; o['s'][rows] = s2; o['z'][rows] = z2
            res['views'][vn][f'win{k}'] = {'nc_rows': int(nc.sum()), 'mean_dnew_nc_rows': float((x2 - x)[nc].mean()), 'mean_dz_nc_rows': float((z2 - z)[nc].mean())}
            lg(f'window {k} {vn}', json.dumps(res['views'][vn][f'win{k}']))
    for vn in VIEWS:
        o, nc = outs[vn], ncs[vn]
        fnc = feed_any(nc & in_win, coff, M_all)
        inn = np.repeat(fnc, M_all)
        ch = {q: ~bits_eq(o[q], st[q]) for q in ('x', 's', 'z')}
        r = res['views'][vn]
        r.update({'P3n_rows_changed_outside_nc': {q: int((ch[q] & ~nc).sum()) for q in ('x', 's')}, 'P3n_z_rows_changed_in_feeds_without_nc': int((ch['z'] & ~inn).sum()),
                  'rows_changed': {q: int(v.sum()) for q, v in ch.items()}, 'feeds_changed': int(feed_any(ch['x'] | ch['s'] | ch['z'], coff, M_all).sum()),
                  'nan_mask_changed_rows': int((np.isnan(o['x']) != np.isnan(st['x'])).sum()),
                  'G1n_before_rows_cold_effective_with_new_id': None,
                  'rows_new_id_and_nc_unchanged_bitwise': int((raw_id_all & nc & in_win & ~ch['x']).sum()),
                  'mean_abs_dz_non_nc_rows_in_nc_feeds': float(np.abs(o['z'] - st['z'])[inn & ~nc].mean())})
        r['P3n_pass'] = bool(sum(r['P3n_rows_changed_outside_nc'].values()) == 0 and r['P3n_z_rows_changed_in_feeds_without_nc'] == 0 and r['nan_mask_changed_rows'] == 0)
        d = f'{HERE}/{vn}F/train'
        os.makedirs(d, exist_ok=True)
        for nm, q in (('pre2_hz_new', 'x'), ('pre2_hz_new_s', 's'), ('pre2_hz_new_z', 'z')):
            tmp = f'{d}/.{nm}.tmp.npy'
            np.save(tmp, o[q]); os.replace(tmp, f'{d}/{nm}.npy')
        for n in sorted(os.listdir(f'{HERE}/{vn}/train')):   # base-view links and the as-asked fe-head files
            if not os.path.lexists(f'{d}/{n}'):
                os.symlink(os.path.realpath(f'{HERE}/{vn}/train/{n}'), f'{d}/{n}')
        os.makedirs(f'{HERE}/{vn}F/aux', exist_ok=True)
        if not os.path.lexists(f'{HERE}/{vn}F/aux/pre2_hz_fe_has.npy'):
            os.symlink(f'{HERE}/{vn}/aux/pre2_hz_fe_has.npy', f'{HERE}/{vn}F/aux/pre2_hz_fe_has.npy')
        lg(f'{vn}F written; P3n {"PASS" if r["P3n_pass"] else "FAIL"}', json.dumps({k_: v_ for k_, v_ in r.items() if not k_.startswith('win')}))
        outs[vn] = None
    res['P0n_pass'] = bool(all(max(p['new_max_abs'], p['new_z_max_abs']) <= 1e-6 for p in res['P0n'].values()))
    json.dump(res, open(f'{HERE}/build_trainF.json', 'w'), indent=1)
    lg('P0n', 'PASS' if res['P0n_pass'] else 'FAIL', '| BUILD F DONE')


if __name__ == '__main__':
    {'val': run_val, 'train': run_train}[sys.argv[1]]()
