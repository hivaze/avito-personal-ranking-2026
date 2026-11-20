"""hz2n label-free gate (DECLARATION.txt D2 + ADDENDUM 1). No cand_target file is opened.
  python gate.py offset            # G4(iii): pre2_fe_noid - pre2_fe on rows without item-ID contribution (train windows, val, test)
  python gate.py full [dirs...]    # G1-G4 for the listed view directories (default: every existing one of feats_n feats_Ln feats_hn feats_nP feats_LnP feats_hnP)
"""
import json
import os
import sys

from enc_common import F, FC, FLAT, HERE, HZ2, N_TRAIN, P, PT, PUB_TS, ROOT, WS, VIEWS, bits_eq, feed_any, lg, mm, DAY, DAY0, MSK
import numpy as np
import polars as pl

nz = lambda x: np.nan_to_num(np.asarray(x, np.float32), nan=0.0)  # noqa: E731


def q(x):
    return [float(v) for v in np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99])]


def run_offset():
    res = {}
    fe, noid, idc = np.load(f'{FC}/train/pre2_fe.npy'), np.load(f'{F}/train/pre2_fe_noid.npy'), np.load(f'{FC}/train/pre2_id_contrib.npy')
    for k in (1, 2, 3):
        rows = np.load(f'{PT}/win{k}/cand_rows.npy')
        d = (noid[rows].astype(np.float64) - fe[rows]); i0 = idc[rows] == 0
        res[f'train_win{k}'] = dict(rows=int(len(rows)), rows_idc0=int(i0.sum()), mean=float(d[i0].mean()), sd=float(d[i0].std()), abs_max=float(np.abs(d[i0]).max()), q=q(d[i0]),
                                    mean_idc_ne0=float(d[~i0].mean()), mean_idc_on_ne0=float(idc[rows][~i0].mean()),
                                    mean_noid_minus_fe_plus_idc_all=float((d + idc[rows]).mean()), sd_fe=float(fe[rows].std()))
        lg(f'train window {k}', json.dumps(res[f'train_win{k}']))
    del fe, noid, idc
    for s in ('val', 'test'):
        fe, noid, idc = np.load(f'{FC}/{s}/pre2_fe.npy'), np.load(f'{F}/{s}/pre2_fe_noid.npy'), nz(np.load(f'{FC}/{s}/pre2_id_contrib.npy'))
        d = noid.astype(np.float64) - fe; i0 = idc == 0
        res[s] = dict(rows=int(len(fe)), rows_idc0=int(i0.sum()), mean=float(d[i0].mean()), sd=float(d[i0].std()), abs_max=float(np.abs(d[i0]).max()), q=q(d[i0]),
                      mean_idc_ne0=float(d[~i0].mean()), mean_idc_on_ne0=float(idc[~i0].mean()), sd_fe=float(fe.std()))
        lg(s, json.dumps(res[s]))
    res['rule_trigger_P_family'] = bool(max(abs(res[f'train_win{k}']['mean']) for k in (1, 2, 3)) > 0.05)
    lg('ADDENDUM 1 rule: build the P family =', res['rule_trigger_P_family'])
    json.dump(res, open(f'{HERE}/gate_offset.json', 'w'), indent=1)


def ms(x, m):
    """(n, mean, sd) of x over mask m in float64."""
    v = x[m].astype(np.float64)
    return (int(v.size), float(v.mean()) if v.size else float('nan'), float(v.std()) if v.size else float('nan'))


def targets():
    out = {}
    fte = pl.read_parquet(f'{FLAT}/test/feeds.parquet')
    tts, Mte = fte['timestamp'].to_numpy().astype(np.int64), fte['M'].to_numpy().astype(np.int64)
    pub_rows = np.repeat(tts < PUB_TS, Mte); m50_rows = np.repeat(Mte >= 50, Mte)
    L = lambda s, n: np.load(f'{FC}/{s}/{n}.npy')  # noqa: E731
    sel = {'val': ('val', None), 'public': ('test', pub_rows), 'private': ('test', ~pub_rows), 'private_M50': ('test', ~pub_rows & m50_rows)}
    for nm, (s, m) in sel.items():
        luo, idc, fs, fz, fe = L(s, 'it_hist_all_luo'), nz(L(s, 'pre2_id_contrib')), L(s, 'pre2_fe_s'), np.clip(L(s, 'pre2_fe_z'), -5, 5), L(s, 'pre2_fe')
        pop, xp = L(s, 'seq_cv_pop'), L(s, 'xp_imp_c50')
        m = np.ones(len(luo), bool) if m is None else m
        cold = m & (luo == 0)
        out[nm] = dict(rows=int(m.sum()), cold_rows=int(cold.sum()), cold_share=float(cold.sum() / m.sum()),
                       G1_cold_and_id=int((cold & (idc != 0)).sum()), nan_fe_rows=int((m & np.isnan(fe)).sum()),
                       fs_all=ms(fs, m), fs_cold=ms(fs, cold), fz_all=ms(fz, m), fz_cold=ms(fz, cold),
                       P_seq_cv_pop_pos_given_cold=float(((pop > 0) & cold).sum() / cold.sum()), P_xp_imp_c50_pos_given_cold=float(((xp > 0) & cold).sum() / cold.sum()),
                       P_seq_cv_pop_pos_given_warm=float(((pop > 0) & m & (luo > 0)).sum() / (m & (luo > 0)).sum()),
                       warm_and_id0_share_of_rows=float((m & (luo > 0) & (idc == 0)).sum() / m.sum()))
        lg('target', nm, json.dumps(out[nm]))
    return out


def new_head(tg):
    """G4(i): rows whose NEW-head score / hmax32 carry an item-ID contribution (raw score != raw no-ID score, any config). Returns the train row flags."""
    idn = np.zeros(N_TRAIN, bool); idh = np.zeros(N_TRAIN, bool)
    for k in (1, 2, 3):
        rows = np.load(f'{PT}/win{k}/cand_rows.npy')
        a = np.zeros(len(rows), bool); b = np.zeros(len(rows), bool)
        for c in ('v3', 'v4'):
            L0 = lambda key: mm(f'{WS}/pre_w{k}r_{c}_{key}_win{k}.npy')  # noqa: E731
            a |= np.asarray(L0('new')) != np.asarray(L0('new_noid'))
            b |= np.asarray(L0('hmax32')) != np.asarray(L0('hmax32_noid'))
        idn[rows] = a; idh[rows] = b
    fte = pl.read_parquet(f'{FLAT}/test/feeds.parquet')
    pub_rows = np.repeat(fte['timestamp'].to_numpy().astype(np.int64) < PUB_TS, fte['M'].to_numpy().astype(np.int64))
    for s, tags in (('val', ('v3At2', 'v4At2')), ('test', ('v3t2', 'v4t2'))):
        luo = np.load(f'{FC}/{s}/it_hist_all_luo.npy')
        a = np.zeros(len(luo), bool); b = np.zeros(len(luo), bool); e = np.zeros(len(luo), bool)
        for t in tags:
            a |= np.load(f'{P}/pre_{t}_new_{s}.npy') != np.load(f'{P}/pre_{t}_new_noid_{s}.npy')
            b |= np.load(f'{P}/pre_{t}_hmax32_{s}.npy') != np.load(f'{P}/pre_{t}_hmax32_noid_{s}.npy')
            e |= np.load(f'{P}/pre_{t}_fe_{s}.npy') != np.load(f'{P}/pre_{t}_fe_noid_{s}.npy')
        for nm, m in ((('val', np.ones(len(luo), bool)),) if s == 'val' else (('public', pub_rows), ('private', ~pub_rows))):
            cold = m & (luo == 0)
            tg[nm].update(P_new_head_id_given_cold=float((a & cold).sum() / cold.sum()), P_hmax32_id_given_cold=float((b & cold).sum() / cold.sum()),
                          P_fe_raw_id_given_cold=float((e & cold).sum() / cold.sum()), P_new_head_id_given_warm=float((a & m & (luo > 0)).sum() / (m & (luo > 0)).sum()))
            lg('target', nm, 'G4(i)', {k_: tg[nm][k_] for k_ in ('P_new_head_id_given_cold', 'P_hmax32_id_given_cold', 'P_fe_raw_id_given_cold', 'P_new_head_id_given_warm')})
    return idn, idh


def run_full(dirs):
    tg = targets()
    idn, idh = new_head(tg)
    f = pl.read_parquet(f'{FLAT}/train/feeds.parquet')
    M = f['M'].to_numpy().astype(np.int64); coff = f['cand_off'].to_numpy().astype(np.int64)
    slot = ((f['timestamp'].to_numpy().astype(np.int64) + MSK) // DAY - DAY0).astype(np.int8)
    rslot = np.repeat(slot, M)
    S = np.load(f'{FC}/train/it_hist_all_luo.npy')
    idc = nz(np.load(f'{FC}/train/pre2_id_contrib.npy')) != 0
    idv = nz(np.load(f'{FC}/train/pre2_id_contrib.npy'))
    fe, noid, fs, fz = np.load(f'{FC}/train/pre2_fe.npy'), np.load(f'{F}/train/pre2_fe_noid.npy'), np.load(f'{FC}/train/pre2_fe_s.npy'), np.load(f'{FC}/train/pre2_fe_z.npy')
    has = np.isfinite(fe)
    pop_s, xp_s = np.load(f'{FC}/train/seq_cv_pop.npy'), np.load(f'{FC}/train/xp_imp_c50.npy')
    # review population (hz2_review/pre2_tell.py): trPool private-like, days >= Sep 13
    pr = np.load(f'{ROOT}/eda/labels_position/out/train_prandom.npz')
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    pool = (p > 0.5) & (M >= 50); pool[hold] = False
    PP = np.load(f'{HZ2}/feats/train/hz2_part_pub.npy')
    rev_rows = np.repeat(pool & ~PP & (slot >= 7), M)
    groups = {'all days': np.ones(N_TRAIN, bool), 'days >= Sep 13': rslot >= 7, 'Sep 9-12': (rslot >= 3) & (rslot <= 6), 'Sep 6-8': rslot <= 2,
              'review pop (trPool private-like >= Sep 13)': rev_rows}
    fgroups = {'all days': np.ones(len(M), bool), 'days >= Sep 13': slot >= 7, 'Sep 9-12': (slot >= 3) & (slot <= 6), 'Sep 6-8': slot <= 2,
               'review pop (trPool private-like >= Sep 13)': pool & ~PP & (slot >= 7)}
    res = {'targets': tg, 'views': {}}
    for dn in dirs:
        base_name = dn[:-1] if dn.endswith('P') else dn
        base, keepf = VIEWS[base_name]
        fam = 'P' if dn.endswith('P') else 'as asked'
        d = f'{HERE}/{dn}/train'
        V = np.load(f'{base}/train/it_hist_hz_all.npy')
        keep = np.load(keepf) if keepf else np.zeros(N_TRAIN, bool)
        eff = np.where(keep, S, V)
        cold = eff == 0
        nc = (V == 0) & (S > 0) & ~keep
        assert np.array_equal(nc, np.load(f'{d}/hz2n_newly_cold.npy'))
        nfe, nfs, nfz = np.load(f'{d}/pre2_hz_fe.npy'), np.load(f'{d}/pre2_hz_fe_s.npy'), np.load(f'{d}/pre2_hz_fe_z.npy')
        hasf = f'{d}/pre2_hz_fe_has.npy' if os.path.exists(f'{d}/pre2_hz_fe_has.npy') else f'{HERE}/{dn}/aux/pre2_hz_fe_has.npy'
        nhas = np.load(hasf)
        if fam == 'as asked':  # declared: still carrying = ID contribution != 0 and the new score is not the no-ID score
            still = idc & ~bits_eq(nfe, noid)
        else:  # P family: residual ID contribution of the new score, new - (fe - id_contrib), beyond float rounding
            still = idc & ~(np.abs(nfe.astype(np.float64) - (fe.astype(np.float64) - idv)) <= 1e-3)
        unchanged_id = idc & bits_eq(nfe, fe)   # second reading: ID-carrying rows whose score is the stored one
        ch_fs, ch_fz, ch_has = ~bits_eq(nfs, fs), ~bits_eq(nfz, fz), nhas != has.astype(np.float32)
        fch = feed_any(ch_fs | ch_fz | ch_has, coff, M); fch_s = feed_any(ch_fs, coff, M)
        fzc_b, fzc_a = np.clip(fz, -5, 5), np.clip(nfz, -5, 5)
        if os.path.exists(f'{base}/train/seq_cv_hz_pop.npy'):
            pop_v = np.where(keep, pop_s, np.load(f'{base}/train/seq_cv_hz_pop.npy')); xp_v = np.where(keep, xp_s, np.load(f'{base}/train/xp_imp_hz_c50.npy'))
        else:
            pop_v, xp_v = pop_s, xp_s
        rv = {'family': fam, 'base': base, 'keep': keepf, 'groups': {}}
        for gn, g in groups.items():
            c = cold & g; ch = c & has
            B, C, D = nc & has & idc & g, nc & has & ~idc & g, (S == 0) & has & g
            r = dict(rows=int(g.sum()), feeds=int(fgroups[gn].sum()), cold_rows=int(c.sum()), cold_share=float(c.sum() / g.sum()), cold_rows_with_score=int(ch.sum()),
                     nc_rows=int((nc & g).sum()), nc_rows_with_score=int((nc & has & g).sum()),
                     G1_before_count=int((c & idc).sum()), G1_before=float((c & idc).sum() / max(c.sum(), 1)),
                     G1_before_share_of_all_rows=float((c & idc).sum() / g.sum()),
                     G1_after_count=int((c & still).sum()), G1_after=float((c & still).sum() / max(c.sum(), 1)),
                     G1_after_count_unchanged_score_reading=int((c & unchanged_id).sum()),
                     G1_as_built_count=int(((S == 0) & g & idc).sum()),
                     G2_fs={'A all view-cold with score': (ms(fs, ch), ms(nfs, ch)), 'B made cold, with ID': (ms(fs, B), ms(nfs, B)),
                            'C made cold, no ID': (ms(fs, C), ms(nfs, C)), 'D as-built cold': (ms(fs, D), ms(nfs, D))},
                     G2_fz={'A all view-cold with score': (ms(fzc_b, ch), ms(fzc_a, ch)), 'B made cold, with ID': (ms(fzc_b, B), ms(fzc_a, B)),
                            'C made cold, no ID': (ms(fzc_b, C), ms(fzc_a, C)), 'D as-built cold': (ms(fzc_b, D), ms(fzc_a, D))},
                     G3_rows_changed=dict(fs=int((ch_fs & g).sum()), fz=int((ch_fz & g).sum()), has=int((ch_has & g).sum()),
                                          fs_share=float((ch_fs & g).sum() / g.sum()), fz_share=float((ch_fz & g).sum() / g.sum())),
                     G3_feeds_changed=dict(any=int((fch & fgroups[gn]).sum()), any_share=float((fch & fgroups[gn]).sum() / fgroups[gn].sum()),
                                           fs=int((fch_s & fgroups[gn]).sum()), fs_share=float((fch_s & fgroups[gn]).sum() / fgroups[gn].sum())),
                     G3_mean_abs_dz_non_nc_rows_changed=float(np.abs(nfz - fz)[ch_fz & ~nc & g].mean()) if (ch_fz & ~nc & g).any() else 0.0,
                     G4_P_new_head_id_given_cold=float((c & idn).sum() / max(c.sum(), 1)), G4_P_hmax32_id_given_cold=float((c & idh).sum() / max(c.sum(), 1)),
                     G4_P_new_head_id_given_nc_with_score=float((nc & has & g & idn).sum() / max((nc & has & g).sum(), 1)),
                     G4_P_seq_cv_pop_pos_given_cold=float((c & (pop_v > 0)).sum() / max(c.sum(), 1)), G4_P_xp_imp_c50_pos_given_cold=float((c & (xp_v > 0)).sum() / max(c.sum(), 1)),
                     G4_P_seq_cv_pop_pos_given_cold_as_built=float(((S == 0) & g & (pop_s > 0)).sum() / max(((S == 0) & g).sum(), 1)),
                     G4_warm_and_id0_share_of_rows_with_score=float((~c & has & g & ~idc).sum() / max((has & g).sum(), 1)))
            # SMD vs targets: (mean - target cold mean) / target all-row SD
            smd = {}
            for key, tk in (('G2_fs', 'fs'), ('G2_fz', 'fz')):
                for pn, (bf, af) in r[key].items():
                    for tn in ('private', 'public', 'val', 'private_M50'):
                        t_mean, t_sd = tg[tn][f'{tk}_cold'][1], tg[tn][f'{tk}_all'][2]
                        smd[f'{tk} | {pn} | vs {tn}'] = (round((bf[1] - t_mean) / t_sd, 4), round((af[1] - t_mean) / t_sd, 4))
            r['SMD_before_after'] = smd
            rv['groups'][gn] = r
            lg(dn, '|', gn, '| rows', r['rows'], 'cold share', round(r['cold_share'], 4), '| G1 before', round(r['G1_before'], 4), f"({r['G1_before_count']})", 'after', r['G1_after'],
               f"({r['G1_after_count']}; unchanged-score reading {r['G1_after_count_unchanged_score_reading']})", '| G3 rows changed fs', round(r['G3_rows_changed']['fs_share'], 4), 'fz',
               round(r['G3_rows_changed']['fz_share'], 4), 'feeds', round(r['G3_feeds_changed']['any_share'], 4))
            lg('     SMD fs (before, after): ' + ' | '.join(f'{k_[5:]}: {v_}' for k_, v_ in smd.items() if k_.startswith('fs') and ('private' in k_ or 'public' in k_) and 'M50' not in k_))
            lg('     SMD fs vs private_M50 (review def): ' + ' | '.join(f'{k_[5:]}: {v_}' for k_, v_ in smd.items() if k_.startswith('fs') and 'M50' in k_))
            lg('     SMD fz (before, after): ' + ' | '.join(f'{k_[5:]}: {v_}' for k_, v_ in smd.items() if k_.startswith('fz') and 'private' in k_ and 'M50' not in k_))
            lg(f"     G4: P(new-head ID | cold) {r['G4_P_new_head_id_given_cold']:.4f}, P(hmax32 ID | cold) {r['G4_P_hmax32_id_given_cold']:.4f}, P(seq_cv_pop > 0 | cold) "
               f"{r['G4_P_seq_cv_pop_pos_given_cold']:.4f} (as built {r['G4_P_seq_cv_pop_pos_given_cold_as_built']:.4f}), P(xp_imp_c50 > 0 | cold) {r['G4_P_xp_imp_c50_pos_given_cold']:.4f}")
        res['views'][dn] = rv
        json.dump(res, open(f'{HERE}/gate_full.json', 'w'), indent=1)
        del V, keep, eff, cold, nc, nfe, nfs, nfz, nhas, still, unchanged_id, ch_fs, ch_fz, ch_has, fzc_b, fzc_a, pop_v, xp_v
    lg('GATE DONE')


if __name__ == '__main__':
    if sys.argv[1] == 'offset':
        run_offset()
    else:
        dirs = sys.argv[2:] or [d for d in ('feats_n', 'feats_Ln', 'feats_hn', 'feats_nP', 'feats_LnP', 'feats_hnP') if os.path.exists(f'{HERE}/{d}/train/pre2_hz_fe_z.npy')]
        run_full(dirs)
