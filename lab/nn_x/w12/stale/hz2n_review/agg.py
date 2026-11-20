"""Aggregates r2b_full_{a,b,c}.json + targets.json into the gate recount (G1, G2, G3) and the N items. Pure arithmetic on my own counts."""
import json, sys
import numpy as np
O = '<repo>/nn_x/w12/stale/hz2n_review'
T = json.load(open(f'{O}/targets.json'))
out = {}
lines = []
P = lambda s: (lines.append(s), print(s))
fc = np.arange(16)
GR = {'all': np.ones(16, bool), 'ge13': (fc // 2) % 2 == 1, 'pool13': fc % 2 == 1, 'win1': fc // 4 == 1, 'win2': fc // 4 == 2, 'win3': fc // 4 == 3, 'nowin': fc // 4 == 0,
      'win2_ge13': (fc // 4 == 2) & ((fc // 2) % 2 == 1), 'win2_lt13': (fc // 4 == 2) & ((fc // 2) % 2 == 0)}
POP = {'W': [0], 'D': [1], 'B': [2], 'E': [3], 'A': [1, 2, 3], 'NC': [2, 3], 'ALL': [0, 1, 2, 3]}
tp, tpub, tpm = T['private'], T['public'], T['private_M50']
for v in ('a', 'b', 'c'):
    try:
        J = json.load(open(f'{O}/r2b_full_{v}.json'))
    except FileNotFoundError:
        continue
    A = {q: np.array(x) for q, x in J['acc'].items()}
    C = J['counters']
    def s(q, g, pop, has=(0, 1)):
        return float(A[q][GR[g]][:, POP[pop]][:, :, list(has)].sum())
    o = {'counters_nonzero_violations': {k: x for k, x in C.items() if (k.startswith(('r1_', 'r2_', 'r3_', 'r4_', 'r5_', 'r6_', 'F_', 'FS_')) and x != 0)}}
    o['counters'] = C
    P(f"\n===== view {v} ({J['cfg']['d']}): rows {C['rows']}, nc {C['nc']} ({C['nc'] / C['rows']:.4f}), keep {C['keep']}, effective cold {C['cold_eff']} ({C['cold_eff'] / C['rows']:.4f}), stored cold {C['S0']}; violations {o['counters_nonzero_violations']}")
    P(f"   (V > S rows {C['V_gt_S']}, keep & V != S {C['keep_and_Vneq_S']}, V > 0 & S == 0: {C['V_pos_S0']}; keep rows with a score {C['keep_rows_with_score']}; idc != 0 & S == 0: {C['idc_ne0_and_S0']}; nc with fe finite / noid NaN {C['nc_fe_finite_noid_nan']})")
    fn, fch = np.array(J['feed_n']), np.array(J['feed_changed'])
    g1 = {}
    P('   G1 / G3     group   cold share  G1 before  G1 after(count)  after2(count)  chg fe_s  chg fe_z  feeds chg   B share of rows  E share of rows')
    for g in ('all', 'ge13', 'pool13', 'win1', 'win2', 'win3'):
        n_all, n_cold = s('n', g, 'ALL'), s('n', g, 'A')
        r = dict(cold_share=n_cold / n_all, before=s('n', g, 'B') / n_cold, after=s('id_after', g, 'A') / n_cold, after_count=int(s('id_after', g, 'A')), after2_count=int(s('id_after2', g, 'A')),
                 id_after_warm_rows=int(s('id_after', g, 'W')), chg_fs=s('fe_s_changed', g, 'ALL') / n_all, chg_fz=s('fe_z_changed', g, 'ALL') / n_all, feeds=float(fch[GR[g]].sum() / fn[GR[g]].sum()),
                 B_share=s('n', g, 'B') / n_all, E_share=s('n', g, 'E') / n_all, E_share_with_score=s('n', g, 'E', (1,)) / n_all, n_rows=int(n_all), n_feeds=int(fn[GR[g]].sum()))
        g1[g] = r
        P(f"   {g:>18}   {r['cold_share']:.4f}     {r['before']:.4f}     {r['after']:.6f} ({r['after_count']})   ({r['after2_count']})    {r['chg_fs']:.4f}    {r['chg_fz']:.4f}   {r['feeds']:.4f}      {r['B_share']:.4f}          {r['E_share']:.4f}")
    o['G1'] = g1
    g2 = {}
    P('   G2 pre2_fe_s SMD vs PRIVATE cold rows [vs public], unit = all-row SD of the target; rows with a score only')
    P('     group    pop      n rows     before -> after   [public before -> after]   | pre2_fe_z (clip 5) mean before -> after (private cold mean %.4f)' % tp['z_cold_mean'])
    for g in ('ge13', 'pool13', 'all', 'win1', 'win2', 'win3'):
        for pop in ('B', 'E', 'NC', 'A', 'D'):
            n = s('n', g, pop, (1,))
            if n == 0:
                continue
            mo, mn = s('fs_old', g, pop, (1,)) / n, s('fs_new', g, pop, (1,)) / n
            tgt = tpm if g == 'pool13' else tp
            r = dict(n=int(n), mean_old=mo, mean_new=mn, smd_old=(mo - tp['fs_cold_mean']) / tp['fs_all_sd'], smd_new=(mn - tp['fs_cold_mean']) / tp['fs_all_sd'],
                     smd_old_M50=(mo - tpm['fs_cold_mean']) / tpm['fs_all_sd'], smd_new_M50=(mn - tpm['fs_cold_mean']) / tpm['fs_all_sd'],
                     smd_old_pub=(mo - tpub['fs_cold_mean']) / tpub['fs_all_sd'], smd_new_pub=(mn - tpub['fs_cold_mean']) / tpub['fs_all_sd'],
                     z_old=s('z_old', g, pop, (1,)) / n, z_new=s('z_new', g, pop, (1,)) / n, dfe=s('dfe', g, pop, (1,)) / n, dfs_sd=s('dfs', g, pop, (1,)) / n / tp['fs_all_sd'],
                     absdz=s('absdz', g, pop, (1,)) / n)
            g2[f'{g}|{pop}'] = r
            P(f"     {g:>7}  {pop:>3}  {int(n):>11}   {r['smd_old']:+.3f} -> {r['smd_new']:+.3f}   [{r['smd_old_pub']:+.3f} -> {r['smd_new_pub']:+.3f}]  (vs private M>=50: {r['smd_old_M50']:+.3f} -> {r['smd_new_M50']:+.3f})"
              f"  | z {r['z_old']:+.4f} -> {r['z_new']:+.4f}  | mean d(pre2_fe) {r['dfe']:+.4f}, d(fe_s) {r['dfs_sd']:+.4f} SD")
    o['G2'] = g2
    # N1: E rows
    n1 = {}
    P('   N1 rows made cold WITHOUT an ID contribution (E): the as-asked view changes them too (test: fe == noid on such rows, shift 0)')
    for g in ('win1', 'win2', 'win3', 'ge13', 'pool13', 'all'):
        n = s('n', g, 'E', (1,)); nall = s('n', g, 'ALL')
        if n == 0: continue
        r = dict(n=int(n), share_of_rows=n / nall, mean_dfe=s('dfe', g, 'E', (1,)) / n, shift_sd=s('dfs', g, 'E', (1,)) / n / tp['fs_all_sd'], dz=(s('z_new', g, 'E', (1,)) - s('z_old', g, 'E', (1,))) / n)
        r['index'] = abs(r['shift_sd']) * r['share_of_rows']
        nb = s('n', g, 'B', (1,))
        r['B_share_of_rows'] = nb / nall; r['B_smd_before'] = (s('fs_old', g, 'B', (1,)) / nb - tp['fs_cold_mean']) / tp['fs_all_sd']; r['B_removed_sd'] = -s('dfs', g, 'B', (1,)) / nb / tp['fs_all_sd']
        r['I0_declared'] = abs(r['B_smd_before']) * r['B_share_of_rows']; r['I0_removed'] = abs(r['B_removed_sd']) * r['B_share_of_rows']
        r['B_dz'] = (s('z_new', g, 'B', (1,)) - s('z_old', g, 'B', (1,))) / nb
        n1[g] = r
        P(f"     {g:>7}: E rows {int(n):>10} = {r['share_of_rows']:.4f} of rows; mean d(pre2_fe) {r['mean_dfe']:+.4f} = {r['shift_sd']:+.4f} target SD; mean dz {r['dz']:+.4f}; index {r['index']:.5f} "
          f"| B rows {r['B_share_of_rows']:.4f} of rows, SMD before {r['B_smd_before']:+.3f}, removed by the fix {r['B_removed_sd']:+.3f} SD, dz {r['B_dz']:+.4f}; I0 declared {r['I0_declared']:.5f}, I0(removed) {r['I0_removed']:.5f} "
          f"-> ratio {r['index'] / r['I0_declared']:.2f} / {r['index'] / r['I0_removed']:.2f}")
    o['N1'] = n1
    # N3 new head (F files): share of B / E rows whose pre2_new changes; SMD of new_s
    n3 = {}
    for g in ('ge13', 'pool13', 'all'):
        for pop in ('B', 'E', 'A', 'D'):
            n = s('n', g, pop, (1,))
            mo, mn = s('news_old', g, pop, (1,)) / n, s('news_new', g, pop, (1,)) / n
            n3[f'{g}|{pop}'] = dict(new_changed_share=s('new_changed', g, pop, (1,)) / n, smd_old=(mo - tp['news_cold_mean']) / tp['news_all_sd'], smd_new=(mn - tp['news_cold_mean']) / tp['news_all_sd'],
                                    newz_old=s('newz_old', g, pop, (1,)) / n, newz_new=s('newz_new', g, pop, (1,)) / n)
    o['N3'] = n3
    P('   N3 new head (F files): share of rows whose pre2_new bits change / pre2_new_s SMD vs private cold before -> after F')
    for kk in ('ge13|B', 'ge13|E', 'ge13|A', 'ge13|D', 'pool13|B', 'pool13|A', 'pool13|D', 'all|B', 'all|D'):
        r = n3[kk]; P(f"     {kk:>10}: changed {r['new_changed_share']:.4f}; SMD {r['smd_old']:+.3f} -> {r['smd_new']:+.3f}; new_z mean {r['newz_old']:+.4f} -> {r['newz_new']:+.4f} (private cold {tp['newz_cold_mean']:+.4f})")
    # N4 seq_cv on cold rows
    n4 = {}
    for g in ('all', 'ge13', 'pool13'):
        for pop in ('A', 'NC', 'D'):
            n = s('n', g, pop)
            n4[f'{g}|{pop}'] = dict(pop_pos=s('scv_pop_pos', g, pop) / n, invd_pos=s('scv_invd_pos', g, pop) / n, norm_pos=s('scv_norm_pos', g, pop) / n, pop_pos_FS=s('scv_pop_pos_FS', g, pop) / n, n=int(n))
    o['N4'] = n4
    P('   N4 P(seq_cv_* > 0 | effective cold) through the as-asked view [pop / invd20 / norm] and seq_cv_pop after FS (view c only)')
    for kk in ('all|A', 'all|NC', 'all|D', 'ge13|A', 'pool13|A'):
        r = n4[kk]; P(f"     {kk:>10}: {r['pop_pos']:.4f} / {r['invd_pos']:.4f} / {r['norm_pos']:.4f}   FS: {r['pop_pos_FS']:.4f}")
    # N5 hmax32
    n5 = {}
    P('   N5 pre2_hmax32 (untouched model input, clip 10): SMD vs private cold rows (unit all-row SD %.4f)' % tp['hm_all_sd'])
    for g in ('ge13', 'pool13', 'all'):
        for pop in ('B', 'E', 'D', 'A', 'W'):
            n = s('n', g, pop, (1,)); m = s('hm', g, pop, (1,)) / n
            n5[f'{g}|{pop}'] = dict(mean=m, smd=(m - tp['hm_cold_mean']) / tp['hm_all_sd'], n=int(n))
        P(f"     {g:>7}: " + '  '.join(f"{pop} {n5[f'{g}|{pop}']['smd']:+.3f}" for pop in ('B', 'E', 'D', 'A', 'W')) + f"   (B - D: {n5[f'{g}|B']['smd'] - n5[f'{g}|D']['smd']:+.3f} SD; share B {s('n', g, 'B') / s('n', g, 'ALL'):.4f})")
    o['N5'] = n5
    out[v] = o
json.dump(out, open(f'{O}/agg.json', 'w'), indent=1)
open(f'{O}/agg.txt', 'w').write('\n'.join(lines) + '\n')
