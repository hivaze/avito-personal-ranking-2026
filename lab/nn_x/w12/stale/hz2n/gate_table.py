"""Compact gate table from gate_full.json + gate_new.json + gate_g4_extra.json -> gate_table.txt"""
import json
H = '<repo>/nn_x/w12/stale/hz2n'
g, gn, gx = json.load(open(f'{H}/gate_full.json')), json.load(open(f'{H}/gate_new.json')), json.load(open(f'{H}/gate_g4_extra.json'))
L = []
t = g['targets']
L.append('TARGETS (as built) [M]: P(pre2 ID contribution != 0 | it_hist_all == 0): val %d of %d, public %d of %d, private %d of %d cold rows; P(new-head ID | cold) %.4f / %.4f / %.4f; '
         'P(seq_cv_pop > 0 | cold) %.4f / %.4f / %.4f; pre2_fe_s cold mean (all-row SD): val %.4f (%.4f), public %.4f (%.4f), private %.4f (%.4f)' % (
             t['val']['G1_cold_and_id'], t['val']['cold_rows'], t['public']['G1_cold_and_id'], t['public']['cold_rows'], t['private']['G1_cold_and_id'], t['private']['cold_rows'],
             t['val']['P_new_head_id_given_cold'], t['public']['P_new_head_id_given_cold'], t['private']['P_new_head_id_given_cold'],
             t['val']['P_seq_cv_pop_pos_given_cold'], t['public']['P_seq_cv_pop_pos_given_cold'], t['private']['P_seq_cv_pop_pos_given_cold'],
             t['val']['fs_cold'][1], t['val']['fs_all'][2], t['public']['fs_cold'][1], t['public']['fs_all'][2], t['private']['fs_cold'][1], t['private']['fs_all'][2]))
GR = [('all days', 'all'), ('days >= Sep 13', '>=13'), ('review pop (trPool private-like >= Sep 13)', 'pool')]
L.append('')
L.append('G1 / G3 (fe head). cold = effective view counter == 0. before -> after = P(fe-head ID contribution != 0 | cold), count after in (). rows chg = share of rows whose pre2_fe_s / pre2_fe_z bits change; feeds = share of feeds with a changed row.')
L.append('%-10s %-5s %10s %9s %22s %9s %9s %7s' % ('dir', 'days', 'cold share', 'G1 before', 'G1 after (count)', 'chg fe_s', 'chg fe_z', 'feeds'))
for dn, v in g['views'].items():
    for gk, gs in GR:
        r = v['groups'][gk]
        L.append('%-10s %-5s %10.4f %9.4f %12.6f (%6d) %9.4f %9.4f %7.4f' % (dn, gs, r['cold_share'], r['G1_before'], r['G1_after'], r['G1_after_count'], r['G3_rows_changed']['fs_share'],
                                                                           r['G3_rows_changed']['fz_share'], r['G3_feeds_changed']['any_share']))
L.append('')
L.append('G2 (fe head): standardised mean difference of pre2_fe_s vs PRIVATE cold rows (public in []), before -> after. A = all view-cold rows with a score, B = made cold AND ID-carrying, '
         'C = made cold, no ID, D = as-built cold (never touched).')
L.append('%-10s %-5s %-34s %-34s %-34s %-16s' % ('dir', 'days', 'B', 'A', 'C', 'D'))
for dn, v in g['views'].items():
    for gk, gs in GR:
        s = v['groups'][gk]['SMD_before_after']
        f = lambda pn: '%+.3f -> %+.3f [%+.3f -> %+.3f]' % (*s[f'fs | {pn} | vs private'], *s[f'fs | {pn} | vs public'])  # noqa: E731
        L.append('%-10s %-5s %-34s %-34s %-34s %+.3f [%+.3f]' % (dn, gs, f('B made cold, with ID'), f('A all view-cold with score'), f('C made cold, no ID'),
                                                             s['fs | D as-built cold | vs private'][0], s['fs | D as-built cold | vs public'][0]))
L.append('')
L.append('G2 for pre2_fe_z (clipped +-5) vs PRIVATE cold rows, before -> after:')
for dn, v in g['views'].items():
    for gk, gs in GR:
        s = v['groups'][gk]['SMD_before_after']
        L.append('%-10s %-5s B %+.3f -> %+.3f | A %+.3f -> %+.3f | C %+.3f -> %+.3f | D %+.3f -> %+.3f' % (dn, gs, *s['fz | B made cold, with ID | vs private'], *s['fz | A all view-cold with score | vs private'],
                                                                                              *s['fz | C made cold, no ID | vs private'], *s['fz | D as-built cold | vs private']))
L.append('')
L.append('G4 (not repaired by the as-asked / P files): P(NEW-head ID contribution | cold), P(seq_cv_pop > 0 | cold), P(xp_imp_c50 > 0 | cold); targets 0 / 0 / 0.18-0.31')
for dn in ('feats_n', 'feats_Ln', 'feats_hn'):
    for gk, gs in GR:
        r = g['views'][dn]['groups'][gk]
        L.append('%-10s %-5s new-head ID %.4f | seq_cv_pop > 0 %.4f | xp_imp_c50 > 0 %.4f | hmax32 ID-dependent %.4f (targets 0.98-0.99)' % (
            dn, gs, r['G4_P_new_head_id_given_cold'], r['G4_P_seq_cv_pop_pos_given_cold'], r['G4_P_xp_imp_c50_pos_given_cold'], r['G4_P_hmax32_id_given_cold']))
L.append('')
L.append('F family, NEW head: G1n = P(new-head ID contribution | cold) before -> after (count); rows chg; SMD of pre2_new_s vs private [public] before -> after for B and A')
for dn, v in gn['views'].items():
    for gk, gs in GR:
        r = v[gk]; s = r['SMD_before_after']
        L.append('%-10s %-5s G1n %.4f -> %.6f (%d) | chg new_s %.4f new_z %.4f | B %+.3f -> %+.3f [%+.3f -> %+.3f] | A %+.3f -> %+.3f [%+.3f -> %+.3f] | C %+.3f | D %+.3f' % (
            dn, gs, r['G1n_before'], r['G1n_after'], r['G1n_after_count'], r['G3n_rows_changed']['new_s'], r['G3n_rows_changed']['new_z'], *s['new_s B vs private'], *s['new_s B vs public'],
            *s['new_s A vs private'], *s['new_s A vs public'], s['new_s C vs private'][0], s['new_s D vs private'][0]))
L.append('')
L.append('View c co-visitation (gate_g4_extra.json): P(x > 0 | view cold) / P(x > 0 | made cold): ' + '; '.join(f'{k} {a:.4f} / {b:.4f}' for k, (a, b) in gx['feats_hn'].items())
         + ' | targets: seq_cv_* 0.0000 exactly; xp_imp_c50 val %.3f public %.3f private %.3f' % (gx['val']['xp_imp_c50'], gx['public']['xp_imp_c50'], gx['private']['xp_imp_c50']))
open(f'{H}/gate_table.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
