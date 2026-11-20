"""Compact tables from cache/gate_<tag>.json (no new computation)."""
import json, sys
from hz2_common import *
TAG = sys.argv[1] if len(sys.argv) > 1 else 'final'
J = json.load(open(CACHE / f'gate_{TAG}.json')); R = J['res']; ref = J['ref']
PRIM = ['it_hist_all_luo', 'seq_cv_pop', 'xp_imp_c50']; SEC = ['it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'seq_cv_invd20', 'seq_cv_norm']
L = []
def P(s=''):
    print(s); L.append(s)
views = [v for v in ('asbuilt', 'hz', 'hz2', 'hz2k', 'hz2L', 'hz2Lk') if v in R['trPool|priv|all']]
for pop in ('trPool', 'trHold', 'trAll'):
    for part in ('priv', 'pub'):
        for sub in ('all', 'late', 'Sep 6-8', 'Sep 9-12', 'Sep 13-14', 'Sep 17-19'):
            r = R[f'{pop}|{part}|{sub}']
            P(f'\n## {pop} | {part}-like | days {sub} | rows {r["hz2"][PRIM[0]]["rows"]:.0f}')
            P('| view | it_hist_all dZ / bucket / dM | seq_cv_pop dZ / bucket / dM | xp_imp_c50 dZ / bucket / dM | secondary max dZ / max dM | G1-G5 | verdict |')
            P('|---|---|---|---|---|---|---|')
            for v in views:
                x = r[v]
                cells = [f'{x[c]["dZ"]:.4f} / {x[c]["big"]:.4f} / {x[c]["dM"]:.4f}' for c in PRIM]
                sec = f'{max(x[c]["dZ"] for c in SEC):.4f} / {max(x[c]["dM"] for c in SEC):.4f}'
                if v == 'asbuilt':
                    g = '-'; vd = '-'
                else:
                    q = r['verdict'][v]; g = ' '.join(('ok' if q[k] else 'FAIL') for k in ('G1', 'G2', 'G3', 'G4', 'G5')); vd = 'PASS' if q['PASS'] else 'FAIL'
                P(f'| {v} | {cells[0]} | {cells[1]} | {cells[2]} | {sec} | {g} | {vd} |')
open(LOGS / f'gate_summary_{TAG}.md', 'w').write('\n'.join(L) + '\n')
