import json, sys
O = '<repo>/nn_x/w12/stale/hz2n_review/r1c'
L = lambda n: json.load(open(f'{O}/{n}.json'))
out = {}
ok = True
for e in ('E0', 'E1', 'E2'):
    a, b = L(f'{e}_6b'), L(f'{e}_6c')
    r = {'HZ_DIR': a['HZ_DIR'], 'KEEP': a['KEEP_PATH'][-40:], 'n_feeds': a['n_feeds'], 'n_rows': a['n_rows']}
    for tag in ('cvnd_pre2s', 'poplight'):
        x, y = a[tag], b[tag]
        eq = {k: x.get(k) == y.get(k) for k in set(x) | set(y)}
        r[tag] = {'all_equal': all(eq.values()), 'unequal_keys': [k for k, v in eq.items() if not v], 'cols': [c[1] for c in x.get('cols', [])], 'assert': x.get('luo_cols_assert'),
                  'n_keep': x.get('n_keep'), 'cells': x.get('cells'), 'batches': x.get('batches'), 'cells_changed_vs_plain': x.get('cells_changed_vs_plain')}
        ok &= all(eq.values())
    out[e] = r
out['R1c_pass'] = ok
# builder directories through 6c: popularity part equal to the base view?
base = {'N_feats_n': 'E1_6b', 'N_feats_nF': 'E1_6b', 'N_feats_Ln': 'E2_6b', 'N_feats_hn': 'E0_6b', 'N_feats_hnF': 'E0_6b', 'N_feats_hnFS': 'E0_6b'}
for n, bname in base.items():
    a, b = L(n), L(bname)
    r = {}
    for tag in ('cvnd_pre2s', 'poplight'):
        x, y = a[tag], b[tag]
        if 'cols' not in x:
            r[tag] = {'assert': x.get('luo_cols_assert')}; continue
        sx = {s['luo']: s for s in x['hz_stats']}; sy = {s['luo']: s for s in y.get('hz_stats', [])}
        common = [k for k in sx if k in sy]
        r[tag] = {'n_cols': len(x['cols']), 'cols': [c[1] for c in x['cols']], 'hz_names': [c[2] for c in x['cols']], 'n_keep': x['n_keep'],
                  'base_cols_stats_equal': all(sx[k] == sy[k] for k in common), 'n_common': len(common),
                  'new_col_stats': {k: {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in sx[k].items() if kk in ('zero_luo', 'zero_hz', 'changed')} for k in sx if k not in sy},
                  'cells': x['cells'], 'cells_changed_vs_plain': x['cells_changed_vs_plain']}
    out[n] = r
json.dump(out, open(f'{O}/../r1c_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
