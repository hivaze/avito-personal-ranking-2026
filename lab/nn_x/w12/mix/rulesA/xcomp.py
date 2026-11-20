"""Cross-composition table of the nested search (reads res_C*.pkl)."""
import pickle, sys
from lib import *
R = {c: pickle.load(open(HERE / f'res_{c}.pkl', 'rb')) for c in COMPS}; fams = families(); names = R['C1']['names']
iM = names.index('C(M50)'); top = int(sys.argv[1]) if len(sys.argv) > 1 else 100
order = np.argsort(-R['C1']['nested'])[:top]
print(f'{"family":16s}|' + '|'.join(f' {c}: ins / nested-plain(z) / nested-C(M50)   ' for c in COMPS) + '| best rule on all val C1 ; C3')
for fi in order:
    row = f'{names[fi]:16s}|'
    for c in COMPS:
        r = R[c]; d = r['vec'][fi] - r['plain_vec']; m, s = se(d); dm = r['vec'][fi] - r['vec'][iM]
        row += f' {r["ins"][fi] - r["plain"]:+.5f} / {m:+.5f} ({m / s:+.1f}) / {dm.mean():+.5f} |'
    row += ' ' + ' '.join(str(decode(x)) for x in R['C1']['ins_rule'][fi]) + '  ;  ' + ' '.join(str(decode(x)) for x in R['C3']['ins_rule'][fi])
    print(row)
for v in ('B-all', 'B-3', 'G'):
    row = f'{"VAR " + v:16s}|'
    for c in COMPS:
        r = R[c]; d = r['var'][v]['vec'] - r['plain_vec']; m, s = se(d); dm = r['var'][v]['vec'] - r['vec'][iM]
        row += f'    --    / {m:+.5f} ({m / s:+.1f}) / {dm.mean():+.5f} |'
    print(row)
row = f'{"fixed MRULE":16s}|'
for c in COMPS:
    r = R[c]; d = r['mrule_vec'] - r['plain_vec']; m, s = se(d); row += f'    --    / {m:+.5f} ({m / s:+.1f}) / {(r["mrule_vec"] - r["vec"][iM]).mean():+.5f} |'
print(row)
