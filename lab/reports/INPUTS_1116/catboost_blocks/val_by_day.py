"""VAL BY DAY: the single-column reads of cols_read.py split by val day (Tue 09-22 = 2 d after the dense cutoff, Thu 09-24 = 4 d) and by M.
Composite coefficients from ALL holdout feeds (train-period labels); the 2-parameter mix is fitted on the val subset read.  -> out/val_by_day.txt"""
from cbi_common import *
import json
meta = Meta(); WV = Wit(meta, 'val'); WH = Wit(meta, 'hold')
FB = ROOT / 'nn_x/w12/cb/famB'; FBN = json.load(open(FB / 'feat/c_names.json'))
def feat(name, w):
    x = np.load(FEATS / ('val' if w.sp == 'val' else 'train') / f'{name}.npy', mmap_mode='r'); return np.asarray(x if w.rows is None else x[w.rows], np.float32)
def kn(name, w):
    a = np.load(FB / f"feat/c_{'val' if w.sp == 'val' else 'hold'}.npy", mmap_mode='r'); return np.asarray(a[:, FBN.index(name)], np.float32)
day = (WV.ts.astype(np.int64) + 3 * 3600) // 86400; days = np.unique(day); wd = {d: ['Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed'][int(d) % 7] for d in days}
subs = {f'{wd[d]} (day {int(d)})': day == d for d in days}; subs['M < 50'] = WV.M < 50; subs['M >= 50'] = WV.M >= 50; subs['all val'] = np.ones(WV.nf, bool)
lines = []
for c, ld in (('xp_ctr_c', feat), ('tte_it_coec', feat), ('tte_it_ctr', feat), ('kn_cos1', kn)):
    Xh = WH.zcol(ld(c, WH)); bh, _ = clogit(np.column_stack([WH.zw16, Xh]), WH, None, ridge=1e-6); comp = WV.zcol(ld(c, WV)) * bh[1]
    for k, m in subs.items():
        d, b2 = nats(comp, WV.zw16, WV, m); s = f'{c:12s} | {k:16s} | feeds {int(m.sum()):5d} | NATS {f2(d)} | beta {b2[1]:+.2f}'
        print(s, flush=True); lines.append(s)
open(CBI_OUT + '/val_by_day.txt', 'w').write('\n'.join(lines) + '\n')
