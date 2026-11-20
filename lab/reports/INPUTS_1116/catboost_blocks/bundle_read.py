"""BUNDLE: are the two surviving reads (item click rate, catalogue density) additive next to W16?  Same protocol as blocks_read.py.
Also on val M >= 50 and with holdout-fitted coefficients.  -> out/bundle_read.txt"""
from cbi_common import *
import json
meta = Meta(); WV = Wit(meta, 'val'); WH = Wit(meta, 'hold')
FB = ROOT / 'nn_x/w12/cb/famB'; FBN = json.load(open(FB / 'feat/c_names.json'))
def feat(name, w):
    x = np.load(FEATS / ('val' if w.sp == 'val' else 'train') / f'{name}.npy', mmap_mode='r'); return np.asarray(x if w.rows is None else x[w.rows], np.float32)
def kn(name, w):
    a = np.load(FB / f"feat/c_{'val' if w.sp == 'val' else 'hold'}.npy", mmap_mode='r'); return np.asarray(a[:, FBN.index(name)], np.float32)
SETS = {'ctr (xp_ctr_c)': [('xp_ctr_c', feat)], 'ctr3 (tte_it_ctr, tte_it_coec, tte_it_pos)': [(c, feat) for c in ('tte_it_ctr', 'tte_it_coec', 'tte_it_pos')],
        'density (kn_cos1, kn_cos20)': [('kn_cos1', kn), ('kn_cos20', kn)],
        'ctr + density (xp_ctr_c, kn_cos1, kn_cos20)': [('xp_ctr_c', feat), ('kn_cos1', kn), ('kn_cos20', kn)],
        'ctr3 + density (5 columns)': [(c, feat) for c in ('tte_it_ctr', 'tte_it_coec', 'tte_it_pos')] + [('kn_cos1', kn), ('kn_cos20', kn)]}
lines = []
for k, cols in SETS.items():
    out = {}; bh = None
    for w in (WH, WV):
        X = np.column_stack([w.zcol(ld(c, w)) for c, ld in cols]); half = w.halves(0); d = np.zeros(w.nf); mx = np.zeros(w.nf)
        for h in (True, False):
            fit = half == h; b, _ = clogit(np.column_stack([w.zw16, X]), w, fit, ridge=1e-6); comp = X @ b[1:]; dd, b2 = nats(comp, w.zw16, w, ~fit); d[~fit] = dd
            mix = w.ndcg(b2[0] * w.zw16 + b2[1] * comp) - w.ndcg(w.zw16); mx[~fit] = mix[~fit]
        out[w.sp] = (f2(d), f2(mx))
        if w.sp == 'hold': bh, _ = clogit(np.column_stack([w.zw16, X]), w, None, ridge=1e-6)
        else:
            comp = X @ bh[1:]; dd, b2 = nats(comp, w.zw16, w); m50 = w.M >= 50; d50, b50 = nats(comp, w.zw16, w, m50)
            out['vh'] = (f2(dd), f'{b2[1]:+.2f}', f2(d50), f'{b50[1]:+.2f}')
    s = f"{k:46s} | hold NATS {out['hold'][0]}, dNDCG {out['hold'][1]} | val NATS {out['val'][0]}, dNDCG {out['val'][1]} | val, holdout coefficients: NATS {out['vh'][0]} (beta {out['vh'][1]}); val M >= 50: {out['vh'][2]} (beta {out['vh'][3]})"
    print(s, flush=True); lines.append(s)
open(CBI_OUT + '/bundle_read.txt', 'w').write('\n'.join(lines) + '\n')
