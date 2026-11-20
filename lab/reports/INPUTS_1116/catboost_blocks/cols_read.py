"""COLS: single-column drill-down of the blocks that read above the noise null in blocks_read.py (item-level label / exposure statistics of
tte_* and W25, the W25 columns excluded by PSI, the density columns).  Same protocol as blocks_read.py with a 1-column (+ missing flag) design:
NATS over the W16 train-only pool, cross-fitted on user halves inside each witness; val also with the coefficients fitted on all holdout feeds.
Output: out/cols_read.txt (+ .json).   32 columns are read here: see noise_null.txt for what the best of that many noise columns reads."""
from cbi_common import *
import json, time
T0 = time.time()
meta = Meta(); WV = Wit(meta, 'val'); WH = Wit(meta, 'hold')
R25 = ROOT / 'nn_x/w25'; FB = ROOT / 'nn_x/w12/cb/famB'
rows25 = np.load(R25 / 'data/train/rows.npy', mmap_mode='r'); pos25 = np.searchsorted(rows25, WH.rows); assert (np.asarray(rows25[pos25]) == WH.rows).all()
G = json.load(open(R25 / 'data/groups.json'))
FBN = {b: json.load(open(FB / f'feat/{b}_names.json')) for b in 'cd'}
def feat(name, w):
    p = FEATS / ('val' if w.sp == 'val' else 'train') / f'{name}.npy'; x = np.load(p, mmap_mode='r'); return np.asarray(x if w.rows is None else x[w.rows], np.float32)
def w25(name, w):
    if w.sp == 'val': return np.load(R25 / f'data/val/{name}.npy')
    return np.asarray(np.load(R25 / f'data/train/{name}.npy', mmap_mode='r')[pos25], np.float32)
def famb(block, c, w):
    a = np.load(FB / f"feat/{block}_{'val' if w.sp == 'val' else 'hold'}.npy", mmap_mode='r'); return np.asarray(a[:, FBN[block].index(c)], np.float32)
COLS = [(c, feat) for c in ('tte_it_imp', 'tte_it_pos', 'tte_it_ctr', 'tte_it_coec', 'tte_c_ctr', 'tte_uc_ctr', 'tte_uc_imp', 'tte_ui_imp', 'tte_ui_pos', 'xp_imp_c50', 'xp_ctr_c', 'it_hist_all_luo')] \
     + [(c, w25) for c in G['G3'] if c.startswith('w25_it_')] + [(c, w25) for c in G['G_excl']] \
     + [(c, lambda n, w: famb('c', n, w)) for c in ('kn_cos1', 'kn_cos20', 'kn_ex20')] + [(c, lambda n, w: famb('d', n, w)) for c in ('kn_cos2', 'dup_n97')]
lines = []; res = {}
def say(s=''):
    print(s, flush=True); lines.append(s)
say(f'{len(COLS)} columns | column | NaN share hold / val | holdout NATS xfit, beta | val NATS xfit, beta | val with holdout-fitted coefficients: NATS, dNDCG, beta')
half = {w.sp: w.halves(0) for w in (WV, WH)}
for c, ld in COLS:
    out = {}; bh = None; nm = None
    for w in (WH, WV):
        x = np.asarray(ld(c, w), np.float64); ok = np.isfinite(x); X = [w.zcol(x)]; names = ['v']
        if 0.005 < 1 - ok.mean() < 0.995:
            zi = w.zcol(ok.astype(np.float64))
            if zi.any(): X.append(zi); names.append('has')
        if nm is not None and names != nm:
            if len(names) < len(nm): X.append(np.zeros(w.n))
            else: X = X[:len(nm)]
        nm = nm or names; X = np.column_stack(X); d = np.zeros(w.nf); bs = []
        for h in (True, False):
            fit = half[w.sp] == h; b, _ = clogit(np.column_stack([w.zw16, X]), w, fit, ridge=1e-6); dd, b2 = nats(X @ b[1:], w.zw16, w, ~fit); d[~fit] = dd; bs.append(float(b2[1]))
        out[w.sp] = dict(nan=float(1 - ok.mean()), nats=sd_se(d), beta=float(np.mean(bs)))
        if w.sp == 'hold': bh, _ = clogit(np.column_stack([w.zw16, X]), w, None, ridge=1e-6); out['coef_hold'] = [float(v) for v in bh[1:]]
        else:
            comp = X @ bh[1:]; dd, b2 = nats(comp, w.zw16, w); mix = w.ndcg(b2[0] * w.zw16 + b2[1] * comp) - w.ndcg(w.zw16)
            out['val_holdcoef'] = dict(nats=sd_se(dd), dndcg=sd_se(mix), beta=float(b2[1]))
    res[c] = out; h, v, vh = out['hold'], out['val'], out['val_holdcoef']
    say(f"{c:22s} | NaN {h['nan']:.3f} / {v['nan']:.3f} | hold {h['nats'][0]:+.5f} +- {h['nats'][1]:.5f}, b {h['beta']:+.2f} | val {v['nats'][0]:+.5f} +- {v['nats'][1]:.5f}, b {v['beta']:+.2f} | val(hold coef) {vh['nats'][0]:+.5f} +- {vh['nats'][1]:.5f}, dNDCG {vh['dndcg'][0]:+.5f} +- {vh['dndcg'][1]:.5f}, b {vh['beta']:+.2f} | holdout coefficients {['%+.4f' % x for x in out['coef_hold']]}")
    json.dump(res, open(CBI_OUT + '/cols_read.json', 'w'), indent=1)
open(CBI_OUT + '/cols_read.txt', 'w').write('\n'.join(lines) + f'\n# elapsed {time.time() - T0:.0f} s\n')
