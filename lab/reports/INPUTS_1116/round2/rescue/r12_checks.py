"""RESCUE r12: checks of the D5 builds (written after r10 / r11; nothing here decides).
  A. independent re-derivation on a sample (plain numpy from the raw arrays; only the window rule is shared): imp5, coec5, gapf5, never5, bt5_seen
     (item is in a source feed of the fit), ns1_5 / ns2_5 (own user removed), uc5, reshown5. Label-free except the click counts of coec5 / uc5.
  B. the D5 natural-val columns against the stored full-depth columns of record (correlation, agreement of the default pattern).
  C. calibration of my use of the lens: the STORED columns of record on natural val under my null model (pool + own inputs + nx3): c06 theta,
     tte_it_coec, kn_cos1, X3 / n_s2 (V-lf), gap_first, tte_uc_ctr, als_exp5 (transductive).  [information; these are the records' columns]
  D. the test-like-only variant of the view (has = slot >= 11): C4 PSI of the forms on pool rows of slots 11-13 against public / private.
CPU only, read-only. Output out/r12_checks.txt (exclusive create)."""
from rs_common import *
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
import lens
import condlib
sys.argv = _argv
nb.set_num_threads(NT)
P = Log(OUT / 'r12_checks.txt')
V = Views(None)
a = V.A['train']
rg = np.random.default_rng(20261116)
CI = {s: np.load(FLAT / s / 'cand_item_id.npy') for s in SPLITS}
# ------------------------------------------------------------------ A. brute-force re-derivation
P('=== A. independent re-derivation on a sample ===')
SAMP = {}
for q, n in (('hold', 400), ('pool', 300), ('val_nat', 300), ('val_twin', 400), ('test_nat', 300)):
    rows, fl, off, M = V.rows(q); wk = V.row_wkey(q); cand = np.nonzero(wk > 0)[0]
    pick = np.sort(rg.choice(cand, n, replace=False)); sp, sel, cut = V.Q[q]
    SAMP[q] = dict(pick=pick, item=CI[sp][rows[pick]].astype(np.int64), cut=cut[fl[pick]], user=V.A[sp]['u'][sel][fl[pick]], frow=rows[pick])
items = np.unique(np.concatenate([s['item'] for s in SAMP.values()]))
fidx = np.load(FLAT / 'train/cand_fidx.npy'); y = np.load(FLAT / 'train/cand_target.npy'); pos = np.load(FLAT / 'train/cand_pos.npy')
slot_r_ok = (~V.ish & (S12[a['slot']] >= 0))
# position prior from non-holdout feeds of slots 0-4 (numpy)
m04 = np.repeat(slot_r_ok & (a['slot'] <= 4), a['M'])
pc = np.bincount(pos[m04], weights=y[m04].astype(np.float64), minlength=int(pos.max()) + 1) / np.maximum(np.bincount(pos[m04], minlength=int(pos.max()) + 1), 1)
del m04
mk = np.isin(CI['train'], items); rr = np.nonzero(mk)[0]; del mk
T = pl.DataFrame({'item': CI['train'][rr].astype(np.int64), 'f': fidx[rr].astype(np.int64), 'y': y[rr].astype(np.float64), 'ex': pc[pos[rr]]})
T = T.with_columns(slot=pl.Series(a['slot'][T['f'].to_numpy()]), hold=pl.Series(V.ish[T['f'].to_numpy()]), user=pl.Series(a['u'][T['f'].to_numpy()]))
praw = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
npos_f = np.add.reduceat(y.astype(np.int64), a['co'])
T = T.with_columns(btsrc=pl.Series(((praw > 0.5) & ~V.ish & (npos_f >= 1))[T['f'].to_numpy()]))
Tn = {c: T[c].to_numpy() for c in T.columns}
bad = {k: 0 for k in ('imp5', 'coec5', 'gapf5', 'never5', 'bt5_seen', 'ns1_5', 'ns2_5', 'uc5', 'reshown5')}; nchk = {k: 0 for k in bad}
# history rows of the sampled items (type 1, source 1|2)
hi = np.load(FLAT / 'train/hist_item_id.npy'); hk = np.nonzero(np.isin(hi, items))[0]
ht = np.load(FLAT / 'train/hist_event_type.npy', mmap_mode='r')[hk]; hs = np.load(FLAT / 'train/hist_event_source.npy', mmap_mode='r')[hk]; hf = np.load(FLAT / 'train/hist_fidx.npy', mmap_mode='r')[hk].astype(np.int64)
k2 = (np.asarray(ht) == 1) & ((np.asarray(hs) == 1) | (np.asarray(hs) == 2))
Hh = dict(item=hi[hk][k2].astype(np.int64), c=np.asarray(hs)[k2].astype(np.int64), f=hf[k2]); Hh['slot'] = a['slot'][Hh['f']]; Hh['u'] = a['u'][Hh['f']]
del hi, hk
cat_tr = np.load(FLAT / 'train/cand_category_id.npy', mmap_mode='r')
for q, s in SAMP.items():
    sp = V.Q[q][0]; G = {n: get(q, n)[s['pick']] for n in bad}
    cc = np.load(FLAT / sp / 'cand_category_id.npy', mmap_mode='r')
    for i in range(len(s['pick'])):
        it = s['item'][i]; w = window(int(s['cut'][i])); u = int(s['user'][i])
        m = (Tn['item'] == it) & ~Tn['hold'] & np.isin(Tn['slot'], w)
        imp = float(m.sum()); clk = Tn['y'][m].sum(); ex = Tn['ex'][m].sum()
        sl = Tn['slot'][m]; gapf = float(len(w) - int(np.nonzero(w == sl.min())[0][0])) if imp > 0 else np.nan
        seen = float(((Tn['item'] == it) & Tn['btsrc'] & np.isin(Tn['slot'], w)).any())
        mh = (Hh['item'] == it) & np.isin(Hh['slot'], w)
        ns = {}
        for c in (1, 2):
            mc = mh & (Hh['c'] == c); uk = np.where(Hh['u'][mc] > 0, Hh['u'][mc], -(Hh['f'][mc] + 1)); uq = np.unique(uk)
            ns[c] = float(len(uq) - (1 if (u > 0 and u in uq) else 0))
        for k, v in (('imp5', imp), ('coec5', (clk + 1) / (ex + 1)), ('gapf5', gapf), ('never5', float(imp == 0)), ('bt5_seen', seen), ('ns1_5', ns[1]), ('ns2_5', ns[2])):
            nchk[k] += 1
            if not (np.isclose(G[k][i], v, rtol=2e-4, atol=1e-5) or (np.isnan(v) and np.isnan(G[k][i]))):
                bad[k] += 1
        if i < 120:      # same-user columns: the user's non-holdout train feeds of the window, straight from the feeds table and the flat arrays
            fu = np.nonzero((a['u'] == u) & ~V.ish & np.isin(a['slot'], w))[0] if u > 0 else np.array([], np.int64)
            if len(fu) == 0:
                uc = np.nan; rs = 0.0
            else:
                rws = np.concatenate([np.arange(a['co'][f], a['co'][f] + a['M'][f]) for f in fu]); c_ = int(cc[s['frow'][i]])
                mcat = np.asarray(cat_tr[rws]) == c_; uc = (y[rws][mcat].sum() + 0.0159 * 20) / (mcat.sum() + 20.0); rs = float((CI['train'][rws] == it).any())
            for k, v in (('uc5', uc), ('reshown5', rs)):
                nchk[k] += 1
                if not (np.isclose(G[k][i], v, rtol=2e-4, atol=1e-6) or (np.isnan(v) and np.isnan(G[k][i]))):
                    bad[k] += 1
    P(f'  {q}: {len(s["pick"])} sampled rows checked')
P('  mismatches / rows checked: ' + ', '.join(f'{k} {bad[k]} / {nchk[k]}' for k in bad))
P(f'  RE-DERIVATION VERDICT: {"PASS" if sum(bad.values()) == 0 else "FAIL"}')
del fidx, y, pos, T, Tn, Hh
# ------------------------------------------------------------------ B. against the stored columns of record (natural val)
P('\n=== B. D5 natural-val columns against the stored full-depth columns of record ===')
c6v = np.load(ROOT / 'nn_x/insights/ccb_1113/scratch/tester3/cache/c06_val.npz'); cv = np.load(ROOT / 'nn_x/w12/cb/famB/feat/c_val.npy')
sv = np.load(ROOT / 'nn_x/codes/exploit/02_s1s2contrast/feats_val.npz'); tg = np.load(ROOT / 'nn_x/insights/timegap_1113/feats/val.npz')
STO = {'bt5': c6v['theta_l3'], 'bt5_seen': c6v['seen_l3'], 'coec5': np.load(ROOT / 'data/feats/val/tte_it_coec.npy'), 'kn1_5': cv[:, 5], 'kn20_5': cv[:, 4],
       'X3_5': sv['val_lf_X3'], 'ns2_5': sv['val_lf_n_s2'], 'gapf5': tg['gap_first'], 'never5': tg['never_seen'], 'uc5': np.load(ROOT / 'data/feats/val/tte_uc_ctr.npy'),
       'als5': np.load(ROOT / 'data/feats/val/als_exp5.npy')}
from scipy.stats import spearmanr
for n, st in STO.items():
    x = get('val_nat', n); st = np.asarray(st, np.float64); ok = np.isfinite(x) & np.isfinite(st)
    P(f'  {n:9s}: present D5 {np.isfinite(x).mean():.4f} / stored {np.isfinite(st).mean():.4f}; Pearson {np.corrcoef(x[ok], st[ok])[0, 1]:+.3f}, Spearman {spearmanr(x[ok][::7], st[ok][::7]).statistic:+.3f} on rows where both are present ({ok.mean():.3f})')
# ------------------------------------------------------------------ C. the stored columns under my null model (natural val)
P('\n=== C. stored columns of record on natural val under the null model pool + own inputs + nx3 (data/feats/val), z / rank form ===')
D = lens.load_split('val', 'TO16'); condlib.sp = 'val'; nm_, cl = condlib.control_block(D)
nz = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nr = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
N0 = condlib.Null(D, nm_, cl); N = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + [nz, nr])
for n, st in STO.items():
    F = lens.forms(np.asarray(st, np.float64), D); F5 = lens.forms(get('val_nat', n), D, log=False) if n != 'ns2_5' else lens.forms(np.log1p(get('val_nat', n)), D, log=False)
    P(f'  stored column for {n:9s}: without nx3 {N0.score(F["z"])["cz"]:+.2f} / {N0.score(F["rk"])["cz"]:+.2f} | with nx3 {N.score(F["z"])["cz"]:+.2f} / {N.score(F["rk"])["cz"]:+.2f}'
      f' || D5 natural column, with nx3 {N.score(F5["z"])["cz"]:+.2f} / {N.score(F5["rk"])["cz"]:+.2f}')
del N, N0, D
# ------------------------------------------------------------------ D. test-like-only variant: label-free C4 on pool rows of slots 11-13
P('\n=== D. test-like-only variant of the view (has = slot >= 11): C4 PSI on POOL rows of slots 11-13 against public / private / (val for scale) ===')
sys.path.insert(0, RS)
R = {q: V.rows(q) for q in ('pool', 'val_nat', 'test_nat')}
epub = np.repeat(V.A['test']['ts'] < hc.PUB_TS, R['test_nat'][3]); slp = np.repeat(a['slot'][V.poolf], R['pool'][3]) >= 11


def psi_bins(a_, b_, binf, nbin):
    pa = np.bincount(binf(a_), minlength=nbin) / len(a_); pb = np.bincount(binf(b_), minlength=nbin) / len(b_)
    pa = np.maximum(pa, 1e-6); pb = np.maximum(pb, 1e-6); return float(((pa - pb) * np.log(pa / pb)).sum())


def spec_binner(val):
    ed = np.unique(np.quantile(val[val != 0], np.linspace(0.1, 0.9, 9)))
    return (lambda x: np.where(x == 0, 0, 1 + np.searchsorted(ed, x, side='right'))), len(ed) + 2


for n, t, fm in (('bt5', None, 'rk'), ('coec5', None, 'rk'), ('kn1_5', None, 'z'), ('ns2_5', 'log1p', 'z'), ('ns2_5', 'log1p', 'rk'), ('als5', None, 'z')):
    f = {}
    for q in R:
        x = get(q, n); x[~np.isfinite(x)] = np.nan
        if t == 'log1p':
            x = np.log1p(x)
        f[q] = lens.feed_z(x, R[q][2], R[q][3], 5.0) if fm == 'z' else lens.feed_rk(x, R[q][2], R[q][3])
    binf, nbin = spec_binner(f['val_nat'])
    P(f'  {n}:{fm}: pool slots 11-13 vs public {psi_bins(f["pool"][slp], f["test_nat"][epub], binf, nbin):.4f}, vs private {psi_bins(f["pool"][slp], f["test_nat"][~epub], binf, nbin):.4f}, vs val {psi_bins(f["pool"][slp], f["val_nat"], binf, nbin):.4f}')
P('DONE')
