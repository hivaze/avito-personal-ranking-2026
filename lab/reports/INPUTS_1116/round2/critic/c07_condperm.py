"""CRITIC2 c07: CONDITIONAL-permutation placebo for the two popularity-type forms (labels opened).
ns2_5 and bt5 are correlated with the ranker's popularity inputs (R2 0.15-0.53) and the lens controls those inputs only linearly (z and rank forms).
Question: could a NON-LINEAR function of the popularity inputs alone produce the read?  Placebo: the raw column is permuted inside the feed AMONG ROWS OF
THE SAME POPULARITY STRATUM (8 within-feed rank bins of a label-free popularity index = OLS prediction of the column's entered form from the within-feed
z of log1p(it_hist_all_luo, it_hist_et1_luo, seq_cv_invd20, seq_cv_pop)); the entered form is recomputed and scored.  The permuted column keeps its whole
relation to popularity and loses everything specific to the item.  100 draws per form and witness.  If the read is specific, the placebo is centred on 0.
A second stratification for bt5: by the exposure count imp5 of the same window (label-free, NOT an input of the ranker): what is left of bt5 is then the
click part of the statistic, the placebo carries the exposure part.
usage: c07_condperm.py    output out/c07_condperm.txt"""
from cc_common import *
P = Log(OUT / 'c07_condperm.txt')
V = rs.Views(None)
ND = 100; NB = 8
INP = ['it_hist_all_luo', 'it_hist_et1_luo', 'seq_cv_invd20', 'seq_cv_pop']
FORMS = [('ns2_5', 'log1p', 'z'), ('bt5', None, 'rk'), ('kn1_5', None, 'z')]
for wit in ('H', 'T'):
    sp = 'hold' if wit == 'H' else 'val'; q = 'hold' if wit == 'H' else 'val_twin'; fsp = 'train' if wit == 'H' else 'val'
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; rows = V.rows(q)[0]
    if wit == 'H':
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
    else:
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
    N = MyNull(D, cl + [nz, nr]); fid = N.fid
    Z = np.column_stack([form(np.asarray(np.load(ROOT / f'nn/feat_cache/{fsp}/{n_}.npy', mmap_mode='r')[rows]).astype(np.float64), off, M, 'log1p', 'z') for n_ in INP] + [np.ones(len(rows), np.float32)]).astype(np.float64)
    imp = rget(q, 'imp5'); view = ~np.isnan(imp)
    rng = np.random.default_rng(555 if wit == 'H' else 556)
    P(f'\n=== witness {wit}: feeds {D["n"]}, rows {len(rows)} (with a view {view.mean():.4f}) ===')
    for col, t, fm in FORMS:
        raw = rget(q, col); x = form(raw, off, M, t, fm); read = N.cz(x)[0]
        beta, *_ = np.linalg.lstsq(Z[view], x[view].astype(np.float64), rcond=None); pred = Z @ beta
        r2 = 1 - ((x[view] - pred[view]) ** 2).sum() / ((x[view] - x[view].mean()) ** 2).sum()
        strat = {'popularity index': np.minimum((NB * (lens.feed_rk(np.where(view, pred, np.nan), off, M) + 0.5)).astype(np.int64), NB - 1)}
        if col == 'bt5':
            strat['exposure imp5 of the window'] = np.minimum((NB * (lens.feed_rk(np.where(view, imp, np.nan), off, M) + 0.5)).astype(np.int64), NB - 1)
        for sn, st in strat.items():
            key = fid * NB + st; ordk = np.argsort(key, kind='stable'); cz_ = []; moved = []
            for s in range(ND):
                o = np.argsort(key + rng.random(len(key)), kind='stable'); xp = np.empty_like(raw); xp[ordk] = raw[o]
                xf = form(xp, off, M, t, fm); cz_.append(N.cz(xf)[0])
                if s < 5:
                    moved.append(float(np.corrcoef(xf[view], x[view])[0, 1]))
            cz_ = np.array(cz_)
            P(f'  {col}:{fm} | read {read:+.2f} | R2 of the form with the popularity index {r2:.3f} | placebo within strata of the {sn}: mean {cz_.mean():+.2f}, sd {cz_.std():.2f}, min {cz_.min():+.2f}, max {cz_.max():+.2f}; '
              f'corr(placebo form, true form) {np.mean(moved):.3f}; (read - mean) / sd = {(read - cz_.mean()) / cz_.std():+.2f}; share of placebos beyond the read {np.mean(np.abs(cz_) >= abs(read)):.3f}')
    del N, D
P('DONE')
