"""CRITIC2 c06: where the information of the three forms sits with respect to the PRIVATE board (labels opened).
 A (label-free)  C3 without the 5%-row-share exemption: non-default share by listing-age bucket, pool rows slots 11-13 against PRIVATE / PUBLIC rows,
                 slots 6-8 against VAL rows; every bucket printed; for bt5 (seen), ns2_5 (> 0), the walk (above the feed minimum).
 B  per-feed scores of c01's null models on T (twin) and N (natural val): effective sample size of the pmix weights; cz by feed cold-share tercile;
    cz at pmix-private weights x the PRIVATE lag mix (twin lags 8 / 9 / 10 d re-weighted to 0.24 / 0.38 / 0.38; lag-7 feeds dropped).
 C  cross-witness transfer in NDCG (column only): coefficients of the joint fit on one witness, applied on the other as pool + (b / b_pool) x;
    paired per-feed delta against the pool, plain and at the pmix weights. Trio and each form alone; nx3 the same way as yardstick.
usage: c06_private.py    output out/c06_private.txt"""
from cc_common import *
import pmix
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import hz2_common as hc
P = Log(OUT / 'c06_private.txt')
V = rs.Views(None)
MAIN = [('kn1_5', None, 'z'), ('ns2_5', 'log1p', 'z'), ('bt5', None, 'rk')]
nm = lambda c: f'{c[0]}:{c[2]}'
# ------------------------------------------------------------------ A
P('=== A. non-default share by listing-age bucket, every bucket (no row-share exemption) ===')
P('buckets: ' + str(hc.AGE_NAMES))
kn = hc.knots(); FC = ROOT / 'nn/feat_cache'; G = {}
for q in ('pool', 'val_nat', 'test_nat'):
    sp = V.Q[q][0]; rows, fl, off, M = V.rows(q); a = V.A[sp]; sel = V.Q[q][1]
    nxr = np.asarray(np.load(FC / sp / 'nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32)
    G[q] = dict(b=hc.age_bucket(hc.age_hours(np.asarray(np.load(FLAT / sp / 'cand_f4.npy', mmap_mode='r')[rows]), kn)), slot=np.repeat(a['slot'][sel], M), big=np.repeat(M >= 50, M),
                bt=rget(q, 'bt5_seen') > 0, ns2=rget(q, 'ns2_5') > 0, walk=nxr > np.repeat(np.minimum.reduceat(nxr, off), M))
    if sp == 'test':
        G[q]['pub'] = np.repeat(a['ts'] < hc.PUB_TS, M)
pl_, vl_, te_ = G['pool'], G['val_nat'], G['test_nat']
PAIRS = [('pool slots 11-13 vs PRIVATE', pl_, pl_['slot'] >= 11, te_, ~te_['pub'] & te_['big']), ('pool slots 11-13 vs PUBLIC', pl_, pl_['slot'] >= 11, te_, te_['pub'] & te_['big']),
         ('pool slots 6-8 vs VAL', pl_, (pl_['slot'] >= 6) & (pl_['slot'] <= 8), vl_, vl_['big'])]
for flag, lab in (('bt', 'bt5 seen'), ('ns2', 'ns2_5 > 0'), ('walk', 'walk above the feed minimum (nx3)')):
    for pn, da, ma, db, mb in PAIRS:
        wa = np.bincount(da['b'][ma], minlength=7) / ma.sum(); wb = np.bincount(db['b'][mb], minlength=7) / mb.sum()
        za = np.array([da[flag][ma & (da['b'] == k)].mean() for k in range(7)]); zb = np.array([db[flag][mb & (db['b'] == k)].mean() for k in range(7)])
        g = np.abs(za - zb)
        P(f'  {lab:34s} {pn:28s}: view {" ".join(f"{x:.3f}" for x in za)} | ref {" ".join(f"{x:.3f}" for x in zb)} | row share view {" ".join(f"{x:.3f}" for x in wa)} ref {" ".join(f"{x:.3f}" for x in wb)} | '
          f'dZ {float((wa * g).sum()):.4f}; largest bucket with share >= 0.05: {float(np.where(wa >= 0.05, g, 0).max()):.4f}; largest of ALL buckets: {g.max():.4f} at {hc.AGE_NAMES[int(g.argmax())]}')
del G
# ------------------------------------------------------------------ B, C
FIT = {}
for wit in ('H', 'T', 'N'):
    sp = 'hold' if wit == 'H' else 'val'; q = {'H': 'hold', 'T': 'val_twin', 'N': 'val_nat'}[wit]
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; rows = V.rows(q)[0]
    if wit == 'H':
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
    elif wit == 'T':
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
    else:
        nz = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nr = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
    X = {nm(c): form(rget(q, c[0]), off, M, c[1], c[2]) for c in MAIN}
    N0 = MyNull(D, cl); N = MyNull(D, cl + [nz, nr], theta0=list(N0.theta) + [0.0, 0.0])
    UF = {k: N.score_f(X[k])[0] for k in X}; UF['nx_rp3_z (null without nx3)'] = N0.score_f(nz)[0]
    if wit != 'H':
        W = pmix.weights(); fid = N.fid
        luo = np.load(ROOT / 'nn/feat_cache/val/it_hist_all_luo.npy'); cold = np.bincount(fid, (luo == 0).astype(np.float64), D['n']) / M; qc = np.quantile(cold, [1 / 3, 2 / 3])
        slot_f = V.A['val']['slot']; cut = V.Q[q][2]; lag = np.array([int(s - rs.window(int(c)).max()) for s, c in zip(slot_f, cut)])
        SL = [('all', None), ('cold share low', (cold <= qc[0]) * 1.0), ('cold mid', ((cold > qc[0]) & (cold <= qc[1])) * 1.0), ('cold high', (cold > qc[1]) * 1.0), ('pmix-public w', W['public']), ('pmix-private w', W['private'])]
        if wit == 'T':
            tgt = {8: 0.24, 9: 0.38, 10: 0.38}; wl = np.zeros(D['n'])
            for d_, t_ in tgt.items():
                wl[lag == d_] = t_ / np.mean(lag == d_)
            SL.append(('pmix-private w x private lag mix', W['private'] * wl))
        P(f'\n=== B. witness {wit}: val feeds {D["n"]}; mean feed cold share {cold.mean():.3f}, terciles at {qc.round(3).tolist()}; pmix cold share val / public / private {W["cs_val"]:.3f} / {W["cs_pub"]:.3f} / {W["cs_priv"]:.3f} ===')
        for n_, w in SL[4:]:
            P(f'  weights {n_}: effective feeds {w.sum() ** 2 / (w ** 2).sum():.0f} of {D["n"]} -> a homogeneous effect keeps {np.sqrt(w.sum() ** 2 / (w ** 2).sum() / D["n"]):.2f} of its cz')
        P('  form | ' + ' | '.join(n_ for n_, _ in SL))
        for k, Uf in UF.items():
            P(f'  {k:28s} | ' + ' | '.join(f'{czw(D, Uf, w):+.2f}' for _, w in SL))
    # joint fits for the transfer
    names = [nm(c) for c in MAIN]
    Nf = MyNull(D, cl + [nz, nr] + [X[k] for k in names], theta0=list(N.theta) + [0.0] * 3)
    FIT[wit] = dict(D=D, X=X, nz=nz, bpool=Nf.theta[0], b={k: t for k, t in zip(names, Nf.theta[-3:])}, bnx=N.theta[-2] / N.theta[0], bpool0=N.theta[0])
    P(f'  joint coefficients on {wit}: pool {Nf.theta[0]:.4f}; ' + ', '.join(f'{k} {t:+.4f}' for k, t in zip(names, Nf.theta[-3:])) + f'; nx_rp3_z in the null model with nx3: {N.theta[-2]:+.4f}, nx_rp3_rk {N.theta[-1]:+.4f}')
    del N0, N, Nf
P('\n=== C. cross-witness transfer, column only: NDCG(pool + (b / b_pool) x) - NDCG(pool), b from the SOURCE witness, read on the TARGET; paired per-feed mean +- 1 SE ===')
W = pmix.weights()
for src, tg in (('H', 'T'), ('H', 'N'), ('T', 'H'), ('N', 'H')):
    S = FIT[src]; Tt = FIT[tg]; D = Tt['D']; names = [nm(c) for c in MAIN]
    sets = [('trio', names)] + [(k, [k]) for k in names]
    for sn, ks in sets:
        s = D['base'].astype(np.float64) + sum((S['b'][k] / S['bpool']) * Tt['X'][k].astype(np.float64) for k in ks)
        d = lens.ndcg(s.astype(np.float32), D) - D['ndcg0']
        if tg == 'H':
            slot_f = V.A['train']['slot'][V.hold]; has = slot_f >= 6; tl = slot_f >= 11
            cells = [f'lag-view feeds {d[has].mean():+.5f} +- {d[has].std() / np.sqrt(has.sum()):.5f}', f'slots 11-13 {d[tl].mean():+.5f} +- {d[tl].std() / np.sqrt(tl.sum()):.5f}']
        else:
            cells = [f'all {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}'] + [f'{wn} {pmix.wmean_se(d, W[wk])[0]:+.5f} +- {pmix.wmean_se(d, W[wk])[1]:.5f}' for wn, wk in (('pmix-public', 'public'), ('pmix-private', 'private'))]
        P(f'  {src} -> {tg} | {sn:9s} | ' + ' | '.join(cells))
P('DONE')
