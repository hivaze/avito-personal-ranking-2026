"""Stage 2: re-evaluate S1 (hfit P11W1 TOonly), S2 (g3gate3r), S3 (PoP3, PoP2M) with own code.  CPU only."""
import os, sys, json, time, pickle
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']: os.environ.setdefault(_k, '12')
src = open('<repo>/nn_x/stack/verify/verify.py').read().split('RES = {}')[0]
exec(compile(src, 'verify.py', 'exec'))  # meta, families, cached family sums, helpers (no re-eval)
import scipy.optimize as so
refs = pickle.load(open(f'{C}/refs.pkl', 'rb')); sp, sph = refs['sp'], refs['sph']
fold_s2 = np.load(f'{C}/fold_s2.npy')
OUT = {}

# ---- better fold-alignment probe: per-feed mean of (ft - to) raw logits = fold-model shift, per fold
for m in ['r6_cd_m02_r101', 'v23_gru_pre2s_s501', 'v26_d192_pre2s_s701']:
    x = np.load(f'{P}/nn_{m}_n2cpv3_val_oof.npy').astype(np.float64) - np.load(f'{P}/nn_{m}_trainonly_val.npy').astype(np.float64)
    fm = np.bincount(fidv, x, len(Mv)) / Mv
    bt = lambda fl: np.var([fm[fl == k].mean() for k in range(5)])
    rnd = [bt(np.random.default_rng(s).permutation(fold_s2)) for s in range(50)]
    log(f'fold probe {m}: between-fold var of mean(ft-to) S2 folds {bt(fold_s2):.3e} | random p95 {np.quantile(rnd, .95):.3e} max {max(rnd):.3e}'
        f' | per-fold means {[round(fm[fold_s2 == k].mean(), 4) for k in range(5)]}')

# ======================= S1: holdout-fitted family weights on the to half only
R1 = json.load(open(f'{R}/nn_x/stack/hfit/logs/P11_W1-W2-W3-W4-W5.json'))
assert R1['families'] == P11
w1 = np.array(R1['variants']['W1']['w']); w2f = [np.array(x) for x in R1['variants']['W1']['w_2fold']]
log('S1 W1 weights', dict(zip(P11, np.round(w1, 4))))
ft_me = {s: half_pool(P11, s, 'ft') for s in ('val', 'test')}
def s1_score(w, split):
    off, M = META[split]
    return 0.5 * Z(off, M, ft_me[split]) + 0.5 * Z(off, M, half_pool(P11, split, 'to', dict(zip(P11, w))))
s1v = s1_score(w1, 'val')
for h in ('ft', 'to'):
    fr = np.load(f'{P}/ens_hfitP11W1to_{h}_val.npy').astype(np.float64)
    mine = ft_me['val'] if h == 'ft' else half_pool(P11, 'val', 'to', dict(zip(P11, w1)))
    log(f'S1 parity ens_hfitP11W1to_{h}_val corr {np.corrcoef(fr, mine)[0, 1]:.8f} max|diff| {np.abs(fr - mine).max():.2e}')
OUT['S1_vs_P11ME'] = report('S1 TOonly vs P11 member-equal (59)', s1v, sp['P11'])
OUT['S1_vs_G3'] = report('S1 TOonly vs G3', s1v, sp['G3'])
OUT['S1_vs_PF7x'] = report('S1 TOonly vs PF7x', s1v, sp['PF7x'])
# holdout, using THEIR 2-fold weights (fold def replicated from hfit.group_folds(uh, 2, 11))
uu = np.unique(uh); r = np.random.default_rng(11).permutation(len(uu)); g = dict(zip(uu, r % 2)); fo2 = np.array([g[u] for u in uh])
Xh = np.stack([FS[(f, 'hold', 'to')] / NF[f] for f in P11], 1)
fidh = np.repeat(np.arange(len(Mh)), Mh)
sh_oos = np.empty(len(yh))
rows_k = [np.repeat(fo2 == k, Mh) for k in range(2)]
for k in range(2): sh_oos[rows_k[k]] = (Xh @ w2f[k])[rows_k[k]]
OUT['S1_hold_oos_theirfolds'] = report('S1 holdout 2-fold OOS (their folds+weights) vs P11 ME', sh_oos, sph['P11'], 'hold')
OUT['S1_hold_in'] = report('S1 holdout IN-SAMPLE W1 vs P11 ME', Xh @ w1, sph['P11'], 'hold')
OUT['S1_hold_oos_vsG3'] = report('S1 holdout 2-fold OOS vs G3 to-pool', sh_oos, sph['G3'], 'hold')

# ---- independent ListNet refit (numpy, float64, simplex via v>=0, lam=0) on the HOLDOUT ONLY
nposh = np.bincount(fidh, yh, len(Mh)); starts = oh
def listnet(v, feeds_mask_rows, nfeeds):
    s = Xh @ v
    mx = np.maximum.reduceat(s, starts); e = np.exp(s - np.repeat(mx, Mh)); Zs = np.add.reduceat(e, starts)
    p = e / np.repeat(Zs, Mh); yn = yh / np.repeat(np.maximum(nposh, 1), Mh)
    lse = mx + np.log(Zs)
    lf = lse - np.add.reduceat(yn * s, starts)
    fm = feeds_mask_rows
    L = (lf * fm).sum() / nfeeds
    G = (((p - yn) * np.repeat(fm, Mh))[:, None] * Xh).sum(0) / nfeeds
    return L, G
def fit_hold(feed_sel):
    fm = np.zeros(len(Mh)); fm[feed_sel] = 1.0; fm[nposh == 0] = 0.0; n = fm.sum()
    v0 = np.array([NF[f] for f in P11], float); v0 /= v0.sum()
    r = so.minimize(lambda v: listnet(v, fm, n), v0, jac=True, method='L-BFGS-B', bounds=[(0, None)] * len(P11),
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-11})
    return r.x / r.x.sum()
wr = fit_hold(np.arange(len(Mh)))
log('S1 my full-holdout ListNet refit', dict(zip(P11, np.round(wr, 4))), '| max|w - W1|', np.abs(wr - w1).max().round(4))
OUT['S1_myrefit_val'] = report('S1 TOonly with MY refit weights vs P11 ME', s1_score(wr, 'val'), sp['P11'])
# my own 2-fold user-grouped OOS (cluster = user, user0 singletons) with 3 fold seeds; + val with half-fit weights
oos_d = []
for seed in (101, 202, 303):
    ug = np.unique(gh); rr = np.random.default_rng(seed).permutation(len(ug)); gg = dict(zip(ug, rr % 2)); fo = np.array([gg[u] for u in gh])
    so_ = np.empty(len(yh)); vs = []
    for k in range(2):
        wk = fit_hold(np.where(fo != k)[0]); rk = np.repeat(fo == k, Mh); so_[rk] = (Xh @ wk)[rk]
        dv = nd(s1_score(wk, 'val')) - nd(sp['P11']); vs.append((dv.mean(), dv.std() / np.sqrt(len(dv))))
    o = report(f'S1 my 2-fold OOS holdout seed {seed}', so_, sph['P11'], 'hold'); oos_d.append(o['d'])
    log(f'   val TOonly with half-fit weights (vs P11 ME): ' + ', '.join(f'{a:+.5f}+-{b:.5f}' for a, b in vs))
OUT['S1_my_oos'] = oos_d
# re-cool confound: same tilt but no fitting = up-weight F7x/F7x70 to the fitted share, rest member-equal
rec = np.array([f in ('F7x', 'F7x70') for f in P11]); share = w1[rec].sum()
wme = np.array([NF[f] for f in P11], float); wme /= wme.sum()
wt = wme.copy(); wt[rec] *= share / wme[rec].sum(); wt[~rec] *= (1 - share) / wme[~rec].sum()
OUT['S1_naive_recool_tilt'] = report(f'S1-ctrl: TOonly recooled share {share:.3f}, rest ME', s1_score(wt, 'val'), sp['P11'])
OUT['S1_naive_recool_tilt_hold'] = report('S1-ctrl holdout', Xh @ wt, sph['P11'], 'hold')

# ======================= S2: g3gate3r
rk_v = np.load(f'{P}/nnstk_g3gate3r_val.npy').astype(np.float64)
ON = f'{R}/nn_x/stack/nnstack/out'
runs = ['g3_gate_b3', 'g3_gate_b3_s1', 'g3_gate_b3_s2']
mz = Z(ov, Mv, np.mean([Z(ov, Mv, np.load(f'{ON}/{r}_val_oof.npy')) for r in runs], 0))
sv_rank_check = feed_spearman(ov, Mv, rk_v, mz)
log(f'S2 rank file vs mean-z of 3 runs: per-feed spearman min {sv_rank_check.min():.6f} mean {sv_rank_check.mean():.6f}')
OUT['S2_vs_G3'] = report('S2 g3gate3r (rank file) vs G3 equal', rk_v, sp['G3'])
OUT['S2_raw_vs_G3'] = report('S2 mean-z of 3 runs (no tiebreak) vs G3', mz, sp['G3'])
for r in runs: report(f'S2 single replicate {r}', np.load(f'{ON}/{r}_val_oof.npy'), sp['G3'])
dd = nd(rk_v) - nd(sp['G3'])
log('S2 per-fold delta', [round(dd[fold_s2 == k].mean(), 5) for k in range(5)])
hz = Z(oh, Mh, np.mean([Z(oh, Mh, np.load(f'{ON}/{r}_hold.npy')) for r in runs[1:]], 0))
OUT['S2_hold'] = report('S2 holdout (val-trained, s1+s2) vs G3 to-pool', hz, sph['G3'], 'hold')
OUT['S2_val_spearman_vs_G3'] = float(feed_spearman(ov, Mv, rk_v, sp['G3']).mean())
log('S2 val per-feed spearman vs G3', OUT['S2_val_spearman_vs_G3'])

# ======================= S3: PoP
def pop(pools, split):
    off, M = META[split]
    if split == 'hold': return np.mean([Z(off, M, half_pool(p, 'hold', 'to')) for p in pools], 0)
    return np.mean([Z(off, M, spec(p, split)) for p in pools], 0)
for tag, pools in [('PoP3', [PF7x, G3, M2]), ('PoP2M', [PF7x, M2])]:
    pv = pop(pools, 'val'); fr = np.load(f'{P}/agg_S3_{tag}_val.npy').astype(np.float64)
    log(f'{tag} parity agg_S3_{tag}_val max|diff| {np.abs(pv - fr).max():.2e}')
    OUT[f'{tag}_vs_U45'] = report(f'{tag} vs U45 member-equal', pv, sp['U45'])
    OUT[f'{tag}_vs_G3'] = report(f'{tag} vs G3', pv, sp['G3'])
    OUT[f'{tag}_vs_PF7x'] = report(f'{tag} vs PF7x', pv, sp['PF7x'])
    ph = pop(pools, 'hold')
    OUT[f'{tag}_hold_vs_U45'] = report(f'{tag} holdout vs U45', ph, sph['U45'], 'hold')
    OUT[f'{tag}_hold_vs_G3'] = report(f'{tag} holdout vs G3', ph, sph['G3'], 'hold')
    # train-only val (to half only) - the part comparable to the holdout
    pto = np.mean([Z(ov, Mv, half_pool(p, 'val', 'to')) for p in pools], 0)
    OUT[f'{tag}_valto_vs_U45'] = report(f'{tag} train-only val vs U45 to', pto, half_pool(U45, 'val', 'to'))
# PoP3 implied effective member weights (approx; z(spec) scale ~1)
pickle.dump(OUT, open(f'{V}/stage2_out.pkl', 'wb'))

# ======================= candidate files
fidx = np.load(FLAT / 'test/cand_fidx.npy'); item = np.load(FLAT / 'test/cand_item_id.npy')
feed_id_flat = ft_['feed_id'].to_numpy()[fidx]
ss = pl.read_parquet(f'{R}/data/raw/sample_submission.parquet')
g3sub = pl.read_parquet(f'{R}/subs/cand_R1_G3_F10F7xF7x70_memeq.parquet')
g3t = 0.5 * Z(ot, Mt, half_pool(G3, 'test', 'ft')) + 0.5 * Z(ot, Mt, half_pool(G3, 'test', 'to'))
st_s2 = Z(ot, Mt, np.mean([Z(ot, Mt, np.load(f'{ON}/{r}_test.npy')) for r in runs], 0))
CANDS = {'S1': ('cand_S1_hfit_P11W1to.parquet', s1_score(w1, 'test')),
         'S2': ('cand_S2_nnstk_g3gate3r.parquet', st_s2),
         'PoP3': ('cand_S3_PoP3.parquet', pop([PF7x, G3, M2], 'test')),
         'PoP2M': ('cand_S3_PoP2M.parquet', pop([PF7x, M2], 'test'))}
log('G3 sub vs my G3 test spec spearman', feed_spearman(ot, Mt, g3sub['score'].to_numpy().astype(np.float64), g3t).min())
for k, (fn, mine) in CANDS.items():
    s = pl.read_parquet(f'{R}/subs/{fn}')
    ok_ids = (s['feed_id'].to_numpy() == feed_id_flat).all() and (s['item_id'].to_numpy() == item).all()
    ok_g3 = (s['feed_id'].to_numpy() == g3sub['feed_id'].to_numpy()).all() and (s['item_id'].to_numpy() == g3sub['item_id'].to_numpy()).all()
    j = ss.select('feed_id', 'item_id').join(s, on=['feed_id', 'item_id'], how='left')
    sc = s['score'].to_numpy().astype(np.float64)
    rho = feed_spearman(ot, Mt, sc, mine); rg = feed_spearman(ot, Mt, sc, g3t)
    # rank validity: within-feed ranks 0..M-1 unique
    uniq = (np.bincount(np.repeat(np.arange(len(Mt)), Mt), sc, len(Mt)) == Mt * (Mt - 1) / 2).all()
    log(f'{k} {fn}: rows {len(s)} (sample {len(ss)}) ids==flat {ok_ids} ids==G3sub {ok_g3} nulls {j["score"].null_count()} finite {np.isfinite(sc).all()} '
        f'ranks0..M-1 {uniq} | spearman vs MY recomputed test min {rho.min():.6f} mean {rho.mean():.6f} frac<0.9999 {(rho < 0.9999).mean():.5f} '
        f'| vs G3 mean {rg.mean():.5f}')
    if k == 'S2':
        rkt = np.load(f'{P}/nnstk_g3gate3r_test.npy').astype(np.float64)
        log(f'   S2 sub vs preds/nnstk_g3gate3r_test ranks spearman min {feed_spearman(ot, Mt, sc, rkt).min():.6f}')
log('stage 2 done')
