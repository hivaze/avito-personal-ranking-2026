"""ft_1115 step C: what the out-of-fold lens cannot see: a val row is scored by ONE fold model (x 3 seeds), a test row by the mean of 5 fold models (x 3 seeds).
 C1  test rows, label-free: fold-to-fold agreement of the fine-tuned branch of W8P20 and FD16P20 (per-fold pool built exactly like the branch).
 C2  the only nested fine-tune on disk (nn_x/solvers/preds/refit/nested_<m>.npz, 3 hard-label members, written 2026-11-11 by r5_refit): for every val feed
     P_i  = mean of 5 inner fold models (each fitted on 64% of val, none saw the feed)  = the analogue of a TEST prediction (bag of 5 fold models)
     P_ii = one model fitted on the whole 80%                                             = the analogue of the OUT-OF-FOLD prediction (second seed)
     prod = preds/nn_<m>_n2cpv3_val_oof.npy (one model on the same 80%)                   = the out-of-fold prediction itself
     -> bag minus single at member, 3-member pool and file level, and along w_ft.
 C3  control variate: N = z(single) - z(bag) of the 3 nested members is a REAL fold-noise field on val rows (label-free for the scored feed). If the fold noise is
     shared across model families (same GroupKFold split), F - beta N is the student FT pool with part of its fold noise removed; measured NDCG change, placebo
     (seed-noise field), symmetric add-noise check, and extrapolation to the removal of 4/5 of the fold noise.
 C4  the two public pairs (W8 vs W8TO, G3 vs G3 TO-only) are not independent readings: correlation of their per-feed val deltas.
Output: out/c_foldavg.txt"""
from lib import *
from sklearn.model_selection import GroupKFold
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
GRID = [0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0]; I5 = GRID.index(0.5)
# ------------------------------------------------------------------ C1
w_('## C1. Fold-to-fold agreement of the fine-tuned branch on TEST rows (label-free)')
w_('branch | mean Pearson r between two fold-branches (10 pairs; pooled over rows) | r(fold k, 5-fold mean) | fold-noise variance of ONE fold-branch as a share of the branch variance | ... as a share of the FT-minus-TO shift variance | same shares after 5-fold averaging')
SPEC = {'W8': (MEM['W8'], [], 0.0), 'W8P20': (MEM['W8'], MEM['P1d'], 0.2), 'FD16': (MEM['FD16'], [], 0.0), 'FD16P20': (MEM['FD16'], MEM['FDP'], 0.2), 'W16': (MEM['W16'], [], 0.0)}
FOLDSTAT = {}
for name, (base, pop, wp) in SPEC.items():
    try:
        TB = np.mean([test_folds(m) for m in base], 0)                                    # 5 x rows; each member row already = per-feed z, mean of 3 seeds
        Fk = np.stack([zt(TB[k]) for k in range(5)])
        if pop:
            TP = np.mean([test_folds(m) for m in pop], 0); Fk = np.stack([(1 - wp) * Fk[k] + wp * zt(TP[k]) for k in range(5)])
    except FileNotFoundError as e:
        w_(f'{name}: test_folds missing ({e})'); continue
    Fm = Fk.mean(0); zto = mpool(base, 'trainonly', 'test')
    if pop: zto = (1 - wp) * zt(zto) + wp * zt(mpool(pop, 'trainonly', 'test'))
    else: zto = zt(zto)
    rs = [np.corrcoef(Fk[i], Fk[j])[0, 1] for i in range(5) for j in range(i)]
    rm = [np.corrcoef(Fk[k], Fm)[0, 1] for k in range(5)]
    dev = Fk - Fm; v_dev = (dev ** 2).mean()                                               # = (4/5) sigma_e^2
    sig2 = v_dev * 5 / 4; vF = Fm.var(); vS = ((Fm - zto) ** 2).mean()
    # is the fold noise the same on public and private rows?
    rowpub = np.repeat(PUB, Mt)
    FOLDSTAT[name] = dict(sig2=float(sig2), vF=float(vF), vS=float(vS))
    w_(f'{name} | {np.mean(rs):.4f} | {np.mean(rm):.4f} | {sig2 / vF:.4%} | {sig2 / vS:.2%} | {sig2 / 5 / vF:.4%} / {sig2 / 5 / vS:.2%}'
       f' | fold-noise variance public rows {(dev[:, rowpub] ** 2).mean() * 5 / 4:.5f} private rows {(dev[:, ~rowpub] ** 2).mean() * 5 / 4:.5f}')
    del TB, Fk
w_('(sigma_e^2 = variance of the fold-specific part of one fold-branch, in per-feed z units; a val out-of-fold row carries sigma_e^2, a test row sigma_e^2 / 5.)')
w_()
# ------------------------------------------------------------------ C2
w_('## C2. Nested fine-tune of 2026-11-11 (3 hard-label members, 1 fine-tune seed): bag of 5 fold models against one model on the same data')
NM = ['v15_cd_all_e10', 'v14_s93', 'v17_pre2s_s102']; NP = R / 'nn_x/solvers/preds/refit'
Zi, Zii, Zp, Zt_ = [], [], [], []
# the outer folds of the nested run must be the production fine-tune folds
g = user_groups(); gk = np.zeros(len(Mv), np.int64)
for k, (_, ho) in enumerate(GroupKFold(5).split(np.arange(len(Mv)), groups=g)): gk[ho] = k
w_('member | outer folds = production folds | NDCG bag (5 x 64%) | one model 80% (refit seed) | one model 80% (stored out-of-fold) | bag - stored OOF | bag - refit | refit - stored OOF (seed A/A)')
for m in NM:
    z = np.load(NP / f'nested_{m}.npz', allow_pickle=True)
    same = bool((z['outer'] == gk).all())
    pi, pii = z['P_i'], z['P_ii']; pr = np.load(P / f'nn_{m}_n2cpv3_val_oof.npy'); to = np.load(P / f'nn_{m}_trainonly_val.npy')
    ei, eii, ep = ev(pi), ev(pii), ev(pr)
    w_(f'{m} | {same} | {ei.mean():.5f} | {eii.mean():.5f} | {ep.mean():.5f} | {f(ei - ep)} | {f(ei - eii)} | {f(eii - ep)}')
    Zi.append(zv(pi)); Zii.append(zv(pii)); Zp.append(zv(pr)); Zt_.append(zv(to))
Zi, Zii, Zp, Zt_ = map(np.array, (Zi, Zii, Zp, Zt_))
em = lambda Z: np.mean([ev(z) for z in Z], 0)
d1 = em(Zi) - em(Zp); d2 = em(Zi) - em(Zii)
w_(f'member level, mean of the 3 members per feed: bag - stored OOF {f(d1)} | bag - refit {f(d2)} | bag - mean of the two single models {f(0.5 * (d1 + d2))}'
   f' | pmix-public {f(0.5 * (d1 + d2), WPUB)} pmix-private {f(0.5 * (d1 + d2), WPRIV)}')
bag, s1, s2, to3 = zv(Zi.mean(0)), zv(Zp.mean(0)), zv(Zii.mean(0)), zv(Zt_.mean(0))
ebag, es1, es2, eto = ev(bag), ev(s1), ev(s2), ev(to3)
w_(f'3-member pool, FT branch alone: bag {ebag.mean():.5f} | single (stored OOF) {es1.mean():.5f} | single (refit) {es2.mean():.5f} | TO pool {eto.mean():.5f}')
w_(f'   bag - single: stored {f(ebag - es1)} | refit {f(ebag - es2)} | mean of both {f(ebag - 0.5 * (es1 + es2))} | pmix-public {f(ebag - 0.5 * (es1 + es2), WPUB)} | pmix-private {f(ebag - 0.5 * (es1 + es2), WPRIV)}')
w_(f'   corr z(FT), z(TO): bag {np.corrcoef(bag, to3)[0, 1]:.4f} single {np.corrcoef(s1, to3)[0, 1]:.4f} | RMS shift bag {np.sqrt(((bag - to3) ** 2).mean()):.4f} single {np.sqrt(((s1 - to3) ** 2).mean()):.4f} {np.sqrt(((s2 - to3) ** 2).mean()):.4f}'
   f' | r(single, bag) {np.corrcoef(s1, bag)[0, 1]:.4f} {np.corrcoef(s2, bag)[0, 1]:.4f} | r(single, single) {np.corrcoef(s1, s2)[0, 1]:.4f}')
w_('w_ft | file with the BAG as FT branch (level) | file with ONE model as FT branch (mean of the two single-model versions) | bag - single, paired | pmix-public | pmix-private')
CURB, CURS = [], []
for w in GRID:
    eb = ev(w * bag + (1 - w) * to3); e1 = ev(w * s1 + (1 - w) * to3); e2 = ev(w * s2 + (1 - w) * to3); es = 0.5 * (e1 + e2); CURB.append(eb); CURS.append(es)
    w_(f'{w:.2f} | {eb.mean():.5f} | {es.mean():.5f} | {f(eb - es)} | {f(eb - es, WPUB)} | {f(eb - es, WPRIV)}')
CURB, CURS = np.array(CURB), np.array(CURS)
w_('value of the FT half (file at w_ft 0.5 minus TO pool): bag ' + f(CURB[I5] - CURB[0]) + ' | single ' + f(CURS[I5] - CURS[0]) + f' | argmax w_ft: bag {GRID[int(CURB.mean(1).argmax())]} single {GRID[int(CURS.mean(1).argmax())]}')
w_('curve against w_ft 0.5: bag ' + ' '.join(f'{w:.2f}:{(CURB[i] - CURB[I5]).mean():+.5f}' for i, w in enumerate(GRID)) + ' || single ' + ' '.join(f'{w:.2f}:{(CURS[i] - CURS[I5]).mean():+.5f}' for i, w in enumerate(GRID)))
np.savez(OUT / 'c_nested_curves.npz', bag=CURB, single=CURS, grid=np.array(GRID))
w_()
# ------------------------------------------------------------------ C3
w_('## C3. Control variate: remove from the student FT pool the part of its fold noise that the nested members share')
N = 0.5 * (s1 + s2) - bag                  # fold noise of the single 80% model (mean of 2 seeds), 3-member pool, per val row
Nseed = 0.5 * (s1 - s2)                    # placebo: pure seed noise (both on the same 80%), same scale family
# member-level fields for the cross-family share
w_(f'noise field N = z(single, mean of 2 seeds) - z(bag): RMS {np.sqrt((N ** 2).mean()):.4f} (variance {N.var():.5f}); placebo seed field RMS {np.sqrt((Nseed ** 2).mean()):.4f};'
   f' corr of the two single-seed fields z(s1)-z(bag), z(s2)-z(bag): {np.corrcoef(s1 - bag, s2 - bag)[0, 1]:.3f}')
w_('pool | corr(shift, N) | beta | variance removed (z units) | fold noise sigma_e^2 on test (C1) | share of sigma_e^2 removed | FT alone: denoised - stored | file 0.5: denoised - stored | pmix-public | pmix-private | placebo (seed field, same beta-rule) file | add-noise check file (+beta N)')
CV = {}
for name, tag in (('W8P20', 'W8P20kd'), ('FD16P20', 'FD16P20kd'), ('W8', 'W8kd'), ('W16', 'W16kd')):
    b_ft, b_to = stored(tag, 'val'); F, T = zv(b_ft).astype(np.float64), zv(b_to).astype(np.float64); S = F - T
    beta = float(S @ N / (N @ N)); rem = beta ** 2 * N.var(); sig2 = FOLDSTAT.get(name, {}).get('sig2', float('nan'))
    bp = float(S @ Nseed / (Nseed @ Nseed))
    e_ft = ev(F); e_fi = ev(0.5 * F + 0.5 * T)
    Fd = zv(F - beta * N); e_ftd = ev(Fd); e_fid = ev(0.5 * Fd + 0.5 * T)
    Fa = zv(F + beta * N); e_fia = ev(0.5 * Fa + 0.5 * T)
    Fp = zv(F - bp * Nseed); e_fip = ev(0.5 * Fp + 0.5 * T)
    share = rem / sig2
    CV[name] = dict(beta=beta, share=float(share), d_ft=e_ftd - e_ft, d_file=e_fid - e_fi, F=F, T=T)
    w_(f'{name} | {np.corrcoef(S, N)[0, 1]:+.3f} | {beta:+.3f} | {rem:.5f} | {sig2:.5f} | {share:.1%} | {f(e_ftd - e_ft)} | {f(e_fid - e_fi)} | {f(e_fid - e_fi, WPUB)} | {f(e_fid - e_fi, WPRIV)}'
       f' | beta_seed {bp:+.3f}: {f(e_fip - e_fi)} | {f(e_fia - e_fi)}')
w_('Extrapolation (gain taken as linear in the removed noise variance): removal of 4/5 of sigma_e^2 = measured gain x 0.8 / share removed.')
for name, c in CV.items():
    k = 0.8 / c['share'] if c['share'] > 0 else float('nan')
    w_(f'  {name}: FT alone {c["d_ft"].mean() * k:+.5f} (+-{se(c["d_ft"]) * k:.5f}) | file at w_ft 0.5 {c["d_file"].mean() * k:+.5f} (+-{se(c["d_file"]) * k:.5f}) | scale factor {k:.2f}')
w_('Curve against w_ft with the denoised FT pool (W8P20, FD16P20): d(w) = file(w; denoised) - file(0.5; denoised), and the correction file(w; denoised) - file(w; stored)')
for name in ('W8P20', 'FD16P20'):
    c = CV[name]; F, T = c['F'], c['T']; Fd = zv(F - c['beta'] * N)
    cur_d = np.array([ev(w * Fd + (1 - w) * T) for w in GRID]); cur_s = np.array([ev(w * zv(F) + (1 - w) * T) for w in GRID])
    np.savez(OUT / f'c_cv_curves_{name}.npz', den=cur_d, sto=cur_s, grid=np.array(GRID), share=c['share'])
    w_(f'  {name}: ' + ' | '.join(f'w {w:.2f}: stored {(cur_s[i] - cur_s[I5]).mean():+.5f} denoised {(cur_d[i] - cur_d[I5]).mean():+.5f} corr {f(cur_d[i] - cur_s[i])}' for i, w in enumerate(GRID)))
w_()
# ------------------------------------------------------------------ C4
w_('## C4. The two public pairs of the FT half are correlated readings')
eW = np.load(OUT / 'b_perfeed_W8.npz'); gW = list(np.round(eW['grid'], 2)); dW = eW['e'][gW.index(0.5)] - eW['e'][gW.index(0.0)]
g_ft, g_to = stored('G3', 'val'); eg5 = ev(0.5 * zv(g_ft) + 0.5 * zv(g_to)); eg0 = ev(zv(g_to)); dG = eg5 - eg0
rho = float(np.corrcoef(dW, dG)[0, 1])
seW, seG = dW.std() / np.sqrt(8562), dG.std() / np.sqrt(8562)
pooled = 0.5 * (0.000477 + 0.000474); se_p = 0.5 * np.sqrt(seW ** 2 + seG ** 2 + 2 * rho * seW * seG)
w_(f'val: FT half of W8 (file - TO pool) {f(dW)} | of G3 (1-seed FT, the scored file) {f(dG)} | per-feed correlation of the two deltas {rho:.3f}')
w_(f'paired public SE by the val per-feed sd: W8 pair {seW:.5f}, G3 pair {seG:.5f}; public readings +0.000477 and +0.000474 -> mean {pooled:+.6f} +- {se_p:.5f} (z {pooled / se_p:.2f}); if they were independent the SE would be {0.5 * np.sqrt(seW ** 2 + seG ** 2):.5f}')
w_(f'pmix-public reading of the same contrasts: W8 {f(dW, WPUB)} | G3 {f(dG, WPUB)}; pmix-private: W8 {f(dW, WPRIV)} | G3 {f(dG, WPRIV)}')
open(OUT / 'c_foldavg.txt', 'w').write('\n'.join(out) + '\n')
