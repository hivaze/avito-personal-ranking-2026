"""D. Val/test parity of the FT-minus-TO shift (W8 students; C3 as reference).
 1. standardised extra-feature drift on test under the val statistics (exactly as SplitData.attach_extra does);
 2. is the shift the same FUNCTION of the inputs on val (OOF) and on test? ridge of the pooled shift on [42 extra features (val-standardised,
    as the model sees them), their per-feed z, z_TO, z_TO^2]: fit on one split, R2 on the other;
 3. shift size by M bucket and by day (test: public Sat / Sun-am, private Sun-pm / Mon / Tue);
 4. val only: does the shift point at the positives (mean shift on positives, per-feed)?"""
from lib import *
sys.path.insert(0, '<repo>/nn_x/w12/recheck/ft_redesign')
N2C = ['centrality', 'it_hist_all_luo', 'it_hist_all_luo_rk', 'it_hist_all_luo_zsl', 'it_hist_all_luo_gap', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'it_hist_et5_luo', 'it_hist_et6_luo', 'it_hist_et7_luo', 'it_hist_et8_luo', 'it_hist_et9_luo', 'it_hist_et10_luo', 'seq_cv_invd20', 'seq_cv_invd20_rk', 'seq_cv_pop', 'seq_max_all', 'seq_top3', 'seq_argmax_hpos', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u', 'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'pre_fe', 'pre_fe_z', 'pre_new', 'pre_new_z', 'pre_hmax32_c', 'pre_hmax8_c', 'pre_fe_noid', 'pre_id_contrib_v2', 'pre_idmax32', 'pre_idmax32_has', 'pre_hmax32_sent']
FC = Path('<repo>/nn_x/w5/feat_cache')
def load(sp): return np.stack([np.load(FC / sp / f'{n}.npy').astype(np.float32) for n in N2C], 1)
Xv, Xt = load('val'), load('test')
logm = (np.nanmin(Xv, 0) >= 0) & (np.nanmax(Xv, 0) > 20)
for X in (Xv, Xt): X[:, logm] = np.log1p(np.maximum(X[:, logm], 0))
mu, sd = np.nanmean(Xv, 0), np.nanstd(Xv, 0) + 1e-6
Sv = np.nan_to_num(np.clip((Xv - mu) / sd, -6, 6)); St = np.nan_to_num(np.clip((Xt - mu) / sd, -6, 6))
print('== 1. extra features on test under val standardisation (mean, sd; val = 0, 1 before clipping); nan share val/test')
bad = []
for j, n in enumerate(N2C):
    m, s = St[:, j].mean(), St[:, j].std(); nv, nt = np.isnan(Xv[:, j]).mean(), np.isnan(Xt[:, j]).mean()
    flag = abs(m - Sv[:, j].mean()) > 0.1 or abs(s / (Sv[:, j].std() + 1e-9) - 1) > 0.15 or abs(nv - nt) > 0.03
    if flag: bad.append(n)
    print(f'  {n:22s} log {int(logm[j])} val mean {Sv[:, j].mean():+.3f} sd {Sv[:, j].std():.3f} | test mean {m:+.3f} sd {s:.3f} | nan {nv:.3f}/{nt:.3f} {"<-- DRIFT" if flag else ""}')
print('  flagged:', bad)
# public / private split of test
pub = np.arange(len(Mt)) < 8562; dayt = (tt.astype(np.int64) + 3 * 3600) // 86400; dt = np.unique(dayt)
print('\n== 1b. the same drift, test public (weekend) vs private, for the flagged + popularity features (mean under val standardisation)')
rp = np.repeat(pub, Mt)
for n in sorted(set(bad + ['it_hist_all_luo', 'seq_cv_pop', 'pre_fe', 'ctx_nd_max'])):
    j = N2C.index(n); print(f'  {n:22s} public {St[rp, j].mean():+.3f} private {St[~rp, j].mean():+.3f}')

def fz(x, M, off):  # per-feed z of every column
    mu_ = np.add.reduceat(x, off, 0) / M[:, None]; d = x - np.repeat(mu_, M, 0)
    s_ = np.sqrt(np.add.reduceat(d * d, off, 0) / M[:, None]) + 1e-6
    return d / np.repeat(s_, M, 0)
def design(S, zto, M, off):
    return np.hstack([S, fz(S, M, off), zto[:, None], (zto ** 2)[:, None], np.ones((len(S), 1))]).astype(np.float64)
def ridge(A, y, lam=10.0):
    G = A.T @ A; G[np.diag_indices_from(G)] += lam; return np.linalg.solve(G, A.T @ y)
r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print('\n== 2. shift as a function of the inputs: ridge on 86 columns; R2 in-split and cross-split')
for k, ms in (('W8', W8), ('C3', C3M)):
    tov, tot = zv(pool(ms, 'to', 'val')), zt(pool(ms, 'to', 'test'))
    sv, st_ = zv(pool(ms, 'ft', 'val')) - tov, zt(pool(ms, 'ft', 'test')) - tot
    Av, At = design(Sv, tov, Mv, ov), design(St, tot, Mt, ot)
    bv, bt = ridge(Av, sv), ridge(At, st_)
    print(f' {k}: fit val -> R2 val {r2(sv, Av @ bv):.3f}, R2 test {r2(st_, At @ bv):.3f} | fit test -> R2 test {r2(st_, At @ bt):.3f}, R2 val {r2(sv, Av @ bt):.3f} | cos(coef val, coef test) {bv[:-1] @ bt[:-1] / np.linalg.norm(bv[:-1]) / np.linalg.norm(bt[:-1]):.3f}')
    pv, ptv = Av @ bt, sv - Av @ bt   # explained by the TEST-fitted map (label-free for val), and the residual
    e0 = ev(tov)
    print(f'    val NDCG: TO {e0.mean():.5f} | TO + explained part {f(ev(tov + pv) - e0)} | TO + residual part {f(ev(tov + ptv) - e0)} | TO + full shift (= FT) {f(ev(tov + sv) - e0)} | TO + 0.5 explained {f(ev(tov + 0.5 * pv) - e0)} | TO + 0.5 residual {f(ev(tov + 0.5 * ptv) - e0)}')
    if k == 'W8': keep = (sv, st_, tov, tot)
sv, st_, tov, tot = keep
print('\n== 3. RMS of the W8 pooled shift by segment')
rms = lambda x: np.sqrt((x ** 2).mean())
for nm, lo, hi in (('M<20', 0, 20), ('M20-49', 20, 50), ('M50-99', 50, 100), ('M>=100', 100, 10 ** 6)):
    a = np.repeat((Mv >= lo) & (Mv < hi), Mv); b = np.repeat((Mt >= lo) & (Mt < hi), Mt)
    print(f'  {nm:8s} val {rms(sv[a]):.4f} | test {rms(st_[b]):.4f} | ratio {rms(st_[b]) / rms(sv[a]):.3f}')
dayv = (tv.astype(np.int64) + 3 * 3600) // 86400
for d in np.unique(dayv):
    if (dayv == d).sum() > 100: print(f'  val day {d - np.unique(dayv)[0]} ({(dayv == d).sum()} feeds): {rms(sv[np.repeat(dayv == d, Mv)]):.4f}')
for nm, m in (('public Sat', pub & (dayt == dt[0])), ('public Sun-am', pub & (dayt == dt[1])), ('private Sun-pm', ~pub & (dayt == dt[1])), ('private Mon', dayt == dt[2]), ('private Tue', dayt == dt[3]), ('PUBLIC', pub), ('PRIVATE', ~pub)):
    print(f'  test {nm:15s} ({m.sum()} feeds): RMS {rms(st_[np.repeat(m, Mt)]):.4f} | corr(z_FT, z_TO) {np.corrcoef(tot[np.repeat(m, Mt)], (tot + st_)[np.repeat(m, Mt)])[0, 1]:.5f}')
print('\n== 4. val: mean shift on positives (z units; per-feed mean over positives, then mean over feeds with a positive)')
for k, ms in (('W8', W8), ('N8', N8), ('C3', C3M)):
    s = zv(pool(ms, 'ft', 'val')) - zv(pool(ms, 'to', 'val')); num = np.add.reduceat(s * yv, ov); npos = np.add.reduceat(yv, ov); ok = npos > 0
    d = num[ok] / npos[ok]; print(f'  {k}: {f(d)}  (M<50 {f((num / np.maximum(npos, 1))[ok & (Mv < 50)])}, M>=50 {f((num / np.maximum(npos, 1))[ok & (Mv >= 50)])})')
