"""w1: independent re-derivation of the calibration claims of reports/NEXT_1115/merge.md (r2). CPU, 12 threads. Val labels are used only to score.
A  correlation definitions, W8 file vs C3 file (test), from the two parquet files and from the model scores
B  10 partners of W8: val, gap, rho (feed-mean / pooled), law on record, refit, measured equal z-mix; law error; W16 + CatBoost
C  refit, leave-one-out, feed bootstrap (own seed) of the 10-partner line and of the 55 / 52-pair lines; break-even gaps
D  rank-file mix with an explicit tie-break (ties to the better file / random / weaker) minus the z-mix
E  CatBoost at small weights (information for the owner's question; fixed lens grid, no selection)"""
import sys, json, time
sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify/r2')
from w_common import *
from metrics import ndcg_per_feed
t0 = time.time()
V, T = Split('val'), Split('test'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
ev = lambda s: my_ndcg(V, s, yv)
w8v, w8t = V.z(fin(V, 'W8kd')), T.z(fin(T, 'W8kd')); e8 = ev(w8v)
off_ = ndcg_per_feed(w8v, yv, V.off, V.M); print(f'own NDCG vs src/metrics on W8 val: max abs per-feed diff {np.abs(off_ - e8).max():.2e}; W8 val {e8.mean():.5f} ({V.n} feeds)')
OUT = {}

# ---------------------------------------------------------------- A
print('\n== A. W8 file vs C3 file, test')
fw, fc = T.read_parquet(ROOT / 'subs/cand_W8kd_r2.parquet'), T.read_parquet(ROOT / 'subs/cand_Q2a8_ftbag3_memeq.parquet')
c3v, c3t = V.z(fin(V, 'Q2a8s3')), T.z(fin(T, 'Q2a8s3'))
print(f'  file order reproduced by preds/ens_*: W8 {(T.rank(fw) == T.rank(w8t)).mean():.6%}  C3 {(T.rank(fc) == T.rank(c3t)).mean():.6%}')
zrw, zrc = T.z(T.rank(fw)), T.z(T.rank(fc)); rs = T.rf(zrw, zrc); rp = T.rf(w8t, c3t); big = T.M >= 50
pool = lambda r, m: float((r[m] * T.M[m]).sum() / T.M[m].sum())
al = np.ones(T.n, bool)
print(f'  pooled Spearman (rank files)        all {pool(rs, al):.4f}  M>=50 {pool(rs, big):.4f}  M<50 {pool(rs, ~big):.4f}   np.corrcoef of concatenated z(rank): {np.corrcoef(zrw, zrc)[0, 1]:.4f}')
print(f'  feed-mean Spearman                  all {rs.mean():.4f}  M>=50 {rs[big].mean():.4f}  M<50 {rs[~big].mean():.4f}')
print(f'  pooled Pearson of z of model scores all {pool(rp, al):.4f}  M>=50 {pool(rp, big):.4f}  M<50 {pool(rp, ~big):.4f}')
print(f'  feed-mean Pearson of z              all {rp.mean():.4f}  M>=50 {rp[big].mean():.4f}  M<50 {rp[~big].mean():.4f}   | val: feed-mean {V.rf(w8v, c3v).mean():.4f} pooled {np.corrcoef(w8v, c3v)[0, 1]:.4f}')
# shape restoration of the partner rank file on OUR sorted z
def restore(S, zours, rank_partner):
    zs = zours[np.lexsort((zours, S.fid))]; return zs[S.off[S.fid] + rank_partner.astype(np.int64)]
zc_h = restore(T, w8t, T.rank(fc)); rr = T.rf(w8t, T.z(zc_h))
print(f'  rank file restored on the W8 shape   feed-mean {rr.mean():.4f}  pooled {pool(rr, al):.4f}')
OUT['A'] = dict(spear_pooled=pool(rs, al), spear_fm=float(rs.mean()), pear_pooled=pool(rp, al), pear_fm=float(rp.mean()), restored_fm=float(rr.mean()))

# ---------------------------------------------------------------- B
print('\n== B. calibration, 10 partners of W8 (val 14,423 feeds; equal z-mix over the better file)')
C = partner_table(V, T); NAMES = list(C); E = {k: ev(C[k][0]) for k in NAMES}
REC, REF = (0.0186, 0.378), (0.0103, 0.304); law = lambda ab, rho, gap: ab[0] * (1 - rho) - ab[1] * abs(gap)
print(f'  {"partner":9s} {"val":>8s} {"gap ± SE":>19s} | rho val fm / pooled | test fm | {"law rec":>8s} {"refit":>8s} | {"measured ± SE":>19s} | err rec (in SE) | err refit')
rows = []
for nm in NAMES:
    pv, pt = C[nm]; e = E[nm]; g = e8 - e; bt = e8 if g.mean() >= 0 else e; mx = ev(0.5 * w8v + 0.5 * pv); d = mx - bt
    rfm, rpl = float(V.rf(w8v, pv).mean()), float(np.corrcoef(w8v, pv)[0, 1]); tfm = float(T.rf(w8t, pt).mean())
    lr, lf = law(REC, rfm, g.mean()), law(REF, rpl, g.mean())
    rows.append(dict(nm=nm, val=float(e.mean()), gap=float(g.mean()), gap_se=se(g), rfm=rfm, rpl=rpl, tfm=tfm, lr=lr, lf=lf, d=float(d.mean()), d_se=se(d)))
    print(f'  {nm:9s} {e.mean():8.5f} {pm(g):>19s} | {rfm:.4f} / {rpl:.4f}     | {tfm:.4f}  | {lr:+8.5f} {lf:+8.5f} | {pm(d):>19s} | {lr - d.mean():+.5f} ({(lr - d.mean()) / se(d):+.2f}) | {lf - d.mean():+.5f}')
er = np.array([r['lr'] - r['d'] for r in rows]); ef = np.array([r['lf'] - r['d'] for r in rows]); ses = np.array([r['d_se'] for r in rows])
print(f'  law on record: mean {er.mean():+.5f} rms {np.sqrt((er ** 2).mean()):.5f} max {np.abs(er).max():.5f} | neural 7 rms {np.sqrt((er[:7] ** 2).mean()):.5f} | non-neural 3 rms {np.sqrt((er[7:] ** 2).mean()):.5f} | max |err|/SE {np.abs(er / ses).max():.2f}')
print(f'  refit        : mean {ef.mean():+.5f} rms {np.sqrt((ef ** 2).mean()):.5f} max {np.abs(ef).max():.5f}')
print(f'  design: corr(1 - rho, |gap|) = {np.corrcoef([1 - r["rfm"] for r in rows], [abs(r["gap"]) for r in rows])[0, 1]:.3f}; partners with rho < 0.95 and |gap| < 0.005: {sum(r["rfm"] < 0.95 and abs(r["gap"]) < 0.005 for r in rows)}')
# CatBoost vs W16
w16v, w16t = V.z(fin(V, 'W16kd')), T.z(fin(T, 'W16kd')); e16 = ev(w16v); cbv, cbt = C['CatBoost']
d = ev(0.5 * w16v + 0.5 * cbv) - e16; g = e16 - E['CatBoost']; r16 = float(V.rf(w16v, cbv).mean()); r16t = float(T.rf(w16t, cbt).mean())
print(f'  W16 val {e16.mean():.5f}; W16 + CatBoost: gap {pm(g)}, rho val fm {r16:.4f} test fm {r16t:.4f} pooled test {np.corrcoef(w16t, cbt)[0, 1]:.4f}, law rec (test rho) {law(REC, r16t, g.mean()):+.5f}, measured {pm(d)}')
tfm8 = float(T.rf(w8t, cbt).mean()); print(f'  W8 + CatBoost with the TEST rho {tfm8:.4f}: law rec {law(REC, tfm8, rows[7]["gap"]):+.5f}; pooled test {np.corrcoef(w8t, cbt)[0, 1]:.4f} refit {law(REF, np.corrcoef(w8t, cbt)[0, 1], rows[7]["gap"]):+.5f}')
bigv = V.M >= 50; dcb = ev(0.5 * w8v + 0.5 * cbv) - e8; print(f'  W8 + CatBoost measured by size: M>=50 {pm(dcb[bigv])}  M<50 {pm(dcb[~bigv])}')
OUT['B'] = rows

# ---------------------------------------------------------------- C
print('\n== C. refits and feed bootstrap (2,000 multinomial resamples, seed 777; all pairs resampled together; better file re-decided in each resample)')
fit = lambda rho, gap, gain: np.linalg.lstsq(np.stack([1 - rho, -np.abs(gap)], 1), gain, rcond=None)[0]
R = np.array([[r['rfm'], r['rpl'], r['gap'], r['d']] for r in rows])
cp = fit(R[:, 1], R[:, 2], R[:, 3]); cf = fit(R[:, 0], R[:, 2], R[:, 3]); print(f'  refit, pooled rho: a {cp[0]:.4f} b {cp[1]:.3f} | feed-mean rho: a {cf[0]:.4f} b {cf[1]:.3f}')
names = ['W8'] + NAMES; Zs = [w8v] + [C[k][0] for k in NAMES]; Es = np.stack([e8] + [E[k] for k in NAMES]); I_, J_, BL, RF = [], [], [], []
for i in range(11):
    for j in range(i + 1, 11): I_.append(i); J_.append(j); BL.append(ev(0.5 * Zs[i] + 0.5 * Zs[j])); RF.append(V.rf(Zs[i], Zs[j]))
I_, J_ = np.array(I_), np.array(J_); BL = np.stack(BL); RF = np.stack(RF)
em = Es.mean(1); gap0 = np.abs(em[I_] - em[J_]); gain0 = BL.mean(1) - np.maximum(em[I_], em[J_]); rho0 = RF.mean(1)
weak = {names.index(k) for k in ('CatBoost', 'XGBoost', 'LLM x2')}; ww = np.array([(i in weak) and (j in weak) for i, j in zip(I_, J_)])
SUB = {'10 partners': I_ == 0, '55 pairs': np.ones(55, bool), '52 pairs': ~ww}
print(f'  pairs (of 55) with rho < 0.95 and gap < 0.005: {int(((rho0 < 0.95) & (gap0 < 0.005)).sum())}; with rho < 0.95 and gap < 0.01: ' + ', '.join(f'{names[i]}+{names[j]} (rho {r:.3f}, gap {g:.4f}, gain {x:+.5f})' for i, j, r, g, x in zip(I_, J_, rho0, gap0, gain0) if r < 0.95 and g < 0.01))
B = 2000; rng = np.random.default_rng(777); CF = {k: np.empty((B, 2)) for k in SUB}
for b in range(B):
    w = rng.multinomial(V.n, np.full(V.n, 1.0 / V.n)) / V.n; e_ = Es @ w; bm = BL @ w; rm = RF @ w; gp = np.abs(e_[I_] - e_[J_]); gn = bm - np.maximum(e_[I_], e_[J_])
    for k, m in SUB.items(): CF[k][b] = fit(rm[m], gp[m], gn[m])
TG = ((0.90, 0.0), (0.90, 0.002), (0.95, 0.0), (0.86, 0.0)); zn = np.random.default_rng(778).standard_normal(B); OUT['C'] = {}
for k, m in SUB.items():
    c = fit(rho0[m], gap0[m], gain0[m]); idx = np.where(m)[0]; loo = []
    for i in idx:
        kk = m.copy(); kk[i] = False; ci = fit(rho0[kk], gap0[kk], gain0[kk]); loo.append(gain0[i] - (ci[0] * (1 - rho0[i]) - ci[1] * gap0[i]))
    lr_ = float(np.sqrt(np.mean(np.square(loo)))); a5, a95 = np.percentile(CF[k][:, 0], [5, 95]); b5, b95 = np.percentile(CF[k][:, 1], [5, 95])
    txt = f'  {k:11s}: a {c[0]:.4f} [{a5:+.4f}, {a95:+.4f}] b {c[1]:.3f} [{b5:.3f}, {b95:.3f}] LOO rms {lr_:.5f}'
    OUT['C'][k] = dict(a=float(c[0]), b=float(c[1]), a_ci=[float(a5), float(a95)], loo=lr_)
    for r_, g_ in TG:
        p_ = CF[k][:, 0] * (1 - r_) - CF[k][:, 1] * g_; lo, hi = np.percentile(p_, [5, 95]); lo2, hi2 = np.percentile(p_ + lr_ * zn, [5, 95]); lo3, hi3 = np.percentile(p_, [2.5, 97.5])
        txt += f'\n       ({r_:.2f}, {g_:.3f}): point {c[0] * (1 - r_) - c[1] * g_:+.5f} line 5-95% [{lo:+.5f}, {hi:+.5f}] with residual [{lo2:+.5f}, {hi2:+.5f}] | 2.5-97.5% [{lo3:+.5f}, {hi3:+.5f}] | share >= 0.0005: {(p_ + lr_ * zn >= 0.0005).mean():.2f}'
        OUT['C'][k][f'{r_}_{g_}'] = [float(lo), float(hi), float(lo2), float(hi2)]
    for r_ in (0.84, 0.90, 0.93):
        g0 = CF[k][:, 0] * (1 - r_) / CF[k][:, 1]; lo, hi = np.percentile(g0, [5, 95]); txt += f'\n       break-even gap at rho {r_:.2f}: 5-95% [{lo:+.4f}, {hi:+.4f}]'
    print(txt)
for r_ in (0.84, 0.90, 0.93, 0.95):
    print(f'  break-even at rho {r_:.2f}: on record gap {REC[0] * (1 - r_) / REC[1]:.4f} -> val {e8.mean() - REC[0] * (1 - r_) / REC[1]:.4f} | refit gap {REF[0] * (1 - r_) / REF[1]:.4f} -> val {e8.mean() - REF[0] * (1 - r_) / REF[1]:.4f}')
print(f'  CatBoost val {E["CatBoost"].mean():.5f}: short of break-even at rho 0.84 by {e8.mean() - REC[0] * 0.16 / REC[1] - E["CatBoost"].mean():+.4f} (on record) / {e8.mean() - REF[0] * 0.16 / REF[1] - E["CatBoost"].mean():+.4f} (refit)')

# ---------------------------------------------------------------- D
print('\n== D. equal mix of W8 scores and a partner RANK file restored on the W8 shape, explicit tie-break, minus the z-mix of the model scores (paired, val)')
key = np.random.default_rng(31).random(V.N); D = []
def order_rank(S, mix, pref):
    o = np.lexsort((key, pref, mix, S.fid)); r = np.empty(S.N); r[o] = np.arange(S.N) - S.off[S.fid[o]]; return r
print(f'  {"partner":9s} tied | {"ties->better":>19s} | {"random":>19s} | {"ties->weaker":>19s} | {"w0.4":>19s} | {"w0.6":>19s} || rank-file mix vs the better file')
for nm in NAMES:
    pv = C[nm][0]; e = E[nm]; pb = (e8.mean() - e.mean()) < 0; zr = restore(V, w8v, V.rank(pv)); zm = ev(0.5 * w8v + 0.5 * pv)
    mix = np.round(0.5 * w8v + 0.5 * zr, 8); o = np.lexsort((mix, V.fid)); ms_ = mix[o]; ff = V.fid[o]; eq = np.zeros(V.N, bool); eq[1:] = (ff[1:] == ff[:-1]) & (ms_[1:] == ms_[:-1]); t = eq.copy(); t[:-1] |= eq[1:]
    better, weaker = (zr, w8v) if pb else (w8v, zr)
    db = ev(order_rank(V, mix, better)) - zm; dw = ev(order_rank(V, mix, weaker)) - zm; dr = ev(order_rank(V, mix, key)) - zm
    d4 = ev(order_rank(V, np.round(0.6 * w8v + 0.4 * zr, 8), better)) - ev(0.6 * w8v + 0.4 * pv); d6 = ev(order_rank(V, np.round(0.4 * w8v + 0.6 * zr, 8), better)) - ev(0.4 * w8v + 0.6 * pv)
    tool = ev(order_rank(V, mix, better)) - (e if pb else e8)
    D.append((db.mean(), dr.mean(), dw.mean(), d4.mean(), d6.mean(), tool.mean()))
    print(f'  {nm:9s} {t.mean():5.1%} | {pm(db):>19s} | {pm(dr):>19s} | {pm(dw):>19s} | {pm(d4):>19s} | {pm(d6):>19s} || {pm(tool)}')
D = np.array(D); print('  mean of 10: ' + ' | '.join(f'{lab} {D[:, i].mean():+.5f} (7 neural {D[:7, i].mean():+.5f}, 3 non-neural {D[7:, i].mean():+.5f})' for i, lab in enumerate(('better', 'random', 'weaker', 'w0.4', 'w0.6'))))
OUT['D'] = D.tolist()

# ---------------------------------------------------------------- E
print('\n== E. CatBoost at small weights (information; the fixed grid of the lens, no weight is selected). (1 - w) z(pool) + w z(CatBoost) minus the pool, paired, val')
for lab, zb, eb in (('W8 final', w8v, e8), ('W16 final', w16v, e16), ('W16 train-only pool', V.z(np.load(PR / 'ens_W16kd_to_val.npy')), None)):
    eb = ev(zb) if eb is None else eb
    print(f'  {lab:20s} ({eb.mean():.5f}): ' + ' | '.join(f'w {w:.2f}: {pm(ev((1 - w) * zb + w * cbv) - eb)}' for w in (0.05, 0.1, 0.2, 0.3, 0.5)))
    d1 = ev(0.9 * zb + 0.1 * cbv) - eb; print(f'  {"":20s} w 0.10 by size: M>=50 {pm(d1[bigv])}  M<50 {pm(d1[~bigv])}')
json.dump(OUT, open(HERE / 'out/w1_calib.json', 'w'), indent=1, default=float)
print(f'\ndone in {time.time() - t0:.0f} s')
