"""V1. Independent re-derivation of the part-A numbers of the 'merge' study (own code; val labels only score blends).
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/next/merge_verify/v1_calib.py > nn_x/w12/next/merge_verify/out/v1_calib.txt 2>&1"""
import sys, json, time; sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify')
from v_common import *
t0 = time.time()
V, T = Sp('val'), Sp('test'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float64), yv, V.off, V.M)
LREC, LREF = (0.0186, 0.378), (0.0103, 0.304); law = lambda ab, rho, gap: ab[0] * (1 - rho) - ab[1] * abs(gap)
w8v, w8t = V.z(fin(V, 'W8kd')), T.z(fin(T, 'W8kd')); e8 = ev(w8v); C = partners(V, T)
print(f'W8 final val {e8.mean():.5f} ({V.n} feeds, {V.N} rows) | test {T.n} feeds, {T.N} rows | public feeds {(T.ts < BOUND).sum()}', flush=True)

# ---------------------------------------------------------------- A. W8 vs C3 on test, four definitions
print('\n== A. W8 file vs C3 file, test: the correlation under each definition')
sw, sc = T.read_file(ROOT / 'subs/cand_W8kd_r2.parquet'), T.read_file(ROOT / 'subs/cand_Q2a8_ftbag3_memeq.parquet')
c3t = C['C3'][1]
print(f'  file == rank of the ens blend: W8 {np.mean(T.rank(w8t) == sw):.6f}  C3 {np.mean(T.rank(c3t) == sc):.6f}')
zrw, zrc = T.z(sw), T.z(sc); big = T.M >= 50; bigr = big[T.fid]
pc = lambda a, b, m=None: float(np.corrcoef(a, b)[0, 1]) if m is None else float(np.corrcoef(a[m], b[m])[0, 1])
print(f'  pooled Spearman (np.corrcoef of z of file ranks): all {pc(zrw, zrc):.4f} | M>=50 {pc(zrw, zrc, bigr):.4f} | M<50 {pc(zrw, zrc, ~bigr):.4f}')
rs = T.rf(zrw, zrc); print(f'  feed-mean Spearman: all {rs.mean():.4f} | M>=50 {rs[big].mean():.4f} | M<50 {rs[~big].mean():.4f}')
print(f'  pooled Pearson of z of the model scores: all {pc(w8t, c3t):.4f} | M>=50 {pc(w8t, c3t, bigr):.4f} | M<50 {pc(w8t, c3t, ~bigr):.4f}')
rp = T.rf(w8t, c3t); print(f'  feed-mean Pearson of z of the model scores: all {rp.mean():.4f} ± {se(rp):.4f} | M>=50 {rp[big].mean():.4f} | M<50 {rp[~big].mean():.4f}')
def restore(S, rank_p, ours_z):
    """partner ranks -> sorted per-feed z of OUR scores at the same rank."""
    srt = ours_z[np.lexsort((ours_z, S.fid))]; return srt[S.off[S.fid] + rank_p.astype(np.int64)]
rh = T.rf(w8t, T.z(restore(T, sc, w8t))); print(f'  shape-restored (C3 file ranks on the W8 shape) feed-mean {rh.mean():.4f} | M-weighted {np.sum(rh * T.M) / T.N:.4f}')
for nm in ('S16kd', 'CatBoost', 'XGBoost', 'LLMx2', 'GRUx6'):
    pt = C[nm][1]; r1 = T.rf(w8t, pt).mean(); r2 = T.rf(w8t, T.z(restore(T, T.rank(pt), w8t))).mean(); r3 = pc(T.z(T.rank(w8t)), T.z(T.rank(pt)))
    print(f'  test {nm:9s}: truth feed-mean Pearson {r1:.4f} | shape-restored {r2:.4f} ({r2 - r1:+.4f}) | pooled Spearman {r3:.4f} ({r3 - r1:+.4f})')

# ---------------------------------------------------------------- B. calibration table on val
print('\n== B. calibration on val: gap, rho (feed-mean | pooled), laws, measured equal z-mix over the better file, paired per feed')
rows = []; PF = {}
key = np.random.default_rng(777).random(V.N)                  # my own tie key (the study used default_rng(0))
r8 = V.rank(w8v); zr8 = V.z(r8)
for nm, (pv, pt) in C.items():
    e = ev(pv); g = e8 - e; rf = V.rf(w8v, pv); rho_fm = rf.mean(); rho_pl = pc(w8v, pv); better = e8 if g.mean() >= 0 else e
    mix = ev(0.5 * w8v + 0.5 * pv); d = mix - better
    rpv = V.rank(pv)
    m1 = (0.5 * zr8.astype(np.float32) + 0.5 * V.z(rpv).astype(np.float32)).astype(np.float32)
    mix_v1 = ev(V.rank(m1, key)); mix_v1_pess = ev(m1)        # v1 way: z of ranks, ties broken at random | left tied (pessimistic)
    mix_v2 = ev(0.5 * w8v + 0.5 * V.z(restore(V, rpv, w8v)))  # v2 way: ours exact, partner ranks on our shape
    rows.append(dict(nm=nm, val=e.mean(), gap=g.mean(), gap_se=se(g), rho_fm=rho_fm, rho_pl=rho_pl, d=d.mean(), d_se=se(d), l_rec=law(LREC, rho_fm, g.mean()), l_ref=law(LREF, rho_pl, g.mean()),
                     v1=(mix_v1 - better).mean(), v1_se=se(mix_v1 - better), v2=(mix_v2 - better).mean(), v2_se=se(mix_v2 - better),
                     pen1=pm(mix_v1 - mix), pen1p=pm(mix_v1_pess - mix), pen2=pm(mix_v2 - mix), pen2m=(mix_v2 - mix).mean(), pen1m=(mix_v1 - mix).mean(),
                     ties=float(np.mean(np.diff(np.sort(m1.astype(np.float64) + V.fid * 1000.0)) == 0))))
    PF[nm] = dict(e=e, mix=mix, rf=rf, z=pv)
    r = rows[-1]
    print(f'  {nm:9s} val {r["val"]:.5f} gap {r["gap"]:+.5f} ± {r["gap_se"]:.5f} | rho fm {rho_fm:.4f} pl {rho_pl:.4f} | law rec {r["l_rec"]:+.5f} refit {r["l_ref"]:+.5f} | measured {r["d"]:+.5f} ± {r["d_se"]:.5f} |'
          f' err rec {r["l_rec"] - r["d"]:+.5f} refit {r["l_ref"] - r["d"]:+.5f}', flush=True)
er = np.array([r['l_rec'] - r['d'] for r in rows]); ef = np.array([r['l_ref'] - r['d'] for r in rows])
print(f'  law on record: mean {er.mean():+.5f} rms {np.sqrt((er ** 2).mean()):.5f} max {np.abs(er).max():.5f} | refit: mean {ef.mean():+.5f} rms {np.sqrt((ef ** 2).mean()):.5f} max {np.abs(ef).max():.5f}')
print(f'  |law error| / paired SE of the measurement: on record {np.round(np.abs(er) / np.array([r["d_se"] for r in rows]), 2).tolist()}')

print('\n== C. blend from RANK files against the z-mix of the model scores (paired per feed: rank mix minus z mix)')
print(f'  {"partner":9s} {"gap":>8s} | {"v1 way - better":>20s} {"v2 way - better":>20s} | {"v1 way - zmix (paired)":>24s} {"v1 ties left tied - zmix":>24s} {"v2 way - zmix (paired)":>24s} | tied rows in the v1 mix')
for r in rows:
    print(f'  {r["nm"]:9s} {r["gap"]:+8.5f} | {r["v1"]:+.5f} ± {r["v1_se"]:.5f}   {r["v2"]:+.5f} ± {r["v2_se"]:.5f} | {r["pen1"]:>24s} {r["pen1p"]:>24s} {r["pen2"]:>24s} | {r["ties"]:.3f}')
gaps = np.array([abs(r['gap']) for r in rows]); p2 = np.array([r['pen2m'] for r in rows]); p1 = np.array([r['pen1m'] for r in rows])
print(f'  v2-way penalty: mean {p2.mean():+.5f} min {p2.min():+.5f} max {p2.max():+.5f} | v1-way: mean {p1.mean():+.5f} min {p1.min():+.5f} max {p1.max():+.5f}')
print(f'  corr(|gap|, v2 penalty) {np.corrcoef(gaps, p2)[0, 1]:+.3f} (Spearman {np.corrcoef(np.argsort(np.argsort(gaps)), np.argsort(np.argsort(p2)))[0, 1]:+.3f}); corr(|gap|, v1 penalty) {np.corrcoef(gaps, p1)[0, 1]:+.3f}')
print('  -- alternative explanation test: is the "rank-file penalty" a property of rank files, or of giving a WEAKER partner the shape of the stronger file?')
print('     (a) roles swapped: OURS = the weaker model (scores), PARTNER = W8 as a rank file restored on the weaker shape; (b) equal-strength pairs, both directions')
def pen(ours, part):
    zm = ev(0.5 * ours + 0.5 * part); rm = ev(0.5 * ours + 0.5 * V.z(restore(V, V.rank(part), ours))); return rm - zm
for nm in ('Pall', 'Cd', 'GRUx6', 'basex4', 'CatBoost', 'LLMx2', 'XGBoost'):
    print(f'     (a) ours = {nm:9s}, partner = W8 ranks: v2-way mix minus z-mix {pm(pen(C[nm][0], w8v))}   | original direction (ours W8, partner {nm} ranks) {pm(pen(w8v, C[nm][0]))}')
for a, b in (('Pall', 'Cd'), ('GRUx6', 'basex4'), ('C3', 'G3'), ('CatBoost', 'LLMx2')):
    ga = PF[a]['e'].mean() - PF[b]['e'].mean()
    print(f'     (b) {a} + {b} (gap {ga:+.5f}, rho fm {V.rf(C[a][0], C[b][0]).mean():.4f}): ours {a}, partner {b} ranks {pm(pen(C[a][0], C[b][0]))} | ours {b}, partner {a} ranks {pm(pen(C[b][0], C[a][0]))}')
print(f'     (b) W8 + S16kd: ours W8, partner S16 ranks {pm(pen(w8v, C["S16kd"][0]))} | ours S16, partner W8 ranks {pm(pen(C["S16kd"][0], w8v))}')

# ---------------------------------------------------------------- D. refit and bootstrap
def fit(rho, gap, gain):
    X = np.stack([1 - rho, -np.abs(gap)], 1); return np.linalg.lstsq(X, gain, rcond=None)[0]
gain = np.array([r['d'] for r in rows]); gap = np.array([r['gap'] for r in rows]); rfm = np.array([r['rho_fm'] for r in rows]); rpl = np.array([r['rho_pl'] for r in rows])
c_pl, c_fm = fit(rpl, gap, gain), fit(rfm, gap, gain)
print(f'\n== D. refit on the 10 partners: pooled rho a {c_pl[0]:.4f} b {c_pl[1]:.3f} | feed-mean rho a {c_fm[0]:.4f} b {c_fm[1]:.3f} | corr(1 - rho, |gap|) {np.corrcoef(1 - rfm, np.abs(gap))[0, 1]:.3f}')
names = list(C); n = V.n; B = 2000; rng = np.random.default_rng(4242)
E = np.stack([PF[k]['e'] for k in names]); MX = np.stack([PF[k]['mix'] for k in names]); RF = np.stack([PF[k]['rf'] for k in names])
TG = ((0.95, 0.0), (0.90, 0.0), (0.90, 0.002), (0.86, 0.0)); cf = np.empty((B, 2)); cf2 = np.empty((B, 2))
for b in range(B):
    w = rng.multinomial(n, np.full(n, 1.0 / n)) / n; em = E @ w; m8 = e8 @ w; g_ = m8 - em; mm = MX @ w; rr = RF @ w
    cf[b] = fit(rr, g_, mm - np.where(gap >= 0, m8, em))      # better file fixed as on the full sample (the study's choice)
    cf2[b] = fit(rr, g_, mm - np.maximum(m8, em))             # better file re-decided in every resample
for lab, c_ in (('better fixed', cf), ('better re-decided', cf2)):
    print(f'  bootstrap ({lab}; {B} resamples of feeds): a sd {c_[:, 0].std():.4f} 5-95% [{np.percentile(c_[:, 0], 5):+.4f}, {np.percentile(c_[:, 0], 95):+.4f}] | b sd {c_[:, 1].std():.3f} [{np.percentile(c_[:, 1], 5):.3f}, {np.percentile(c_[:, 1], 95):.3f}] | corr {np.corrcoef(c_.T)[0, 1]:.3f}')
    for r_, g_ in TG:
        pr = c_[:, 0] * (1 - r_) - c_[:, 1] * g_
        print(f'     ({r_:.2f}, {g_:.3f}): on record {law(LREC, r_, g_):+.5f} | refit fm {law(c_fm, r_, g_):+.5f} | 5-95% [{np.percentile(pr, 5):+.5f}, {np.percentile(pr, 95):+.5f}] | 2.5-97.5% [{np.percentile(pr, 2.5):+.5f}, {np.percentile(pr, 97.5):+.5f}]')

# ---------------------------------------------------------------- E. 55 pairs
print('\n== E. all 55 pairs of the 11 models (equal z-mix over the better member, val feed-mean rho)')
Z = [w8v] + [C[k][0] for k in names]; Es = [e8] + [PF[k]['e'] for k in names]; nm11 = ['W8'] + names; pr = []; MIX = []; RFP = []
for i in range(11):
    for j in range(i + 1, 11):
        mx = ev(0.5 * Z[i] + 0.5 * Z[j]) if i > 0 else PF[names[j - 1]]['mix']; rf_ = V.rf(Z[i], Z[j]); g = Es[i].mean() - Es[j].mean()
        pr.append((i, j, rf_.mean(), abs(g), (mx - (Es[i] if g >= 0 else Es[j])).mean(), se(mx - (Es[i] if g >= 0 else Es[j])))); MIX.append(mx); RFP.append(rf_)
pr = np.array(pr); c55 = fit(pr[:, 2], pr[:, 3], pr[:, 4]); res = pr[:, 4] - (c55[0] * (1 - pr[:, 2]) - c55[1] * pr[:, 3]); resR = pr[:, 4] - (LREC[0] * (1 - pr[:, 2]) - LREC[1] * pr[:, 3])
print(f'  55 pairs: a {c55[0]:.4f} b {c55[1]:.3f} | rms residual {np.sqrt((res ** 2).mean()):.5f} | law on record on the 55: rms {np.sqrt((resR ** 2).mean()):.5f} | pairs with rho < 0.95 and gap < 0.005: {int(((pr[:, 2] < 0.95) & (pr[:, 3] < 0.005)).sum())}')
top = (pr[:, 0] == 0) | (pr[:, 1] == 0) | (pr[:, 0] == 5) | (pr[:, 1] == 5); hi = pr[:, 2] >= 0.95
for lab, m in (('pairs with W8 or S16 (19)', top), ('pairs among the 8 neural pools, rho >= 0.95', hi), ('pairs with rho < 0.95 (any model of CatBoost / XGBoost / LLM)', ~hi), ('pairs of two non-neural models (3)', (pr[:, 0] >= 8) & (pr[:, 1] >= 8))):
    if m.sum() >= 2:
        c_ = fit(pr[m, 2], pr[m, 3], pr[m, 4]); print(f'  subset {lab}: {int(m.sum())} pairs | own fit a {c_[0]:.4f} b {c_[1]:.3f} | law on record minus measured: mean {(-resR[m]).mean():+.5f} rms {np.sqrt((resR[m] ** 2).mean()):.5f}')
# leverage: which pairs carry the diversity coefficient of the 55-pair fit
X = np.stack([1 - pr[:, 2], -pr[:, 3]], 1); H = X @ np.linalg.inv(X.T @ X) @ X.T; lev = np.diag(H); o = np.argsort(-lev)[:6]
print('  highest-leverage pairs of the 55-pair fit: ' + '; '.join(f'{nm11[int(pr[k, 0])]}+{nm11[int(pr[k, 1])]} (h {lev[k]:.2f}, rho {pr[k, 2]:.3f}, gap {pr[k, 3]:.4f}, meas {pr[k, 4]:+.5f})' for k in o))
drop3 = ~((pr[:, 0] >= 8) & (pr[:, 1] >= 8)); c52 = fit(pr[drop3, 2], pr[drop3, 3], pr[drop3, 4])
print(f'  55-pair fit WITHOUT the 3 pairs of two non-neural models (52 pairs): a {c52[0]:.4f} b {c52[1]:.3f} -> at (0.90, 0) {law(c52, 0.90, 0):+.5f}, at (0.90, 0.002) {law(c52, 0.90, 0.002):+.5f}  (55 pairs: {law(c55, 0.90, 0):+.5f}, {law(c55, 0.90, 0.002):+.5f})')
EE = np.stack(Es); MIX = np.stack(MIX); RFP = np.stack(RFP); I_ = pr[:, 0].astype(int); J_ = pr[:, 1].astype(int); rng = np.random.default_rng(99); c5 = np.empty((B, 2)); c5d = np.empty((B, 2))
for b in range(B):
    w = rng.multinomial(n, np.full(n, 1.0 / n)) / n; em = EE @ w; bm = MIX @ w; rm = RFP @ w; gp = np.abs(em[I_] - em[J_]); gn = bm - np.maximum(em[I_], em[J_]); c5[b] = fit(rm, gp, gn); c5d[b] = fit(rm[drop3], gp[drop3], gn[drop3])
for lab, c_ in (('55 pairs', c5), ('52 pairs (no pair of two non-neural models)', c5d)):
    print(f'  bootstrap {lab}: a 5-95% [{np.percentile(c_[:, 0], 5):+.4f}, {np.percentile(c_[:, 0], 95):+.4f}] b [{np.percentile(c_[:, 1], 5):.3f}, {np.percentile(c_[:, 1], 95):.3f}]')
    for r_, g_ in TG:
        q = c_[:, 0] * (1 - r_) - c_[:, 1] * g_; print(f'     ({r_:.2f}, {g_:.3f}): 5-95% [{np.percentile(q, 5):+.5f}, {np.percentile(q, 95):+.5f}]')
print(f'\n[done in {time.time() - t0:.0f} s]')
