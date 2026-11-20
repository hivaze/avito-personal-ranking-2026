"""w2: independent re-derivation of the drift table of reports/NEXT_1115/merge.md (r2), its A/A null, and two controls the study did not run. No labels.
A  cold-minus-warm offset (score-z), candidate minus W8, by val day / test-day group / PUBLIC / PRIVATE; contrasts; chi2 over the 5 test-day groups
B  agreement of the two FILES (per-feed Spearman, same top-1), PUBLIC / PRIVATE
C  A/A null: random 8-vs-8 splits of the 16 wide students (own seed, 60 splits): sd of the z of the contrasts
D  control 1: SE of the PRIVATE - PUBLIC contrast under time-block resampling (blocks of 1 hour) instead of independent feeds
E  control 2: is the W8P20 step at the public bound a composition effect? contrast inside strata of feed size x cold share; and the two halves of Sunday
F  CatBoost vs W16: Spearman, top-1, cold offset (rank-z) by PUBLIC / PRIVATE"""
import sys, time
sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify/r2')
from w_common import *
P = lambda *a: print(*a, flush=True)
t0 = time.time(); V, T = Split('val'), Split('test'); coldT, coldV = T.cold(), V.cold()
DN = {20718: 'val Tue', 20720: 'val Thu', 20722: 'Sat', 20723: 'Sun', 20724: 'Mon', 20725: 'Tue'}
GT = {'Sat (pub)': (T.day == 20722) & T.public, 'Sun am (pub)': (T.day == 20723) & T.public, 'Sun pm (priv)': (T.day == 20723) & ~T.public, 'Mon (priv)': T.day == 20724, 'Tue (priv)': T.day == 20725}
P('test-day groups: ' + ', '.join(f'{k} {int(m.sum())}' for k, m in GT.items()) + f' | public {int(T.public.sum())} private {int((~T.public).sum())} | feeds outside the 5 groups: {int(T.n - sum(m.sum() for m in GT.values()))}')
GV = {'val Tue': V.day == 20718, 'val Thu': V.day == 20720}; P('val: ' + ', '.join(f'{k} {int(m.sum())}' for k, m in GV.items()))


def cw(S, cold, z):
    c = cold.astype(np.float64); nc = S.fsum(c); nw = S.M - nc; both = (nc > 0) & (nw > 0)
    g = S.fsum(z * c) / np.maximum(nc, 1) - S.fsum(z * (1 - c)) / np.maximum(nw, 1); return np.where(both, g, np.nan)


def ms(x, m):
    v = x[m & np.isfinite(x)]; return float(v.mean()), se(v), len(v)


def con(x, a, b):
    ma, sa, _ = ms(x, a); mb, sb, _ = ms(x, b); s = float(np.hypot(sa, sb)); return ma - mb, s, (ma - mb) / s


def chi2(x):
    g = [ms(x, m) for m in GT.values()]; w = np.array([1 / q[1] ** 2 for q in g]); mu = (w * np.array([q[0] for q in g])).sum() / w.sum(); return float(sum(((q[0] - mu) / q[1]) ** 2 for q in g))


def scores(S, tag):
    return S.z(np.load(PR / f'ens_{tag}_to_{S.name}.npy')) if tag == 'W8TOkd' else S.z(fin(S, tag))


CANDS = (('W16', 'W16kd', 'subs/cand_W16kd_r3.parquet'), ('W8P20', 'W8P20kd', 'subs/cand_W8P20kd_r3.parquet'), ('W8TO', 'W8TOkd', 'subs/cand_W8TOkd_r3.parquet'),
         ('S16', 'S16kd', 'subs/cand_S16kd_r2.parquet'), ('C3', 'Q2a8s3', 'subs/cand_Q2a8_ftbag3_memeq.parquet'))
rT, rV = scores(T, 'W8kd'), scores(V, 'W8kd'); fW8 = T.read_parquet(ROOT / 'subs/cand_W8kd_r2.parquet'); rkW8 = T.rank(fW8)
P(f'W8 file order reproduced by preds/ens_W8kd_*: {(rkW8 == T.rank(rT)).mean():.6%}')
cwrT, cwrV = cw(T, coldT, rT), cw(V, coldV, rV); XS = {}
P('\n== A. cold-minus-warm offset, candidate minus W8, z of the model scores (mean ± SE over feeds)')
P(f'  {"cand":6s} | ' + ' | '.join(f'{k:>16s}' for k in list(GV) + list(GT) + ['PUBLIC', 'PRIVATE']) + ' | PRIVATE-PUBLIC (z) | last-val Tue (z) | last-first (z) | chi2(4)')
top1 = lambda S, s: np.lexsort((-s, S.fid))[S.off]
for nm, tag, path in CANDS:
    cT, cV = scores(T, tag), scores(V, tag); x = cw(T, coldT, cT) - cwrT; xv = cw(V, coldV, cV) - cwrV; XS[nm] = x
    cells = [ms(xv, m) for m in GV.values()] + [ms(x, m) for m in GT.values()] + [ms(x, T.public), ms(x, ~T.public)]
    c1 = con(x, ~T.public, T.public); lv = ms(x, GT['Tue (priv)']); vt = ms(xv, GV['val Tue']); s3 = float(np.hypot(lv[1], vt[1])); c2 = con(x, GT['Tue (priv)'], GT['Sat (pub)'])
    P(f'  {nm:6s} | ' + ' | '.join(f'{c[0]:+.4f} ± {c[1]:.4f}' for c in cells) + f' | {c1[0]:+.4f} ± {c1[1]:.4f} ({c1[2]:+.1f}) | {lv[0] - vt[0]:+.4f} ± {s3:.4f} ({(lv[0] - vt[0]) / s3:+.1f}) | {c2[0]:+.4f} ({c2[2]:+.1f}) | {chi2(x):.1f}')
P('\n== B. agreement of the two FILES (per-feed Spearman, same top-1 item), mean over feeds')
for nm, tag, path in CANDS:
    f = T.read_parquet(ROOT / path); rk = T.rank(f); sp = T.rf(T.z(rk), T.z(rkW8)); t1 = (top1(T, f) == top1(T, fW8)).astype(float)
    same = (rk == T.rank(scores(T, tag))).mean()
    P(f'  {nm:6s} Spearman PUBLIC {sp[T.public].mean():.4f} PRIVATE {sp[~T.public].mean():.4f} | by day ' + ' / '.join(f'{sp[m].mean():.4f}' for m in GT.values()) + f' | same top-1 all {t1.mean():.1%} PUBLIC {t1[T.public].mean():.1%} PRIVATE {t1[~T.public].mean():.1%} | file order reproduced by preds/ens_*: {same:.5%}')

P('\n== C. A/A null of the cold-offset contrasts: 60 random 8-vs-8 splits of the 16 wide students (train-only test predictions), seed 4242')
W = ROOT / 'nn_x/w5/preds'; Z = np.stack([T.z(np.load(W / f'nn_w5_e10_{a}_s{s}_trainonly_test.npy')) for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)])
CW = np.stack([cw(T, coldT, z) for z in Z])         # cw is linear in z, but the pool is re-standardised per feed: recomputed below, CW is only used for the check
rng = np.random.default_rng(4242); res = []
for it in range(60):
    p = rng.permutation(16); a, b = T.z(Z[p[:8]].mean(0)), T.z(Z[p[8:]].mean(0)); x = cw(T, coldT, a) - cw(T, coldT, b)
    c1 = con(x, ~T.public, T.public); c2 = con(x, GT['Tue (priv)'], GT['Sat (pub)']); res.append((c1[0], c1[2], c2[0], c2[2], chi2(x), ms(x, np.ones(T.n, bool))[0] / ms(x, np.ones(T.n, bool))[1]))
res = np.array(res)
P(f'  PRIVATE - PUBLIC: mean diff {res[:, 0].mean():+.5f}, sd of diff {res[:, 0].std():.5f}, sd of z {res[:, 1].std():.2f}, rms of z {np.sqrt((res[:, 1] ** 2).mean()):.2f}, max |z| {np.abs(res[:, 1]).max():.2f}')
P(f'  last - first day: mean diff {res[:, 2].mean():+.5f}, sd of diff {res[:, 2].std():.5f}, sd of z {res[:, 3].std():.2f}, rms of z {np.sqrt((res[:, 3] ** 2).mean()):.2f}, max |z| {np.abs(res[:, 3]).max():.2f}')
P(f'  chi2 over the 5 day groups: mean {res[:, 4].mean():.2f} (expected 4), max {res[:, 4].max():.1f}, share above 9.49 (p 0.05): {(res[:, 4] > 9.49).mean():.2f}; share above 14.9 / 15.6 / 18.5: {(res[:, 4] > 14.9).mean():.2f} / {(res[:, 4] > 15.6).mean():.2f} / {(res[:, 4] > 18.5).mean():.2f}')
P(f'  z of the overall level (all test feeds): sd {res[:, 5].std():.2f}, rms {np.sqrt((res[:, 5] ** 2).mean()):.2f}')
P('  note: the 60 splits share members (16 students), so they are not independent draws; the sd carries roughly ± 15-20%.')

P('\n== D. control 1: SE of PRIVATE - PUBLIC under resampling of 1-hour time blocks (1,000 resamples, blocks drawn inside public and inside private)')
hr = T.ts // 3600
def block_se(x, n_boot=1000, seed=9):
    r = np.random.default_rng(seed); out = []
    for part in (T.public, ~T.public):
        ok = part & np.isfinite(x); hs, inv = np.unique(hr[ok], return_inverse=True); sx = np.bincount(inv, x[ok]); nx = np.bincount(inv).astype(float); k = len(hs)
        draws = r.integers(0, k, (n_boot, k)); out.append(sx[draws].sum(1) / nx[draws].sum(1))
    return float(np.std(out[1] - out[0])), [len(np.unique(hr[T.public])), len(np.unique(hr[~T.public]))]
for nm in ('W16', 'W8P20', 'W8TO', 'S16', 'C3'):
    c1 = con(XS[nm], ~T.public, T.public); bs, nb = block_se(XS[nm]); P(f'  {nm:6s} PRIVATE - PUBLIC {c1[0]:+.5f}: feed-level SE {c1[1]:.5f} (z {c1[2]:+.1f}) | hour-block SE {bs:.5f} (z {c1[0] / bs:+.1f}; ratio {bs / c1[1]:.2f}; blocks public / private {nb})')
xa = cw(T, coldT, T.z(Z[:8].mean(0))) - cw(T, coldT, T.z(Z[8:].mean(0))); c1 = con(xa, ~T.public, T.public); bs, _ = block_se(xa); P(f'  A/A (kd05w192 x8 vs kd05t25w192 x8): {c1[0]:+.5f}, feed-level SE {c1[1]:.5f}, hour-block SE {bs:.5f} (ratio {bs / c1[1]:.2f})')

P('\n== E. control 2: W8P20 minus W8 cold offset, PUBLIC vs PRIVATE inside strata (feed size x cold share of the feed); and the two halves of Sunday')
x = XS['W8P20']; cs = T.fsum(coldT.astype(float)) / T.M; okx = np.isfinite(x)
mb = np.digitize(T.M, [20, 50, 100, 200]); cb_ = np.digitize(cs, [0.2, 0.35, 0.5, 0.65]); num = den = 0.0; var = 0.0; tab = []
for i in range(5):
    for j in range(5):
        m = (mb == i) & (cb_ == j) & okx; a, b = m & T.public, m & ~T.public
        if a.sum() < 30 or b.sum() < 30: continue
        d = x[b].mean() - x[a].mean(); v = x[b].var() / b.sum() + x[a].var() / a.sum(); wgt = m.sum(); num += wgt * d; den += wgt; var += wgt ** 2 * v; tab.append((i, j, int(a.sum()), int(b.sum()), d))
P(f'  stratified PRIVATE - PUBLIC (25 strata of M x cold share, weighted by stratum size): {num / den:+.5f} ± {np.sqrt(var) / den:.5f} | unstratified {con(x, ~T.public, T.public)[0]:+.5f} ± {con(x, ~T.public, T.public)[1]:.5f} | strata used {len(tab)}, negative in {sum(t[4] < 0 for t in tab)}')
P(f'  mean cold share of the feeds: public {cs[T.public].mean():.3f} private {cs[~T.public].mean():.3f}; share of feeds with M < 50: public {(T.M[T.public] < 50).mean():.3f} private {(T.M[~T.public] < 50).mean():.3f}')
c = con(x, GT['Sun pm (priv)'], GT['Sun am (pub)']); P(f'  Sunday only: pm (private) - am (public) {c[0]:+.5f} ± {c[1]:.5f} (z {c[2]:+.1f})')
c = con(x, GT['Sun am (pub)'], GT['Sat (pub)']); P(f'  inside public: Sun am - Sat {c[0]:+.5f} ± {c[1]:.5f} (z {c[2]:+.1f})')
c = con(x, GT['Tue (priv)'], GT['Sun pm (priv)']); P(f'  inside private: Tue - Sun pm {c[0]:+.5f} ± {c[1]:.5f} (z {c[2]:+.1f})')
hod = ((T.ts + 3 * 3600) % 86400) // 3600; am = hod < 11
for lab, dm in (('Sat', T.day == 20722), ('Mon', T.day == 20724), ('Tue', T.day == 20725)):
    c = con(x, dm & ~am, dm & am); P(f'  time-of-day placebo, {lab}: after 11:00 MSK - before {c[0]:+.5f} ± {c[1]:.5f} (z {c[2]:+.1f}); feeds {int((dm & am).sum())} / {int((dm & ~am).sum())}')

P('\n== F. CatBoost (e3_xm_yeti) vs the W16 file')
f16 = T.read_parquet(ROOT / 'subs/cand_W16kd_r3.parquet'); cbt = np.load(ROOT / 'nn_x/cb26/data/e3_xm_yeti_test.npy').astype(np.float64); r16, rcb = T.rank(f16), T.rank(cbt)
sp = T.rf(T.z(rcb), T.z(r16)); t1 = (top1(T, cbt) == top1(T, f16)).astype(float); xr = cw(T, coldT, T.z(rcb)) - cw(T, coldT, T.z(r16))
P(f'  Spearman all {sp.mean():.4f} ± {se(sp):.4f} PUBLIC {sp[T.public].mean():.4f} PRIVATE {sp[~T.public].mean():.4f} | same top-1 {t1.mean():.1%} | cold offset (rank-z) all {ms(xr, np.ones(T.n, bool))[0]:+.4f} ± {ms(xr, np.ones(T.n, bool))[1]:.4f} PUBLIC {ms(xr, T.public)[0]:+.4f} PRIVATE {ms(xr, ~T.public)[0]:+.4f} last day {ms(xr, GT["Tue (priv)"])[0]:+.4f} chi2 {chi2(xr):.1f}')
w8t = rT; cz = T.z(cbt); rr = T.rf(w8t, cz)
P(f'  CatBoost vs W8 law-rho (feed-mean Pearson of z) by group: ' + ' | '.join(f'{k} {rr[m].mean():.4f} ± {se(rr[m]):.4f}' for k, m in GT.items()) + f' | PUBLIC {rr[T.public].mean():.4f} PRIVATE {rr[~T.public].mean():.4f} | M<50 {rr[T.M < 50].mean():.4f} M>=50 {rr[T.M >= 50].mean():.4f}')
P(f'\ndone in {time.time() - t0:.0f} s')
