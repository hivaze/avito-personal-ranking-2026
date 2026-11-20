"""w4: controls the study did not run. CPU, no GPU; val labels only to score.
A  A/A null of the LEVEL of the cold offset (two same-recipe 8-pools): is the ± of a level column a valid SE?
B  matched null for the W8P20 design: 0.8 z(W8 TO pool) + 0.2 z(4 OTHER wide students) vs z(W8 TO pool), all 70 choices of 4 from s955-958 x 2 arms;
   the real admixture at the same (train-only) level: 0.8 z(W8 TO pool) + 0.2 z(pop1kd TO pool)
C  a monotone re-scaling of a file (probabilities instead of logits): what the law-rho and the law become, and what the z-mix really does on val
D  the feeds with M < 20 (not looked at by the tool's distribution check): what damage there does to the law
E  rebuild of two cached inputs of the study"""
import sys, time, itertools
sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify/r2')
from w_common import *
P = lambda *a: print(*a, flush=True)
t0 = time.time(); V, T = Split('val'), Split('test'); coldT = T.cold(); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64); ev = lambda s: my_ndcg(V, s, yv)
GT = {'Sat': (T.day == 20722) & T.public, 'Sun am': (T.day == 20723) & T.public, 'Sun pm': (T.day == 20723) & ~T.public, 'Mon': T.day == 20724, 'Tue': T.day == 20725}
def cw(z):
    c = coldT.astype(np.float64); nc = T.fsum(c); nw = T.M - nc; both = (nc > 0) & (nw > 0)
    return np.where(both, T.fsum(z * c) / np.maximum(nc, 1) - T.fsum(z * (1 - c)) / np.maximum(nw, 1), np.nan)
def ms(x, m): v = x[m & np.isfinite(x)]; return float(v.mean()), se(v)
def con(x, a, b): ma, sa = ms(x, a); mb, sb = ms(x, b); s = float(np.hypot(sa, sb)); return ma - mb, s, (ma - mb) / s
def chi2(x):
    g = [ms(x, m) for m in GT.values()]; w = np.array([1 / q[1] ** 2 for q in g]); mu = (w * np.array([q[0] for q in g])).sum() / w.sum(); return float(sum(((q[0] - mu) / q[1]) ** 2 for q in g))
al = np.ones(T.n, bool); W = ROOT / 'nn_x/w5/preds'
names16 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)]
Z = {m: T.z(np.load(W / f'nn_{m}_trainonly_test.npy')) for m in names16}

P('== A. A/A null of the LEVEL of the cold offset: 60 random 8-vs-8 splits of the 16 wide students (seed 4242), all test feeds')
rng = np.random.default_rng(4242); lv = []; ZZ = np.stack([Z[m] for m in names16])
for it in range(60):
    p = rng.permutation(16); x = cw(T.z(ZZ[p[:8]].mean(0))) - cw(T.z(ZZ[p[8:]].mean(0))); m_, s_ = ms(x, al); lv.append((m_, s_, ms(x, T.public)[0], ms(x, ~T.public)[0]))
lv = np.array(lv); P(f'  level (pool A - pool B): sd over splits {lv[:, 0].std():.5f}; mean feed-level SE {lv[:, 1].mean():.5f}; ratio {lv[:, 0].std() / lv[:, 1].mean():.1f}; max |level| {np.abs(lv[:, 0]).max():.5f}; sd on PUBLIC {lv[:, 2].std():.5f}, on PRIVATE {lv[:, 3].std():.5f}')
cwm = np.array([ms(cw(Z[m]), al)[0] for m in names16]); P(f'  single members: mean cold-minus-warm offset (own z) {cwm.mean():+.4f}, sd between the 16 members {cwm.std():.4f} (kd05w192 {cwm[:8].mean():+.4f}, kd05t25w192 {cwm[8:].mean():+.4f})')

P('\n== B. matched null of the W8P20 design at the train-only level (PRIVATE - PUBLIC contrast of the cold offset, candidate minus the W8 TO pool)')
w8 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in (951, 952, 953, 954)]; rest = [m for m in names16 if m not in w8]
p8 = np.mean([Z[m] for m in w8], 0); z8 = T.z(p8); cw8 = cw(z8); nul = []
for combo in itertools.combinations(rest, 4):
    x = cw(T.z(0.8 * z8 + 0.2 * T.z(np.mean([Z[m] for m in combo], 0)))) - cw8; c = con(x, ~T.public, T.public); c2 = con(x, GT['Sun pm'], GT['Sun am']); nul.append((c[0], c[1], c[2], chi2(x), ms(x, al)[0], c2[2]))
nul = np.array(nul)
P(f'  null, 70 choices of 4 same-view students: contrast mean {nul[:, 0].mean():+.5f}, sd {nul[:, 0].std():.5f}; mean feed-level SE {nul[:, 1].mean():.5f}; sd of z {nul[:, 2].std():.2f}, rms of z {np.sqrt((nul[:, 2] ** 2).mean()):.2f}, max |z| {np.abs(nul[:, 2]).max():.2f}; chi2 mean {nul[:, 3].mean():.1f}, max {nul[:, 3].max():.1f}; level sd {nul[:, 4].std():.5f}; Sunday pm - am z: sd {nul[:, 5].std():.2f}')
AB = ROOT / 'nn_x/abl/preds'; pp = T.z(np.mean([T.z(np.load(AB / f'nn_abl_e10_pop1kd05t25_s{s}_trainonly_test.npy')) for s in (951, 952, 953, 954)], 0))
x = cw(T.z(0.8 * z8 + 0.2 * pp)) - cw8; c = con(x, ~T.public, T.public)
P(f'  real pop1kd admixture, train-only: level {ms(x, al)[0]:+.5f}; PUBLIC {ms(x, T.public)[0]:+.5f} PRIVATE {ms(x, ~T.public)[0]:+.5f}; contrast {c[0]:+.5f} ± {c[1]:.5f} (nominal z {c[2]:+.1f}; in null sd of the contrast {c[0] / nul[:, 0].std():+.1f}; z / null sd of z {c[2] / nul[:, 2].std():+.1f}); chi2 {chi2(x):.1f} (null max {nul[:, 3].max():.1f})')
P('  by day group: ' + ' | '.join(f'{k} {ms(x, m)[0]:+.5f} ± {ms(x, m)[1]:.5f}' for k, m in GT.items()))

P('\n== C. a monotone re-scaling (probabilities instead of logits): law-rho against the W8 scores (feed-mean Pearson of z), law on record, measured z-mix on val')
w8v, w8t = V.z(fin(V, 'W8kd')), T.z(fin(T, 'W8kd')); c3v, c3t = V.z(fin(V, 'Q2a8s3')), T.z(fin(T, 'Q2a8s3')); e8 = ev(w8v); e3 = ev(c3v)
def softmax(S, z, a):
    e = np.exp(a * z - np.maximum.reduceat(a * z, S.off)[S.fid]); return e / S.fsum(e)[S.fid]
P(f'  {"partner file":38s} | Spearman fm | law-rho test fm (val) | law on record at gap 0 / 0.001032 | gate 2 at 0.001032 | val: equal z-mix vs W8')
for lab, pt, pv, gap in (('C3 logits (as on disk)', c3t, c3v, 0.001032), ('C3 softmax(1.0 z)', softmax(T, c3t, 1.0), softmax(V, c3v, 1.0), 0.001032), ('C3 softmax(1.5 z)', softmax(T, c3t, 1.5), softmax(V, c3v, 1.5), 0.001032),
                          ('C3 softmax(3.0 z)', softmax(T, c3t, 3.0), softmax(V, c3v, 3.0), 0.001032), ('W8 itself as exp(2 z) (same order)', np.exp(2 * w8t), np.exp(2 * w8v), 0.0),
                          ('W8 itself as softmax(1.0 z)', softmax(T, w8t, 1.0), softmax(V, w8v, 1.0), 0.0)):
    zt_, zv_ = T.z(pt), V.z(pv); rho = float(T.rf(w8t, zt_).mean()); rv = float(V.rf(w8v, zv_).mean()); sp = float(T.rf(T.z(T.rank(w8t)), T.z(T.rank(pt))).mean())
    l0, lg = 0.0186 * (1 - rho), 0.0186 * (1 - rho) - 0.378 * gap; d = ev(0.5 * w8v + 0.5 * zv_) - e8
    P(f'  {lab:38s} | {sp:.4f}      | {rho:.4f} ({rv:.4f})       | {l0:+.5f} / {lg:+.5f}              | {"met" if lg - 0.0005 >= 0 else "not met":7s}            | {pm(d)} (z {d.mean() / se(d):+.2f})')

P('\n== D. feeds with M < 20 (the distribution check of the tool reads feeds with M >= 20 only)')
small = T.M < 20; P(f'  feeds with M < 20: test {int(small.sum())} ({small.mean():.1%}), public {int((small & T.public).sum())} ({(small & T.public).sum() / T.public.sum():.1%}); val {int((V.M < 20).sum())} ({(V.M < 20).mean():.1%})')
rs0 = T.rf(T.z(T.rank(w8t)), T.z(T.rank(c3t))); r0 = float(T.rf(w8t, c3t).mean())
rg = np.random.default_rng(2026)
for lab, hit in (('all M < 20 feeds negated', small), ('30% of the M < 20 feeds negated', small & (rg.random(T.n) < 0.30)), ('10% of the M < 20 feeds negated', small & (rg.random(T.n) < 0.10))):
    z = c3t.copy(); hr = np.repeat(hit, T.M); z[hr] = -z[hr]; rs = T.rf(T.z(T.rank(w8t)), T.z(T.rank(z))); rho = float(T.rf(w8t, z).mean()); big = T.M >= 20
    P(f'  C3 scores, {lab:32s}: {hit.mean():.2%} of all feeds | mean Spearman {rs.mean():.4f} | share of M >= 20 feeds below 0.3: {(rs[big] < 0.3).mean():.3%} | share of M < 20 feeds with Spearman < 0: {(rs[small] < 0).mean():.2%} | law-rho {rho:.4f} (clean {r0:.4f}) -> law rises by {0.0186 * (r0 - rho):+.5f}')
P('  genuine partners, share of M < 20 feeds with Spearman < 0 against W8 (study c2_extend T10): neural 0.000%, CatBoost 0.387%, LLM 0.494%, XGBoost 1.031%')

P('\n== E. rebuild of two cached inputs of the study (nn_x/w12/next/merge/cache)')
Mc = ROOT / 'nn_x/w12/next/merge/cache'
a = T.read_parquet(Mc / 'NOTSUB_c3_test_scores.parquet'); b = fin(T, 'Q2a8s3').astype(np.float32).astype(np.float64); P(f'  NOTSUB_c3_test_scores.parquet vs my 0.5 z(FT) + 0.5 z(TO) of preds/ens_Q2a8s3_*: max abs diff {np.abs(a - b).max():.2e}; same within-feed order {(T.rank(a) == T.rank(b)).mean():.5%}')
a = V.read_parquet(Mc / 'NOTSUB_catboost_val_ranks.parquet'); b = V.rank(np.load(ROOT / 'nn_x/cb26/data/e3_xm_yeti_val.npy').astype(np.float64)); P(f'  NOTSUB_catboost_val_ranks.parquet vs my within-feed rank of e3_xm_yeti_val.npy: identical rows {(a == b).mean():.5%}; per-feed Spearman min {V.rf(V.z(a), V.z(b)).min():.6f}')
P(f'\ndone in {time.time() - t0:.0f} s')
