"""POST HOC block 9: P6 (in-sample family-wise p of the origin winner), V1 (contrast spread of same-recipe placebo pools vs real), V2 (5-fold-seed sign-flip null)."""
import sys, time
sys.path.insert(0, '<repo>/nn_x/w12/mix/review')
from rlib import *
t0 = time.time(); mv = meta('val'); g = groups(); M = mv['M']; H = mv['H']; big = M >= 50; n = mv['n']; y = labels(); ng = g.max() + 1
KEYS = ('C1', 'C2', 'C3', 'C4'); D0 = {k: tensor(k) - tensor(k)[:, [PLAIN]] for k in KEYS}
cs = cold_pmix('val'); GATES = [big.astype(np.int64), (cs >= 0.30).astype(np.int64), (cs >= 0.45).astype(np.int64)]
W5 = [comb(w, 0.5) for w in (0, .1, .2, .3, .4)]; R75 = np.array([[a, b] for a in W5 for b in W5]); one = np.zeros(n, np.int64)
p = lambda obs, null: (1 + (np.asarray(null) >= obs - 1e-15).sum()) / (1 + len(null))


def mx(Dsel, gates):
    return max(best_rule(block_sums(Dsel, one, cell, 2, 1)[0], R75)[0] for cell in gates) / n


OBJ = {'C3': D0['C3'], 'mean(C3,C4)': 0.5 * (D0['C3'] + D0['C4']), 'mean(C1,C2)': 0.5 * (D0['C1'] + D0['C2']), 'mean(C1..C4)': sum(D0[k] for k in KEYS) / 4, 'C1': D0['C1'], 'C2': D0['C2']}
null = {k: [[], []] for k in OBJ}; obs = {k: np.where(big, OBJ[k][:, C04], 0.0).mean() for k in OBJ}
for d in range(400):
    s = (np.random.default_rng([20261115, d]).integers(0, 2, ng) * 2.0 - 1.0)[g][:, None]
    for k in OBJ:
        Ds = s * OBJ[k]; null[k][0].append(mx(Ds, GATES)); null[k][1].append(mx(Ds, GATES[:1]))
print('=== P6: in-sample family-wise p of the fixed M-rule against the null maximum of the origin scan (75 rules) and of the M gate alone (25 rules) ===')
for k in OBJ:
    a, b = np.array(null[k][0]), np.array(null[k][1])
    print(f'  {k}: fixed M-rule {obs[k]:+.5f} | real in-sample max of the 75-rule scan {mx(OBJ[k], GATES):+.5f} | null max(75) mean {a.mean():+.5f} q95 {np.quantile(a, .95):+.5f} -> p {p(obs[k], a):.3f} | null max(25, M gate) mean {b.mean():+.5f} q95 {np.quantile(b, .95):+.5f} -> p {p(obs[k], b):.3f}')

print('\n=== V1: per-feed sd of the fixed-rule contrast, same-recipe placebo pools vs real pools ===')


def zmem(name):
    return (Z('val', np.load(P / f'nn_{name}_{SUF["val"][0]}.npy')), Z('val', np.load(P / f'nn_{name}_{SUF["val"][1]}.npy')))


for base, real in (('FD16', 'C1'), ('W16', 'C3')):
    zc = [zmem(m) for m in MEM[base]]; dr = np.where(big, D0[real][:, C04], 0.0); dall = D0[real][:, C04]
    print(f'  real {real}: per-feed sd of (M-rule - plain) {dr.std():.5f} (SE {dr.std() / np.sqrt(n):.5f}); of D = E(.4,.5)-E(.2,.5) on all feeds {dall.std():.5f}; of E(.2,.7)-E(.2,.5) {D0[real][:, comb(0.2, 0.7)].std():.5f}')
    for si, idx in enumerate(([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15])):
        rest = [i for i in range(16) if i not in idx]
        q = np.stack([Z('val', np.mean([zc[i][0] for i in rest], 0)), Z('val', np.mean([zc[i][1] for i in rest], 0)), Z('val', np.mean([zc[i][0] for i in idx], 0)), Z('val', np.mean([zc[i][1] for i in idx], 0))])
        e = {w: ndcg_feed(final('val', q, w), y, mv['off'], M) for w in (0.2, 0.25, 0.4)}; em = np.where(big, e[0.4], e[0.2]); d2 = em - e[0.2]; da = e[0.4] - e[0.2]
        e7 = ndcg_feed(final('val', q, 0.2, 0.7), y, mv['off'], M) - e[0.2]
        rho = np.mean([np.corrcoef(q[1], q[3])[0, 1]])
        print(f'    placebo split {si} ({base} members {idx} as the pseudo pool): M-rule - (0.2) {d2.mean():+.5f}, per-feed sd {d2.std():.5f} (ratio to real {d2.std() / dr.std():.2f}); D all feeds sd {da.std():.5f} (ratio {da.std() / dall.std():.2f});'
              f' w_ft .7 vs .5 sd {e7.std():.5f} (FT / TO not swapped here); row corr of base and pseudo pool (train-only) {rho:.4f}')
    qr = pools4(real); print(f'    real pools: row corr of base and popularity-light pool (train-only) {np.corrcoef(qr[1], qr[3])[0, 1]:.4f}, (fine-tuned) {np.corrcoef(qr[0], qr[2])[0, 1]:.4f}')

print('\n=== V2: sign-flip null with 5 fold seeds averaged (100 draws) vs 1 seed, C1 ===')
FL5 = [folds(g, s) for s in (7001, 7002, 7003, 7004, 7005)]; hq = np.quantile(H, [.2, .35, .5, .65, .8]); fams = [f for f in fam_defs(dict(M=M, H=H), hq) if f[0] in ('C(M50)', 'F(Hq)', 'A', 'MSTEP50', 'G(M3q)', 'K(M70,Hq)')]
acc = {f[0]: [[], []] for f in fams}
for d in range(100):
    s = (np.random.default_rng([20261115, d]).integers(0, 2, ng) * 2.0 - 1.0)[g][:, None]; r = nested(s * D0['C1'], fams, FL5, want_vec=False)
    for nm in r: acc[nm][0].append(r[nm]['seed'][0]); acc[nm][1].append(r[nm]['nested'])
real5 = nested(D0['C1'], fams, FL5, want_vec=False)
for nm in acc:
    a, b = np.array(acc[nm][0]), np.array(acc[nm][1]); print(f'  {nm}: null sd 1 seed {a.std():.5f} | 5 seeds averaged {b.std():.5f} (mean {b.mean():+.5f}) | real 5-seed nested {real5[nm]["nested"]:+.5f} -> p {p(real5[nm]["nested"], b):.3f}')
mx5 = np.max([acc[nm][1] for nm in acc], 0); ob = max(real5[nm]['nested'] for nm in acc); print(f'  family-wise over these 6 (5 seeds): observed max {ob:+.5f}, null q95 {np.quantile(mx5, .95):+.5f}, p {p(ob, mx5):.3f}')
print(f'done {time.time() - t0:.0f}s')
