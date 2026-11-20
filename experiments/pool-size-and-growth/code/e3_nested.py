# Learned combiners over 132 members, estimated nested. The outer folds are the fine-tune's own 5
# user-grouped folds, so weights fitted on four folds score the fifth. Forms: greedy selection
# with replacement, non-negative least squares and ridge, on member columns or on arm pools, over
# all members or over the members trained on all data.
# How it was run: on CPU from lab/nn_x/w12/ens_1115, 866 s; the command line is not on record.
# Output: out/e3_nested.log and out/nested.json.
# What it produced: nested differences to the member-equal 60-member structure from -0.00083 +-
# 0.00042 to +0.00020 +- 0.00034, all below the 20-member reference file; in sample the same forms
# read up to +0.00177 (results/ens_e3_nested.log).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/ens_1115/e3_nested.py
"""e3 (e): learned combiners, NESTED estimate. Outer folds = the fine-tune's own 5 folds (GroupKFold by user, user-0 feeds as singletons; Meta.ft_folds), so no user is on both sides
and every out-of-fold fine-tune prediction is scored by weights fitted on the other four folds only. Nothing is tuned on the outer test fold; ridge lambda is chosen on inner folds
(leave-one-training-fold-out). Hyper-parameters fixed BEFORE the first run: Caruana 64 steps with replacement from the empty pool; NNLS on feed-centred labels with row weight 1 / M.
Columns: per member (or per arm pool) TWO columns: z of the fine-tuned prediction and z of the train-only prediction.
Column sets: ALL = every member with a fine-tune (132 members) | FD = members trained on all data only (60).
Forms (10 = 5 x 2 column sets): car_member, car_arm, nnls_member, nnls_arm, ridge_arm.
Compared on the same feeds with the member-equal structures A and B and FD16P20 (no fitted quantity).  Output out/nested.json."""
from ens_common import *
import time, collections
from scipy.optimize import nnls
import pmix
m = Meta(); st = Store(m); WTS = pmix.weights(); t0 = time.time(); fold = m.ft_folds(); NF = 5
print('fold sizes', np.bincount(fold), flush=True)
y = m.yv; off = m.ov; M = m.Mv; fid = m.fid['val']
npos = np.add.reduceat(y, off); DISC = 1.0 / np.log2(np.arange(2, M.max() + 3)); IDCG = np.array([DISC[:int(k)].sum() if k > 0 else 0.0 for k in npos])

@nb.njit(parallel=True, cache=True)
def cand_scores(cur, Zc, y, off, M, feeds, idcg, disc):
    nc = Zc.shape[0]; out = np.zeros(nc)
    for c in nb.prange(nc):
        tot = 0.0
        for fi in range(len(feeds)):
            f = feeds[fi]
            if idcg[f] <= 0: tot += 1.0; continue
            o = off[f]; mm = M[f]; s = np.empty(mm)
            for k in range(mm): s[k] = -(cur[o + k] + Zc[c, o + k])
            order = np.argsort(s); dcg = 0.0
            for k in range(mm):
                if y[o + order[k]] > 0: dcg += disc[k]
            tot += dcg / idcg[f]
        out[c] = tot / len(feeds)
    return out

Zp = lambda x: m.z('val', x)
A_base = W16 + FD16; A_pop = P1d + P1w + FDP12 + FDPd + FDP2 + FDPF; B_base = FD16 + FDXHZ; B_pop = FDP12 + FDPd + FDP2 + FDPF
def final(base, pop, w=0.2):
    br = [(1 - w) * Zp(st.pool(k, base)) + w * Zp(st.pool(k, pop)) for k in ('ft_val', 'to_val')]; return 0.5 * Zp(br[0]) + 0.5 * Zp(br[1])
eB = m.ev(final(B_base, B_pop)); eA = m.ev(final(A_base, A_pop)); eFD = m.ev(final(FD16, FDP4))
stored = lambda tag: 0.5 * Zp(np.load(P / f'ens_{tag}_ft_val.npy')) + 0.5 * Zp(np.load(P / f'ens_{tag}_to_val.npy')); eW8 = m.ev(stored('W8P20kd'))
print(f'B {eB.mean():.5f} A {eA.mean():.5f} FD16P20 {eFD.mean():.5f} W8P20 {eW8.mean():.5f}', flush=True)

ARMS = collections.OrderedDict([('e10 kd05w192', W16[:8]), ('e10 kd05t25w192', W16[8:]), ('fd kd05w192', FD16[:8]), ('fd kd05t25w192', FD16[8:]), ('fdxhz kd05w192', FDXHZ[:8]), ('fdxhz kd05t25w192', FDXHZ[8:]),
                                ('e10 hz@last', HZL8), ('e10 pop1 d128', P1d), ('e10 pop1 wide', P1w), ('fd pop1 wide', FDP12), ('fd pop1 d128', FDPd), ('fd pop2 wide', FDP2), ('fd popfree wide', FDPF),
                                ('kd05 d128', e10('kd05')), ('kd05t25 d128', e10('kd05t25')), ('kd08 d128', e10('kd08')), ('F7x hard', F7x), ('F7x70 hard', F7x70), ('F10 gru hard', F10), ('pop1 hard', POP1H)])
assert sum(len(v) for v in ARMS.values()) == 132 and all(st.has('ft_val', x) for v in ARMS.values() for x in v)
FDARMS = [a for a in ARMS if a.startswith('fd')]
def cols(level, cset):
    arms = list(ARMS) if cset == 'ALL' else FDARMS; names = []; X = []
    for a in arms:
        if level == 'arm':
            for k in ('ft_val', 'to_val'): X.append(Zp(st.pool(k, ARMS[a])).astype(np.float64)); names.append(f'{a}|{k[:2]}')
        else:
            for n_ in ARMS[a]:
                for k in ('ft_val', 'to_val'): X.append(np.asarray(st.get(k, n_), np.float64)); names.append(f'{n_}|{k[:2]}')
    return names, np.stack(X)

yc = (y - np.repeat(npos / M, M)).astype(np.float64); rw = np.repeat(1.0 / M, M)          # feed-centred label, row weight 1 / M
def rows_of(feeds):
    msk = np.zeros(len(M), bool); msk[feeds] = True; return np.repeat(msk, M)
def fit_caruana(X, feeds, steps=64):
    cur = np.zeros(X.shape[1]); cnt = np.zeros(X.shape[0]); path = []
    for s in range(steps):
        sc = cand_scores(cur, X, y, off, M, feeds, IDCG, DISC); j = int(np.argmax(sc)); cnt[j] += 1; cur += X[j]; path.append(float(sc[j]))
    return cnt / steps, path
def gram(X, feeds):
    r = rows_of(feeds); Xr = X[:, r] * np.sqrt(rw[r]); return Xr @ Xr.T, Xr @ (yc[r] * np.sqrt(rw[r]))
def fit_nnls(X, feeds):
    G, b = gram(X, feeds); G = G + 1e-8 * np.trace(G) / len(G) * np.eye(len(G)); L = np.linalg.cholesky(G); w, _ = nnls(L.T, np.linalg.solve(L, b), maxiter=20000); return w
def fit_ridge(X, feeds, lam):
    G, b = gram(X, feeds); return np.linalg.solve(G + lam * np.trace(G) / len(G) * np.eye(len(G)), b)
LAMS = (0.003, 0.03, 0.3, 3.0)
def score(X, w): return m.ev((w[:, None] * X).sum(0).astype(np.float32)) if False else m.ev(np.asarray(w @ X, np.float32))

RES = {}
for cset in ('FD', 'ALL'):
    for level in ('arm', 'member'):
        names, X = cols(level, cset); print(f'--- columns {cset} / {level}: {X.shape}', flush=True)
        forms = ['car', 'nnls'] + (['ridge'] if level == 'arm' else [])
        for form in forms:
            nest = np.zeros(len(M)); W_ = []; extra = {}
            for k in range(NF):
                tr = np.where(fold != k)[0]; te = fold == k
                if form == 'car': w, path = fit_caruana(X, tr)
                elif form == 'nnls': w = fit_nnls(X, tr)
                else:
                    inner = {lam: 0.0 for lam in LAMS}
                    for j in range(NF):
                        if j == k: continue
                        itr = np.where((fold != k) & (fold != j))[0]; ite = fold == j
                        for lam in LAMS: inner[lam] += score(X, fit_ridge(X, itr, lam))[ite].sum()
                    lam = max(inner, key=inner.get); w = fit_ridge(X, tr, lam); extra.setdefault('lam', []).append(lam)
                e = score(X, w); nest[te] = e[te]; W_.append(w / np.abs(w).sum())
                print(f'  {form} fold {k}: train {e[~te].mean():.5f} test {e[te].mean():.5f} | B on the test fold {eB[te].mean():.5f} | nonzero {int((w != 0).sum())} [{time.time() - t0:.0f}s]', flush=True)
            # in-sample fit on all feeds (the optimistic number)
            allf = np.arange(len(M))
            if form == 'car': wall, _ = fit_caruana(X, allf)
            elif form == 'nnls': wall = fit_nnls(X, allf)
            else: wall = fit_ridge(X, allf, max(set(extra['lam']), key=extra['lam'].count))
            ins = score(X, wall); key = f'{form}_{level}_{cset}'; Wm = np.mean(W_, 0); top = sorted(zip(names, Wm), key=lambda t: -abs(t[1]))[:12]
            def cmp(e, ref):
                d = e - ref; pp, sp = pmix.wmean_se(d, WTS['public']); pr, sr = pmix.wmean_se(d, WTS['private']); return dict(d=float(d.mean()), se=se(d), pub=float(pp), pub_se=float(sp), priv=float(pr), priv_se=float(sr))
            fold_d = [float((nest - eB)[fold == k].mean()) for k in range(NF)]
            wsum = {'ft': float(sum(w_ for n_, w_ in zip(names, Wm) if n_.endswith('|ft'))), 'to': float(sum(w_ for n_, w_ in zip(names, Wm) if n_.endswith('|to')))}
            RES[key] = dict(ncols=len(names), nested=float(nest.mean()), insample=float(ins.mean()), vsB=cmp(nest, eB), vsA=cmp(nest, eA), vsFD16P20=cmp(nest, eFD), vsW8P20=cmp(nest, eW8), insample_vsB=cmp(ins, eB),
                            fold_d_vsB=fold_d, top=[(n_, float(w_)) for n_, w_ in top], branch_weight=wsum, weights_all={n_: float(w_) for n_, w_ in zip(names, wall / np.abs(wall).sum())}, **extra)
            r = RES[key]
            print(f'{key}: nested {r["nested"]:.5f} in-sample {r["insample"]:.5f} | nested vs B {r["vsB"]["d"]:+.5f}+-{r["vsB"]["se"]:.5f} (folds {" ".join(f"{x:+.5f}" for x in fold_d)}) pmix {r["vsB"]["pub"]:+.5f} / {r["vsB"]["priv"]:+.5f} | '
                  f'vs FD16P20 {r["vsFD16P20"]["d"]:+.5f}+-{r["vsFD16P20"]["se"]:.5f} vs W8P20 {r["vsW8P20"]["d"]:+.5f}+-{r["vsW8P20"]["se"]:.5f} | in-sample vs B {r["insample_vsB"]["d"]:+.5f} | FT weight share {wsum["ft"]:.2f} | top {[(n_[:30], round(w_, 3)) for n_, w_ in top[:6]]}', flush=True)
            json.dump(RES, open(OUT / 'nested.json', 'w'), indent=0)
print(f'done {time.time() - t0:.0f}s')
