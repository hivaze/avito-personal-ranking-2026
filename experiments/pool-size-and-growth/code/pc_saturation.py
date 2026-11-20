# The pool-size law. It evaluates all 255 subsets of the 8 wide distilled students, averages the
# per-feed NDCG by subset size n, fits NDCG(n) = a - b / n and extrapolates to 11, 16, 32 and
# infinitely many members with a paired per-feed standard error. A second part checks the forecast
# against three new students that had finished training.
# How it was run: on CPU from lab/nn_x/w12/recheck/pool_composition; the command line is not on
# record. Output: out/saturation.log and out/saturation.json.
# What it produced: train-only branch on val, a = 0.40181 and b = 0.00250; on the holdout
# a = 0.32472 and b = 0.00215. 8 to 16 members is worth +0.00016 val and +0.00013 holdout by the
# law (results/pool_composition_saturation.log).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/recheck/pool_composition/pc_saturation.py
"""(4) Saturation of the pool with the number of d192 students.
 A. all 255 subsets of the 8 wide students (W8): mean per-feed NDCG by size n = 1..8 on val / holdout (TO branch), val FT branch, val final 0.5/0.5.
    Fit NDCG(n) = a - b/n (exact under the ambiguity law); extrapolate to 11, 16, 32, inf with a per-feed paired SE (the extrapolation is a
    linear combination of per-feed size means). The subset mean is an unbiased estimate of E[NDCG(n)] over seeds for n <= 8.
 B. check against the 3 new wide students (kd05w192 s955, kd05t25w192 s955/s957): subsets of the 11 (<= 120 random per size) and the actual W11.
 C. leave-one-out and single-member table of W8."""
from pc_common import *
from itertools import combinations
m = Meta(); st = Store(m); G = groups(); rng = np.random.default_rng(1114)
W8 = G['w192'] + G['t25w192']; NEW = G['w192new'] + G['t25w192new']; W11 = W8 + NEW
Y = dict(val=(m.yv, m.ov, m.Mv), hold=(m.yh, m.oh, m.Mh))
ev = lambda sp, s: ndcg_fast(s, *Y[sp])
Z = {sp: np.stack([np.asarray(st.get(sp, v)) for v in W11]).astype(np.float32) for sp in ('val', 'hold')}
ZL = {sp: np.stack([np.asarray(st.get(sp, v + '@last')) for v in W8]).astype(np.float32) for sp in ('val', 'hold')}
FT = np.stack([m.z('val', np.load(P/f'nn_{v}_n2cpv3s3_val_oof.npy')) for v in W8]).astype(np.float32)
zv = lambda x: m.z('val', x)
res = {}


def curve(mat, sp, idx_pool, sizes, max_sub=None, fn=None):
    """per-size mean per-feed NDCG over subsets of idx_pool; returns (sizes x n_feeds), n_subsets per size, sd across subsets of the subset means."""
    out = []; ns = []; sds = []
    for n in sizes:
        subs = list(combinations(idx_pool, n))
        if max_sub and len(subs) > max_sub: subs = [subs[i] for i in rng.choice(len(subs), max_sub, replace=False)]
        acc = np.zeros(len(Y[sp][2])); means = []
        for s in subs:
            e = fn(s) if fn else ev(sp, mat[list(s)].mean(0)); acc += e; means.append(e.mean())
        out.append(acc / len(subs)); ns.append(len(subs)); sds.append(float(np.std(means)))
    return np.array(out), ns, sds


def fit_extrap(C, sizes, targets):
    """OLS of size means on [1, -1/n]; returns coefficient rows so that value(N) = sum_n w_n(N) * mean_n (linear in the per-feed size means)."""
    X = np.stack([np.ones(len(sizes)), -1.0 / np.array(sizes)], 1); Pinv = np.linalg.pinv(X)        # (2, len(sizes))
    a = Pinv[0] @ C; b = Pinv[1] @ C                                                                # per-feed a, b
    out = {}
    for N in targets:
        v = a - (b / N if np.isfinite(N) else 0.0); out[N] = v
    return a, b, out


sizes = list(range(1, 9))
for tag, sp, fn in (('TO val', 'val', None), ('TO hold', 'hold', None),
                    ('TO val @last', 'val', lambda s: ev('val', ZL['val'][list(s)].mean(0))), ('TO hold @last', 'hold', lambda s: ev('hold', ZL['hold'][list(s)].mean(0))),
                    ('FT val', 'val', lambda s: ev('val', FT[list(s)].mean(0))),
                    ('FINAL val (0.5 zFT + 0.5 zTO)', 'val', lambda s: ev('val', 0.5 * zv(FT[list(s)].mean(0)) + 0.5 * zv(Z['val'][list(s)].mean(0))))):
    C, ns, sds = curve(Z[sp], sp, range(8), sizes, fn=fn)
    a, b, ex = fit_extrap(C, sizes, (8, 11, 16, 32, np.inf)); w8 = C[-1]
    big = (Y[sp][2] >= 50)
    r = dict(sizes=sizes, mean=[float(c.mean()) for c in C], n_subsets=ns, sd_across_subsets=sds, a=float(a.mean()), b=float(b.mean()),
             fit_resid=[float(c.mean() - (a.mean() - b.mean() / n)) for c, n in zip(C, sizes)])
    print(f'\n== {tag}: size curve from all subsets of W8')
    for n, c, k, s in zip(sizes, C, ns, sds):
        d = c - w8; print(f'  n={n}: {c.mean():.5f}  (vs W8 {d.mean():+.5f}+-{se(d):.5f}; {k} subsets, sd of subset means {s:.5f}; fit resid {c.mean() - (a.mean() - b.mean()/n):+.6f})')
    print(f'  fit NDCG(n) = a - b/n: a {a.mean():.5f} b {b.mean():.5f}')
    r['extrap'] = {}
    for N in (11, 16, 32, np.inf):
        d = ex[N] - w8; d2 = ex[N] - ex[8]
        r['extrap'][str(N)] = dict(value=float(ex[N].mean()), d_vs_w8=float(d.mean()), se=se(d), d_vs_fit8=float(d2.mean()), se_fit8=se(d2))
        print(f'  extrapolated n={N}: {ex[N].mean():.5f}  delta vs actual W8 {d.mean():+.5f}+-{se(d):.5f} | vs fitted n=8 {d2.mean():+.5f}+-{se(d2):.5f}' + (f' | M>=50 {d2[big].mean():+.5f} M<50 {d2[~big].mean():+.5f}' if sp == 'val' else ''))
    # robustness: fit on sizes 1..4 only, predict 8 (is the 1/n law predictive out of range?)
    a4, b4, ex4 = fit_extrap(C[:4], sizes[:4], (8,)); d = ex4[8] - w8
    print(f'  out-of-range check: fit on n=1..4 predicts n=8 {ex4[8].mean():.5f} vs actual {w8.mean():.5f} (error {d.mean():+.5f}+-{se(d):.5f})'); r['oor_err'] = float(d.mean()); r['oor_se'] = se(d)
    res[tag] = r

print('\n== B. the 3 new wide students as an out-of-sample check (TO branch)')
for sp in ('val', 'hold'):
    e8 = ev(sp, Z[sp][:8].mean(0)); e11 = ev(sp, Z[sp].mean(0)); d = e11 - e8
    print(f'  {sp}: actual W11 {e11.mean():.5f} vs W8 {e8.mean():.5f}: {d.mean():+.5f}+-{se(d):.5f}; law predicts {res["TO " + sp]["extrap"]["11"]["d_vs_fit8"]:+.5f}')
    C, ns, sds = curve(Z[sp], sp, range(11), list(range(1, 12)), max_sub=120)
    a, b, ex = fit_extrap(C, list(range(1, 12)), (16, 32, np.inf))
    print(f'  {sp}: 11-student curve ' + ' '.join(f'{c.mean():.5f}' for c in C))
    print(f'  {sp}: fit on 11: a {a.mean():.5f} b {b.mean():.5f}; n=16 {ex[16].mean():.5f}, n=32 {ex[32].mean():.5f}, inf {ex[np.inf].mean():.5f}; W16 - W8(actual) {fmt(ex[16]-e8)}, inf - W8 {fmt(ex[np.inf]-e8)}')
    res[f'W11 {sp}'] = dict(curve=[float(c.mean()) for c in C], a=float(a.mean()), b=float(b.mean()), w11_minus_w8=float(d.mean()), w11_minus_w8_se=se(d),
                            n16=float(ex[16].mean()), n32=float(ex[32].mean()), inf=float(ex[np.inf].mean()))
    # the new members alone vs random W8 members: are they exchangeable?
    for i, v in enumerate(W11): print(f'     {v:28s} {sp} {ev(sp, Z[sp][i]).mean():.5f}')
    # replace-k test: pools of 8 drawn from the 11 that contain all 3 new members vs W8
    subs = [s for s in combinations(range(11), 8) if {8, 9, 10} <= set(s)]
    acc = np.zeros(len(e8))
    for s in subs: acc += ev(sp, Z[sp][list(s)].mean(0))
    d = acc / len(subs) - e8; print(f'  {sp}: mean of the {len(subs)} 8-pools containing the 3 new students vs W8: {fmt(d)} (seed exchangeability; expected 0)')
    res[f'swap3 {sp}'] = dict(d=float(d.mean()), se=se(d))

print('\n== C. leave-one-out of W8 (TO), val | hold')
for i, v in enumerate(W8):
    keep = [k for k in range(8) if k != i]
    dv = ev('val', Z['val'][keep].mean(0)) - ev('val', Z['val'][:8].mean(0)); dh = ev('hold', Z['hold'][keep].mean(0)) - ev('hold', Z['hold'][:8].mean(0))
    print(f'  drop {v:26s}: val {fmt(dv)} | hold {fmt(dh)}')
json.dump(res, open(OUT/'saturation.json', 'w'), indent=1)
