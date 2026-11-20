"""ft_1115 step B: final NDCG against the fine-tune weight w_ft for the stored branch files of W8P20 and FD16P20 (references W8, W16, FD16-like pools
are read from stored branch files where they exist).  file(w) = w z(FT pool) + (1 - w) z(TO pool); FT pool on val = out-of-fold (one fold model x 3 seeds per feed).
Views: plain val, pmix-public, pmix-private (src/pmix.py label-free feed weights). All deltas paired per feed against w = 0.5.
Also: paired public SE of each delta (sd of the val per-feed delta / sqrt(8562)), test correlation to the w = 0.5 file, nested pick of w (user-grouped).
Output: out/b_wft.txt, out/b_perfeed_<tag>.npz"""
from lib import *
GRID = [0.0, 0.15, 0.25, 0.35, 0.5, 0.6, 0.65, 0.75, 0.85, 1.0]
TAGS = [('W8P20', 'W8P20kd'), ('FD16P20', 'FD16P20kd'), ('W8', 'W8kd'), ('W16', 'W16kd'), ('FD8P20', 'FD8P20kd'), ('FD16P20vf', 'FD16P20vfkd'), ('G3s3', 'G3s3'), ('C3', 'Q2a8s3')]
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
E = {}; TC = {}
for name, tag in TAGS:
    cache = OUT / f'b_perfeed_{name}.npz'
    b_ft, b_to = stored(tag, 'val'); zf, zo = zv(b_ft), zv(b_to)
    if cache.exists(): e = np.load(cache)['e']
    else:
        e = np.stack([ev(w * zf + (1 - w) * zo) for w in GRID]); np.savez(cache, e=e, grid=np.array(GRID))
    E[name] = e
    t_ft, t_to = stored(tag, 'test'); tf, to_ = zt(t_ft), zt(t_to); ref = 0.5 * tf + 0.5 * to_
    TC[name] = [float(np.corrcoef(w * tf + (1 - w) * to_, ref)[0, 1]) for w in GRID]
    # per-feed Spearman-like agreement on public feeds: share of public feeds whose top-1 item is unchanged against the 0.5 file
    w_(f'## {name} (preds/ens_{tag}_*): final val at w_ft 0.5 = {e[GRID.index(0.5)].mean():.5f} | FT pool alone {e[-1].mean():.5f} | TO pool alone {e[0].mean():.5f}'
       f' | corr z(FT), z(TO): val {np.corrcoef(zf, zo)[0, 1]:.4f} test {np.corrcoef(tf, to_)[0, 1]:.4f} | RMS shift val {np.sqrt(((zf - zo) ** 2).mean()):.4f} test {np.sqrt(((tf - to_) ** 2).mean()):.4f}')
    i5 = GRID.index(0.5)
    w_('w_ft | val level | val d vs 0.5 | pmix-public d | pmix-private d | M<50 d | M>=50 d | paired public SE of d | test corr to 0.5 file')
    big = Mv >= 50
    for i, w in enumerate(GRID):
        d = e[i] - e[i5]
        w_(f'{w:.2f} | {e[i].mean():.5f} | {f(d)} | {f(d, WPUB)} | {f(d, WPRIV)} | {f(d[~big])} | {f(d[big])} | {d.std() / np.sqrt(8562):.5f} | {TC[name][i]:.4f}')
    w_()
# ---- nested pick of w (user-grouped 5-fold, 40 repeats with shuffled group assignment); pick by plain val / pmix-public / pmix-private weights on the 4 training folds
g = user_groups(); ug, gi = np.unique(g, return_inverse=True); rng = np.random.default_rng(1115)
w_('## Nested pick of w_ft (grid above; 5 user-grouped folds, 40 random fold assignments; the weight is picked on 4 folds by the stated criterion and scored on the 5th; delta against 0.5;'
   ' sd over the 40 repeats; the paired per-feed SE of a fixed-weight delta is 0.0002-0.0004)')
w_('pool | criterion | mean picked w (sd) | share of picks > 0.5 | out-of-sample val d | out-of-sample pmix-public d | out-of-sample pmix-private d')
NEST = {}
for name in E:
    e = E[name]; i5 = GRID.index(0.5); dall = e - e[i5]
    for crit, wt in (('plain val', np.ones(len(Mv))), ('pmix-public', WPUB), ('pmix-private', WPRIV)):
        res = []; picks = []
        for r in range(40):
            fold = rng.integers(0, 5, len(ug))[gi]
            oos = np.zeros(len(Mv))
            for k in range(5):
                tr = fold != k; te = ~tr
                sc = (dall[:, tr] * wt[tr]).sum(1) / wt[tr].sum(); j = int(np.argmax(sc)); picks.append(GRID[j]); oos[te] = dall[j, te]
            res.append((oos.mean(), np.average(oos, weights=WPUB), np.average(oos, weights=WPRIV)))
        res = np.array(res); picks = np.array(picks); NEST[(name, crit)] = (res.mean(0), res.std(0), picks.mean())
        w_(f'{name} | {crit} | {picks.mean():.2f} ({picks.std():.2f}) | {(picks > 0.5).mean():.2f} | {res[:, 0].mean():+.5f} (sd {res[:, 0].std():.5f}) | {res[:, 1].mean():+.5f} (sd {res[:, 1].std():.5f}) | {res[:, 2].mean():+.5f} (sd {res[:, 2].std():.5f})')
w_()
# ---- bootstrap over user groups: distribution of the argmax weight and of d(0.65), d(0.75) under each view
w_('## User-group bootstrap (2000 draws): where the maximum of the curve sits, and the sign of the two candidate weights')
w_('pool | view | P(argmax < 0.5) | P(argmax = 0.5) | P(argmax > 0.5) | median argmax | P(d(0.65) > 0) | P(d(0.75) > 0) | P(d(0.35) > 0)')
ng = len(ug); order = np.argsort(gi); cnt = np.bincount(gi, minlength=ng)
for name in ('W8P20', 'FD16P20', 'W8', 'W16', 'FD8P20', 'FD16P20vf'):
    e = E[name]; i5 = GRID.index(0.5); dall = e - e[i5]
    for view, wt in (('plain val', np.ones(len(Mv))), ('pmix-public', WPUB), ('pmix-private', WPRIV)):
        # per-group sums of weight * delta and of weight
        num = np.stack([np.bincount(gi, weights=dall[i] * wt, minlength=ng) for i in range(len(GRID))]); den = np.bincount(gi, weights=wt, minlength=ng)
        am = []; d65 = []; d75 = []; d35 = []
        for b in range(2000):
            c = np.bincount(rng.integers(0, ng, ng), minlength=ng).astype(np.float64)
            m = (num @ c) / (den @ c); am.append(GRID[int(np.argmax(m))]); d65.append(m[GRID.index(0.65)]); d75.append(m[GRID.index(0.75)]); d35.append(m[GRID.index(0.35)])
        am = np.array(am)
        w_(f'{name} | {view} | {(am < 0.5).mean():.2f} | {(am == 0.5).mean():.2f} | {(am > 0.5).mean():.2f} | {np.median(am):.2f} | {(np.array(d65) > 0).mean():.2f} | {(np.array(d75) > 0).mean():.2f} | {(np.array(d35) > 0).mean():.2f}')
open(OUT / 'b_wft.txt', 'w').write('\n'.join(out) + '\n')
