import json, math
import numpy as np
from scipy import stats
W = '<repo>/lab/nn_x/w12/closures_scan/'
D = json.load(open(W + 'work/merged.json'))
def f(x):
    return None if x in (None, 'None', '') else float(x)
CLOSED = ('no effect', 'hurt', 'inconclusive')
cl = [d for d in D if d['outcome'] in CLOSED and d['rtype'] == 'model']
S = [d for d in cl if f(d['val']) is not None and f(d['se'])]
d_ = np.array([f(d['val']) for d in S]); s_ = np.array([f(d['se']) for d in S]); z_ = d_ / s_
print('N rows with SE (closed model changes):', len(S), ' SE percentiles 10/25/50/75/90:', np.round(np.percentile(s_, [10, 25, 50, 75, 90]), 5))
ZBAR = 2.0  # closed rows did not pass; approximate the pass region by z >= 2 (truncation correction)

def dens(d, s, z, delta):
    """density of observed d given true delta, conditioned on 'closed' (z<ZBAR) when the row is below the bar"""
    tr = np.where(z < ZBAR, stats.norm.cdf((ZBAR * s - delta) / s), 1.0)
    return stats.norm.pdf(d, delta, s) / tr

# negative component: delta = -|N(0, tau^2)|, discretised
dgrid = np.concatenate([np.linspace(0, -0.002, 101)[1:], np.geomspace(0.002, 0.08, 120)[1:] * -1])
ddel = np.abs(np.diff(np.concatenate([[0.0], dgrid])))
Kneg = np.stack([dens(d_, s_, z_, dj) for dj in dgrid], axis=1)  # rows x grid
taus = np.geomspace(0.0002, 0.02, 40)
def neg_comp(tau, K=Kneg):
    w = stats.norm.pdf(dgrid / tau) * ddel; w = w / w.sum()
    return K @ w
NEG = np.stack([neg_comp(t) for t in taus], axis=1)  # rows x taus
C0 = dens(d_, s_, z_, 0.0)

def fit_three(Dlt, C0=C0, NEG=NEG, d=d_, s=s_, z=z_):
    C1 = dens(d, s, z, Dlt)
    pp = np.linspace(0, 0.9, 91)[:, None, None]; pn = np.linspace(0, 0.99, 100)[None, :, None]
    best = (-np.inf, None)
    prof = np.full(pp.shape[0], -np.inf)
    for ti in range(NEG.shape[1]):
        p0 = 1 - pp - pn
        ok = p0 >= 0
        L = p0[..., None] * C0 + pp[..., None] * C1 + pn[..., None] * NEG[:, ti]
        ll = np.where(ok, np.log(np.clip(L, 1e-300, None)).sum(-1), -np.inf)
        prof = np.maximum(prof, ll.max(axis=1)[:, 0] if ll.ndim == 3 else ll.max(axis=1))
        i, j, _ = np.unravel_index(np.argmax(ll), ll.shape)
        if ll[i, j, 0] > best[0]:
            best = (ll[i, j, 0], (float(pp[i, 0, 0]), float(pn[0, j, 0]), float(taus[ti])))
    llmax, (bp, bn, bt) = best
    ci = pp[:, 0, 0][prof >= llmax - 1.92]
    ti = int(np.argmin(np.abs(taus - bt)))
    c0 = (1 - bp - bn) * C0; c1 = bp * C1; c2 = bn * NEG[:, ti]
    post = c1 / (c0 + c1 + c2)
    return dict(pi_pos=bp, pi_neg=bn, pi_null=1 - bp - bn, tau=bt, ci=[float(ci.min()), float(ci.max())], post=post, ll=float(llmax))

out = {}
for Dlt in (0.0003, 0.0005):
    r = fit_three(Dlt)
    n = len(d_)
    print(f'\nTHREE-GROUP null / +{Dlt} / negative half-normal on {n} rows: pi_null {r["pi_null"]:.2f} pi_pos {r["pi_pos"]:.2f} pi_neg {r["pi_neg"]:.2f} tau {r["tau"]:.5f}')
    print(f'  expected count of +{Dlt} effects (sum of posteriors) {r["post"].sum():.1f}; profile 95% CI for pi_pos [{r["ci"][0]:.2f}, {r["ci"][1]:.2f}] -> count {r["ci"][0]*n:.0f}-{r["ci"][1]*n:.0f}')
    top = np.argsort(-r['post'])[:12]
    print('  highest posteriors (line, post, z):', [(S[i]['line'], round(float(r['post'][i]), 2), round(float(z_[i]), 2)) for i in top])
    out[f'three_{Dlt}'] = dict(n=n, pi_pos=r['pi_pos'], pi_neg=r['pi_neg'], pi_null=r['pi_null'], tau=r['tau'], ci=r['ci'], exp=float(r['post'].sum()),
                               post={S[i]['line']: float(r['post'][i]) for i in range(n)})

# TWO-GROUP null vs +Delta on rows not closed as a clear loss
NC = np.array([d['reason'] != 'clear loss' for d in S])
d2, s2, z2 = d_[NC], s_[NC], z_[NC]
for Dlt in (0.0003, 0.0005):
    c0 = dens(d2, s2, z2, 0.0); c1 = dens(d2, s2, z2, Dlt)
    grid = np.linspace(0, 1, 1001)
    ll = np.array([np.log((1 - g) * c0 + g * c1).sum() for g in grid])
    g = grid[np.argmax(ll)]; ci = grid[ll >= ll.max() - 1.92]
    post = g * c1 / ((1 - g) * c0 + g * c1)
    lr = stats.norm.pdf(d2, Dlt, s2) / stats.norm.pdf(d2, 0, s2)
    print(f'\nTWO-GROUP null vs +{Dlt} on {len(d2)} rows not closed as clear loss: pi_pos {g:.3f} 95% CI [{ci.min():.2f}, {ci.max():.2f}] -> count {post.sum():.1f} (CI {ci.min()*len(d2):.0f}-{ci.max()*len(d2):.0f})')
    print(f'  per-row LR(+{Dlt} vs 0) > 1: {(lr>1).sum()}, > 3: {(lr>3).sum()}, < 1/3: {(lr<1/3).sum()}; mean z of these rows {z2.mean():+.2f} (sd {z2.std(ddof=1):.2f})')
    out[f'two_{Dlt}'] = dict(n=int(len(d2)), pi_pos=float(g), ci=[float(ci.min()), float(ci.max())], exp=float(post.sum()),
                             lr_gt1=int((lr > 1).sum()), lr_gt3=int((lr > 3).sum()), lr_lt3=int((lr < 1 / 3).sum()), meanz=float(z2.mean()), sdz=float(z2.std(ddof=1)))

# Simple moment check: if k of the n rows carried +Delta, mean z rises by (1/n) sum_{real} Delta/s
mz = z2.mean(); se_mz = 1 / math.sqrt(len(z2))
avg_shift = np.mean(0.0003 / s2)
print(f'\nMoment check (rows not clear loss): mean z {mz:+.3f} +- {se_mz:.3f}; a real +0.0003 shifts its row by {avg_shift:.2f} z on average;'
      f' upper 95% of mean z = {mz + 1.96*se_mz:+.3f} -> share real <= {(mz + 1.96*se_mz)/avg_shift:.2f} if all other rows were exact nulls')
out['moment'] = dict(meanz=float(mz), se=float(se_mz), shift=float(avg_shift), upper_share=float((mz + 1.96 * se_mz) / avg_shift))

# Power of the closing tests for +0.0003 / +0.0005, approximated as z >= 2 on validation alone
pw = 1 - stats.norm.cdf(2 - 0.0003 / s_); pw5 = 1 - stats.norm.cdf(2 - 0.0005 / s_)
lr_closure = (1 - pw) / (1 - stats.norm.sf(2))
print(f'\nPower (z>=2) at +0.0003: median {np.median(pw):.3f} IQR {np.percentile(pw,25):.3f}-{np.percentile(pw,75):.3f} max {pw.max():.3f}; rows >= 0.5: {(pw>=0.5).sum()} of {len(pw)}')
print(f'Power at +0.0005: median {np.median(pw5):.3f}; rows >= 0.5: {(pw5>=0.5).sum()}; rows >= 0.8: {(pw5>=0.8).sum()}')
print(f'LR carried by "did not pass" (real +0.0003 vs null): median {np.median(lr_closure):.3f}, min {lr_closure.min():.3f}')
out['power'] = dict(median=float(np.median(pw)), q25=float(np.percentile(pw, 25)), q75=float(np.percentile(pw, 75)), max=float(pw.max()),
                    n50=int((pw >= 0.5).sum()), n=len(pw), median5=float(np.median(pw5)), n50_5=int((pw5 >= 0.5).sum()), n80_5=int((pw5 >= 0.8).sum()),
                    lr_closure_median=float(np.median(lr_closure)), lr_closure_min=float(lr_closure.min()),
                    se_pct=[float(x) for x in np.percentile(s_, [10, 25, 50, 75, 90])])
out['power_rows'] = {S[i]['line']: float(pw[i]) for i in range(len(S))}
json.dump(out, open(W + 'work/mixture.json', 'w'), indent=1)
