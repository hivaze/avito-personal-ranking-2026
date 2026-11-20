"""Refit the 2-D mixture with variance floors (04g diverged for M in [10,50)), estimate the M distribution of random-order
train feeds (sum of posteriors per M bin) and compare to val/test; per-feed P(random) saved for all scored train feeds."""
from common import *
import json
from scipy.stats import norm
d = np.load(f'{OUT}/04g_train_prandom.npz'); z = d['z'].astype(np.float64); gs = d['gap_share'].astype(np.float64)
F = feeds('train'); M = F['M'].to_numpy(); lg = np.log(gs + 0.01)
def em2(z, g, iters=500, floor=0.05):
    pi = 0.15; mu, s = np.median(z) + 0.5, np.std(z)
    ar, br = np.quantile(g, 0.2), np.std(g); ap, bp = np.median(g), np.std(g)
    for _ in range(iters):
        a = pi * norm.pdf(z) * norm.pdf(g, ar, br); b = (1 - pi) * norm.pdf(z, mu, s) * norm.pdf(g, ap, bp)
        r = a / (a + b + 1e-300); w = 1 - r
        pi = r.mean()
        mu = (w * z).sum() / w.sum(); s = max(np.sqrt((w * (z - mu) ** 2).sum() / w.sum()), floor)
        ar = (r * g).sum() / r.sum(); br = max(np.sqrt((r * (g - ar) ** 2).sum() / r.sum()), floor)
        ap = (w * g).sum() / w.sum(); bp = max(np.sqrt((w * (g - ap) ** 2).sum() / w.sum()), floor)
    return dict(pi=pi, mu=mu, s=s, ar=ar, br=br, ap=ap, bp=bp), r
post = np.full(len(M), np.nan); res = {'bins': []}
edges = [5, 10, 20, 35, 50, 75, 100, 150, 200, 250, 301]
V = feeds('val')['M'].to_numpy(); T = feeds('test')['M'].to_numpy()
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (M >= lo) & (M < hi) & ~np.isnan(z)
    pars, r = em2(z[m], lg[m])
    post[np.where(m)[0]] = r
    nb_all = ((M >= lo) & (M < hi)).sum()
    b = dict(M=f'[{lo},{hi})', n_scored=int(m.sum()), pi=float(pars['pi']), mu_z_prod=float(pars['mu']), sd_z_prod=float(pars['s']),
             gapshare_random=float(np.exp(pars['ar']) - 0.01), gapshare_prod=float(np.exp(pars['ap']) - 0.01),
             frac_post_gt_0_9=float(np.mean(r > 0.9)), frac_post_gt_0_5=float(np.mean(r > 0.5)),
             est_random_feeds_all=float(pars['pi'] * nb_all), train_share=float(nb_all / len(M)),
             val_share=float(((V >= lo) & (V < hi)).mean()), test_share=float(((T >= lo) & (T < hi)).mean()))
    res['bins'].append(b); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}, flush=True)
tot = sum(b['est_random_feeds_all'] for b in res['bins'])
for b in res['bins']:
    b['random_train_Mshare'] = b['est_random_feeds_all'] / tot
print('M distribution: bin, train_all, train_random(est), val, test')
for b in res['bins']:
    print(b['M'], round(b['train_share'], 3), round(b['random_train_Mshare'], 3), round(b['val_share'], 3), round(b['test_share'], 3))
tot50 = sum(b['est_random_feeds_all'] for b in res['bins'] if int(b['M'][1:].split(',')[0]) >= 50)
res['est_random_feeds_total'] = tot; res['est_random_feeds_M50plus'] = tot50
res['n_M50plus'] = int((M >= 50).sum())
print('estimated random-order feeds in train (all M, small-M bins unreliable):', tot, ' M>=50:', tot50, 'of', res['n_M50plus'])
np.savez(f'{OUT}/train_prandom.npz', fidx=np.arange(len(M)), p_random=post.astype(np.float32), z=z.astype(np.float32), gap_share=gs.astype(np.float32))
json.dump(res, open(f'{OUT}/04h_refit.json', 'w'), indent=1)
