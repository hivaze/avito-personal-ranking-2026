"""Apply the pos-proxy to (almost) all train feeds, combine proxy z with gap share in a 2-D two-component mixture
(random component: z ~ N(0,1) fixed) per M bin, and save per-feed P(random-order) for train."""
from featlib import *
import lightgbm as lgb, time, json
from scipy.stats import norm
nb.set_num_threads(40)
from importlib import import_module
p2 = import_module('02_position')
t0 = time.time()
F = feeds('train'); nF = len(F)
S = subsample_feeds(F, 600_000, seed=11); AB = S[:400_000]
selA = S[:200_000]
dA, offA, MA = gather('train', selA, F, ['item_id', 'pos'])
tA = pl.DataFrame({'item_id': dA['item_id'], 'pos': dA['pos'], 'fid': np.repeat(np.arange(len(selA)), MA), 'M': np.repeat(MA, MA)})
tA = tA.with_columns(((pl.col('pos').rank('ordinal').over('fid') - 1) / (pl.col('M') - 1)).alias('prank'))
te_item = tA.group_by('item_id').agg(pl.col('prank').mean().alias('item_prank'), pl.len().alias('item_n'))
bst = lgb.Booster(model_file=f'{OUT}/04b_posproxy.txt'); FEAT = bst.feature_name()
rest = np.setdiff1d(np.arange(nF), AB)
z_all = np.full(nF, np.nan); rho_all = np.full(nF, np.nan)
CH = 300_000
for c0 in range(0, len(rest), CH):
    sel = rest[c0:c0 + CH]
    df, off, M, sel = build('train', sel)
    df = add_hist_sim(df, 'train', sel, off, M)
    df = df.join(te_item, on='item_id', how='left').sort('fid', maintain_order=True)
    pred = bst.predict(df.select(FEAT).to_numpy().astype(np.float32), num_threads=40)
    rho, *_ = p2.per_feed_stats(df['pos'].to_numpy().astype(np.float64), pred[None, :].astype(np.float64), df['target'].to_numpy(), off.astype(np.int64), M.astype(np.int64))
    rho_all[sel] = rho[:, 0]; z_all[sel] = rho[:, 0] * np.sqrt(M - 1)
    print(c0, time.time() - t0, flush=True)
    del df
Mall = F['M'].to_numpy(); off_all = F['cand_off'].to_numpy()
ngap, relgap, gt10 = gap_stats(np.asarray(cand('train', 'pos')).astype(np.int64), off_all, Mall)
gs = ngap / (ngap + Mall); lg = np.log(gs + 0.01)

def em2(z, g, iters=400):
    pi = 0.15; mu, s = np.median(z) + 0.5, np.std(z)
    ar, br = np.quantile(g, 0.2), np.std(g); ap, bp = np.median(g), np.std(g)
    for _ in range(iters):
        a = pi * norm.pdf(z) * norm.pdf(g, ar, br); b = (1 - pi) * norm.pdf(z, mu, s) * norm.pdf(g, ap, bp)
        r = a / (a + b + 1e-300); w = 1 - r
        pi = r.mean()
        mu = (w * z).sum() / w.sum(); s = np.sqrt((w * (z - mu) ** 2).sum() / w.sum())
        ar = (r * g).sum() / r.sum(); br = np.sqrt((r * (g - ar) ** 2).sum() / r.sum())
        ap = (w * g).sum() / w.sum(); bp = np.sqrt((w * (g - ap) ** 2).sum() / w.sum())
    return dict(pi=pi, mu=mu, s=s, ar=ar, br=br, ap=ap, bp=bp), r

def em1(z, iters=300):
    pi, mu, s = 0.2, np.median(z), np.std(z)
    for _ in range(iters):
        a = pi * norm.pdf(z); b = (1 - pi) * norm.pdf(z, mu, s); r = a / (a + b); w = 1 - r
        pi = r.mean(); mu = (w * z).sum() / w.sum(); s = np.sqrt((w * (z - mu) ** 2).sum() / w.sum())
    return pi, r

post = np.full(nF, np.nan); res = {'bins': []}
edges = [5, 10, 20, 35, 50, 75, 100, 150, 200, 250, 301]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (Mall >= lo) & (Mall < hi) & ~np.isnan(z_all)
    pars, r = em2(z_all[m], lg[m]); pi1, _ = em1(z_all[m])
    post[np.where(m)[0]] = r
    b = dict(M=f'[{lo},{hi})', n=int(m.sum()), pi_2d=float(pars['pi']), pi_zonly=float(pi1), mu_z_prod=float(pars['mu']), sd_z_prod=float(pars['s']),
             gapshare_random=float(np.exp(pars['ar']) - 0.01), gapshare_prod=float(np.exp(pars['ap']) - 0.01),
             frac_post_gt_0_9=float(np.mean(r > 0.9)), frac_post_gt_0_5=float(np.mean(r > 0.5)), mean_post=float(r.mean()))
    res['bins'].append(b); print(b, flush=True)
valid = ~np.isnan(post)
res['overall'] = dict(n_scored=int(valid.sum()), expected_random_feeds=float(np.nansum(post)), share=float(np.nanmean(post)),
                      n_post_gt_0_9=int(np.sum(post > 0.9)), n_post_gt_0_5=int(np.sum(post > 0.5)),
                      cands_post_gt_0_5=int(Mall[post > 0.5].sum()))
print(res['overall'])
np.savez(f'{OUT}/04g_train_prandom.npz', fidx=np.arange(nF), post=post.astype(np.float32), z=z_all.astype(np.float32), gap_share=gs.astype(np.float32), ngap=ngap.astype(np.int16))
json.dump(res, open(f'{OUT}/04g_full_posterior.json', 'w'), indent=1)
