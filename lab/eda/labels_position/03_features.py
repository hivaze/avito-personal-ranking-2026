"""What are f1..f5? Marginals per split, discreteness/ties, item-level vs user-level variance, (feed,category) constancy,
relation to history category counts, correlation with pos (train) and between features."""
from common import *
import json, time
nb.set_num_threads(40)
res = {}
FE = [f'f{k}' for k in range(1, 6)]

@nb.njit(parallel=True, cache=True)
def within_feed_tie_frac(x, off, M):
    """fraction of candidates whose value is shared with another candidate in same feed"""
    out = np.zeros(len(M))
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        s = np.sort(x[o:o + m]); c = 0
        for i in range(m):
            if (i > 0 and s[i] == s[i - 1]) or (i + 1 < m and s[i] == s[i + 1]):
                c += 1
        out[f] = c / m
    return out

for split in ['train', 'val', 'test']:
    F = feeds(split)
    if split == 'train':
        sel = subsample_feeds(F, 300_000, seed=1)
        d, off, M = gather(split, sel, F, FE + ['category_id', 'item_id', 'pos', 'target'])
        user = F['user_id'].to_numpy()[sel]
    else:
        off = F['cand_off'].to_numpy(); M = F['M'].to_numpy(); user = F['user_id'].to_numpy()
        d = {k: np.asarray(cand(split, k)) for k in FE + ['category_id', 'item_id'] + (['target'] if split == 'val' else [])}
    r = {}
    for k in FE:
        x = d[k]
        u, cnts = np.unique(x, return_counts=True)
        top = np.argsort(-cnts)[:5]
        r[k] = {'n_unique': int(len(u)), 'min': float(x.min()), 'max': float(x.max()), 'mean': float(x.mean()),
                'q': [float(v) for v in np.quantile(x, [.01, .1, .25, .5, .75, .9, .99])],
                'top_values': [[float(u[i]), float(cnts[i] / len(x))] for i in top],
                'frac_eq_min': float((x == x.min()).mean()), 'frac_eq_max': float((x == x.max()).mean()),
                'within_feed_tie_frac': float(within_feed_tie_frac(x.astype(np.float64), off.astype(np.int64), M.astype(np.int64)).mean()),
                'hist10': np.histogram(x, bins=10, range=(0, 1))[0].tolist()}
    # pairwise spearman correlations between features (+category) at candidate level (sample 2M)
    rng = np.random.default_rng(0); ii = rng.choice(len(d['f1']), size=min(2_000_000, len(d['f1'])), replace=False)
    df = pl.DataFrame({k: d[k][ii] for k in FE})
    r['corr_spearman'] = df.select([pl.corr(a, b, method='spearman').alias(f'{a}_{b}') for a in FE for b in FE if a < b]).to_dicts()[0]
    # (feed,category) constancy: fraction of (feed,cat) groups with >=2 items where feature is constant
    fid = np.repeat(np.arange(len(M)), M)
    g = pl.DataFrame({'fid': fid, 'cat': d['category_id'], **{k: d[k] for k in FE}, 'item': d['item_id'], 'user': np.repeat(user, M)})
    gg = g.group_by(['fid', 'cat']).agg([pl.len().alias('n')] + [pl.col(k).n_unique().alias(k) for k in FE]).filter(pl.col('n') >= 2)
    r['feedcat_const_frac'] = {k: float((gg[k] == 1).mean()) for k in FE}
    # same for random pairs of items in same feed regardless of category (baseline)
    gf = g.group_by('fid').agg([pl.len().alias('n')] + [pl.col(k).n_unique().alias(k) for k in FE])
    r['feed_const_frac'] = {k: float((gf[k] == 1).mean()) for k in FE}
    r['feed_nunique_ratio'] = {k: float((gf[k] / gf['n']).mean()) for k in FE}
    # item-level: eta^2 = share of variance explained by item_id (items occurring in >=3 feeds)
    gi = g.group_by('item').agg(pl.len().alias('n'), *[pl.col(k).var(ddof=0).alias(k + '_v') for k in FE],
                               *[pl.col(k).n_unique().alias(k + '_nu') for k in FE]).filter(pl.col('n') >= 3)
    gs = g.join(gi.select('item'), on='item')
    tot = {k: float(gs[k].var(ddof=0)) for k in FE}
    r['item_eta2'] = {k: float(1 - (gi[k + '_v'] * gi['n']).sum() / gi['n'].sum() / tot[k]) for k in FE}
    r['item_const_frac'] = {k: float((gi[k + '_nu'] == 1).mean()) for k in FE}
    r['n_items_ge3'] = int(len(gi))
    # (user,cat) eta2 vs (feed,cat) eta2
    for key, cols in [('feedcat', ['fid', 'cat']), ('usercat', ['user', 'cat']), ('cat', ['cat'])]:
        gc = g.group_by(cols).agg(pl.len().alias('n'), *[pl.col(k).var(ddof=0).alias(k + '_v') for k in FE])
        totall = {k: float(g[k].var(ddof=0)) for k in FE}
        r[f'{key}_eta2'] = {k: float(1 - (gc[k + '_v'] * gc['n']).sum() / gc['n'].sum() / totall[k]) for k in FE}
    if 'pos' in d:
        r['corr_with_pos_spearman_global'] = {k: float(pl.DataFrame({'a': d[k][ii], 'b': d['pos'][ii]}).select(pl.corr('a', 'b', method='spearman')).item()) for k in FE}
    if 'target' in d:
        # mean feature value positives vs negatives
        t = d['target'].astype(bool)
        r['mean_pos_vs_neg'] = {k: [float(d[k][t].mean()), float(d[k][~t].mean())] for k in FE}
    res[split] = r
    print(split, json.dumps({k: (v if not isinstance(v, dict) or k not in FE else {kk: vv for kk, vv in v.items() if kk in ('n_unique', 'within_feed_tie_frac', 'top_values', 'q')}) for k, v in r.items()}, indent=0)[:6000], flush=True)

json.dump(res, open(f'{OUT}/03_features.json', 'w'), indent=1)
