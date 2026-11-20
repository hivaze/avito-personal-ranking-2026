# Scores about forty embedding-derived signals one at a time on val: cosine of the candidate to the
# user vector, to the mean of the last k history items, recency-weighted history similarity and
# others. Each signal is used alone as the ranking score with a random tie-break and compared per
# feed with the expected NDCG of a random order and with f1.
# How it was run: from lab/eda/embeddings (imports common.py and emb_features.py). The command line
# is not on record; outputs are 02_val_signals.log and val_signals.csv next to the script.
# What it produced: results/emb_val_signals.csv. Best single signal: recency-weighted history
# similarity with a half-life of 5 events, val NDCG 0.3492 against 0.3366 for f1.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/eda/embeddings/02_val_signals.py
"""Single-signal exact NDCG on val for embedding-derived scores (random tie-breaking),
paired comparison vs exact expected-random NDCG and vs f1."""
import numpy as np
import polars as pl
import time
from common import ndcg_per_feed
from emb_features import load_split, compute_features

t0 = time.time()
D = load_split('val')
print('loaded', time.time() - t0)
F, names = compute_features(D)
print('features', time.time() - t0, F.shape)
np.save('<repo>/eda/embeddings/val_feats.npy', F)
open('<repo>/eda/embeddings/val_feat_names.txt', 'w').write('\n'.join(names))

M = D['M']; off = D['coff']; t = D['c_target'].astype(np.float64)
nf = len(M)
fid = np.repeat(np.arange(nf), M)
rng = np.random.default_rng(123)
tieb = rng.random(len(t))

# exact expected NDCG under random order
disc_cum = np.cumsum(1.0 / np.log2(np.arange(2, 400)))
npos = np.bincount(fid, weights=t, minlength=nf)
exp_dcg = npos * disc_cum[M - 1] / M
idcg = disc_cum[np.maximum(npos.astype(int), 1) - 1]
rand_pf = np.where(npos > 0, exp_dcg / idcg, 1.0)
print(f'expected random NDCG = {rand_pf.mean():.4f}')


def to_rank_score(s):
    s = np.asarray(s, np.float64).copy()
    nan = ~np.isfinite(s)
    if nan.any():
        s[nan] = -1e18  # undefined -> bottom, ties broken randomly
    order = np.lexsort((tieb, s, fid))
    r = np.empty(len(s)); r[order] = np.arange(len(s))
    return r


def nd(s):
    return ndcg_per_feed(to_rank_score(s), t, off, M)


f1_pf = nd(D['c_f1'])
res = []
defined = np.isfinite(F)
for k, nm in enumerate(names):
    s = F[:, k]
    # coverage: feeds where signal is non-constant (defined for >=2 distinct values)
    sf = np.where(np.isfinite(s), s, np.nan)
    dfe = pl.DataFrame({'f': fid, 's': sf}).group_by('f').agg(pl.col('s').drop_nans().n_unique().alias('nu')).sort('f')
    cov = (dfe['nu'].to_numpy() >= 2)
    for sign in (1, -1):
        pf = nd(sign * s)
        d = pf - rand_pf
        d1 = pf - f1_pf
        res.append(dict(signal=nm, sign=sign, ndcg=pf.mean(), lift_vs_rand=d.mean(), se_rand=d.std(ddof=1) / np.sqrt(nf),
                        vs_f1=d1.mean(), se_f1=d1.std(ddof=1) / np.sqrt(nf), coverage=cov.mean(),
                        ndcg_cov=pf[cov].mean(), rand_cov=rand_pf[cov].mean()))
R = pl.DataFrame(res)
# keep best sign per signal
best = R.sort('ndcg', descending=True).unique(subset='signal', keep='first').sort('ndcg', descending=True)
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_float('full')
print(best.with_columns([pl.col(c).round(4) for c in ['ndcg', 'lift_vs_rand', 'se_rand', 'vs_f1', 'se_f1', 'coverage', 'ndcg_cov', 'rand_cov']]))
R.write_csv('<repo>/eda/embeddings/val_signals.csv')
print(f'f1 NDCG {f1_pf.mean():.4f}')

# within-feed Spearman correlation of top signals with f1..f5 (mean over feeds)
def feed_spearman(a, b):
    ra = to_rank_score(a); rb = to_rank_score(b)
    # convert global ranks to within-feed ranks
    ra = ra - np.repeat(off, M); rb = rb - np.repeat(off, M)
    ma = np.bincount(fid, ra, nf) / M; mb = np.bincount(fid, rb, nf) / M
    da = ra - ma[fid]; db = rb - mb[fid]
    num = np.bincount(fid, da * db, nf); den = np.sqrt(np.bincount(fid, da * da, nf) * np.bincount(fid, db * db, nf))
    ok = den > 0
    return (num[ok] / den[ok]).mean()

print('\nwithin-feed Spearman with f1..f5 / user_dot')
top = ['user_dot', 'user_cos', 'item_norm', 'hmean_cos', 'last10_cos', 'max_sim', 'smax_t0.1', 'rec_hl20', 'centrality_loo', 'cand_cat_centrality']
for nm in top:
    s = F[:, names.index(nm)]
    s = np.where(np.isfinite(s), s, -9)
    print(nm, ' '.join(f'{c}:{feed_spearman(s, D["c_" + c]):+.3f}' for c in ['f1', 'f2', 'f3', 'f4', 'f5']),
          f'user_dot:{feed_spearman(s, F[:, names.index("user_dot")]):+.3f}')
print('done', time.time() - t0)
