"""Covariate drift of embedding features: train subsample vs val vs test (quantiles), val first-30% vs last-70% as public/private proxy."""
import numpy as np
from emb_features import load_split, compute_features
C = '<repo>/eda/embeddings/cache'
names = open(f'{C}/feat_names.txt').read().split('\n')
Dt = load_split('test'); Fte, _ = compute_features(Dt)
np.save(f'{C}/test_feats.npy', Fte)
Fva = np.load('<repo>/eda/embeddings/val_feats.npy')
Ftr = np.load(f'{C}/train_feats.npy', mmap_mode='r')
sel = np.random.default_rng(0).choice(len(Ftr), 3_000_000, replace=False); sel.sort()
Ftr = np.asarray(Ftr[sel])
print(f'{"feature":16s} {"split":6s} {"nan":>6s} {"q10":>8s} {"q50":>8s} {"q90":>8s} {"mean":>8s}')
for nm in ['user_dot', 'rec_hl5', 'last5_cos', 'max_sim', 'hmean_cos', 'centrality_loo', 'n_sim_gt0.9', 'type1_max']:
    j = names.index(nm)
    for sp, F in [('train', Ftr), ('val', Fva), ('test', Fte)]:
        x = F[:, j]; ok = np.isfinite(x)
        q = np.quantile(x[ok], [0.1, 0.5, 0.9])
        print(f'{nm:16s} {sp:6s} {1 - ok.mean():6.3f} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f} {x[ok].mean():8.3f}')
