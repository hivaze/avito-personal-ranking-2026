"""Stage 7: SE-estimator sensitivity for the 4 accepted deltas: iid, user-clustered, cluster bootstrap (B=4000), bootstrap percentile 2.5%."""
import os, json
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']: os.environ.setdefault(_k, '12')
src = open('<repo>/nn_x/stack/verify/verify.py').read().split('RES = {}')[0]
exec(compile(src, 'verify.py', 'exec'))
_, inv = np.unique(gv, return_inverse=True); G = inv.max() + 1; ng = np.bincount(inv).astype(float)
W = np.random.default_rng(11).multinomial(G, np.ones(G) / G, size=4000).astype(np.float32)
w1 = np.array(json.load(open(f'{R}/nn_x/stack/hfit/logs/P11_W1-W2-W3-W4-W5.json'))['variants']['W1']['w'])
s1 = 0.5 * Z(ov, Mv, half_pool(P11, 'val', 'ft')) + 0.5 * Z(ov, Mv, half_pool(P11, 'val', 'to', dict(zip(P11, w1))))
C = {'S1 vs P11ME': (s1, spec(P11, 'val')), 'S2 vs G3': (np.load(f'{P}/nnstk_g3gate3r_val.npy'), spec(G3, 'val')),
     'PoP3 vs U45': (np.load(f'{P}/agg_S3_PoP3_val.npy'), spec(U45, 'val')), 'PoP2M vs U45': (np.load(f'{P}/agg_S3_PoP2M_val.npy'), spec(U45, 'val'))}
for n, (a, b) in C.items():
    d = nd(a) - nd(b); m, i, c = se_pair(d, gv)
    bs = (W @ np.bincount(inv, d, G).astype(np.float32)) / (W @ ng.astype(np.float32))
    # per-fold (FT folds) jackknife SE
    f5 = np.load(f'{C}/fold_s2.npy') if False else np.load(f'<repo>/nn_x/stack/verify/cache/fold_s2.npy')
    jk = np.array([d[f5 != k].mean() for k in range(5)]); sej = np.sqrt(4 / 5 * ((jk - jk.mean()) ** 2).sum())
    log(f'{n:13s} d {m:+.6f} | SE iid {i:.6f} (t {m / i:.2f}) cl {c:.6f} (t {m / c:.2f}) boot {bs.std():.6f} (t {m / bs.std():.2f}) '
        f'fold-jackknife {sej:.6f} (t {m / sej:.2f}) | boot P(d<=0) {(bs <= 0).mean():.4f} 2.5% {np.quantile(bs, .025):+.6f}')
