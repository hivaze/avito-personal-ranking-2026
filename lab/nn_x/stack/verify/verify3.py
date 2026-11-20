"""Stage 3: S2 transfer diagnostics - ft/to agreement val vs test (public/private); size of the stacker's change on val vs test."""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']: os.environ.setdefault(_k, '12')
src = open('<repo>/nn_x/stack/verify/verify.py').read().split('RES = {}')[0]
exec(compile(src, 'verify.py', 'exec'))
pub = ft_['timestamp'].to_numpy() < 1790497920
for name, fams in [('G3', G3), ('PF7x', PF7x)]:
    a_v = feed_spearman(ov, Mv, half_pool(fams, 'val', 'ft'), half_pool(fams, 'val', 'to'))
    a_t = feed_spearman(ot, Mt, half_pool(fams, 'test', 'ft'), half_pool(fams, 'test', 'to'))
    log(f'{name} per-feed spearman(ft pool, to pool): val {a_v.mean():.4f} (M<50 {a_v[Mv < 50].mean():.4f}) | test public {a_t[pub].mean():.4f} private {a_t[~pub].mean():.4f} (M<50 {a_t[Mt < 50].mean():.4f})')
ON = f'{R}/nn_x/stack/nnstack/out'; runs = ['g3_gate_b3', 'g3_gate_b3_s1', 'g3_gate_b3_s2']
g3v = spec(G3, 'val'); g3t = spec(G3, 'test')
sv = np.load(f'{P}/nnstk_g3gate3r_val.npy').astype(np.float64); st = np.load(f'{P}/nnstk_g3gate3r_test.npy').astype(np.float64)
rv = feed_spearman(ov, Mv, sv, g3v); rt = feed_spearman(ot, Mt, st, g3t)
log(f'S2 spearman vs G3: val {rv.mean():.5f} (M<50 {rv[Mv < 50].mean():.5f}, M>=50 {rv[Mv >= 50].mean():.5f}) | test public {rt[pub].mean():.5f} private {rt[~pub].mean():.5f} (M<50 {rt[Mt < 50].mean():.5f}, M>=50 {rt[Mt >= 50].mean():.5f})')
log(f'M<50 share: val {(Mv < 50).mean():.3f} test public {(Mt[pub] < 50).mean():.3f} private {(Mt[~pub] < 50).mean():.3f}')
for tag, fn in [('S1', None), ('PoP3', 'agg_S3_PoP3'), ('PoP2M', 'agg_S3_PoP2M')]:
    if fn is None: continue
    v = np.load(f'{P}/{fn}_val.npy').astype(np.float64); t = np.load(f'{P}/{fn}_test.npy').astype(np.float64)
    a = feed_spearman(ov, Mv, v, g3v); b = feed_spearman(ot, Mt, t, g3t)
    log(f'{tag} spearman vs G3: val {a.mean():.5f} | test public {b[pub].mean():.5f} private {b[~pub].mean():.5f}')
