"""Stage 6: val delta by feed cold-share tercile (private test is colder: feed cold share val .38, public .43, private .46)."""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']: os.environ.setdefault(_k, '12')
src = open('<repo>/nn_x/stack/verify/verify.py').read().split('RES = {}')[0]
exec(compile(src, 'verify.py', 'exec'))
cs = np.bincount(fidv, cold.astype(float), len(Mv)) / Mv
q = np.quantile(cs, [1 / 3, 2 / 3]); bins = np.digitize(cs, q)
g3 = spec(G3, 'val'); u45 = spec(U45, 'val'); p11 = spec(P11, 'val')
w1 = np.array(__import__('json').load(open(f'{R}/nn_x/stack/hfit/logs/P11_W1-W2-W3-W4-W5.json'))['variants']['W1']['w'])
s1 = 0.5 * Z(ov, Mv, half_pool(P11, 'val', 'ft')) + 0.5 * Z(ov, Mv, half_pool(P11, 'val', 'to', dict(zip(P11, w1))))
C = {'S1 vs P11ME': (s1, p11), 'S1 vs G3': (s1, g3), 'S2 vs G3': (np.load(f'{P}/nnstk_g3gate3r_val.npy'), g3),
     'PoP3 vs U45': (np.load(f'{P}/agg_S3_PoP3_val.npy'), u45), 'PoP3 vs G3': (np.load(f'{P}/agg_S3_PoP3_val.npy'), g3),
     'PoP2M vs U45': (np.load(f'{P}/agg_S3_PoP2M_val.npy'), u45)}
log(f'cold-share tercile cuts {q.round(3).tolist()}')
for n, (a, b) in C.items():
    d = nd(a) - nd(b); out = []
    for k in range(3):
        m_, i_, c_ = se_pair(d[bins == k], gv[bins == k]); out.append(f'T{k} {m_:+.5f}+-{max(i_, c_):.5f}')
    # linear trend of delta on cold share
    sl = np.polyfit(cs, d, 1)[0]
    log(f'{n:14s} ' + '  '.join(out) + f'  | slope d/dcoldshare {sl:+.5f}')
