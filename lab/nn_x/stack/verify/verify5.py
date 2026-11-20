"""Stage 5: Study C multiplicity over all 9 PoP variants (NEW = the 8 members named in the Study C summary)."""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']: os.environ.setdefault(_k, '12')
src = open('<repo>/nn_x/stack/verify/verify4.py').read().split('# ---- Study B')[0]
exec(compile(src, 'verify4.py', 'exec'))
FAM['NEW'] = ['v27_gru_m02_s511', 'v27_gru_m02_s513', 'v27_gru_m02_s515', 'v28_lr1_m02_s411', 'v28_lr1_m02_s413', 'v28_lr1_m02_s415', 'v29_d192_m02_s711', 'v29_d192_m02_s712']
NF['NEW'] = 8
for split, halves in [('val', ('ft', 'to')), ('hold', ('to',))]:
    for h in halves: FS[('NEW', split, h)] = famsum('NEW', split, h)
ALLg = U45 + ['NEW']; N1 = G3 + ['NEW']; NEWs = ['NEW']
def popv(pools, split='val'):
    if split == 'hold': return np.mean([Z(oh, Mh, half_pool(p, 'hold', 'to')) for p in pools], 0)
    return np.mean([Z(ov, Mv, spec(p, 'val')) for p in pools], 0)
V9 = {'PoP(PF7x,G3)': ([PF7x, G3], U45), 'PoP(PF7x,M2)': ([PF7x, M2], U45), 'PoP3': ([PF7x, G3, M2], U45),
      'PoP(PF7x,ALLg)': ([PF7x, ALLg], ALLg), 'PoP(G3,ALLg)': ([G3, ALLg], ALLg), 'PoP(PF7x,N1)': ([PF7x, N1], ALLg),
      'PoP(G3,N1)': ([G3, N1], N1), 'P5a': ([PF7x, G3, M2, NEWs], ALLg), 'P5b': ([PF7x, M2, NEWs], ALLg)}
names, D = [], []
for n, (pools, ref) in V9.items():
    d = nd(popv(pools)) - nd(spec(ref, 'val')); names.append(n); D.append(d)
    dh = nd(popv(pools, 'hold'), 'hold') - nd(spec(ref, 'hold'), 'hold')
    m, i, c = se_pair(d, gv); mh, ih, ch = se_pair(dh, gh)
    log(f'{n:16s} val {m:+.6f} (t {m / max(i, c):+.2f})  hold {mh:+.6f} (t {mh / max(ih, ch):+.2f})')
log('STUDY C, 9 PoP variants:'); maxt(D, names, 'PoP3'); maxt(D, names, 'PoP(PF7x,M2)')
