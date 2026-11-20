import sys; sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review')
from rv_common import *
wv, wh = Wit('val'), Wit('hold')
S = ROOT / 'nn_x/w12/cb/famS/staged'; P = ROOT / 'nn_x/w12/cb/famS/preds'
def E(t):
    d = S if '@' in t else P
    return wv.ndcg(np.load(d / f'{t}_val.npy')), wh.ndcg(np.load(d / f'{t}_hold.npy'))
def cmp(a, b, note=''):
    (av, ah), (bv, bh) = E(a), E(b); dv, dh = av - bv, ah - bh
    print(f'{a} - {b}: {dv.mean():+.5f}+-{se(dv):.5f} | {dh.mean():+.5f}+-{se(dh):.5f} | z {(dv.mean() / se(dv) + dh.mean() / se(dh)) / np.sqrt(2):+.2f}  [levels {av.mean():.5f}/{ah.mean():.5f} vs {bv.mean():.5f}/{bh.mean():.5f}] {note}', flush=True)
L = 's1f_rmse_d8_lr030_s1long'; F = 's1f_rmse_d8_lr030_s0'
cmp(f'{L}@1761', f'{L}@1400', 'study: -0.00013+-0.00033 | +0.00010+-0.00017')
cmp(f'{L}@1400', f'{L}@886', 'study: +0.00151+-0.00042 | +0.00088+-0.00022')
cmp(f's3f_pca_d8_lr030_s0@721', f'{F}@721', 'study: -0.00018+-0.00087 | +0.00036+-0.00049')
cmp('s2_rmse_d10_lr030_s0', f'{F}@721', 'study: -0.00152+-0.00075 | -0.00065+-0.00044')
cmp('s2_rmse_d10_lr030_s0@300', f'{F}@300', 'study: +0.00006+-0.00072 | +0.00099+-0.00044')
cmp('s4b_qsm_soft_lr030_s0', F, 'study: +0.00052+-0.00084 | +0.00124+-0.00047')
cmp('s4b_qsm_soft_lr030_s0@300', f'{F}@300', 'study: +0.00185+-0.00084 | +0.00264+-0.00049')
cmp('s4b_qsm_soft_lr030_s0', 's4_qsm_kdmix_lr030_s0', 'study: +0.00435+-0.00103 | +0.00359+-0.00055')
cmp('s1_rmse_d8_lr010_s0', F, 'study: +0.00049+-0.00065 | +0.00026+-0.00037')
cmp('s4y_yeti_teacher_lr030_s0', 's4b_qsm_soft_lr030_s0@300', 'study: -0.00175+-0.00079 | -0.00313+-0.00046')
