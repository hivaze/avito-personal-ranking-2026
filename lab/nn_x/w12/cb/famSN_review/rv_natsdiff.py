import sys; sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review')
from rv_common import *
wv, wh = Wit('val'), Wit('hold'); zwv, zwh = w16(wv), w16(wh)
P = ROOT / 'nn_x/w12/cb/famS/preds'; R = ROOT / 'nn_x/w12/cb/infra/out'
S3 = ['s1f_rmse_d8_lr030_s0', 's1f_rmse_d8_lr030_s1', 's1f_rmse_d8_lr030_s2']
X = {'hard': (np.load(R / 'ts_lr020_s0_val.npy'), np.load(R / 'ts_lr020_s0_hold.npy')),
     'bundle': (np.mean([wv.z(np.load(P / f'{t}_val.npy')) for t in S3], 0), np.mean([wh.z(np.load(P / f'{t}_hold.npy')) for t in S3], 0))}
for t in ('s1f_rmse_d8_lr030_s0', 's3f_pca_d8_lr030_s0', 's4b_qsm_soft_lr030_s0', 's4_qsm_kdmix_lr030_s0', 's1_rmse_d8_lr010_s0', 's2_rmse_d10_lr030_s0', 's4y_yeti_teacher_lr030_s0', 's1f_rmse_d8_lr030_s1long'): X[t] = (np.load(P / f'{t}_val.npy'), np.load(P / f'{t}_hold.npy'))
L = {t: (wv.clogit(np.column_stack([zwv, wv.z(v[0])]))[1], wh.clogit(np.column_stack([zwh, wh.z(v[1])]))[1]) for t, v in X.items()}
def d(a, b):
    dv = L[b][0] - L[a][0]; dh = L[b][1] - L[a][1]; z = (dv.mean() / se(dv) + dh.mean() / se(dh)) / np.sqrt(2)
    print(f'NATS({a}) - NATS({b}): {dv.mean():+.5f}+-{se(dv):.5f} | {dh.mean():+.5f}+-{se(dh):.5f} | z {z:+.2f}', flush=True)
for t in X:
    if t != 'hard': d(t, 'hard')
d('s3f_pca_d8_lr030_s0', 's1f_rmse_d8_lr030_s0'); d('s4b_qsm_soft_lr030_s0', 's1f_rmse_d8_lr030_s0'); d('bundle', 's1f_rmse_d8_lr030_s0')
