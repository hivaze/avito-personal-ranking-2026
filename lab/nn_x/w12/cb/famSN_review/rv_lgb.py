import sys; sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review')
from rv_common import *
wv, wh = Wit('val'), Wit('hold'); zwv, zwh = w16(wv), w16(wh)
P = ROOT / 'nn_x/w12/cb/famS/preds'
L = {'lgb_K134k_r600': ('nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_val.npy', 'nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_trH.npy'),
     'lgb_KL0_60k_r2400': ('nn_x/w12/cb/audit/out/KL0_60k_r2400_val.npy', 'nn_x/w12/cb/audit/out/KL0_60k_r2400_trH.npy')}
E = {}
for k, (a, b) in L.items():
    xv, xh = np.load(ROOT / a), np.load(ROOT / b); o = read(wv, wh, zwv, zwh, xv, xh, k); print(line(o)); E[k] = (wv.ndcg(xv), wh.ndcg(xh))
S3 = ['s1f_rmse_d8_lr030_s0', 's1f_rmse_d8_lr030_s1', 's1f_rmse_d8_lr030_s2']
X = {'bundle': (np.mean([wv.z(np.load(P / f'{t}_val.npy')) for t in S3], 0), np.mean([wh.z(np.load(P / f'{t}_hold.npy')) for t in S3], 0))}
for t in ('s1_rmse_d8_lr010_s0', 's1f_rmse_d8_lr030_s1long', 's4b_qsm_soft_lr030_s0'): X[t] = (np.load(P / f'{t}_val.npy'), np.load(P / f'{t}_hold.npy'))
for t, (xv, xh) in X.items():
    ev, eh = wv.ndcg(xv), wh.ndcg(xh)
    for k, (lv, lh) in E.items():
        dv, dh = ev - lv, eh - lh; print(f'{t} - {k}: {dv.mean():+.5f}+-{se(dv):.5f} | {dh.mean():+.5f}+-{se(dh):.5f} | z {(dv.mean() / se(dv) + dh.mean() / se(dh)) / np.sqrt(2):+.2f}', flush=True)
# best-of-N look at the level: the four "ways" and their max
print('levels:', {t: (round(float(wv.ndcg(v[0]).mean()), 5), round(float(wh.ndcg(v[1]).mean()), 5)) for t, v in X.items()})
