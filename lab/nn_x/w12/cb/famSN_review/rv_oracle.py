"""Break attempt: (1) in-sample upper bound: conditional logit of [W16, EVERY full-scale tree score on disk of family S + the hard-label reference + the two LightGBM students];
(2) cross-witness NATS: coefficients fitted on one witness, gain measured on the other (no in-sample optimism); (3) best fixed weight in hindsight for the hard-label tree and the bundle."""
import sys; sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review')
from rv_common import *
wv, wh = Wit('val'), Wit('hold'); zwv, zwh = w16(wv), w16(wh)
P = ROOT / 'nn_x/w12/cb/famS/preds'; R = ROOT / 'nn_x/w12/cb/infra/out'
tags = ['s1_rmse_d8_lr010_s0', 's1f_rmse_d8_lr030_s0', 's1f_rmse_d8_lr030_s2', 's1f_rmse_d8_lr030_s1long', 's3f_pca_d8_lr030_s0', 's2_rmse_d10_lr030_s0', 's4_qsm_kdmix_lr030_s0', 's4b_qsm_soft_lr030_s0', 's4y_yeti_teacher_lr030_s0']
Zv = {t: wv.z(np.load(P / f'{t}_val.npy')) for t in tags}; Zh = {t: wh.z(np.load(P / f'{t}_hold.npy')) for t in tags}
Zv['hard'] = wv.z(np.load(R / 'ts_lr020_s0_val.npy')); Zh['hard'] = wh.z(np.load(R / 'ts_lr020_s0_hold.npy'))
for k, (a, b) in {'lgbK': ('nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_val.npy', 'nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_trH.npy'), 'lgbKL0': ('nn_x/w12/cb/audit/out/KL0_60k_r2400_val.npy', 'nn_x/w12/cb/audit/out/KL0_60k_r2400_trH.npy')}.items():
    Zv[k] = wv.z(np.load(ROOT / a)); Zh[k] = wh.z(np.load(ROOT / b))
names = list(Zv)
for w, zw, Z, nm in ((wv, zwv, Zv, 'val'), (wh, zwh, Zh, 'hold')):
    _, l1 = w.clogit(zw); b, l2 = w.clogit(np.column_stack([zw] + [Z[t] for t in names])); d = l1 - l2
    print(f'(1) {nm}: [W16 + {len(names)} tree scores] in-sample NATS {d.mean():+.5f}+-{se(d):.5f} (bar 0.0016); coefficients W16 {b[0]:+.3f}, ' + ', '.join(f'{t} {v:+.3f}' for t, v in zip(names, b[1:])), flush=True)
def loss(w, X, b):
    X = np.asarray(X, np.float64); s = X @ b; mx = np.maximum.reduceat(s, w.off); lse = mx + np.log(np.add.reduceat(np.exp(s - mx[w.fid]), w.off)); t = w.y / w.npos[w.fid]
    return -(np.bincount(w.fid, t * s, w.nf) - lse)
print('(2) cross-witness NATS (coefficients from the OTHER witness):')
for t in ('hard', 's1f_rmse_d8_lr030_s0', 's4b_qsm_soft_lr030_s0', 's4_qsm_kdmix_lr030_s0'):
    bv1, _ = wv.clogit(zwv); bv2, _ = wv.clogit(np.column_stack([zwv, Zv[t]])); bh1, _ = wh.clogit(zwh); bh2, _ = wh.clogit(np.column_stack([zwh, Zh[t]]))
    dv = loss(wv, zwv[:, None], bh1) - loss(wv, np.column_stack([zwv, Zv[t]]), bh2); dh = loss(wh, zwh[:, None], bv1) - loss(wh, np.column_stack([zwh, Zh[t]]), bv2)
    print(f'   {t}: val (holdout coefficients {bh2[0]:.3f}, {bh2[1]:.3f}) {dv.mean():+.5f}+-{se(dv):.5f} | holdout (val coefficients {bv2[0]:.3f}, {bv2[1]:.3f}) {dh.mean():+.5f}+-{se(dh):.5f}', flush=True)
print('(3) weights in hindsight (information; the rule fixes 0.1):')
e0v, e0h = wv.ndcg(zwv), wh.ndcg(zwh)
for t in ('hard', 's4b_qsm_soft_lr030_s0'):
    for wg in (0.03, 0.05, 0.08, 0.1, 0.15):
        dv = wv.ndcg((1 - wg) * zwv + wg * Zv[t]) - e0v; dh = wh.ndcg((1 - wg) * zwh + wg * Zh[t]) - e0h
        print(f'   {t} w {wg}: {dv.mean():+.5f}+-{se(dv):.5f} | {dh.mean():+.5f}+-{se(dh):.5f} | z {(dv.mean() / se(dv) + dh.mean() / se(dh)) / np.sqrt(2):+.2f}', flush=True)
