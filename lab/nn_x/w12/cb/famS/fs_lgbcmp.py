"""Paired per-feed NDCG difference of famS tags against the two LightGBM students on disk (evaluation only).
usage: python fs_lgbcmp.py <tag> [<tag> ...]  -> reads/cmp_lgb_<tag>.json"""
import sys, json
import numpy as np
sys.path.insert(0, '<repo>/nn_x/w12/cb'); sys.path.insert(0, '<repo>/nn_x/w12/cb/famS')
from nats import se
from fs_read import P, nd, R
L = {'lgb_K134k_r600': ('nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_val.npy', 'nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_trH.npy'),
     'lgb_KL0_60k_r2400': ('nn_x/w12/cb/audit/out/KL0_60k_r2400_val.npy', 'nn_x/w12/cb/audit/out/KL0_60k_r2400_trH.npy')}
E = {k: (nd('val', np.load('<repo>/' + a)), nd('hold', np.load('<repo>/' + b))) for k, (a, b) in L.items()}
for t in sys.argv[1:]:
    tv, th = P(t); ev, eh = nd('val', tv), nd('hold', th); out = {}
    for k, (lv, lh) in E.items():
        dv, dh = ev - lv, eh - lh; zp = (dv.mean() / se(dv) + dh.mean() / se(dh)) / np.sqrt(2)
        out[k] = dict(val=float(dv.mean()), val_se=float(se(dv)), hold=float(dh.mean()), hold_se=float(se(dh)), pooled_z=float(zp))
        print(f"[{t} - {k}] val {dv.mean():+.5f} +- {se(dv):.5f} | holdout {dh.mean():+.5f} +- {se(dh):.5f} | pooled z {zp:+.2f}")
    (R / f'famS/reads/cmp_lgb_{t}.json').write_text(json.dumps(out))
