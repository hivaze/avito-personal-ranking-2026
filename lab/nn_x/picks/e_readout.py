"""Wave-5 stage E readout: TRUNC-PROD 10-epoch from-scratch PAIRS (w5_e10_{base,trunc}_s951-954), train-only members.
Rule PRE-DECLARED in CURRENT_STATUS 19:3x UTC 2026-11-12 (before any E run finished):
  TRUNC PASSES iff ALL: pooled per-seed paired member val delta > 0; pooled holdout delta > +1 SE;
  no val slice (M<50, M>=50, Tue, Thu) < -2 SE; pmix private-mix >= -1 SE.   (selected checkpoints; last ckpt reported)
Also reported: 4-member pool trunc vs base (val, holdout).  Usage: e_readout.py [arm=trunc] -> prints PASS/FAIL, writes e_readout_<arm>.json"""
import sys, json; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
ARM = sys.argv[1] if len(sys.argv) > 1 else 'trunc'
W = Path('<repo>/nn_x/w5/preds'); SEEDS = [951, 952, 953, 954]
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT/'val/cand_target.npy')
Mh, oh = np.load(W/'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W/'holdout/holdout_y.npy')
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv); eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)
se = lambda d: d.std() / np.sqrt(len(d))
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]; zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]
tue = (fv['timestamp'].to_numpy() + 10800) // 86400 < 20720
SL = [('M<50', Mv < 50), ('M>=50', Mv >= 50), ('Tue', tue), ('Thu', ~tue)]
PV, PH, dv, dh, dvl, dhl = {}, {}, [], [], [], []
for s in SEEDS:
    b, t = f'w5_e10_base_s{s}', f'w5_e10_{ARM}_s{s}'
    for v in (b, t):
        PV[v] = np.load(W/f'nn_{v}_trainonly_val.npy'); PH[v] = np.load(W/f'holdout/nn_{v}_holdout.npy')
    eb, et, hb, ht = ev(PV[b]), ev(PV[t]), eh(PH[b]), eh(PH[t])
    dv.append(et - eb); dh.append(ht - hb)
    print(f'seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {et.mean():.5f} hold {ht.mean():.5f} | d val {(et-eb).mean():+.5f} d hold {(ht-hb).mean():+.5f}')
    try:  # last-checkpoint view (reported only)
        lb, lt = ev(np.load(W/f'nn_{b}@last_trainonly_val.npy')), ev(np.load(W/f'nn_{t}@last_trainonly_val.npy'))
        hlb, hlt = eh(np.load(W/f'holdout/nn_{b}@last_holdout.npy')), eh(np.load(W/f'holdout/nn_{t}@last_holdout.npy'))
        print(f'         last ckpt: d val {(lt-lb).mean():+.5f} d hold {(hlt-hlb).mean():+.5f}')
    except FileNotFoundError:
        pass
dv = np.mean(dv, 0); dh = np.mean(dh, 0)
zs = {n: dv[m].mean() / max(se(dv[m]), 1e-9) for n, m in SL}
Wt = pmix.weights(); pm, pse = pmix.wmean_se(dv, Wt['private'])
print(f'POOLED member: val {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean()/se(dv):+.2f}) | holdout {dh.mean():+.5f}+-{se(dh):.5f} (z {dh.mean()/se(dh):+.2f}) | pmix {pm:+.5f}+-{pse:.5f} | ' + ' '.join(f'{n} {z:+.1f}' for n, z in zs.items()))
pb = ev(zv(np.mean([zv(PV[f'w5_e10_base_s{s}']) for s in SEEDS], 0))); pt = ev(zv(np.mean([zv(PV[f'w5_e10_{ARM}_s{s}']) for s in SEEDS], 0)))
hb = eh(zh(np.mean([zh(PH[f'w5_e10_base_s{s}']) for s in SEEDS], 0))); ht = eh(zh(np.mean([zh(PH[f'w5_e10_{ARM}_s{s}']) for s in SEEDS], 0)))
print(f'4-member POOL: base val {pb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {pt.mean():.5f} hold {ht.mean():.5f} | d val {(pt-pb).mean():+.5f}+-{se(pt-pb):.5f} d hold {(ht-hb).mean():+.5f}+-{se(ht-hb):.5f}')
ok = dv.mean() > 0 and dh.mean() > se(dh) and min(zs.values()) >= -2 and pm >= -pse
print(f'VERDICT {ARM}: ' + ('PASS' if ok else 'FAIL'))
json.dump({'arm': ARM, 'pass': bool(ok), 'val': float(dv.mean()), 'hold': float(dh.mean()), 'hold_se': float(se(dh)), 'pmix': float(pm), 'slices': zs},
          open(f'/tmp/scratchpad/e_readout_{ARM}.json', 'w'), indent=1)
