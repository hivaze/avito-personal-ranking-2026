"""Pop-free family STAGE 1 readout (information only; pre-declared CURRENT_STATUS 03:50 UTC 2026-11-13 "POP-FREE FAMILY").
abl_e_readout.py generalised to the seeds that exist: arm abl_e10_<ARM>_s<seed> (nn_x/abl/preds, default ARM xdrop_pop)
vs w5_e10_base_s<seed> (nn_x/w5/preds), seeds 951-954, missing seeds skipped (train-only selected checkpoints; the last
checkpoint and stale-val are reported where their files exist). Reports per seed and pooled: member val / holdout /
pmix private-mix / val slices (M<50, M>=50, Tue, Thu); the n-member pool arm vs base (val, holdout); and the within-feed
z-score correlation (Pearson over all candidates of per-feed rank-z scores, src/stack_val.feed_rank_z) of the pop-free
pool with the base pool and with the G3 train-only pool (preds/ens_G3_to_val.npy; holdout: z-mean of the G3 members'
preds/holdout files). No verdict: stage 2 (decide9.py) is the decision.
Usage: readout_popfree.py [arm=xdrop_pop] -> prints, writes nn_x/abl/popfree/readout_popfree.json"""
import sys, json; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
ARM = sys.argv[1] if len(sys.argv) > 1 else 'xdrop_pop'
W = Path('<repo>/nn_x/w5/preds'); WA = Path('<repo>/nn_x/abl/preds'); P = Path('<repo>/preds')
OUT = Path('<repo>/nn_x/abl/popfree/readout_popfree.json')
ALL = [951, 952, 953, 954]
have = lambda s: all(p.exists() for p in [WA/f'nn_abl_e10_{ARM}_s{s}_trainonly_val.npy', WA/f'holdout/nn_abl_e10_{ARM}_s{s}_holdout.npy',
                                            W/f'nn_w5_e10_base_s{s}_trainonly_val.npy', W/f'holdout/nn_w5_e10_base_s{s}_holdout.npy'])
SEEDS = [s for s in ALL if have(s)]
print(f'arm {ARM}: seeds present {SEEDS} (missing {[s for s in ALL if s not in SEEDS]})')
assert SEEDS, 'no complete seed'
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT/'val/cand_target.npy')
Mh, oh = np.load(W/'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W/'holdout/holdout_y.npy')
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv); eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)
se = lambda d: d.std() / np.sqrt(len(d))
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]; zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]
corr = lambda a, b: float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])
tue = (fv['timestamp'].to_numpy() + 10800) // 86400 < 20720
SL = [('M<50', Mv < 50), ('M>=50', Mv >= 50), ('Tue', tue), ('Thu', ~tue)]
PV, PH, dv, dh, rows, mcorr = {}, {}, [], [], {}, []
for s in SEEDS:
    b, t = f'w5_e10_base_s{s}', f'abl_e10_{ARM}_s{s}'
    for v in (b, t):
        D = W if v.startswith('w5_') else WA; PV[v] = np.load(D/f'nn_{v}_trainonly_val.npy'); PH[v] = np.load(D/f'holdout/nn_{v}_holdout.npy')
    eb, et, hb, ht = ev(PV[b]), ev(PV[t]), eh(PH[b]), eh(PH[t])
    dv.append(et - eb); dh.append(ht - hb); c = corr(zv(PV[t]), zv(PV[b])); mcorr.append(c)
    print(f'seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {et.mean():.5f} hold {ht.mean():.5f} | d val {(et-eb).mean():+.5f} d hold {(ht-hb).mean():+.5f} | member z-corr to base {c:.4f}')
    rows[s] = dict(base_val=float(eb.mean()), base_hold=float(hb.mean()), arm_val=float(et.mean()), arm_hold=float(ht.mean()), d_val=float((et-eb).mean()), d_hold=float((ht-hb).mean()), zcorr_base=c)
    try:  # last-checkpoint view (reported only)
        lb, lt = ev(np.load(W/f'nn_{b}@last_trainonly_val.npy')), ev(np.load(WA/f'nn_{t}@last_trainonly_val.npy'))
        hlb, hlt = eh(np.load(W/f'holdout/nn_{b}@last_holdout.npy')), eh(np.load(WA/f'holdout/nn_{t}@last_holdout.npy'))
        print(f'         last ckpt: d val {(lt-lb).mean():+.5f} d hold {(hlt-hlb).mean():+.5f}'); rows[s].update(last_d_val=float((lt-lb).mean()), last_d_hold=float((hlt-hlb).mean()))
    except FileNotFoundError:
        pass
    try:  # stale-val view (reported only; files written by stale_pred.py when the lead runs it)
        sb, st = ev(np.load(W/f'stale/nn_{b}_staleval.npy')), ev(np.load(WA/f'stale/nn_{t}_staleval.npy'))
        print(f'         stale-val: base {sb.mean():.5f} {ARM} {st.mean():.5f} d {(st-sb).mean():+.5f}+-{se(st-sb):.5f}'); rows[s].update(stale_d_val=float((st-sb).mean()))
    except FileNotFoundError:
        pass
dv = np.mean(dv, 0); dh = np.mean(dh, 0)
zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
Wt = pmix.weights(); pm, pse = pmix.wmean_se(dv, Wt['private'])
print(f'POOLED member ({len(SEEDS)} seeds): val {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean()/se(dv):+.2f}) | holdout {dh.mean():+.5f}+-{se(dh):.5f} (z {dh.mean()/se(dh):+.2f}) | pmix {pm:+.5f}+-{pse:.5f} (z {pm/pse:+.2f}) | ' + ' '.join(f'{n} {z:+.1f}' for n, z in zs.items()))
zb = np.mean([zv(PV[f'w5_e10_base_s{s}']) for s in SEEDS], 0); zt = np.mean([zv(PV[f'abl_e10_{ARM}_s{s}']) for s in SEEDS], 0)
zhb = np.mean([zh(PH[f'w5_e10_base_s{s}']) for s in SEEDS], 0); zht = np.mean([zh(PH[f'abl_e10_{ARM}_s{s}']) for s in SEEDS], 0)
pb, pt, hb, ht = ev(zv(zb)), ev(zv(zt)), eh(zh(zhb)), eh(zh(zht))
n = len(SEEDS)
print(f'{n}-member POOL: base val {pb.mean():.5f} hold {hb.mean():.5f} | {ARM} val {pt.mean():.5f} hold {ht.mean():.5f} | d val {(pt-pb).mean():+.5f}+-{se(pt-pb):.5f} d hold {(ht-hb).mean():+.5f}+-{se(ht-hb):.5f}')
# within-feed z-score correlations: pop-free pool vs base pool, vs the G3 train-only pool (val: preds/ens_G3_to_val.npy)
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]; F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]; F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]
g3v = zv(np.load(P/'ens_G3_to_val.npy')); g3h = zh(np.mean([zh(np.load(P/f'holdout/nn_{m}_holdout.npy')) for m in F10 + F7x + F7x70], 0))
C = dict(pool_vs_base_val=corr(zv(zt), zv(zb)), pool_vs_G3_val=corr(zv(zt), g3v), base_vs_G3_val=corr(zv(zb), g3v),
         pool_vs_base_hold=corr(zh(zht), zh(zhb)), pool_vs_G3_hold=corr(zh(zht), g3h), base_vs_G3_hold=corr(zh(zhb), g3h), member_vs_base_val_mean=float(np.mean(mcorr)))
print(f'WITHIN-FEED Z-CORR (Pearson over candidates): {ARM} pool vs base pool val {C["pool_vs_base_val"]:.4f} hold {C["pool_vs_base_hold"]:.4f} | '
      f'{ARM} pool vs G3 train-only pool val {C["pool_vs_G3_val"]:.4f} hold {C["pool_vs_G3_hold"]:.4f} | (reference: base pool vs G3 val {C["base_vs_G3_val"]:.4f} hold {C["base_vs_G3_hold"]:.4f}; '
      f'mean per-seed member-vs-base val {C["member_vs_base_val_mean"]:.4f})')
flags = dict(member_val_pos=bool(dv.mean() > 0), hold_gt_1se=bool(dh.mean() > se(dh)), slices_ok=bool(min(zs.values()) >= -2), pmix_ok=bool(pm >= -pse))
print(f'INFORMATION ONLY (the E member rule, not applied to a diversity family): {flags}; the decision is decide9.py (stage 2).')
json.dump({'arm': ARM, 'seeds': SEEDS, 'per_seed': rows, 'member_val': float(dv.mean()), 'member_val_se': float(se(dv)), 'member_hold': float(dh.mean()), 'member_hold_se': float(se(dh)),
           'pmix': float(pm), 'pmix_se': float(pse), 'slices_z': zs, 'pool_d_val': float((pt-pb).mean()), 'pool_d_val_se': float(se(pt-pb)), 'pool_d_hold': float((ht-hb).mean()),
           'pool_d_hold_se': float(se(ht-hb)), 'pool_base_val': float(pb.mean()), 'pool_arm_val': float(pt.mean()), 'pool_base_hold': float(hb.mean()), 'pool_arm_hold': float(ht.mean()),
           'zcorr': C, 'e_rule_flags_information_only': flags}, open(OUT, 'w'), indent=1)
print('written', OUT)
