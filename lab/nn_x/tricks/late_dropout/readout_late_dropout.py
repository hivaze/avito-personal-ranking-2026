"""LATE-DROPOUT 10-epoch pair readout (rule: nn_x/tricks/late_dropout/PREDECLARED.txt, written 2026-11-13 04:50:54 UTC before
any arm run). Template: nn_x/abl/popfree/readout_popfree.py (abl_e_readout pattern). CPU only.

  final      (default) arm abl_e10_latedrop_s951/952 (nn_x/abl/preds) vs w5_e10_base_s951/952 (nn_x/w5/preds), SELECTED
             checkpoints: pooled per-feed member val / holdout deltas, val slices, pmix, diversity clause
             rho_x = corr(LD pool {951,952}, base pool {953,954}) <= rho_bb - 0.005 with
             rho_bb = corr(base pool {951,952}, base pool {953,954}); verdict PASS / FAIL; info: @last view, SE_cal,
             same-seed and member-level correlations, holdout correlations, corr to G3, 2-member pool deltas, gate numbers.
  killed     the plateau kill gate fired (gate.json + the waiter's kill record in argv[2]): RESULT = FAIL (kill gate).
  incomplete a run is gone without its json (or never launched); argv[2] = reason: RESULT = INCOMPLETE.
Writes nn_x/tricks/late_dropout/RESULT.txt (+ result.json); SETUP.txt (the setup record) is appended to RESULT.txt.
Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12 .venv/bin/python readout_late_dropout.py [final|killed|incomplete] [msg]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/src')
import numpy as np  # noqa: E402
from common import FLAT, feed_meta  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

OUTD = Path('<repo>/nn_x/tricks/late_dropout')
SETUPD = OUTD
TEST_ARM = os.environ.get('LD_READOUT_TEST_ARM', '')  # plumbing test only: another finished abl_e10 arm, output elsewhere
if TEST_ARM:
    OUTD = Path(os.environ['LD_READOUT_TEST_OUT'])
    OUTD.mkdir(parents=True, exist_ok=True)
W = Path('<repo>/nn_x/w5/preds')
WA = Path('<repo>/nn_x/abl/preds')
P = Path('<repo>/preds')
LOGA = Path('<repo>/nn_x/abl/logs')
LOGW = Path('<repo>/nn_x/w5/logs')
MODE = sys.argv[1] if len(sys.argv) > 1 else 'final'
MSG = sys.argv[2] if len(sys.argv) > 2 else ''
SEEDS, REF = [951, 952], [953, 954]
ARM = lambda s: f'abl_e10_{TEST_ARM or "latedrop"}_s{s}'  # noqa: E731
BASE = lambda s: f'w5_e10_base_s{s}'  # noqa: E731
SIG_RUN = dict(val=0.00035, hold=0.00015)  # r7 A/A sigma_run (abl_eval.py convention), information only
L = []


def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def finish(res):
    res['written_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    setup = (SETUPD / 'SETUP.txt').read_text() if (SETUPD / 'SETUP.txt').exists() else ''
    (OUTD / 'RESULT.txt').write_text('\n'.join(L) + '\n\n' + '=' * 100 + '\nSETUP RECORD (written before the runs)\n' + setup)
    json.dump(res, open(OUTD / 'result.json', 'w'), indent=1, default=float)
    out(f'written {OUTD / "RESULT.txt"} and result.json')


def gate_info():
    g = OUTD / 'gate.json'  # (TEST_ARM: the test output dir, normally without a gate.json)
    return json.load(open(g)) if g.exists() else None


now = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
out(f'LATE-DROPOUT 10-epoch pair RESULT ({MODE}; computed {now}; rule nn_x/tricks/late_dropout/PREDECLARED.txt, 04:50:54 UTC)'
    + (f' [PLUMBING TEST with arm {TEST_ARM}: NOT a LATE-DROPOUT result]' if TEST_ARM else ''))
G = gate_info()
if G:
    out('PLATEAU KILL GATE (step 60000; SE_gate {:.5f}, threshold d < {:+.5f} on BOTH seeds): '.format(G['se_gate'], G['threshold'])
        + ' | '.join(f"s{s}: LD {G['per_seed'][str(s)]['h_ld']:.5f} base {G['per_seed'][str(s)]['h_base']:.5f} d "
                     f"{G['per_seed'][str(s)]['d']:+.5f} ({G['per_seed'][str(s)]['d'] / G['se_gate']:+.2f} SE_gate)"
                     f"{' BELOW' if G['per_seed'][str(s)]['below'] else ''}" for s in SEEDS)
        + f" -> {G['decision']}")
    for s in SEEDS:
        v = G['per_seed'][str(s)]
        if v.get('val_ld') is not None:
            out(f'   (info) s{s} step-60000 val: LD {v["val_ld"]:.5f} base {v["val_base"]:.5f} d {v["val_ld"] - v["val_base"]:+.5f}')
if MODE == 'killed':
    out('VERDICT: FAIL (pre-declared plateau kill gate fired at step 60000 on both seeds; both runs stopped by exact PID before '
        'the cooldown; no readout of partial runs)')
    out(f'kill record: {MSG}')
    finish(dict(verdict='FAIL', reason='plateau kill gate', gate=G, kill_record=MSG))
    sys.exit(0)
if MODE == 'incomplete':
    out(f'VERDICT: INCOMPLETE (no PASS / FAIL): {MSG}')
    out('A re-run needs a new pre-declaration (PREDECLARED.txt rule).')
    finish(dict(verdict='INCOMPLETE', reason=MSG, gate=G))
    sys.exit(0)

# ------------------------------------------------------------------------------------------------ final readout
from stack_val import feed_rank_z  # noqa: E402
import pmix  # noqa: E402

fv, Mv, ov = feed_meta('val')
ov = ov.astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy')
Mh, oh = np.load(W / 'holdout/holdout_meta.npy')
Mh, oh = Mh.astype(np.int64), oh.astype(np.int64)
yh = np.load(W / 'holdout/holdout_y.npy')
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
se = lambda d: float(d.std() / np.sqrt(len(d)))  # noqa: E731
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
corr = lambda a, b: float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])  # noqa: E731
tue = (fv['timestamp'].to_numpy() + 10800) // 86400 < 20720
SL = [('M<50', Mv < 50), ('M>=50', Mv >= 50), ('Tue', tue), ('Thu', ~tue)]
assert len(Mv) == 14423 and len(Mh) == 30000


def files(v, last=False):
    D = W if v.startswith('w5_') else WA
    t = '@last' if last else ''
    return D / f'nn_{v}{t}_trainonly_val.npy', D / f'holdout/nn_{v}{t}_holdout.npy'


PV, PH = {}, {}
for v in [ARM(s) for s in SEEDS] + [BASE(s) for s in SEEDS + REF]:
    fvp, fhp = files(v)
    PV[v], PH[v] = np.load(fvp), np.load(fhp)
    assert PV[v].shape == yv.shape and PH[v].shape == yh.shape, v
out(f'files: arm {[ARM(s) for s in SEEDS]} (nn_x/abl/preds, selected checkpoint) vs {[BASE(s) for s in SEEDS]} (nn_x/w5/preds); '
    f'diversity reference pool {[BASE(s) for s in REF]}')
rows, dvs, dhs = {}, [], []
for s in SEEDS:
    a, b = ARM(s), BASE(s)
    ja, jb = json.load(open(LOGA / f'{a}.json')), json.load(open(LOGW / f'{b}.json'))
    eb, et, hb, ht = ev(PV[b]), ev(PV[a]), eh(PH[b]), eh(PH[a])
    dv_s, dh_s = et - eb, ht - hb
    dvs.append(dv_s)
    dhs.append(dh_s)
    rows[s] = dict(base_val=float(eb.mean()), arm_val=float(et.mean()), base_hold=float(hb.mean()), arm_hold=float(ht.mean()),
                   d_val=float(dv_s.mean()), d_val_se=se(dv_s), d_hold=float(dh_s.mean()), d_hold_se=se(dh_s),
                   arm_best_step=ja['best_step'], base_best_step=jb['best_step'], arm_minutes=ja['train_minutes'],
                   base_minutes=jb['train_minutes'], member_zcorr_same_seed=corr(zv(PV[a]), zv(PV[b])))
    out(f'seed {s}: base val {eb.mean():.5f} hold {hb.mean():.5f} (sel step {jb["best_step"]}) | LD val {et.mean():.5f} hold '
        f'{ht.mean():.5f} (sel step {ja["best_step"]}) | d val {dv_s.mean():+.5f}+-{se(dv_s):.5f} d hold {dh_s.mean():+.5f}'
        f'+-{se(dh_s):.5f} | train min LD {ja["train_minutes"]:.1f} base {jb["train_minutes"]:.1f}')
    try:  # last-checkpoint view (information only)
        lv, lh = files(a, last=True)
        bv, bh = files(b, last=True)
        lb, lt, hlb, hlt = ev(np.load(bv)), ev(np.load(lv)), eh(np.load(bh)), eh(np.load(lh))
        rows[s].update(last_d_val=float((lt - lb).mean()), last_d_hold=float((hlt - hlb).mean()), last_arm_val=float(lt.mean()),
                       last_arm_hold=float(hlt.mean()), last_base_val=float(lb.mean()), last_base_hold=float(hlb.mean()))
        rows[s]['_last_dv'], rows[s]['_last_dh'] = lt - lb, hlt - hlb
        out(f'         (info) @last: LD val {lt.mean():.5f} hold {hlt.mean():.5f} | d val {(lt - lb).mean():+.5f} d hold '
            f'{(hlt - hlb).mean():+.5f}')
    except FileNotFoundError as e:
        out(f'         (info) @last view unavailable: {e}')
dv, dh = np.mean(dvs, 0), np.mean(dhs, 0)
sv, sh = se(dv), se(dh)
zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-12)) for n, m in SL}
sl_d = {n: (float(dv[m].mean()), se(dv[m])) for n, m in SL}
Wt = pmix.weights()
pm, pse = pmix.wmean_se(dv, Wt['private'])
pm, pse = float(pm), float(pse)
secal_v, secal_h = float(np.sqrt(sv ** 2 + SIG_RUN['val'] ** 2 / 2)), float(np.sqrt(sh ** 2 + SIG_RUN['hold'] ** 2 / 2))
out(f'POOLED member (2 seeds, selected): val {dv.mean():+.5f}+-{sv:.5f} (z {dv.mean() / sv:+.2f}) | holdout {dh.mean():+.5f}'
    f'+-{sh:.5f} (z {dh.mean() / sh:+.2f}) | pmix {pm:+.5f}+-{pse:.5f} (z {pm / pse:+.2f}) | slices '
    + ' '.join(f'{n} {sl_d[n][0]:+.5f} (z {z:+.2f})' for n, z in zs.items()))
out(f'   (info) SE_cal val {secal_v:.5f} (z {dv.mean() / secal_v:+.2f}), holdout {secal_h:.5f} (z {dh.mean() / secal_h:+.2f})')
if all('_last_dv' in rows[s] for s in SEEDS):
    ldv, ldh = np.mean([rows[s]['_last_dv'] for s in SEEDS], 0), np.mean([rows[s]['_last_dh'] for s in SEEDS], 0)
    lpm, lpse = pmix.wmean_se(ldv, Wt['private'])
    out(f'   (info) @last pooled: val {ldv.mean():+.5f}+-{se(ldv):.5f} (z {ldv.mean() / se(ldv):+.2f}) | holdout {ldh.mean():+.5f}'
        f'+-{se(ldh):.5f} (z {ldh.mean() / se(ldh):+.2f}) | pmix {lpm:+.5f}+-{lpse:.5f} | slices '
        + ' '.join(f'{n} z {ldv[m].mean() / se(ldv[m]):+.2f}' for n, m in SL))
    last_pooled = dict(d_val=float(ldv.mean()), d_val_se=se(ldv), d_hold=float(ldh.mean()), d_hold_se=se(ldh), pmix=float(lpm),
                       pmix_se=float(lpse))
else:
    last_pooled = None
for s in SEEDS:
    rows[s].pop('_last_dv', None)
    rows[s].pop('_last_dh', None)

# diversity (val, primary) + information correlations
ZV = {v: zv(PV[v]) for v in PV}
ZH = {v: zh(PH[v]) for v in PH}
pool_v = lambda vs: zv(np.mean([ZV[v] for v in vs], 0))  # noqa: E731
pool_h = lambda vs: zh(np.mean([ZH[v] for v in vs], 0))  # noqa: E731
LDp, B12, B34 = [ARM(s) for s in SEEDS], [BASE(s) for s in SEEDS], [BASE(s) for s in REF]
pLD, pB12, pB34 = pool_v(LDp), pool_v(B12), pool_v(B34)
hLD, hB12, hB34 = pool_h(LDp), pool_h(B12), pool_h(B34)
rho_x, rho_bb = corr(pLD, pB34), corr(pB12, pB34)
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]
F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]
F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]
g3v = zv(np.load(P / 'ens_G3_to_val.npy'))
g3h = zh(np.mean([zh(np.load(P / f'holdout/nn_{m}_holdout.npy')) for m in F10 + F7x + F7x70], 0))
C = dict(rho_x=rho_x, rho_bb=rho_bb, rho_x_hold=corr(hLD, hB34), rho_bb_hold=corr(hB12, hB34),
         same_seed_pool_LD_vs_B12_val=corr(pLD, pB12), same_seed_pool_LD_vs_B12_hold=corr(hLD, hB12),
         member_rho_x_cross_seed_val=0.5 * (corr(ZV[ARM(951)], ZV[BASE(952)]) + corr(ZV[ARM(952)], ZV[BASE(951)])),
         member_rho_bb_val=corr(ZV[BASE(951)], ZV[BASE(952)]), member_rho_aa_val=corr(ZV[ARM(951)], ZV[ARM(952)]),
         member_same_seed_val=0.5 * (rows[951]['member_zcorr_same_seed'] + rows[952]['member_zcorr_same_seed']),
         LD_pool_vs_G3_val=corr(pLD, g3v), B12_pool_vs_G3_val=corr(pB12, g3v), LD_pool_vs_G3_hold=corr(hLD, g3h),
         B12_pool_vs_G3_hold=corr(hB12, g3h))
out(f'DIVERSITY (val, within-feed rank-z, Pearson over all val candidates): rho_x = corr(LD pool{{951,952}}, base pool{{953,954}}) '
    f'{rho_x:.5f} | rho_bb = corr(base pool{{951,952}}, base pool{{953,954}}) {rho_bb:.5f} | margin rho_bb - rho_x '
    f'{rho_bb - rho_x:+.5f} (need >= +0.005)')
out(f'   (info) holdout: rho_x {C["rho_x_hold"]:.5f} rho_bb {C["rho_bb_hold"]:.5f} | same-seed pool LD vs base{{951,952}} val '
    f'{C["same_seed_pool_LD_vs_B12_val"]:.5f} hold {C["same_seed_pool_LD_vs_B12_hold"]:.5f} | member (r6/r7 convention) '
    f'rho_x cross-seed {C["member_rho_x_cross_seed_val"]:.5f} vs rho_bb {C["member_rho_bb_val"]:.5f} (arm-arm '
    f'{C["member_rho_aa_val"]:.5f}, same-seed {C["member_same_seed_val"]:.5f})')
out(f'   (info) corr to the G3 train-only pool: LD pool val {C["LD_pool_vs_G3_val"]:.5f} hold {C["LD_pool_vs_G3_hold"]:.5f} | '
    f'base pool{{951,952}} val {C["B12_pool_vs_G3_val"]:.5f} hold {C["B12_pool_vs_G3_hold"]:.5f}')
ppv, ppb, phv, phb = ev(pLD), ev(pB12), eh(hLD), eh(hB12)
pool = dict(LD_val=float(ppv.mean()), B12_val=float(ppb.mean()), d_val=float((ppv - ppb).mean()), d_val_se=se(ppv - ppb),
            LD_hold=float(phv.mean()), B12_hold=float(phb.mean()), d_hold=float((phv - phb).mean()), d_hold_se=se(phv - phb))
out(f'   (info) 2-member POOL LD vs base{{951,952}}: val {pool["LD_val"]:.5f} vs {pool["B12_val"]:.5f} d {pool["d_val"]:+.5f}+-'
    f'{pool["d_val_se"]:.5f} | hold {pool["LD_hold"]:.5f} vs {pool["B12_hold"]:.5f} d {pool["d_hold"]:+.5f}+-{pool["d_hold_se"]:.5f}')

clauses = {
    '(1) pooled member val delta > 0': bool(dv.mean() > 0),
    '(2) pooled holdout delta > +2 SE': bool(dh.mean() > 2 * sh),
    '(3) no val slice z < -2 (M<50, M>=50, Tue, Thu)': bool(min(zs.values()) >= -2),
    '(4) pmix >= -1 SE': bool(pm >= -pse),
    '(5) rho_x <= rho_bb - 0.005 (diversity)': bool(rho_x <= rho_bb - 0.005),
}
for k, v in clauses.items():
    out(f'   {k}: {"PASS" if v else "FAIL"}')
verdict = 'PASS' if all(clauses.values()) else 'FAIL'
out(f'VERDICT: {verdict} ({sum(clauses.values())}/5 clauses). '
    + ('Pool material only: reaches a pick only via the decide6 holdout-led pool rule vs G3 + presub_gate; s953/954 only under '
       'a new declaration.' if verdict == 'PASS' else 'LATE-DROPOUT is CLOSED at the 10-epoch pair (information readout; '
                                                       'nothing adopted).'))
finish(dict(verdict=verdict, clauses=clauses, per_seed={str(k): v for k, v in rows.items()}, member_val=float(dv.mean()),
            member_val_se=sv, member_hold=float(dh.mean()), member_hold_se=sh, se_cal_val=secal_v, se_cal_hold=secal_h,
            pmix=pm, pmix_se=pse, slices_z=zs, slices_d={k: list(v) for k, v in sl_d.items()}, corr=C, pool2=pool,
            last_pooled=last_pooled, gate=G))
