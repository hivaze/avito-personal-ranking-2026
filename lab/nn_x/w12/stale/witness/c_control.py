"""witness controls (DECLARATION.txt block 1 (1)). Labelled: opens holdout labels and val labels through the read-outs.
  A. lagged holdout: my own CPU fp32 scoring of 3 staged members == staged files; then the reviewer's subset numbers from my read-out.
  B. stale-val on CPU: (b1) CPU without overrides vs the stored GPU bf16 train-only val predictions; (b2) CPU stale-val vs stored GPU stale-val files.
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python c_control.py [A|B|AB]
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import lagged_score as LS
import stale_val_cpu as SVC
import wt_read as WR

which = sys.argv[1] if len(sys.argv) > 1 else 'AB'
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


P(f'=== witness controls {now()} ===')
if 'A' in which:
    P('\n##### CONTROL A: lagged holdout tool #####')
    L = LS.Lag(); hpos = L.hpos
    # metric / z against the project's functions (one member, subset rows)
    sys.path.insert(0, str(ROOT / 'src'))
    from metrics import ndcg_per_feed
    from stack_val import feed_rank_z
    x = np.load(stored_holdout('w5_e10_kd05w192_s951'))[hpos]
    z1 = feed_z(L.off, L.M, x); z2 = feed_rank_z(L.off, L.M, np.ascontiguousarray(x, np.float32))[:, 1]
    n1 = ndcg_fast(z1, L.y, L.off, L.M); n2 = ndcg_per_feed(z2, L.y, L.off, L.M)
    P(f' my feed_z vs stack_val.feed_rank_z[:,1]: max |diff| {np.abs(z1 - z2).max():.2e}; my ndcg_fast vs src/metrics.ndcg_per_feed: max |diff| {np.abs(n1 - n2).max():.2e} ({len(n1)} feeds)')
    ctl = ['w5_e10_kd05w192_s952', 'w5_e10_kd05t25w192_s953', 'abl_e10_pop1kd05t25_s952']
    imp = (PRED / 'IMPORTED.txt').read_text() if (PRED / 'IMPORTED.txt').exists() else ''
    okA = True
    for v in ctl:
        for view, src in (('lag0', STUDY / f'pred_lag0_{v}.npy'), ('priv', STUDY / f'pred_priv_v3_{v}.npy'), ('pub', VERIFY / f'pred_pub_v3_late_{v}.npy')):
            assert f'{view} {v} ' not in imp, f'{v} {view} was imported, not scored by me'
            mine = np.load(LS.pred_path(view, v)); st = np.load(src); st = st[hpos] if len(st) != len(mine) else st
            a = ndcg_fast(feed_z(L.off, L.M, mine), L.y, L.off, L.M); b = ndcg_fast(feed_z(L.off, L.M, st), L.y, L.off, L.M)
            md = float(np.abs(mine - st).max()); dn = float(a.mean() - b.mean()); ok = md <= 1e-3 and abs(dn) <= 2e-5; okA &= ok
            P(f' {v:28s} {view:5s}: max |logit diff| {md:.2e}, corr {np.corrcoef(mine, st)[0, 1]:.8f}, subset NDCG mine {a.mean():.5f} staged {b.mean():.5f} (diff {dn:+.6f}) -> {"ok" if ok else "FAIL"}')
    LS.import_staged(P)
    W8, P4 = [(1.0, G['W8'])], [(1.0, G['pop1kd'])]
    res = {}
    for view in ('lag0', 'priv', 'pub'):
        ew, ep = L.level(W8, view), L.level(P4, view); res[view] = (ew, ep)
        P(f' view {view:5s}: W8 pool {ew.mean():.5f} | pop1kd pool {ep.mean():.5f} | pop1kd - W8 {fmt(ep - ew)}')
    for view in ('priv', 'pub'):
        P(f' {view}: W8 pool change {fmt(res[view][0] - res["lag0"][0])}; pop1kd pool change {fmt(res[view][1] - res["lag0"][1])}')
    S = {}
    for w in (0.1, 0.2, 0.3):
        mix = [(1 - w, G['W8']), (w, G['pop1kd'])]
        c = L.compare(mix, W8)
        S[w] = c
        P(f' S(w {w}): lag0 {"%+.5f +- %.5f" % c["d_lag0"]} | priv {"%+.5f +- %.5f" % c["d_priv"]} | pub {"%+.5f +- %.5f" % c["d_pub"]} | '
          f'D_priv {"%+.5f +- %.5f" % c["D_priv"]} | D_pub {"%+.5f +- %.5f" % c["D_pub"]}')
    t1 = abs(res['lag0'][0].mean() - 0.32334) <= 2e-5 and abs(res['priv'][0].mean() - 0.32002) <= 2e-5
    t2 = abs(S[0.2]['d_priv'][0] - 0.00024) <= 2e-5 and abs(S[0.2]['d_priv'][1] - 0.00031) <= 1e-5
    P(f' reviewer targets: W8 pool 0.32334 -> 0.32002 [{t1}]; S_lag(w 0.2, priv) +0.00024 +- 0.00031 [{t2}]; S_0 -0.00050 +- 0.00032; D +0.00074 +- 0.00040; pub S_lag +0.00018 +- 0.00031, D +0.00068 +- 0.00039')
    P(f' CONTROL A: {"VALID" if okA and t1 and t2 else "NOT VALID"}')
if 'B' in which:
    P('\n##### CONTROL B: stale-val on CPU #####')
    wt = WR.Witness(); V = wt.V
    ev = lambda x: ndcg_fast(feed_z(V.off, V.M, x), V.y, V.off, V.M)
    okB1 = True
    for v in ('w5_cd_hzcd_r101', 'w5_cd_m02ref_r101', 'abl_pop1_s23', 'abl_base_s23', 'w5_e10_kd05w192_s951', 'abl_e10_pop1kd05t25_s951'):
        if not SVC.sv_path(v, 'base').exists():
            P(f' b1 {v}: CPU base file missing'); continue
        c = np.load(SVC.sv_path(v, 'base')); g = np.load(stored_val(v)); a, b = ev(c), ev(g)
        cr = float(np.corrcoef(c, g)[0, 1]); ok = abs(a.mean() - b.mean()) <= 1e-3 and cr >= 0.9995; okB1 &= ok
        P(f' b1 {v:28s}: CPU no-override val NDCG {a.mean():.5f} vs stored GPU {b.mean():.5f} (paired {fmt(a - b)}), corr {cr:.6f}, max |d| {np.abs(c - g).max():.2e} -> {"ok" if ok else "FAIL"}')
    P(f' CONTROL b1: {"VALID" if okB1 else "NOT VALID"}')
    okB2 = True; lev = {}
    for v in ('w5_cd_hzcd_r101', 'w5_cd_m02ref_r101', 'abl_pop1_s23', 'abl_base_s23'):
        if not SVC.sv_path(v, 'stale').exists():
            P(f' b2 {v}: CPU stale file missing'); continue
        c = np.load(SVC.sv_path(v, 'stale')); g = np.load(stored_stale(v)); a, b = ev(c), ev(g); lev[v] = (a, b)
        cr = float(np.corrcoef(c, g)[0, 1]); ok = abs(a.mean() - b.mean()) <= 1e-3; okB2 &= ok
        P(f' b2 {v:28s}: CPU stale-val NDCG {a.mean():.5f} vs stored GPU stale-val {b.mean():.5f} (paired {fmt(a - b)}), corr {cr:.6f}, max |d| {np.abs(c - g).max():.2e} -> {"ok" if ok else "FAIL"}')
    for a_, b_ in (('w5_cd_hzcd_r101', 'w5_cd_m02ref_r101'), ('abl_pop1_s23', 'abl_base_s23')):
        if a_ in lev and b_ in lev:
            dc = lev[a_][0] - lev[b_][0]; dg = lev[a_][1] - lev[b_][1]; dd = dc - dg; ok = abs(dd.mean()) <= se(dd); okB2 &= ok
            P(f' b2 pair {a_} - {b_}: CPU {fmt(dc)} | GPU {fmt(dg)} | CPU - GPU {fmt(dd)} -> {"ok" if ok else "FAIL (outside 1 SE)"}')
    P(f' CONTROL b2: {"VALID" if okB2 else "NOT VALID"}')
(LOGS / f'control_{which}.txt').write_text('\n'.join(OUT) + '\n')
