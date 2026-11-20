"""CAL: what the round-2 NATS screen and the linear residual read for input blocks whose value AS A TRAINED INPUT of the neural ranker is known.
For a block B: 'abl' = stored members trained WITHOUT B, 'twin' = stored members trained WITH B (same seeds, same recipe otherwise).
  true      = NDCG(pool twin) - NDCG(pool abl)                         (trained-input value, pool level, paired per feed)
  info      = NATS(pool twin | pool abl)                               (nats the with-input pool adds to the without-input pool)
  tree      = NATS(T | pool abl) - NATS(T | pool twin), T = stored hard-label tree that HAS the block among its features
              (the round-2 screen applied to a pool that lacks a known block)
  linear    = NATS(composite of B's columns | pool abl), composite coefficients fitted on the other user half (2-fold, user-grouped)
Usage: cal_posctrl.py [control ...]   -> out/cal_posctrl.txt, out/cal_posctrl.json
"""
from cbi_common import *
import json, time
T0 = time.time()
OUT = CBI_OUT
meta = Meta(); WV = Wit(meta, 'val'); WH = Wit(meta, 'hold')
POP8 = 'seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo'.split(',')
PRE2 = 'pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32'.split(',')
CTXND = 'ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_h,ctx_nd_knn_h_dev,ctx_nd_knn_u,ctx_nd_knn_u_dev,ctx_nd_max,ctx_nd_top5h,ctx_nd_top5u,ctx_nd_ucent'.split(',')
CTXH = 'ctx_hn_bestrk,ctx_hn_e1_rrw,ctx_hn_margin,ctx_hn_rrw,ctx_hn_soft05,ctx_hn_top1w,ctx_hs_d7,ctx_hs_e1,ctx_hs_e1k64,ctx_hs_k128,ctx_hs_k128t10,ctx_hs_k256,ctx_hs_mutual,ctx_hs_t10'.split(',')
S4 = (951, 952, 953, 954)
CTRL = {
    # name: (abl variants, twin variants, columns, description)
    'pop8_e10': ([f'abl_e10_xdrop_pop_s{s}' for s in S4], [f'w5_e10_base_s{s}' for s in S4], POP8, '8 popularity / co-visitation inputs, 10 epochs, 4 seeds'),
    'pre2_e10': ([f'abl_e10_xdrop_pre2_s{s}' for s in (951, 952)], [f'w5_e10_base_s{s}' for s in (951, 952)], PRE2, 'pre2 encoder scalars, 10 epochs, 2 seeds'),
    'imp1_e10': ([f'abl_e10_xdrop_pop_s{s}' for s in S4], [f'abl_e10_pop1_s{s}' for s in S4], ['xp_imp_c50'], 'ONE exposure count (tte_it_imp capped) added to the pop-free view, 10 epochs, 4 seeds'),
    'ctr1_e10': ([f'abl_e10_xdrop_pop_s{s}' for s in S4], [f'abl_e10_pop2_s{s}' for s in S4], ['xp_ctr_c'], 'ONE item click rate (tte_it_ctr) added to the pop-free view, 10 epochs, 4 seeds'),
    'pop8_e2': ([f'abl_xdrop_pop_s{s}' for s in (23, 24)], [f'abl_base_s{s}' for s in (23, 24)], POP8, '8 popularity inputs, 2 epochs, 2 seeds'),
    'ctxnd_e2': ([f'abl_xdrop_ctxnd_s{s}' for s in (23, 24)], [f'abl_base_s{s}' for s in (23, 24)], CTXND, '10 ctx_nd inputs, 2 epochs, 2 seeds'),
    'ctxh_e2': ([f'abl_xdrop_ctxhn_hs_s{s}' for s in (23, 24)], [f'abl_base_s{s}' for s in (23, 24)], CTXH, 'ctx_hn + ctx_hs inputs, 2 epochs, 2 seeds'),
    'pre2_e2': ([f'abl_xdrop_pre2_s{s}' for s in (23, 24)], [f'abl_base_s{s}' for s in (23, 24)], PRE2, 'pre2 encoder scalars, 2 epochs, 2 seeds'),
    'imp1_e2': ([f'abl_xdrop_pop_s{s}' for s in (23, 24)], [f'abl_pop1_s{s}' for s in (23, 24)], ['xp_imp_c50'], 'ONE exposure count on the pop-free view, 2 epochs, 2 seeds'),
    'ctr1_e2': ([f'abl_xdrop_pop_s{s}' for s in (23, 24)], [f'abl_pop2_s{s}' for s in (23, 24)], ['xp_ctr_c'], 'ONE item click rate on the pop-free view, 2 epochs, 2 seeds'),
}
TREES = {
    'B0 (CPU YetiRank, 177 features)': ('nn_x/w12/cb/infra/out/ts_lr020_s0_val.npy', 'nn_x/w12/cb/infra/out/ts_lr020_s0_hold.npy'),
    'B3 (B0 + famB a,b,c; seed 0)': ('nn_x/w12/cb/famB/preds/b3_abc_s0_val.npy', 'nn_x/w12/cb/famB/preds/b3_abc_s0_hold.npy'),
    'cb26 e3 repro (GPU YetiRank, 177)': ('nn_x/w25/out/A0_cb26repro_val.npy', 'nn_x/w25/out/A0_cb26repro_trH.npy'),
}


def mpath(v, sp):
    cands = []
    if sp == 'val':
        if v.startswith('w5_'): cands = [W / f'nn_{v}_trainonly_val.npy']
        else: cands = [AB / f'nn_{v}_trainonly_val.npy', P / f'nn_{v}_trainonly_val.npy']
    else:
        cands = [(W if v.startswith('w5_') else AB) / f'holdout/nn_{v}_holdout.npy']
    for c in cands:
        if c.exists(): return c
    return None


def col(name, w):
    for d in (FEATS, FEATS_NEW):
        p = d / ('val' if w.sp == 'val' else 'train') / f'{name}.npy'
        if p.exists():
            x = np.load(p, mmap_mode='r'); return np.asarray(x if w.rows is None else x[w.rows], np.float32)
    return None


res = {}; lines = []
def say(s):
    print(s, flush=True); lines.append(s)

names = sys.argv[1:] or list(CTRL)
tz = {w.sp: {k: w.z(np.load(ROOT / (v[0] if w.sp == 'val' else v[1]))) for k, v in TREES.items()} for w in (WV, WH)}
for k in names:
    ab, tw, cols, desc = CTRL[k]; res[k] = {'desc': desc}
    say(f'\n=== {k}: {desc}')
    for w in (WV, WH):
        fa = [mpath(v, w.sp) for v in ab]; ft = [mpath(v, w.sp) for v in tw]
        if any(f is None for f in fa + ft):
            say(f'  [{w.sp}] missing member files: ' + ', '.join(v for v, f in zip(ab + tw, fa + ft) if f is None)); continue
        za = w.pool(fa); zt = w.pool(ft); ea = w.ndcg(za); et = w.ndcg(zt); true = et - ea
        d_info, b_info = nats(zt, za, w); _, la = clogit(za, w); _, lt = clogit(zt, w)
        r = dict(n_feeds=w.nf, ndcg_abl=float(ea.mean()), ndcg_twin=float(et.mean()), true=sd_se(true), info=sd_se(d_info), loss_diff=sd_se(la - lt),
                 corr=float(np.corrcoef(za, zt)[0, 1]))
        say(f'  [{w.sp}] pool abl {ea.mean():.5f}  pool twin {et.mean():.5f}  TRUE trained-input value {f2(true)} (z {true.mean() / se(true):+.1f}); corr(abl, twin) {r["corr"]:.4f}')
        say(f'        INFO  NATS(twin | abl) {f2(d_info)}; single-score loss(abl) - loss(twin) {f2(la - lt)} nats/feed')
        r['tree'] = {}
        for tn, zT in tz[w.sp].items():
            d_a, b_a = nats(zT, za, w); d_t, b_t = nats(zT, zt, w); dd = d_a - d_t
            m_a = w.ndcg(0.9 * za + 0.1 * zT) - ea; m_t = w.ndcg(0.9 * zt + 0.1 * zT) - et
            r['tree'][tn] = dict(nats_abl=sd_se(d_a), nats_twin=sd_se(d_t), diff=sd_se(dd), mix_abl=sd_se(m_a), mix_twin=sd_se(m_t))
            say(f'        TREE  {tn:34s} NATS | abl {f2(d_a)}   | twin {f2(d_t)}   difference {f2(dd)} (z {dd.mean() / se(dd):+.1f});  0.9/0.1 blend: abl {f2(m_a)}, twin {f2(m_t)}')
        # linear composite of the block's own columns, cross-fitted on user halves
        X = []; used = []
        for c in cols:
            x = col(c, w)
            if x is None: say(f'        (column not in the feature store: {c})'); continue
            X.append(w.zcol(x)); used.append(c)
            if np.nanmin(x) >= 0 and not c.endswith('_z'):
                X.append(w.zcol(np.log1p(x))); used.append(c + ':log1p')
        if X:
            X = np.column_stack(X); half = w.halves(0); comp = np.zeros(w.n); dl = np.zeros(w.nf)
            for h in (True, False):
                fit = half == h; b, _ = clogit(np.column_stack([za, X]), w, fit); comp_h = X @ b[1:]
                d, b2 = nats(comp_h, za, w, ~fit); dl[~fit] = d
            b_all, l_all = clogit(np.column_stack([za, X]), w); d_in = la - l_all
            r['linear'] = dict(xfit=sd_se(dl), insample=sd_se(d_in), ncol=int(X.shape[1]))
            say(f'        LINEAR composite of {X.shape[1]} columns ({len(cols)} inputs) over pool abl: cross-fitted {f2(dl)}; in-sample joint fit {f2(d_in)}')
            if true.mean() != 0:
                say(f'        SHARES of the info NATS: tree B0 difference {r["tree"][list(TREES)[0]]["diff"][0] / d_info.mean():.2f}; linear cross-fitted {dl.mean() / d_info.mean():.2f}')
        res[k][w.sp] = r
    json.dump(res, open(OUT + '/cal_posctrl.json', 'w'), indent=1)
open(OUT + '/cal_posctrl.txt', 'a').write('\n'.join(lines) + f'\n# elapsed {time.time() - T0:.0f} s\n')
