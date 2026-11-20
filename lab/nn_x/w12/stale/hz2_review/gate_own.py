"""hz2_review: the reviewer's own gate numbers (label-free). No import from nn_x/w12/stale/hz2; rules from DECLARATION.txt (GATE block) and REVIEW_RULES.txt R2 / R3.
Zero share / mean by listing-age bucket of view rows (as built, hz half dose, hz2, hz2L, hz2k) against PRIVATE and PUBLIC test rows (feeds with M >= 50).
Extra (R3): shape distance, joint tell with the unswapped pre2 inputs, scan of the other xt inputs.
Usage: CUDA_VISIBLE_DEVICES="" python gate_own.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
import sys, time, json
sys.dont_write_bytecode = True
import numpy as np
import polars as pl

R = '<repo>'
FLAT = f'{R}/data/flat'; FC = f'{R}/nn/feat_cache'; HZ2 = f'{R}/nn_x/w12/stale/hz2'; HERE = f'{R}/nn_x/w12/stale/hz2_review'
MSK, DAY, DAY0 = 3 * 3600, 86400, 20702
T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


kn = np.load(f'{R}/data/feats/life_f4_age_knots.npy').astype(np.float64)
EDG = np.array([0, 1, 3, 7, 10, 24, 76, np.inf])
BN = ['<1d', '1-3d', '3-7d', '7-10d', '10-24d', '24-76d', '>76d']


def bucket(f4):
    x = f4.astype(np.float64); u = x * 64
    k = np.clip(np.floor(u).astype(int), 1, 62); w = u - k
    a = kn[k] * (1 - w) + kn[k + 1] * w
    d = np.where(x < 63 / 64, a / 24.0, 1e9)
    return np.clip(np.searchsorted(EDG, d, side='right') - 1, 0, 6)


def rows_of(off, M):
    return np.repeat(off - np.concatenate([[0], np.cumsum(M)[:-1]]), M) + np.arange(int(M.sum()))


def tf(name, x):
    x = np.asarray(x).astype(np.float32)
    if name.startswith('has:') or name.endswith('_rk'):
        pass
    elif name.endswith('_z'):
        x = np.clip(x, -5, 5)
    elif name.startswith(('seq_cv', 'it_hist')):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


ftr = pl.read_parquet(f'{FLAT}/train/feeds.parquet'); fte = pl.read_parquet(f'{FLAT}/test/feeds.parquet')
Mtr = ftr['M'].to_numpy().astype(np.int64); otr = ftr['cand_off'].to_numpy().astype(np.int64)
dtr = (ftr['timestamp'].to_numpy().astype(np.int64) + MSK) // DAY - DAY0
pr = np.load(f'{R}/eda/labels_position/out/train_prandom.npz')
assert (pr['fidx'] == np.arange(len(ftr))).all()
p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
cand = np.where((p > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))      # nn_x/w5/train.py --holdout 30000 (feat_new/hz_common.holdout_feeds)
pool = (p > 0.5) & (Mtr >= 50); pool[hold] = False
PARTPUB = np.load(f'{HZ2}/feats/train/hz2_part_pub.npy'); KEEP = np.load(f'{HZ2}/feats/train/hz2_keep_rows.npy', mmap_mode='r')
lg('trPool feeds', int(pool.sum()), '| holdout feeds', len(hold), '| holdout days >= Sep 13', int((dtr[hold] >= 7).sum()), '| keep rows', int(np.count_nonzero(KEEP)))
lg('train feeds share: Sep 6-8', round(float((dtr <= 2).mean()), 4), 'Sep 9-12', round(float(((dtr >= 3) & (dtr <= 6)).mean()), 4), '>= Sep 13', round(float((dtr >= 7).mean()), 4),
   '| pool feeds share: Sep 6-8', round(float((dtr[pool] <= 2).mean()), 4), '>= Sep 13', round(float((dtr[pool] >= 7).mean()), 4))

tts = fte['timestamp'].to_numpy().astype(np.int64); Mte = fte['M'].to_numpy().astype(np.int64); ote = fte['cand_off'].to_numpy().astype(np.int64)
PUB_TS = 1790497920
PRIM = ['it_hist_all_luo', 'seq_cv_pop', 'xp_imp_c50']
HZN = {'it_hist_all_luo': 'it_hist_hz_all', 'seq_cv_pop': 'seq_cv_hz_pop', 'xp_imp_c50': 'xp_imp_hz_c50', 'seq_cv_invd20': 'seq_cv_hz_invd20', 'seq_cv_norm': 'seq_cv_hz_norm',
       'it_hist_et1_luo': 'it_hist_hz_et1', 'it_hist_et2_luo': 'it_hist_hz_et2', 'it_hist_et3_luo': 'it_hist_hz_et3', 'it_hist_et4_luo': 'it_hist_hz_et4'}
XT = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo",
      "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent",
      "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10",
      "pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32", "has:pre2_fe"]      # nn_x/w5/train.py XT_PRESETS['cvnd_pre2s']


def load(split, name, rows):
    if name.startswith('has:'):
        return (~np.isnan(np.asarray(np.load(f'{FC}/{split}/{name[4:]}.npy', mmap_mode='r')[rows]))).astype(np.float32)
    return np.asarray(np.load(f'{FC}/{split}/{name}.npy', mmap_mode='r')[rows])


REF = {}
for part, sel in (('priv', (tts >= PUB_TS) & (Mte >= 50)), ('pub', (tts < PUB_TS) & (Mte >= 50))):
    fs = np.flatnonzero(sel); rw = rows_of(ote[fs], Mte[fs])
    b = bucket(np.load(f'{FLAT}/test/cand_f4.npy')[rw])
    REF[part] = dict(rows=rw, b=b, n_feeds=len(fs), fo=np.repeat(np.arange(len(fs)), Mte[fs]))
    lg(f'reference {part}: feeds {len(fs)} rows {len(rw)}')


def stat(vb, vx, rb, rx):
    w = np.bincount(vb, minlength=7) / len(vb)
    zv = np.array([(vx[vb == k] == 0).mean() if (vb == k).any() else np.nan for k in range(7)])
    zr = np.array([(rx[rb == k] == 0).mean() for k in range(7)])
    sd = rx.std()
    mv = np.array([vx[vb == k].mean() if (vb == k).any() else np.nan for k in range(7)]); mr = np.array([rx[rb == k].mean() for k in range(7)])
    gz = np.abs(zv - zr); gm = np.abs(mv - mr) / sd
    # shape: TV distance of the value binned at 0.25 (zero = its own bin)
    tv = np.zeros(7)
    for k in range(7):
        a = vx[vb == k]; c = rx[rb == k]
        if len(a) == 0:
            continue
        ba = np.where(a == 0, 0, 1 + np.floor(np.clip(a, 0, 12) / 0.25)).astype(np.int64); bc = np.where(c == 0, 0, 1 + np.floor(np.clip(c, 0, 12) / 0.25)).astype(np.int64)
        ha = np.bincount(ba, minlength=60) / len(ba); hc = np.bincount(bc, minlength=60) / len(bc)
        tv[k] = 0.5 * np.abs(ha - hc).sum()
    return dict(dZ=float(np.nansum(w * gz)), big=float(np.max(np.where(w >= 0.05, gz, 0.0))), dM=float(np.nansum(w * gm)), dTV=float(np.sum(w * tv)), w=w, zv=zv, zr=zr, mv=mv, mr=mr, tv=tv)


RES = {}
for part, psel in (('priv', ~PARTPUB), ('pub', PARTPUB)):
    for days, dsel in (('late', dtr >= 7), ('all', np.ones(len(dtr), bool)), ('sep9_12', (dtr >= 3) & (dtr <= 6)), ('sep6_8', dtr <= 2), ('sep17_19', dtr >= 11)):
        fs = np.flatnonzero(pool & psel & dsel)
        rw = rows_of(otr[fs], Mtr[fs])
        vb = bucket(np.asarray(np.load(f'{FLAT}/train/cand_f4.npy', mmap_mode='r')[rw]))
        keep = np.asarray(KEEP[rw])
        lg(f'--- trPool {part}-like, days {days}: feeds {len(fs)} rows {len(rw)}; keep rows {float(keep.mean()):.4f}')
        for col in PRIM:
            rx = tf(col, load('test', col, REF[part]['rows']))
            asb = tf(col, load('train', col, rw))
            views = {'asbuilt': asb}
            if col == 'it_hist_all_luo':
                views['hz'] = tf(col, np.asarray(np.load(f'{R}/data/feats_new/train/it_hist_hz_all.npy', mmap_mode='r')[rw]))
            h2 = tf(col, np.asarray(np.load(f'{HZ2}/feats/train/{HZN[col]}.npy', mmap_mode='r')[rw]))
            views['hz2'] = h2
            views['hz2k'] = np.where(keep, asb, h2)
            views['hz2L'] = tf(col, np.asarray(np.load(f'{HZ2}/feats_L/train/{HZN[col]}.npy', mmap_mode='r')[rw]))
            for vn, vx in views.items():
                s = stat(vb, vx, REF[part]['b'], rx)
                RES[f'{part}|{days}|{col}|{vn}'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
                lg(f'  {col:16s} {vn:8s} dZ {s["dZ"]:.3f} bucket {s["big"]:.3f} dM {s["dM"]:.3f} dTV {s["dTV"]:.3f} | zero share by age ' + ' '.join(f'{z:.3f}' for z in s['zv'])
                   + (' | REF ' + ' '.join(f'{z:.3f}' for z in s['zr']) if vn == 'asbuilt' else '') + (' | TV by age ' + ' '.join(f'{z:.2f}' for z in s['tv']) if vn in ('hz2', 'asbuilt') else ''))
        if part == 'priv' and days in ('late', 'all'):
            # feed-level cold share (it_hist_all == 0)
            fo = np.repeat(np.arange(len(fs)), Mtr[fs])
            for vn in ('asbuilt', 'hz', 'hz2', 'hz2k'):
                col = 'it_hist_all_luo'
                vx = {'asbuilt': lambda: tf(col, load('train', col, rw)), 'hz': lambda: tf(col, np.asarray(np.load(f'{R}/data/feats_new/train/it_hist_hz_all.npy', mmap_mode='r')[rw])),
                      'hz2': lambda: tf(col, np.asarray(np.load(f'{HZ2}/feats/train/it_hist_hz_all.npy', mmap_mode='r')[rw])),
                      'hz2k': lambda: np.where(keep, tf(col, load('train', col, rw)), tf(col, np.asarray(np.load(f'{HZ2}/feats/train/it_hist_hz_all.npy', mmap_mode='r')[rw])))}[vn]()
                cs = np.bincount(fo, weights=(vx == 0)) / Mtr[fs]
                lg(f'  feed cold share {vn}: q10 {np.quantile(cs, .1):.3f} q50 {np.quantile(cs, .5):.3f} q90 {np.quantile(cs, .9):.3f}')
            rx = tf('it_hist_all_luo', load('test', 'it_hist_all_luo', REF['priv']['rows']))
            cs = np.bincount(REF['priv']['fo'], weights=(rx == 0)) / np.bincount(REF['priv']['fo'])
            lg(f'  feed cold share PRIVATE test feeds (M >= 50): q10 {np.quantile(cs, .1):.3f} q50 {np.quantile(cs, .5):.3f} q90 {np.quantile(cs, .9):.3f}')
        if part == 'priv' and days == 'late':
            # R3c joint tell with the unswapped pre2 inputs
            rb = REF['priv']['b']
            for nm in ('has:pre2_fe', 'pre2_id_contrib'):
                try:
                    tv_ = load('train', nm, rw); te_ = load('test', nm, REF['priv']['rows'])
                except Exception as e:
                    lg('  joint tell: cannot load', nm, repr(e)[:100]); continue
                tv_ = np.nan_to_num(np.asarray(tv_, np.float32), nan=0.0) != 0; te_ = np.nan_to_num(np.asarray(te_, np.float32), nan=0.0) != 0
                rx = tf('it_hist_all_luo', load('test', 'it_hist_all_luo', REF['priv']['rows'])) == 0
                asb0 = tf('it_hist_all_luo', load('train', 'it_hist_all_luo', rw)) == 0
                h20 = tf('it_hist_all_luo', np.asarray(np.load(f'{HZ2}/feats/train/it_hist_hz_all.npy', mmap_mode='r')[rw])) == 0
                lg(f'  JOINT {nm} != 0 given it_hist_all == 0: PRIVATE {float(te_[rx].mean()):.4f} (n {int(rx.sum())}) | train as built {float(tv_[asb0].mean()):.4f} | train hz2 {float(tv_[h20].mean()):.4f} (n {int(h20.sum())})'
                   f' || unconditional: PRIVATE {float(te_.mean()):.4f} train {float(tv_.mean()):.4f} || share of ALL rows with it_hist_all == 0 and {nm} != 0: PRIVATE {float((te_ & rx).mean()):.4f} as built {float((tv_ & asb0).mean()):.4f} hz2 {float((tv_ & h20).mean()):.4f}')
                for k in range(7):
                    a = (vb == k) & h20; c = (rb == k) & rx
                    lg(f'     {BN[k]:7s} P({nm} != 0 | cold): private {float(te_[c].mean()) if c.any() else float("nan"):.3f}  hz2 {float(tv_[a].mean()) if a.any() else float("nan"):.3f}  as built {float(tv_[(vb == k) & asb0].mean()) if ((vb == k) & asb0).any() else float("nan"):.3f}')
            # R3d scan of the unswapped xt inputs
            out = []
            for nm in XT[8:]:
                rx = tf(nm, load('test', nm, REF['priv']['rows'])); vx = tf(nm, load('train', nm, rw))
                s = stat(vb, vx, rb, rx)
                out.append((s['dM'], nm, s['dZ']))
            out.sort(reverse=True)
            lg('  SCAN unswapped xt inputs, as built train rows vs PRIVATE rows (dM in private SD, dZ): ' + '; '.join(f'{nm} {dm:.3f}/{dz:.3f}' for dm, nm, dz in out))
json.dump(RES, open(f'{HERE}/gate_own.json', 'w'))
lg('GATE_OWN DONE')
