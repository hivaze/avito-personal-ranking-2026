"""N6: generic scan of all 34 xt inputs of cvnd_pre2s (+ xp_imp_c50) AS THE MODEL SEES THEM through a view (my own gather: where(keep, stored, view file), attach_xt transform),
on effective-cold train rows vs cold rows of val / public / private. Looks for (1) identity-type combinations: P(x > 0 | cold) exactly 0 or 1 on every target but not in the view,
(2) level shifts |SMD| > 0.15 that the as-built cold rows (D) of the same feeds do not share. Label-free.
usage: n6_scan.py <tag> <HZCD_DIR> <keep or ''>"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'
tag, HZ, KP = sys.argv[1], sys.argv[2], sys.argv[3]
FC, HZ2 = f'{R}/nn/feat_cache', f'{R}/nn_x/w12/stale/hz2'
CVRAW = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]
CVND = CVRAW + ["ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h",
                "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw",
                "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]
NAMES = CVND + ["pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32", "has:pre2_fe", "xp_imp_c50"]
SWAP = {"it_hist_all_luo": "it_hist_hz_all", "it_hist_et1_luo": "it_hist_hz_et1", "it_hist_et2_luo": "it_hist_hz_et2", "it_hist_et3_luo": "it_hist_hz_et3", "it_hist_et4_luo": "it_hist_hz_et4",
        "seq_cv_invd20": "seq_cv_hz_invd20", "seq_cv_pop": "seq_cv_hz_pop", "seq_cv_norm": "seq_cv_hz_norm", "xp_imp_c50": "xp_imp_hz_c50",
        "pre2_fe_z": "pre2_hz_fe_z", "pre2_fe_s": "pre2_hz_fe_s", "has:pre2_fe": "pre2_hz_fe_has", "pre2_new_z": "pre2_hz_new_z", "pre2_new_s": "pre2_hz_new_s"}
mm = lambda p: np.load(p, mmap_mode='r')

def tf(name, x):
    x = np.asarray(x).astype(np.float32)
    if name.startswith('has:') or name.endswith('_rk'): pass
    elif name.endswith('_z'): x = np.clip(x, -5, 5)
    elif name.startswith(('seq_cv', 'it_hist')): x = np.log1p(np.maximum(x, 0))
    else: x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)

def stored(split, name, rows):
    if name.startswith('has:'):
        return tf(name, (~np.isnan(mm(f'{FC}/{split}/{name[4:]}.npy')[rows])).astype(np.float32))
    return tf(name, mm(f'{FC}/{split}/{name}.npy')[rows])

ftr = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet')
M = ftr['M'].to_numpy().astype(np.int64); off = ftr['cand_off'].to_numpy().astype(np.int64)
slot = (ftr['timestamp'].to_numpy().astype(np.int64) + 3 * 3600) // 86400 - 20702
pr = np.load(f'{R}/eda/labels_position/out/train_prandom.npz')
p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
pool = (p > 0.5) & (M >= 50); pool[hold] = False
PP = np.load(f'{HZ2}/feats/train/hz2_part_pub.npy')
rng = np.random.default_rng(20261115)
pops_f = {'pool13': np.sort(rng.choice(np.flatnonzero(pool & ~PP & (slot >= 7)), 30000, replace=False)),
          'ge13': np.sort(rng.choice(np.flatnonzero(slot >= 7), 30000, replace=False))}
rows_of = lambda fs: np.repeat(off[fs] - np.concatenate([[0], np.cumsum(M[fs])[:-1]]), M[fs]) + np.arange(int(M[fs].sum()))
res = {'tag': tag, 'HZ': HZ, 'keep': KP, 'swapped': [n for n in NAMES if os.path.exists(f'{HZ}/train/{SWAP.get(n, "__none__")}.npy')]}
# targets
T = {}
for split in ('val', 'test'):
    f = pl.read_parquet(f'{R}/data/flat/{split}/feeds.parquet'); Ms = f['M'].to_numpy().astype(np.int64); ts = f['timestamp'].to_numpy().astype(np.int64)
    fid = np.repeat(np.arange(len(Ms)), Ms); allr = np.arange(int(Ms.sum()))
    luo = np.load(f'{FC}/{split}/it_hist_all_luo.npy')
    X = {n: stored(split, n, allr) for n in NAMES}
    parts = {'val': np.ones(len(Ms), bool)} if split == 'val' else {'public': ts < 1790497920, 'private': ts >= 1790497920, 'private_M50': (ts >= 1790497920) & (Ms >= 50)}
    for pn, fm in parts.items():
        rm = fm[fid]; cold = rm & (luo == 0)
        T[pn] = {n: dict(mean=float(X[n][cold].mean()), sd_all=float(X[n][rm].std()), ppos=float((X[n][cold] > 0).mean()), pzero=float((X[n][cold] == 0).mean()), mean_warm=float(X[n][rm & (luo > 0)].mean())) for n in NAMES}
        T[pn]['_cold_share'] = float(cold.sum() / rm.sum())
        idc = np.nan_to_num(np.load(f'{FC}/{split}/pre2_id_contrib.npy'), nan=0.0)
        T[pn]['_P_id_warm'] = float((idc[rm & (luo > 0)] != 0).mean())
res['targets'] = T
for pn, fs in pops_f.items():
    rows = rows_of(fs)
    S = np.asarray(mm(f'{FC}/train/it_hist_all_luo.npy')[rows]); V = np.asarray(mm(f'{HZ}/train/it_hist_hz_all.npy')[rows])
    keep = np.asarray(mm(KP)[rows]) if KP else np.zeros(len(rows), bool)
    idc = np.nan_to_num(np.asarray(mm(f'{FC}/train/pre2_id_contrib.npy')[rows]), nan=0.0)
    nc = (V == 0) & (S > 0) & ~keep
    cold = np.where(keep, S, V) == 0
    P = {'A': cold, 'NC': nc, 'B': nc & (idc != 0), 'E': nc & (idc == 0), 'D': S == 0}
    tg = T['private_M50'] if pn == 'pool13' else T['private']
    o = {'_rows': len(rows), '_cold_share': float(cold.mean()), '_P_id_warm_view': float((idc[~cold] != 0).mean()), '_P_id_warm_asbuilt': float((idc[S > 0] != 0).mean()), '_n': {k: int(v.sum()) for k, v in P.items()}}
    for n in NAMES:
        st = stored('train', n, rows)
        hzp = f'{HZ}/train/{SWAP.get(n, "__none__")}.npy'
        x = np.where(keep, st, tf(SWAP[n], mm(hzp)[rows])) if os.path.exists(hzp) else st
        x16 = x.astype(np.float16).astype(np.float32)  # as stored on the GPU
        r = {}
        for k, m_ in P.items():
            r[k] = dict(mean=float(x16[m_].mean()), ppos=float((x16[m_] > 0).mean()), pzero=float((x16[m_] == 0).mean()), smd=float((x16[m_].mean() - tg[n]['mean']) / (tg[n]['sd_all'] + 1e-12)),
                        smd_stored=float((st[m_].mean() - tg[n]['mean']) / (tg[n]['sd_all'] + 1e-12)))
        r['swapped'] = os.path.exists(hzp)
        o[n] = r
    res[pn] = o
json.dump(res, open(f'{R}/nn_x/w12/stale/hz2n_review/n6_scan_{tag}.json', 'w'), indent=1)
# print
for pn in pops_f:
    o = res[pn]; tgn = 'private_M50' if pn == 'pool13' else 'private'
    print(f"\n== {tag} / train {pn}: rows {o['_rows']}, view cold share {o['_cold_share']:.4f} (target {T[tgn]['_cold_share']:.4f}); P(id | view-warm) {o['_P_id_warm_view']:.4f}, as built {o['_P_id_warm_asbuilt']:.4f}, target {T[tgn]['_P_id_warm']:.4f}; n {o['_n']}")
    print(f"{'input':>18} sw | P(x>0|cold): val public private | view A    NC     D   | SMD vs {tgn} cold: A      NC      B      E      D   | NC stored-values SMD")
    for n in NAMES:
        r = o[n]
        ident = all(T[t][n]['ppos'] in (0.0, 1.0) for t in ('val', 'public', 'private'))
        bad_ident = ident and any(abs(r[k]['ppos'] - T['private'][n]['ppos']) > 0 for k in ('A', 'NC'))
        lvl = max(abs(r['NC']['smd'] - r['D']['smd']), abs(r['B']['smd'] - r['D']['smd']))
        flag = ('  <-- IDENTITY BROKEN' if bad_ident else '') + ('  <-- |NC or B - D| > 0.15' if lvl > 0.15 else '')
        print(f"{n:>18} {'S' if r['swapped'] else ' '}  | {T['val'][n]['ppos']:.4f} {T['public'][n]['ppos']:.4f} {T['private'][n]['ppos']:.4f} | {r['A']['ppos']:.4f} {r['NC']['ppos']:.4f} {r['D']['ppos']:.4f} | "
              f"{r['A']['smd']:+.3f} {r['NC']['smd']:+.3f} {r['B']['smd']:+.3f} {r['E']['smd']:+.3f} {r['D']['smd']:+.3f} | {r['NC']['smd_stored']:+.3f}{flag}")
