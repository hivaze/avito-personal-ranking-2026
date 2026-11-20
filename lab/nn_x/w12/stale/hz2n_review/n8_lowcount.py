"""N8 (added after the N6 scan showed P(ID | view-warm) 0.86 / 0.78 vs 0.74 on private): the joint of the (view) counter LEVEL with the encoder ID contribution.
P(pre2 ID contribution != 0 | counter bin) on private / public / val rows (as built) vs train rows as built vs train rows through views a and c (after hz2n the bin 0 is 0 by construction).
Label-free. Counter bins: 0, (0,1], (1,2], (2,3], (3,5], (5,10], (10,30], >30."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = '<repo>'
FC, HZ2 = f'{R}/nn/feat_cache', f'{R}/nn_x/w12/stale/hz2'
mm = lambda p: np.load(p, mmap_mode='r')
EDGES = np.array([0, 1, 2, 3, 5, 10, 30])
LAB = ['0', '(0,1]', '(1,2]', '(2,3]', '(3,5]', '(5,10]', '(10,30]', '>30']
def binc(c):  # 0 -> bin 0; (0,1] -> 1; ...
    return np.where(c <= 0, 0, 1 + np.searchsorted(EDGES[1:], c, side='left'))
def tab(c, idc, extra=None):
    b = binc(c); n = np.bincount(b, minlength=8).astype(float); k = np.bincount(b, idc != 0, minlength=8)
    o = {'share': (n / n.sum()).round(4).tolist(), 'P_id': np.where(n > 0, k / np.maximum(n, 1), np.nan).round(4).tolist()}
    return o
out = {}
for split in ('val', 'test'):
    f = pl.read_parquet(f'{R}/data/flat/{split}/feeds.parquet'); Ms = f['M'].to_numpy().astype(np.int64); ts = f['timestamp'].to_numpy().astype(np.int64)
    fid = np.repeat(np.arange(len(Ms)), Ms)
    luo = np.load(f'{FC}/{split}/it_hist_all_luo.npy'); idc = np.nan_to_num(np.load(f'{FC}/{split}/pre2_id_contrib.npy'), nan=0.0)
    parts = {'val': np.ones(len(Ms), bool)} if split == 'val' else {'public': ts < 1790497920, 'private': ts >= 1790497920, 'private_M50': (ts >= 1790497920) & (Ms >= 50)}
    for pn, fm in parts.items():
        rm = fm[fid]; out[pn] = tab(luo[rm], idc[rm])
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
rows_of = lambda fs: np.repeat(off[fs] - np.concatenate([[0], np.cumsum(M[fs])[:-1]]), M[fs]) + np.arange(int(M[fs].sum()))
for pn, fs in (('pool13', np.sort(rng.choice(np.flatnonzero(pool & ~PP & (slot >= 7)), 30000, replace=False))), ('ge13', np.sort(rng.choice(np.flatnonzero(slot >= 7), 30000, replace=False)))):
    rows = rows_of(fs)
    S = np.asarray(mm(f'{FC}/train/it_hist_all_luo.npy')[rows]); idc = np.nan_to_num(np.asarray(mm(f'{FC}/train/pre2_id_contrib.npy')[rows]), nan=0.0)
    out[f'train_{pn}_as_built'] = tab(S, idc)
    for v, base, kp in (('a', f'{HZ2}/feats/train', f'{HZ2}/feats/train/hz2_keep_rows.npy'), ('c', f'{R}/data/feats_new/train', None)):
        V = np.asarray(mm(f'{base}/it_hist_hz_all.npy')[rows]); keep = np.asarray(mm(kp)[rows]) if kp else np.zeros(len(rows), bool)
        eff = np.where(keep, S, V)
        nc = (V == 0) & (S > 0) & ~keep
        out[f'train_{pn}_view_{v}_before_hz2n'] = tab(eff, idc)
        out[f'train_{pn}_view_{v}_after_hz2n'] = tab(eff, np.where(nc, 0.0, idc))
        out[f'train_{pn}_view_{v}_noninteger_share_of_warm'] = float((eff[eff > 0] != np.round(eff[eff > 0])).mean())
json.dump(out, open(f'{R}/nn_x/w12/stale/hz2n_review/n8_lowcount.json', 'w'), indent=1)
print('P(ID contribution != 0 | counter bin)   [share of rows in the bin]')
print(f"{'population':>40} " + ' '.join(f'{l:>14}' for l in LAB))
for k, o in out.items():
    if isinstance(o, dict):
        print(f'{k:>40} ' + ' '.join(f"{a:6.3f} [{b:5.3f}]" for a, b in zip(o['P_id'], o['share'])))
    else:
        print(f'{k:>40} {o:.4f}')
