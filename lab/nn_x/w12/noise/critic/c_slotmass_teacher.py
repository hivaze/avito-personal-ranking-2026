"""Critic check: does the member's softmax on production train feeds already carry the production position bias?
Compares, over display-slot bins (last displayed item excluded), the click-mass shape of
  actual production clicks, the flat-examination counterfactual (clicks / e(slot)), and the member's expected mass (row0 / row1).
Member scores: synthesis/out/p_token_w5_fd10_kd05w192_s951_prod_M50+.npz (3,000 seen production feeds, M >= 50).
e(slot): three published estimates (within-feed clean pool, p_oracle; d_position old pool; p_curve v2 & p_set).
Counterfactual also from a 300,000-feed sample of the same production pool (less click noise).
Teacher version: same feeds, stored teacher_mean11.  Usage: python c_slotmass_teacher.py -> out/c_slotmass_teacher.txt
"""
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path('<repo>'); FLAT = ROOT / 'artifacts/data/flat/train'
SYN = ROOT / 'lab/nn_x/w12/noise/synthesis/out'; OUT = ROOT / 'lab/nn_x/w12/noise/critic/out'; OUT.mkdir(exist_ok=True)
f = pl.read_parquet(FLAT / 'feeds.parquet'); M = f['M'].to_numpy().astype(np.int64); off = f['cand_off'].to_numpy().astype(np.int64)
tgt = np.load(FLAT / 'cand_target.npy', mmap_mode='r'); pos = np.load(FLAT / 'cand_pos.npy', mmap_mode='r')
p_old = np.nan_to_num(np.load(ROOT / 'artifacts/exploration/train_prandom.npz')['p_random'].astype(np.float32), nan=-1)
p_v2 = np.load(ROOT / 'lab/eda/labels_position/out/train_prandom_v2.npz')['p_random'].astype(np.float32)
prod = np.flatnonzero((M >= 50) & (p_old >= 0) & (p_old < 0.05) & (p_v2 < 0.05))
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 199), (200, 10**6)]; NB = len(BINS)
E = {'within-feed clean pool (p_oracle)': [1.924, 1.816, 1.481, 1.355, 1.276, 1.155, 1.026, 0.951, 0.908, 0.873],
     'holdout ref pool (p_holdout_oracle)': [1.863, 1.781, 1.49, 1.335, 1.279, 1.161, 1.028, 0.95, 0.909, 0.875],
     'old p>0.9 pool (d_position)': [2.171, 1.925, 1.552, 1.474, 1.332, 1.205, 1.050, 0.938, 0.883, 0.827]}


def layout(fids):
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fids])
    fidx = np.repeat(np.arange(len(fids)), M[fids])
    order = np.lexsort((np.asarray(pos[rows]), fidx))
    start = np.concatenate([[0], np.cumsum(M[fids])[:-1]])
    rk = np.empty(len(rows), np.int64); rk[order] = np.arange(len(rows)) - np.repeat(start, M[fids])
    b = np.full(len(rows), -1)
    for k, (lo, hi) in enumerate(BINS):
        b[(rk >= lo) & (rk <= hi)] = k
    return rows, b, rk == np.repeat(M[fids], M[fids]) - 1, start, fidx


def binsum(w, b, keep, fidx, nf):
    """per-feed x bin sums (nf x NB) of w over kept rows"""
    out = np.zeros((nf, NB)); np.add.at(out, (fidx[keep], b[keep]), w[keep]); return out


tv = lambda a, c: 0.5 * np.abs(a - c).sum()
norm = lambda h: h / h.sum()
z = np.load(SYN / 'p_token_w5_fd10_kd05w192_s951_prod_M50+.npz'); ids = z['ids']
rows, b, last, start, fidx = layout(ids); y = np.asarray(tgt[rows], np.float64); nf = len(ids)
K = np.add.reduceat(y, start); keep = ~last
T = np.asarray(np.load(ROOT / 'artifacts/teachers/teacher_mean11.npy', mmap_mode='r')[rows], np.float64)
def sm(s):
    mx = np.maximum.reduceat(s, start); e = np.exp(s - mx[fidx]); return e / np.add.reduceat(e, start)[fidx]
pt = sm(T); ps = sm(z['s0'].astype(np.float64))
Y = binsum(y, b, keep, fidx, nf); QT = binsum(pt * K[fidx], b, keep, fidx, nf)
TGT = binsum(0.5 * y / K[fidx] + 0.5 * pt, b, keep, fidx, nf)       # the KD target as trained (each feed sums to 1)
CLK = binsum(y / K[fidx], b, keep, fidx, nf)                         # click half alone, same per-feed normalisation
L = ['teacher_mean11 (the stored teacher of the students; production domain row on these feeds) on the same 3,000 production feeds',
     f'teacher softmax shape   {np.round(norm(QT.sum(0)), 4).tolist()}',
     f'KD target (0.5/0.5)     {np.round(norm(TGT.sum(0)), 4).tolist()}',
     f'click half (y/K)        {np.round(norm(CLK.sum(0)), 4).tolist()}',
     f'within-feed Pearson(teacher, student row0) mean {np.mean([np.corrcoef(T[s_:s_+m], z["s0"][s_:s_+m])[0,1] for s_, m in zip(start, M[ids])]):.4f}']
rng = np.random.default_rng(2); W = rng.poisson(1.0, (400, nf)).astype(np.float64)
for name, ev in E.items():
    ev = np.array(ev); cf = norm((Y / ev).sum(0)); cfk = norm((CLK / ev).sum(0)); qt = norm(QT.sum(0)); tg = norm(TGT.sum(0)); ck = norm(CLK.sum(0))
    bs = [tv(norm(w @ QT), norm(w @ (Y / ev))) for w in W]
    L.append(f' e = {name}: TV teacher vs CF {tv(qt, cf):.4f} ± {np.std(bs):.4f} (top slot teacher/CF {qt[0]/cf[0]:.2f}); '
             f'TV click half vs its CF {tv(ck, cfk):.4f}; TV KD target vs CF {tv(tg, cfk):.4f}')
(OUT / 'c_slotmass_teacher.txt').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
