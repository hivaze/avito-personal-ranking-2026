"""Critic check: does the member's softmax on production train feeds already carry the production position bias?
Compares, over display-slot bins (last displayed item excluded), the click-mass shape of
  actual production clicks, the flat-examination counterfactual (clicks / e(slot)), and the member's expected mass (row0 / row1).
Member scores: synthesis/out/p_token_w5_fd10_kd05w192_s951_prod_M50+.npz (3,000 seen production feeds, M >= 50).
e(slot): three published estimates (within-feed clean pool, p_oracle; d_position old pool; p_curve v2 & p_set).
Counterfactual also from a 300,000-feed sample of the same production pool (less click noise).
Usage: python c_slotmass.py -> out/c_slotmass.txt
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
Y = binsum(y, b, keep, fidx, nf)
Q = {}
for r in ('s0', 's1'):
    s = z[r].astype(np.float64); mx = np.maximum.reduceat(s, start); e = np.exp(s - mx[fidx]); p = e / np.add.reduceat(e, start)[fidx] * K[fidx]
    Q[r] = binsum(p, b, keep, fidx, nf)
rng = np.random.default_rng(1); B = 400; W = rng.poisson(1.0, (B, nf)).astype(np.float64)
L = ['3,000 seen production feeds (M >= 50) scored by w5_fd10_kd05w192_s951; slot-bin click-mass shapes, last displayed item excluded',
     f'bins {BINS}', f'actual          {np.round(norm(Y.sum(0)), 4).tolist()}',
     f'member row0     {np.round(norm(Q["s0"].sum(0)), 4).tolist()}', f'member row1     {np.round(norm(Q["s1"].sum(0)), 4).tolist()}']
for name, ev in E.items():
    ev = np.array(ev); CF = Y / ev
    a, c, q0, q1 = norm(Y.sum(0)), norm(CF.sum(0)), norm(Q['s0'].sum(0)), norm(Q['s1'].sum(0))
    bt = {k: [] for k in ('a-cf', 'r0-cf', 'r1-cf', 'top r0/cf', 'top r1/cf')}
    for w in W:
        aa, cc, x0, x1 = norm(w @ Y), norm(w @ CF), norm(w @ Q['s0']), norm(w @ Q['s1'])
        bt['a-cf'].append(tv(aa, cc)); bt['r0-cf'].append(tv(x0, cc)); bt['r1-cf'].append(tv(x1, cc))
        bt['top r0/cf'].append(x0[0] / cc[0]); bt['top r1/cf'].append(x1[0] / cc[0])
    sd = {k: np.std(v) for k, v in bt.items()}
    L.append(f'\n e = {name}: counterfactual (same feeds) {np.round(c, 4).tolist()}')
    L.append(f'  TV actual vs CF {tv(a, c):.4f} ± {sd["a-cf"]:.4f} | row0 vs CF {tv(q0, c):.4f} ± {sd["r0-cf"]:.4f} | row1 vs CF {tv(q1, c):.4f} ± {sd["r1-cf"]:.4f}')
    L.append(f'  top-slot mass: actual {a[0]:.4f} CF {c[0]:.4f} row0 {q0[0]:.4f} ({q0[0]/c[0]:.2f}x CF ± {sd["top r0/cf"]:.2f}) row1 {q1[0]:.4f} ({q1[0]/c[0]:.2f}x ± {sd["top r1/cf"]:.2f})')
    L.append(f'  slots 0-4 mass: actual {a[:4].sum():.4f} CF {c[:4].sum():.4f} row0 {q0[:4].sum():.4f} row1 {q1[:4].sum():.4f}')

# counterfactual from a larger sample of the same pool (click noise small); model shape still from the 3,000 feeds
big = np.sort(rng.choice(prod, 300_000, replace=False))
rb_, bb, lb, sb, fb = layout(big); yb = np.asarray(tgt[rb_], np.float64)
Yb = np.zeros(NB); np.add.at(Yb, bb[~lb], yb[~lb])
L.append(f'\n300,000-feed sample of the pool: actual {np.round(norm(Yb), 4).tolist()}')
for name, ev in E.items():
    c = norm(Yb / np.array(ev)); q0, q1 = norm(Q['s0'].sum(0)), norm(Q['s1'].sum(0))
    L.append(f'  e = {name}: TV actual vs CF {tv(norm(Yb), c):.4f} | row0 vs CF {tv(q0, c):.4f} | row1 vs CF {tv(q1, c):.4f} | top slot CF {c[0]:.4f}, row0/CF {q0[0]/c[0]:.2f}, row1/CF {q1[0]/c[0]:.2f}; slots 0-4 CF {c[:4].sum():.4f}')
(OUT / 'c_slotmass.txt').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
