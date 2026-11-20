"""Context for the label MI (val, arm A): MI(click; top-1 expert) vs MI(click; the model's own within-feed rank decile) vs MI(click; category),
in bits, net of a permutation null; H(click) for scale."""
import sys, numpy as np
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import FLAT, OUT, pred_paths
import polars as pl
rng = np.random.default_rng(0)
def H(p): p = p[p > 0]; return float(-(p * np.log2(p)).sum())
def mi(a, e):
    na, ne = a.max() + 1, e.max() + 1
    c = np.bincount(a * ne + e, minlength=na * ne).reshape(na, ne).astype(float); p = c / c.sum()
    return H(p.sum(1)) + H(p.sum(0)) - H(p.ravel())
mn = lambda a, e: mi(a, e) - mi(a, rng.permutation(e))
y = np.load(FLAT / 'val/cand_target.npy').astype(np.int64)
M = pl.read_parquet(FLAT / 'val/feeds.parquet')['M'].to_numpy().astype(np.int64); off = np.concatenate([[0], np.cumsum(M)[:-1]])
cat = np.load(FLAT / 'val/cand_category_id.npy').astype(np.int64)
print(f'H(click) {H(np.bincount(y) / len(y)):.4f} bits; MI(click; category) {mn(y, cat):.5f}')
for s in (951, 952, 953, 954):
    v = f'w5_e10_kd05w192moeffn_s{s}'
    sc = np.load(pred_paths(v)[0])
    rk = np.empty(len(sc), np.int64)
    for o, m in zip(off, M):
        r = np.argsort(np.argsort(-sc[o:o + m], kind='stable'), kind='stable'); rk[o:o + m] = np.minimum(10 * r // m, 9)
    line = f'{v}: MI(click; own rank decile) {mn(y, rk):.5f}'
    for b in ('cross.0', 'sets.0', 'cross.1', 'sets.1'):
        e = np.load(OUT / 'A' / f'route_{v}_val_{b}.npz')['sel'][:, 0].astype(np.int64)
        line += f' | {b} MI(click; top-1 expert) {mn(y, e):.5f}'
    print(line, flush=True)
