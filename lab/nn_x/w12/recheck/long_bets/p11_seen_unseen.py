"""P11: is function matching input-limited? Student pool (teacher_h1 = 4 kd05w192 students, trained on teacher_mean11) vs its
teacher on SEEN exploration train feeds vs UNSEEN holdout feeds (same population: p_random > 0.9, M >= 50).
Per-feed Pearson corr of logits, over all candidates and over non-clicked candidates only (removes the shared click fit).
Control: two disjoint hard-label groups (teacher_a = r101,r102 vs teacher_e) seen vs unseen = shared label memorisation alone."""
from lb_common import *
import polars as pl
T = ROOT / 'nn_x/w12/kd/out'
ftr = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['M', 'cand_off']); M = ftr['M'].to_numpy().astype(np.int64); off = ftr['cand_off'].to_numpy().astype(np.int64)
pr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0); pr[M < 50] = 0
sel = np.where((pr > 0.9) & (M >= 50))[0]; hold = np.sort(np.random.default_rng(12345).choice(sel, 30000, replace=False))
seen = np.sort(np.random.default_rng(1).choice(np.setdiff1d(sel, hold), 30000, replace=False))
y = np.load(FLAT / 'train/cand_target.npy', mmap_mode='r')
F = {k: np.load(T / f'teacher_{k}.npy', mmap_mode='r') for k in ('mean11', 'h1', 'h2', 'a', 'e', 'oof')}
def stats(feeds, ka, kb):
    ca = []; cn = []
    for i in feeds:
        s = slice(off[i], off[i] + M[i]); a = np.asarray(F[ka][s], np.float64); b = np.asarray(F[kb][s], np.float64); neg = np.asarray(y[s]) == 0
        ca.append(np.corrcoef(a, b)[0, 1]); cn.append(np.corrcoef(a[neg], b[neg])[0, 1])
    return np.array(ca), np.array(cn)
for nm, ka, kb in (('students h1 (kd05w192 x4) vs their teacher mean11', 'h1', 'mean11'), ('students h2 (kd05t25w192 x4) vs mean11', 'h2', 'mean11'),
                   ('students h1 vs students h2', 'h1', 'h2'), ('hard-label control: a (r101,r102) vs e', 'a', 'e'), ('students h1 vs OOF teacher', 'h1', 'oof')):
    sa, sn = stats(seen, ka, kb); ha, hn = stats(hold, ka, kb)
    print(f'{nm:52s} all cands: seen {sa.mean():.4f} unseen {ha.mean():.4f} (1-rho {1-sa.mean():.4f} vs {1-ha.mean():.4f}) | non-clicked only: seen {sn.mean():.4f} unseen {hn.mean():.4f} (1-rho {1-sn.mean():.4f} vs {1-hn.mean():.4f})')
