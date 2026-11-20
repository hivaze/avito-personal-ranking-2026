"""M. Side check: inference precision. The stored predictions come from bf16-autocast inference; the CPU pilot's epoch-0 predictions are the SAME
weights scored in float32. Ties, agreement and paired NDCG on the held-out feeds scored so far (label use: evaluation only)."""
from lib import *
for m in ('w5_e10_kd05w192_s951', 'w5_e10_kd05t25w192_s951'):
    O = np.load(OUT / f'ftcpu_{m}_A_base_oof.npy', mmap_mode='r'); done = np.load(OUT / f'ftcpu_{m}_A_base_done.npy'); rows = np.repeat(done, Mv)
    Ms = Mv[done]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]); y = yv[rows]
    c = np.asarray(O[0])[rows].astype(np.float64); g = raw(m, 'to', 'val')[rows].astype(np.float64)
    def ties(x):
        fid = np.repeat(np.arange(len(Ms)), Ms); o = np.lexsort((x, fid)); xs, fs = x[o], fid[o]
        t = (np.diff(xs) == 0) & (np.diff(fs) == 0); return t.mean(), len(np.unique(fid[o][1:][t])) / len(Ms)
    zc = feed_rank_z(offs, Ms, c.astype(np.float32))[:, 1]; zg = feed_rank_z(offs, Ms, g.astype(np.float32))[:, 1]
    ec, eg = ndcg_per_feed(c, y, offs, Ms), ndcg_per_feed(g, y, offs, Ms)
    # optimistic-tie NDCG of the stored preds (positives first among ties) to size the tie penalty
    ego = ndcg_per_feed(g + 1e-7 * y, y, offs, Ms)
    print(f'{m}: feeds {done.sum()} | tied adjacent pairs: fp32 {ties(c)[0]:.5f} (feeds with a tie {ties(c)[1]:.3f}), stored bf16 {ties(g)[0]:.5f} (feeds with a tie {ties(g)[1]:.3f})')
    print(f'   per-feed-z corr fp32 vs stored {np.corrcoef(zc, zg)[0,1]:.6f}, RMS diff {np.sqrt(((zc-zg)**2).mean()):.4f} | NDCG fp32 {ec.mean():.5f} stored {eg.mean():.5f} paired {f(ec - eg)} | stored with optimistic ties {ego.mean():.5f} ({f(ego - eg)})')
# all feeds: ties in stored TO / FT preds of the W8 members and in the pool
for nm, x in [('member TO ' + W8[0], raw(W8[0], 'to', 'val')), ('member FT ' + W8[0], raw(W8[0], 'ft', 'val')), ('W8 TO pool', pool(W8, 'to', 'val')), ('W8 file', 0.5 * zv(pool(W8, 'ft', 'val')) + 0.5 * zv(pool(W8, 'to', 'val')))]:
    x = np.asarray(x, np.float64); e = ev(x); eo = ndcg_per_feed(x + 1e-9 * yv * (np.abs(x).max() + 1), yv, ov, Mv)
    fid = np.repeat(np.arange(len(Mv)), Mv); o = np.lexsort((x, fid)); t = (np.diff(x[o]) == 0) & (np.diff(fid[o]) == 0)
    print(f'{nm:40s} tied adjacent pairs {t.mean():.5f} | NDCG pessimistic {e.mean():.5f} optimistic {eo.mean():.5f} (gap {eo.mean() - e.mean():+.5f})')
xt_ = np.asarray(0.5 * zt(pool(W8, 'ft', 'test')) + 0.5 * zt(pool(W8, 'to', 'test')), np.float32).astype(np.float64); fid = np.repeat(np.arange(len(Mt)), Mt); o = np.lexsort((xt_, fid)); t = (np.diff(xt_[o]) == 0) & (np.diff(fid[o]) == 0)
print(f'W8 file on test (float32): tied adjacent pairs {t.mean():.6f}')
