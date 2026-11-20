"""V6. The direction of the tool's accidental tie-break depends on the SCALE of our raw scores (algebra: z2 / z1 = 1 + 1e-9 (1 / s - 1), s = per-feed sd of the raw scores)."""
import sys; sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify')
from v_common import *
V = Sp('val'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64); ev = lambda s: ndcg_per_feed(np.asarray(s, np.float64), yv, V.off, V.M)
def zf(S, x):
    x = np.asarray(x, np.float64); mu = np.bincount(S.fid, x, S.n) / S.M; d = x - mu[S.fid]; sd = np.sqrt(np.bincount(S.fid, d * d, S.n) / S.M) + 1e-9; return d / sd[S.fid]
raw8 = 0.5 * zf(V, np.load(P / 'ens_W8kd_ft_val.npy')) + 0.5 * zf(V, np.load(P / 'ens_W8kd_to_val.npy'))
p = zf(V, 0.5 * zf(V, np.load(P / 'ens_Pall_ft_val.npy')) + 0.5 * zf(V, np.load(P / 'ens_Pall_to_val.npy'))); rp = V.rank(p).astype(np.int64); e8 = ev(raw8)
for scale in (1.0, 3.0, 0.3):
    z1 = zf(V, raw8 * scale); z2 = zf(V, z1); z2s = z2[np.lexsort((z2, V.fid))]; m = 0.5 * z1 + 0.5 * z2s[V.off[V.fid] + rp]
    exact = 0.5 * z1 + 0.5 * z1[np.lexsort((z1, V.fid))][V.off[V.fid] + rp]; r = V.rank(m)
    print(f'our raw scores x {scale}: median z2/z1 - 1 = {np.median(z2 / z1 - 1):+.2e} | order == [ties -> partner] {np.mean(r == V.rank(exact, p)):.4f}, == [ties -> ours] {np.mean(r == V.rank(exact, z1)):.4f} | equal mix with Pall ranks minus W8: {pm(ev(m) - e8)}')
