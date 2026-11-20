"""Order-structure test independent of features: same-category adjacency in pos order vs random-permutation expectation.
Per-feed z-score; exploration feeds should give z~N(0,1)."""
from common import *
import json, time
nb.set_num_threads(40)

@nb.njit(parallel=True, cache=True)
def adj_stats(pos, cat, off, M):
    n = len(M)
    obs = np.zeros(n); expv = np.zeros(n); var = np.zeros(n)
    for f in nb.prange(n):
        o = off[f]; m = M[f]
        idx = np.argsort(pos[o:o + m])
        c = cat[o:o + m]
        s = 0
        for k in range(m - 1):
            if c[idx[k]] == c[idx[k + 1]]:
                s += 1
        obs[f] = s
        cnt = np.zeros(64)
        for k in range(m):
            cnt[c[k]] += 1
        p = 0.0
        for k in range(64):
            p += cnt[k] * (cnt[k] - 1)
        p /= (m * (m - 1))
        expv[f] = (m - 1) * p
        var[f] = (m - 1) * p * (1 - p)  # binomial approx (lower bound-ish); exact via MC below
    return obs, expv, var

@nb.njit(parallel=True, cache=True)
def adj_mc(cat, off, M, nsim, seed):
    """Monte-Carlo mean/sd of same-category adjacency count under random permutation."""
    n = len(M)
    mu = np.zeros(n); sd = np.zeros(n)
    for f in nb.prange(n):
        np.random.seed(seed + f)
        o = off[f]; m = M[f]
        c = cat[o:o + m].copy()
        s1 = 0.0; s2 = 0.0
        for it in range(nsim):
            np.random.shuffle(c)
            s = 0
            for k in range(m - 1):
                if c[k] == c[k + 1]:
                    s += 1
            s1 += s; s2 += s * s
        mu[f] = s1 / nsim
        sd[f] = np.sqrt(max(s2 / nsim - mu[f] ** 2, 1e-9))
    return mu, sd

if __name__ == '__main__':
    t0 = time.time()
    F = feeds('train'); off = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
    pos = np.asarray(cand('train', 'pos')); cat = np.asarray(cand('train', 'category_id')).astype(np.int64)
    obs, expv, var = adj_stats(pos, cat, off, M)
    mu, sd = adj_mc(cat, off, M, 200, 12345)
    z = (obs - mu) / sd
    print('done', time.time() - t0)
    np.savez(f'{OUT}/05_adj.npz', obs=obs, mu=mu, sd=sd, z=z)
    ok = sd > 0.3
    print('mean obs/exp ratio', obs.sum() / mu.sum(), 'mean z', np.nanmean(z[ok]))
    for lo, hi in [(5, 20), (20, 50), (50, 100), (100, 200), (200, 301)]:
        m = (M >= lo) & (M < hi) & ok
        zz = z[m]
        h, _ = np.histogram(zz, np.linspace(-6, 14, 41))
        print(f'M[{lo},{hi}) n={m.sum()} obs/exp={obs[m].sum() / mu[m].sum():.3f} z q01..q50:', np.quantile(zz, [.01, .05, .1, .25, .5, .75, .9]).round(2), 'P(z<0)=', np.mean(zz < 0).round(4))
        print('   hist z [-6,14] step .5:', (h / h.sum() * 1000).astype(int).tolist())
