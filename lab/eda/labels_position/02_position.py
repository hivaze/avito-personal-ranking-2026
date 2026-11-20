"""Position bias in train: rate vs pos, gaps, NDCG of pos order, per-feed Spearman(pos, f_k)."""
from common import *
import json, time
nb.set_num_threads(40)

@nb.njit(cache=True)
def avg_rank(x):
    n = len(x)
    idx = np.argsort(x, kind='mergesort')
    r = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        v = 0.5 * (i + j)
        for k in range(i, j + 1):
            r[idx[k]] = v
        i = j + 1
    return r

@nb.njit(cache=True)
def pearson(a, b):
    n = len(a)
    ma = a.mean(); mb = b.mean()
    sab = 0.0; saa = 0.0; sbb = 0.0
    for i in range(n):
        da = a[i] - ma; db = b[i] - mb
        sab += da * db; saa += da * da; sbb += db * db
    if saa <= 0 or sbb <= 0:
        return np.nan
    return sab / np.sqrt(saa * sbb)

@nb.njit(parallel=True, cache=True)
def per_feed_stats(pos, feats, target, off, M):
    n = len(M)
    nf = feats.shape[0]
    rho = np.full((n, nf), np.nan)
    rho_t = np.full(n, np.nan)       # spearman(pos, target)
    minpos = np.zeros(n, np.int32); maxpos = np.zeros(n, np.int32)
    ndup = np.zeros(n, np.int32)
    nuniq = np.zeros((n, nf), np.int32)
    for f in nb.prange(n):
        o = off[f]; m = M[f]
        p = pos[o:o + m].astype(np.float64)
        minpos[f] = int(p.min()); maxpos[f] = int(p.max())
        rp = avg_rank(p)
        ps = np.sort(p)
        d = 0
        for i in range(1, m):
            if ps[i] == ps[i - 1]:
                d += 1
        ndup[f] = d
        for k in range(nf):
            x = feats[k, o:o + m].astype(np.float64)
            rx = avg_rank(x)
            rho[f, k] = pearson(rp, rx)
            xs = np.sort(x); u = 1
            for i in range(1, m):
                if xs[i] != xs[i - 1]:
                    u += 1
            nuniq[f, k] = u
        tt = target[o:o + m].astype(np.float64)
        rho_t[f] = pearson(rp, avg_rank(tt))
    return rho, rho_t, minpos, maxpos, ndup, nuniq

if __name__ == '__main__':
    t0 = time.time()
    F = feeds('train'); off = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
    pos = np.asarray(cand('train', 'pos')); target = np.asarray(cand('train', 'target'))
    print('loaded', time.time() - t0, flush=True)
    res = {}
    # rate vs pos
    cnt = np.bincount(pos, minlength=300); posc = np.bincount(pos, weights=target, minlength=300)
    res['pos_count'] = cnt.tolist(); res['pos_npos'] = posc.tolist()
    # rate vs pos normalized by feed rate (removes M composition): expected = feed rate
    fr = np.add.reduceat(target, off, dtype=np.int64) / M
    exp_ = np.repeat(fr, M)
    expc = np.bincount(pos, weights=exp_, minlength=300)
    res['pos_expected'] = expc.tolist()
    # by M bins
    Mrep = np.repeat(M, M)
    for lo, hi in [(5, 10), (10, 30), (30, 60), (60, 100), (100, 200), (200, 301)]:
        mm = (Mrep >= lo) & (Mrep < hi)
        c = np.bincount(pos[mm], minlength=300); pc = np.bincount(pos[mm], weights=target[mm], minlength=300)
        res[f'pos_rate_M{lo}_{hi}'] = [c.tolist(), pc.tolist()]
    print('rate vs pos', time.time() - t0, flush=True)
    feats = np.stack([np.asarray(cand('train', f'f{k}')) for k in range(1, 6)])
    rho, rho_t, minpos, maxpos, ndup, nuniq = per_feed_stats(pos, feats, target, off, M)
    print('per feed', time.time() - t0, flush=True)
    np.savez(f'{OUT}/02_perfeed_train.npz', rho=rho.astype(np.float32), rho_t=rho_t.astype(np.float32), minpos=minpos,
             maxpos=maxpos, ndup=ndup, nuniq=nuniq.astype(np.int16))
    # pos ndcg
    import metrics
    nd_pos = ndcg_pf(-pos.astype(np.float64), target, off, M)
    nd_rand = ndcg_pf(np.random.default_rng(0).random(len(pos)), target, off, M)
    np.savez(f'{OUT}/02_ndcg_pos_train.npz', nd_pos=nd_pos, nd_rand=nd_rand)
    res['ndcg_pos_asc'] = float(nd_pos.mean()); res['ndcg_random'] = float(nd_rand.mean())
    res['ndcg_pos_desc'] = float(ndcg_pf(pos.astype(np.float64), target, off, M).mean())
    print(res['ndcg_pos_asc'], res['ndcg_random'], res['ndcg_pos_desc'], time.time() - t0)
    json.dump(res, open(f'{OUT}/02_position.json', 'w'))
