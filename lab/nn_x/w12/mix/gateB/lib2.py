"""gateB: exact fast NDCG, grid tensor, gate features, user-grouped folds."""
from lib import *
import numba as nb

WP = np.round(np.arange(0, 0.8001, 0.05), 2); WF = np.round(np.arange(0.2, 0.8001, 0.05), 2)   # 17 x 13 grid
IP = {round(float(w), 2): i for i, w in enumerate(WP)}; IF = {round(float(w), 2): j for j, w in enumerate(WF)}
FEATS = ['lM', 'lH', 'u0', 'cold', 'f4m', 'mcs', 'cih', 'rho']
BOUND = 1790497920


@nb.njit(parallel=True, cache=False)
def ndcg_fast(s, y, off, M):
    """binary gain, log2 discount, pessimistic ties (negatives first), feeds without positives -> 1; == src/metrics.ndcg_per_feed for 0/1 targets."""
    n = len(M); out = np.ones(n)
    for f in nb.prange(n):
        o = off[f]; m = M[f]; npos = 0
        for k in range(m):
            if y[o + k] > 0: npos += 1
        if npos == 0: continue
        idcg = 0.0
        for r in range(npos): idcg += 1.0 / np.log2(r + 2.0)
        dcg = 0.0
        for k in range(m):
            if y[o + k] > 0:
                sk = s[o + k]; r = 0
                for j in range(m):
                    sj = s[o + j]
                    if sj > sk: r += 1
                    elif sj == sk and j != k and (y[o + j] <= 0 or j < k): r += 1
                dcg += 1.0 / np.log2(r + 2.0)
        out[f] = dcg / idcg
    return out


@nb.njit(parallel=True, cache=False)
def zfeed(x, off, M):
    """per-feed z-score, float32, same arithmetic as stack_val.feed_rank_z[:, 1]."""
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; v = x[o:o + m]; mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m): out[o + k] = (v[k] - mu) / sd
    return out


@nb.njit(parallel=True, cache=False)
def rankfeed(x, off, M):
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; r = np.argsort(np.argsort(x[o:o + m]))
        for k in range(m): out[o + k] = r[k] / (m - 1) if m > 1 else 0.5
    return out


@nb.njit(parallel=True, cache=False)
def softmaxfeed(x, off, M, T):
    out = np.empty(len(x), np.float64)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; mx = -1e30
        for k in range(m):
            if x[o + k] > mx: mx = x[o + k]
        s = 0.0
        for k in range(m):
            out[o + k] = np.exp((x[o + k] - mx) / T); s += out[o + k]
        for k in range(m): out[o + k] /= s
    return out


def score(sp, q, wp_feed, wf_feed):
    """final score for per-feed weights (arrays of length n_feeds or scalars), build_r4 form, float32."""
    f, M, off = meta(sp)
    wp = np.repeat(np.asarray(wp_feed, np.float32), M) if np.ndim(wp_feed) else np.float32(wp_feed)
    wf = np.repeat(np.asarray(wf_feed, np.float32), M) if np.ndim(wf_feed) else np.float32(wf_feed)
    bf = zfeed(((1 - wp) * q['Bft'] + wp * q['Pft']).astype(np.float32), off, M); bt = zfeed(((1 - wp) * q['Bto'] + wp * q['Pto']).astype(np.float32), off, M)
    return (wf * bf + (1 - wf) * bt).astype(np.float32)


def ev_w(q, wp_feed, wf_feed, y):
    f, M, off = meta('val')
    return ndcg_fast(score('val', q, wp_feed, wf_feed), y, off, M)


def tensor(q, y, tag=None):
    """E[feed, i(w_pop), j(w_ft)] exact per-feed NDCG on the grid."""
    if tag is not None and (CACHE / f'E_{tag}.npy').exists(): return np.load(CACHE / f'E_{tag}.npy')
    f, M, off = meta('val'); E = np.empty((len(M), len(WP), len(WF)), np.float64)
    for i, wp in enumerate(WP):
        wp = np.float32(wp)
        bf = zfeed(((1 - wp) * q['Bft'] + wp * q['Pft']).astype(np.float32), off, M); bt = zfeed(((1 - wp) * q['Bto'] + wp * q['Pto']).astype(np.float32), off, M)
        for j, wf in enumerate(WF):
            wf = np.float32(wf); E[:, i, j] = ndcg_fast((wf * bf + (1 - wf) * bt).astype(np.float32), y, off, M)
    if tag is not None: np.save(CACHE / f'E_{tag}.npy', E)
    return E


@nb.njit(parallel=True, cache=False)
def feedcorr(a, b, off, M):
    out = np.zeros(len(M))
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; x = a[o:o + m].astype(np.float64); y = b[o:o + m].astype(np.float64)
        x = x - x.mean(); y = y - y.mean(); d = np.sqrt((x * x).sum() * (y * y).sum())
        out[f] = (x * y).sum() / d if d > 0 else 0.0
    return out


def features(sp, q):
    """8 label-free gate features (FEATS order) for split sp; rho from the given pools' train-only branch."""
    f, M, off = meta(sp); r = np.load(CACHE / 'feat_raw.npz')
    g = lambda k: r[f'{sp}__{k}']
    assert (g('M') == M).all()
    X = np.stack([np.log(M.astype(np.float64)), np.log1p(g('H')), g('uid0'), g('cold_luo'), g('f4_mean'), g('max_cat_share'), g('cand_cat_in_hist_share'), feedcorr(q['Bto'], q['Pto'], off, M)], 1)
    return X


def folds(g, seed, k=5):
    """user-grouped fold id per feed (g = group index per feed)."""
    ng = g.max() + 1; rng = np.random.default_rng(seed); perm = rng.permutation(ng); fo = np.empty(ng, np.int64); fo[perm] = np.arange(ng) % k
    return fo[g]


def pmixw():
    import pmix
    return pmix.weights()
