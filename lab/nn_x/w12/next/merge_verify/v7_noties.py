"""V7. Is there a rank-only cost that does NOT come from ties, and does it grow as the correlation falls? Pairs of similar strength, weights without ties
(0.4 / 0.6) and the three tie-breaks at 0.5. Value = mix with the partner as a rank file restored on OUR shape minus the z-mix of the model scores at the same weight.
Also: chi-square of the W16-vs-W8 cold offset over the 5 test-day groups (coarse check of the drift SEs)."""
import sys; sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify')
from v_common import *
V, T = Sp('val'), Sp('test'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64); ev = lambda s: ndcg_per_feed(np.asarray(s, np.float64), yv, V.off, V.M)
C = partners(V, T); C['W8'] = (V.z(fin(V, 'W8kd')), None); k1 = np.random.default_rng(777).random(V.N)
def restore(ours, part): return ours[np.lexsort((ours, V.fid))][V.off[V.fid] + V.rank(part).astype(np.int64)]
print(f'{"ours + partner(ranks)":26s} {"rho":>6s} {"gap":>8s} | {"w 0.4 (no ties)":>19s} {"w 0.6 (no ties)":>19s} | {"w 0.5 ties->ours":>19s} {"random":>19s} {"ties->partner":>19s}')
for a, b in (('W8', 'S16kd'), ('Pall', 'Cd'), ('C3', 'G3'), ('GRUx6', 'basex4'), ('basex4', 'GRUx6'), ('CatBoost', 'LLMx2'), ('LLMx2', 'CatBoost'), ('LLMx2', 'XGBoost'), ('XGBoost', 'LLMx2'), ('W8', 'GRUx6'), ('W8', 'CatBoost')):
    o, p = C[a][0], C[b][0]; R = restore(o, p); g = ev(o).mean() - ev(p).mean(); out = []
    for w in (0.4, 0.6): out.append(pm(ev((1 - w) * o + w * R) - ev((1 - w) * o + w * p)))
    m = 0.5 * o + 0.5 * R; zm = ev(0.5 * o + 0.5 * p)
    for key in (o, k1, p): out.append(pm(ev(V.rank(m, key)) - zm))
    print(f'{a + " + " + b:26s} {V.rf(o, p).mean():6.4f} {g:+8.5f} | ' + ' '.join(f'{x:>19s}' for x in out[:2]) + ' | ' + ' '.join(f'{x:>19s}' for x in out[2:]), flush=True)
cold = (np.nan_to_num(np.asarray(np.load(FEATS / 'test/it_hist_all_luo.npy'), np.float64), nan=0.0) == 0).astype(np.float64)
def cw(z):
    nc = np.add.reduceat(cold, T.off); nw = T.M - nc; ok = (nc > 0) & (nw > 0); g = np.add.reduceat(z * cold, T.off) / np.maximum(nc, 1) - np.add.reduceat(z * (1 - cold), T.off) / np.maximum(nw, 1); return np.where(ok, g, np.nan)
x = cw(T.z(fin(T, 'W16kd'))) - cw(T.z(fin(T, 'W8kd'))); pub = T.ts < BOUND
G = [(T.day == 20722), (T.day == 20723) & pub, (T.day == 20723) & ~pub, T.day == 20724, T.day == 20725]
mu = np.array([np.nanmean(x[m]) for m in G]); s = np.array([np.nanstd(x[m]) / np.sqrt(np.isfinite(x[m]).sum()) for m in G]); w = 1 / s ** 2; mbar = (w * mu).sum() / w.sum()
print(f'W16 - W8 cold offset by test-day group: {np.round(mu, 5).tolist()} SE {np.round(s, 5).tolist()} | chi-square vs a common mean {(w * (mu - mbar) ** 2).sum():.2f} on 4 df')
