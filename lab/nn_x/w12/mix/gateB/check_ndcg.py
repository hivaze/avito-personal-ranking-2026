from lib2 import *
import time
y = labels(); f, M, off = meta('val'); print('targets unique', np.unique(y), 'rows', len(y), 'feeds', len(M), 'dtype', y.dtype)
q = pools4('val', None, None, tag='FD16P4'); rng = np.random.default_rng(0)
tests = {'plain': score('val', q, 0.2, 0.5), 'pft': q['Pft'], 'rounded(ties)': np.round(q['Bto'], 1).astype(np.float32), 'const(all ties)': np.zeros(len(y), np.float32), 'rand': rng.normal(size=len(y)).astype(np.float32)}
ok = True
for k, s in tests.items():
    a = ndcg_per_feed(s, y, off, M); t = time.time(); b = ndcg_fast(s, y.astype(np.float32), off, M); dt = time.time() - t
    print(f'{k:16s} ref {a.mean():.8f} fast {b.mean():.8f} max|d| {np.abs(a - b).max():.2e}  {dt * 1e3:.0f} ms'); ok &= np.abs(a - b).max() < 1e-12
# score() vs lib.final
a = final('val', q, 0.2, 0.5); b = score('val', q, 0.2, 0.5); print('score vs lib.final max|d|', np.abs(a - b).max())
Mrow = np.repeat(M, M); a = final('val', q, np.where(Mrow >= 50, 0.4, 0.2).astype(np.float32), 0.5); b = score('val', q, np.where(M >= 50, 0.4, 0.2), 0.5); print('mrule max|d|', np.abs(a - b).max(), ev(b).mean())
ok &= np.abs(a - b).max() < 1e-6
print('NDCG CHECK', 'PASS' if ok else 'FAIL')
t = time.time(); E = tensor(q, y.astype(np.float32), tag='FD16P4'); print('tensor', E.shape, f'{time.time() - t:.0f}s', 'E[0.2,0.5]', E[:, IP[0.2], IF[0.5]].mean())
X = features('val', q); print('features', X.shape, X.mean(0).round(3), X.std(0).round(3))
