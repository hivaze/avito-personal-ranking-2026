"""Per-segment val NDCG (f1 baseline, random) and segment shares in val / test-public / test-private."""
from common import *
from featlib import subset
NID = 111_400_000
rng = np.random.default_rng(5)
def offs(Mx): return np.r_[0, np.cumsum(Mx)[:-1]]
trc = np.bincount(np.asarray(cand('train', 'item_id')), minlength=NID)
tru = feeds('train')['user_id'].unique().to_numpy()
S = {}
for s in ['val', 'test']:
    D = subset(s); f = D['f']
    seg = {}
    u = f['user_id'].to_numpy()
    seg['user'] = np.where(u == 0, 'user0', np.where(np.isin(u, tru), 'known(in train)', 'new'))
    H = D['H']; seg['H'] = np.select([H == 0, H < 64, H < 256], ['0', '1-63', '64-255'], '256')
    hr = f['hour'].to_numpy(); seg['hour'] = np.select([hr < 6, hr < 12, hr < 18], ['00-06', '06-12', '12-18'], '18-24')
    cov = np.bincount(D['cf'], weights=(trc[D['C']['item_id']] > 0)) / D['M']
    seg['train_cov'] = np.select([cov < 0.4, cov < 0.6, cov < 0.8], ['<40%', '40-60%', '60-80%'], '>=80%')
    S[s] = (D, seg)
D, seg = S['val']
t = D['C']['target'].astype(float); o = offs(D['M']); nz = rng.random(len(t)) * 1e-7
f1 = ndcg_per_feed(D['C']['f1'] + nz, t, o, D['M'])
rnd = np.mean([ndcg_per_feed(rng.random(len(t)), t, o, D['M']) for _ in range(10)], axis=0)
Dt, segt = S['test']
ts = Dt['f']['timestamp'].to_numpy(); od = np.argsort(ts, kind='stable'); pub = np.zeros(len(ts), bool); pub[od[:int(round(0.3 * len(ts)))]] = True
for k in seg:
    print(f'== segment by {k}')
    for v in sorted(set(seg[k])):
        m = seg[k] == v
        print(f'  {v:16s} val share {m.mean()*100:5.1f}% | f1 {f1[m].mean():.4f} rand {rnd[m].mean():.4f} lift {f1[m].mean()-rnd[m].mean():+.4f} (n={m.sum()}) | test-pub share {np.mean(segt[k][pub]==v)*100:5.1f}%  test-priv share {np.mean(segt[k][~pub]==v)*100:5.1f}%')
