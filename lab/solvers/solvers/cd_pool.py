# Proxy for the WSD cooldown exploration pool composition (CPU only).
# Base: CE MLP on 300K mixed train feeds (Sep 9-19), 3 epochs, cosine LR (as loss_zoo.py 'ce').
# Cooldown emulation: in the last cd_frac of the steps, a batch is replaced by a pool batch with probability cd_mix.
# Pools: rand50 = p_random > 0.5 & M >= 50 (the NN's current pool definition, nn/train.py:160,260);
#        randall = rand50 + 'random-like' short feeds (p_random > 0.5 & M < 50; detector not identifiable there).
# Usage: python cd_pool.py --pool rand50|randall|none --seeds 1 --cd_mix 0.3
import sys, time, argparse
import numpy as np, torch
sys.path.insert(0, '<repo>/solvers/solvers')
from proxy_lib import *
ap = argparse.ArgumentParser()
ap.add_argument('--pool', default='rand50', choices=['none', 'rand50', 'randall', 'rand50lo'])
ap.add_argument('--seeds', default='1')
ap.add_argument('--cd_frac', type=float, default=0.2)
ap.add_argument('--cd_mix', type=float, default=0.3)
ap.add_argument('--threads', type=int, default=5)
a = ap.parse_args(); torch.set_num_threads(a.threads)
OUT = '<repo>/solvers/out/solvers'; NAMES = CONTENT + COLLAB
f, M, off = feed_meta('train'); day = day_of(f['timestamp'].to_numpy()); p = np.nan_to_num(prandom(), nan=0.0)
rng = np.random.default_rng(0)
hold_c = np.where((p > 0.9) & (M >= 50))[0]; hold = np.sort(rng.choice(hold_c, 20000, replace=False))
pool_all = np.setdiff1d(np.where(day >= 3)[0], hold)
tr = np.sort(rng.choice(pool_all, 300000, replace=False))  # identical to loss_zoo.py --pool all (same rng draws)
if a.pool == 'none': cdp = np.array([], np.int64)
elif a.pool == 'rand50': cdp = np.setdiff1d(np.where((day >= 3) & (p > 0.5) & (M >= 50))[0], hold)
elif a.pool == 'rand50lo': cdp = np.setdiff1d(np.where((day >= 3) & (p > 0.2) & (M >= 50))[0], hold)
else: cdp = np.setdiff1d(np.where((day >= 3) & (p > 0.5))[0], hold)
ids = np.union1d(tr, cdp)  # every feed we need features for
t0 = time.time()
rows = rows_of(off, M, ids); X = load_X('train', rows, NAMES)
Y = np.asarray(np.load(FLAT / 'train/cand_target.npy', mmap_mode='r')[rows], np.float32)
rh = rows_of(off, M, hold); Xh = load_X('train', rh, NAMES)
fv, Mv, ov = feed_meta('val'); Xv = load_X('val', np.arange(int(Mv.sum())), NAMES)
# normalisation from the 300K base feeds only (identical to loss_zoo.py)
pos_in = np.searchsorted(ids, tr); Mi = M[ids]; oi = np.concatenate([[0], np.cumsum(Mi)[:-1]])
base_rows = np.concatenate([np.arange(oi[k], oi[k] + Mi[k]) for k in pos_in])
mu = np.nanmean(X[base_rows], 0); sd = np.nanstd(X[base_rows], 0) + 1e-6; hasnan = np.isnan(X[base_rows]).any(0)
def prep(Z):
    nanm = np.isnan(Z).astype(np.float32); Z = (Z - mu) / sd; Z = np.nan_to_num(Z, nan=0.0); Z = np.clip(Z, -8, 8)
    return np.concatenate([Z, nanm[:, hasnan]], 1).astype(np.float32)
X, Xh, Xv = prep(X), prep(Xh), prep(Xv)
cd_in = np.searchsorted(ids, cdp)
print('loaded', X.shape, 'pool', a.pool, len(cdp), f'{time.time()-t0:.0f}s', flush=True)
def batches(rng, idx, B=256):
    order = idx[rng.permutation(len(idx))]; out = []
    for i in range(0, len(order), B * 20):
        ch = order[i:i + B * 20]; ch = ch[np.argsort(Mi[ch])]
        for j in range(0, len(ch), B): out.append(ch[j:j + B])
    rng.shuffle(out); return out
def pad(k):
    L = int(Mi[k].max()); idx = oi[k][:, None] + np.arange(L)[None, :]; m = np.arange(L)[None, :] < Mi[k][:, None]
    idx = np.where(m, idx, 0); return torch.from_numpy(X[idx]), torch.from_numpy(Y[idx] * m), torch.from_numpy(m)
def ce(s, y, m):
    lg = s.masked_fill(~m, -1e4); K = y.sum(1); l = -(y * torch.log_softmax(lg, 1)).sum(1) / K.clamp_min(1)
    v = (K > 0).float(); return (l * v).sum() / v.sum()
def predict(net, Z):
    with torch.no_grad(): return np.concatenate([net(torch.from_numpy(Z[i:i + 500000]))[:, 0].numpy() for i in range(0, len(Z), 500000)])
for seed in [int(s) for s in a.seeds.split(',')]:
    torch.manual_seed(seed); rng = np.random.default_rng(seed); mix_rng = np.random.default_rng(1000 + seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Linear(64, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    bl = [b for e in range(3) for b in batches(rng, pos_in)]; T = len(bl); t_cool = int(T * (1 - a.cd_frac))
    cdb, cdi = (batches(mix_rng, cd_in) if len(cd_in) else []), 0; npool = 0; t1 = time.time()
    for st, b in enumerate(bl):
        for g in opt.param_groups: g['lr'] = 2e-3 * 0.5 * (1 + np.cos(np.pi * st / T))
        if len(cdb) and st >= t_cool and mix_rng.random() < a.cd_mix:
            if cdi >= len(cdb): cdb, cdi = batches(mix_rng, cd_in), 0
            b = cdb[cdi]; cdi += 1; npool += 1
        Xb, y, m = pad(b); l = ce(net(Xb)[:, :, 0], y, m)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    pv = per_feed(predict(net, Xv), 'val'); ph = per_feed(predict(net, Xh), 'train', hold)
    np.savez(f'{OUT}/zoo_CD{a.pool}_ce_s{seed}.npz', val=pv.astype(np.float32), hold=ph.astype(np.float32))
    print(seed, a.pool, 'val %.5f hold %.5f' % (pv.mean(), ph.mean()), 'pool batches', npool, f'{time.time()-t1:.0f}s', flush=True)
