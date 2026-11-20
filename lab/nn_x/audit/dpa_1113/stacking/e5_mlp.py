"""E5 (b): tiny residual MLP second level. score = t * pool + MLP(x) (32 hidden, GELU, dropout 0.1, zero-init output),
listwise softmax CE (multi-positive, feed-normalised), AdamW lr 3e-3 wd 1e-2, batch 256 feeds. Nested on the FT folds:
per outer fold, train on 3 inner folds, early-stop (patience 3, max 15 epochs) on the 4th; full-val fit for the holdout/test
uses the median early-stopped epoch count. Usage: e5_mlp.py ARM FORM   (ARM B0..B3 inputs as feats.py)."""
from lib import *
import time, json
import torch, torch.nn as nn
from feats import build
torch.set_num_threads(12)

arm, form = sys.argv[1], sys.argv[2]
T0 = time.time()
V = S('val')
X, names, pool = build(V, arm, form)

def padded(sp, X, pool):
    Mx = int(sp.M.max()); n = sp.n
    idx = np.full((n, Mx), -1, np.int64)
    pos = np.arange(Mx)[None, :]
    rows = sp.off[:, None] + pos
    msk = pos < sp.M[:, None]
    idx[msk] = rows[msk]
    Xp = np.zeros((n, Mx, X.shape[1]), np.float32); Xp[msk] = X[idx[msk]]
    Pp = np.zeros((n, Mx), np.float32); Pp[msk] = pool[idx[msk]]
    Yp = np.zeros((n, Mx), np.float32); Yp[msk] = sp.y[idx[msk]]
    return torch.from_numpy(Xp), torch.from_numpy(Pp), torch.from_numpy(Yp), torch.from_numpy(msk), idx, msk

class Net(nn.Module):
    def __init__(self, d, mu, sd):
        super().__init__()
        self.register_buffer('mu', torch.tensor(mu)); self.register_buffer('sd', torch.tensor(sd))
        self.f = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)
        self.t = nn.Parameter(torch.tensor(2.0))
    def forward(self, x, p):
        return self.t * p + self.f((x - self.mu) / self.sd).squeeze(-1)

def loss_fn(s, y, m):
    s = s.masked_fill(~m, -1e9)
    ls = torch.log_softmax(s, 1)
    npos = y.sum(1); keep = npos > 0
    l = -(ls * y).sum(1)[keep] / npos[keep]
    return l.mean()

def predict(net, Xp, Pp, Mp):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xp), 2048):
            out.append(net(Xp[i:i + 2048], Pp[i:i + 2048]))
    net.train(); return torch.cat(out).numpy()

def flat(sc, idx, msk, nrows):
    o = np.zeros(nrows); o[idx[msk]] = sc[msk]; return o

Xp, Pp, Yp, Mp, idx, msk = padded(V, X, pool)
nr = len(V.y)

def train(tr_feeds, es_feeds, seed, epochs_fixed=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    rtr = np.repeat(tr_feeds, V.M)
    mu = X[rtr].mean(0); sd = X[rtr].std(0) + 1e-6
    net = Net(X.shape[1], mu, sd)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    fids = np.where(tr_feeds)[0]
    best, best_state, best_ep, bad = -1, None, 0, 0
    E = epochs_fixed or 15
    for ep in range(1, E + 1):
        perm = rng.permutation(fids)
        for i in range(0, len(perm), 256):
            b = torch.from_numpy(perm[i:i + 256])
            l = loss_fn(net(Xp[b], Pp[b]), Yp[b], Mp[b])
            opt.zero_grad(); l.backward(); opt.step()
        if epochs_fixed is None:
            sc = flat(predict(net, Xp, Pp, Mp), idx, msk, nr)
            v = V.ev(sc)[es_feeds].mean()
            if v > best + 1e-7:
                best, best_ep, bad = v, ep, 0; best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 3: break
    if epochs_fixed is None:
        net.load_state_dict(best_state)
    return net, best_ep

tag = f'{arm}_{form}'
eps = []
for s in (1, 2, 3):
    oof = np.zeros(nr)
    for k in range(5):
        inner = [j for j in range(5) if j != k]; esk = inner[(k + s) % 4]
        net, ep = train((V.fold != k) & (V.fold != esk), V.fold == esk, 1000 * s + k)
        eps.append(ep)
        sc = flat(predict(net, Xp, Pp, Mp), idx, msk, nr); te = np.repeat(V.fold == k, V.M)
        oof[te] = sc[te]
        print(f'[{time.time()-T0:.0f}s] seed {s} fold {k} es-epoch {ep} t {float(net.t):.2f}', flush=True)
    np.save(CACHE / f'oof_mlp_{tag}_s{s}.npy', V.ev(oof)); np.save(CACHE / f'scr_mlp_{tag}_s{s}_val.npy', oof.astype(np.float32))
Efull = int(np.median(eps)); print('median es epochs', Efull, eps, flush=True)
targets = [S('test')] + ([S('hold')] if form == 'to' else [])
for s in (1, 2, 3):
    net, _ = train(np.ones(V.n, bool), None, 7000 + s, epochs_fixed=Efull)
    for sp in targets:
        Xs, _, ps = build(sp, arm, form)
        Xq, Pq, Yq, Mq, iq, mq = padded(sp, Xs, ps)
        sc = flat(predict(net, Xq, Pq, Mq), iq, mq, len(sp.y))
        np.save(CACHE / f'scr_mlp_{tag}_s{s}_{sp.name}.npy', sc.astype(np.float32))
        if sp.name == 'hold':
            np.save(CACHE / f'hold_mlp_{tag}_s{s}.npy', sp.ev(sc))
print(f'[{time.time()-T0:.0f}s] done {tag}', flush=True)
