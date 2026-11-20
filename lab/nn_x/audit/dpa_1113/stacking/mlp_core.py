# shared MLP pieces (exec'd by e5h_mlp_hold.py); identical architecture / optimiser / loss to e5_mlp.py
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
    return (-(ls * y).sum(1)[keep] / npos[keep]).mean()

def predict(net, Xp, Pp):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xp), 2048):
            out.append(net(Xp[i:i + 2048], Pp[i:i + 2048]))
    net.train(); return torch.cat(out).numpy()

def flat(sc, idx, msk, nrows):
    o = np.zeros(nrows); o[idx[msk]] = sc[msk]; return o

def train_on(sp, X, Xp, Pp, Yp, Mp, idx, msk, tr_feeds, es_feeds, seed, epochs_fixed=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    rtr = np.repeat(tr_feeds, sp.M)
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
            sc = flat(predict(net, Xp, Pp), idx, msk, len(sp.y))
            v = sp.ev(sc)[es_feeds].mean()
            if v > best + 1e-7:
                best, best_ep, bad = v, ep, 0; best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 3: break
    if epochs_fixed is None:
        net.load_state_dict(best_state)
    return net, best_ep
