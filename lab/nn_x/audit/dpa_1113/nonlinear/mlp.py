"""(c') Tiny residual MLP on top of z(G3): s = g + MLP([std(x), emb(cat)]) (last layer zero-init -> starts at G3),
listwise softmax CE (per-feed normalised), AdamW, early stopping per epoch on an inner 20%-of-users split of the training
folds; user-grouped 5-fold OOF on val (same outer folds as gbdt.py); holdout = model trained on all val.
Control = level 0 (input [g] only, no category). CPU only, 12 threads.
Usage: python mlp.py <level> <seed> [hidden] [dropout] [wd]
"""
import sys, time
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, folds5, ndcg, D  # noqa
from featmat import build
torch.set_num_threads(12)
t0 = time.time()
level, seed = int(sys.argv[1]), int(sys.argv[2])
hid = int(sys.argv[3]) if len(sys.argv) > 3 else 64
drop = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
wd = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
LR = float(sys.argv[6]) if len(sys.argv) > 6 else 2e-4
tag = f"L{level}_s{seed}_h{hid}_d{drop}_wd{wd}_lr{LR}"
dv = load("val"); dh = load("hold")
Xv, names, cats = build(dv, level); Xh, _, _ = build(dh, level)
cont = [i for i in range(len(names)) if i not in cats and names[i] != "vert"]
catcol = names.index("cat") if "cat" in names else None
print(f"[{tag}] {len(cont)} continuous inputs, cat {catcol is not None} ({time.time()-t0:.0f}s)", flush=True)
fold = folds5(dv, seed=0)


class Net(nn.Module):
    def __init__(self, nin, use_cat):
        super().__init__()
        self.use_cat = use_cat
        self.emb = nn.Embedding(64, 8) if use_cat else None
        self.f = nn.Sequential(nn.Linear(nin + (8 if use_cat else 0), hid), nn.GELU(), nn.Dropout(drop),
                               nn.Linear(hid, hid // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(hid // 2, 1))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, x, c):
        if self.use_cat:
            x = torch.cat([x, self.emb(c)], -1)
        return self.f(x).squeeze(-1)


def batches(d, feeds, bs, rng=None):
    feeds = np.asarray(feeds)
    if rng is not None:
        feeds = rng.permutation(feeds)
    for i in range(0, len(feeds), bs):
        yield feeds[i:i + bs]


def gather(d, X, mu, sd, fb):
    M = d["M"][fb]; Mx = M.max(); off = d["off"][fb]
    ar = np.arange(Mx)
    mask = ar[None, :] < M[:, None]
    idx = np.where(mask, off[:, None] + ar[None, :], off[:, None])
    x = np.clip((X[idx][..., cont] - mu) / sd, -6, 6)
    c = X[idx][..., catcol].astype(np.int64) if catcol is not None else np.zeros(idx.shape, np.int64)
    return (torch.from_numpy(x.astype(np.float32)), torch.from_numpy(c), torch.from_numpy(mask),
            torch.from_numpy(d["g"][idx].astype(np.float32)), torch.from_numpy(d["y"][idx].astype(np.float32)), idx, mask)


def predict(net, d, X, mu, sd, feeds):
    net.eval()
    out = {}
    with torch.no_grad():
        for fb in batches(d, feeds, 512):
            x, c, m, g, y, idx, mask = gather(d, X, mu, sd, fb)
            s = g + net(x, c)
            out_idx = idx[mask]; out_val = s.numpy()[mask]
            out[len(out)] = (out_idx, out_val)
    rows = np.concatenate([v[0] for v in out.values()]); vals = np.concatenate([v[1] for v in out.values()])
    return rows, vals


def sub_ndcg(d, rows, vals, feeds):
    from metrics import ndcg_per_feed
    s = np.zeros(len(d["y"]), np.float32); s[rows] = vals
    M = d["M"][feeds]; off0 = d["off"][feeds]
    rr = np.concatenate([np.arange(o, o + m) for o, m in zip(off0, M)])
    off = np.concatenate([[0], np.cumsum(M)[:-1]])
    return ndcg_per_feed(s[rr], d["y"][rr], off, M).mean(), ndcg_per_feed(d["g"][rr].astype(np.float32), d["y"][rr], off, M).mean()


def fit(d, X, tr_feeds, sd_seed):
    torch.manual_seed(sd_seed)
    rng = np.random.default_rng(sd_seed)
    uid = d["uid"][tr_feeds]
    uu = np.unique(uid[uid != 0]); inner_u = set(rng.choice(uu, len(uu) // 5, replace=False).tolist())
    inn = np.array([(u in inner_u) if u != 0 else (rng.random() < 0.2) for u in uid.tolist()])
    fa, fb = tr_feeds[~inn], tr_feeds[inn]
    ra = np.concatenate([np.arange(o, o + m) for o, m in zip(d["off"][fa], d["M"][fa])])
    mu = X[ra][:, cont].mean(0); sd = X[ra][:, cont].std(0) + 1e-6
    net = Net(len(cont), catcol is not None)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=wd)
    r, v = predict(net, d, X, mu, sd, fb)
    best, base = sub_ndcg(d, r, v, fb)  # identity (zero-init output) is a candidate: no structure -> stays G3
    bstate, bep = {k: t.clone() for k, t in net.state_dict().items()}, -1
    step = 0
    for ep in range(6):
        for fbb in batches(d, fa, 128, rng):
            net.train()
            x, c, m, g, y, _, _ = gather(d, X, mu, sd, fbb)
            s = (g + net(x, c)).masked_fill(~m, -1e9)
            lp = torch.log_softmax(s, -1)
            npos = y.sum(-1).clamp_min(1)
            loss = -((y * lp).sum(-1) / npos).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 25 == 0:
                r, v = predict(net, d, X, mu, sd, fb)
                val, _ = sub_ndcg(d, r, v, fb)
                if val > best:
                    best, bstate, bep = val, {k: t.clone() for k, t in net.state_dict().items()}, step
    net.load_state_dict(bstate)
    return net, mu, sd, bep, best - base


oof = np.zeros(len(dv["y"]))
for k in range(5):
    te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
    net, mu, sd, bep, gin = fit(dv, Xv, tr, seed * 10 + k)
    r, v = predict(net, dv, Xv, mu, sd, te); oof[r] = v
    print(f"  fold {k}: best step {bep} inner gain {gin:+.5f} ({time.time()-t0:.0f}s)", flush=True)
base = ndcg(dv, dv["g"]); new = ndcg(dv, oof); dl = new - base
print(f"[{tag}] VAL OOF {new.mean():.5f} vs G3 {base.mean():.5f}: delta {dl.mean():+.5f} +- {dl.std()/np.sqrt(len(dl)):.5f}")
np.save(f"{D}/mlp_oof_{tag}.npy", oof.astype(np.float32))
net, mu, sd, bep, gin = fit(dv, Xv, np.arange(dv["nf"]), seed * 10 + 9)
r, v = predict(net, dh, Xh, mu, sd, np.arange(dh["nf"]))
sh = np.zeros(len(dh["y"])); sh[r] = v
bh = ndcg(dh, dh["g"]); nh = ndcg(dh, sh); dd = nh - bh
print(f"[{tag}] HOLD (val-trained, epoch {bep}) {nh.mean():.5f} vs {bh.mean():.5f}: delta {dd.mean():+.5f} +- {dd.std()/np.sqrt(len(dd)):.5f}")
np.save(f"{D}/mlp_hold_{tag}.npy", sh.astype(np.float32))
print(f"done {time.time()-t0:.0f}s")
