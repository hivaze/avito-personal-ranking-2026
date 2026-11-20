"""Bilinear / history-query models WITH a learnable logit scale (init 20) -- 06_bilinear.py had scores in [-1,1],
so the listwise softmax was nearly uniform and its gradient ~ mean(pos)-mean(all), misaligned with NDCG.
Reports held-out-train and val NDCG; also 5-fold val CV."""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.argv = [sys.argv[0], '220000']
exec(open('06_bilinear.py').read().split("t0 = time.time()")[0])  # reuse defs (prep, batch, loss_fn, predict, Model)


class Scaled(nn.Module):
    def __init__(self, kind, rank=None):
        super().__init__()
        self.inner = Model(kind)
        self.logscale = nn.Parameter(torch.tensor(np.log(20.0), dtype=torch.float32))

    def forward(self, U, P, C):
        return self.inner(U, P, C) * self.logscale.exp()


t0 = time.time()
Dv = prep(load_split('val'))
rand_v = np.random.default_rng(0).random(len(Dv['t_np'])) * 1e-6
fidx = np.sort(np.random.default_rng(42).choice(len(feeds('train')), 220000, replace=False))
Dt = prep(load_split('train', fidx))
nft = len(Dt['M_np'])
perm = np.random.default_rng(3).permutation(nft)
tr_f = torch.tensor(np.sort(perm[20000:]), device=dev); ho_f = np.sort(perm[:20000])
rand_t = np.random.default_rng(1).random(len(Dt['t_np'])) * 1e-6
ho_mask = np.zeros(nft, bool); ho_mask[ho_f] = True
nfv = len(Dv['M_np'])


def ho_ndcg(m):
    p = predict(m, Dt, torch.tensor(ho_f, device=dev))
    return ndcg_per_feed(p + rand_t, Dt['t_np'], Dt['off_np'], Dt['M_np'])[ho_mask].mean()


def val_pf(m):
    return ndcg_per_feed(predict(m, Dv) + rand_v, Dv['t_np'], Dv['off_np'], Dv['M_np'])


dot_pf = None
for kind, lr in [('bilinear', 3e-4), ('bias', 3e-4), ('hist', 3e-4), ('deep', 3e-4)]:
    torch.manual_seed(0)
    m = Scaled(kind).to(dev)
    if dot_pf is None:
        dot_pf = val_pf(m)
        print(f'init (=dot): heldout-train {ho_ndcg(m):.4f} val {dot_pf.mean():.4f}', flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0)
    n = len(tr_f)
    for ep in range(3):
        pp = tr_f[torch.randperm(n, device=dev)]
        tl = 0; k = 0
        for i in range(0, n, 256):
            U, P, C, t, mask = batch(Dt, pp[i:i + 256])
            loss = loss_fn(m(U, P, C), t, mask)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); k += 1
        pf = val_pf(m); d = pf - dot_pf
        print(f'[{kind}] ep{ep} loss {tl / k:.4f} scale {m.logscale.exp().item():.1f} heldout-train {ho_ndcg(m):.4f} val {pf.mean():.4f} '
              f'vs dot {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(nfv):.4f})', flush=True)
    np.save(f'<repo>/eda/embeddings/cache/val_pred_scaled_{kind}.npy', predict(m, Dv))

folds = np.random.default_rng(7).integers(0, 5, nfv)
for kind in ['bilinear', 'bias', 'hist']:
    oof = np.zeros(len(Dv['t_np']))
    for k in range(5):
        torch.manual_seed(k)
        m = Scaled(kind).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0)
        trf = torch.tensor(np.where(folds != k)[0], device=dev)
        I = torch.eye(128, device=dev)
        for ep in range(8):
            pp = trf[torch.randperm(len(trf), device=dev)]
            for i in range(0, len(pp), 128):
                U, P, C, t, mask = batch(Dv, pp[i:i + 128])
                loss = loss_fn(m(U, P, C), t, mask) + 1e-3 * ((m.inner.W - I) ** 2).sum()
                if kind == 'hist':
                    loss = loss + 1e-3 * (m.inner.A ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        p = predict(m, Dv, torch.tensor(np.where(folds == k)[0], device=dev))
        sel = np.repeat(folds == k, Dv['M_np']); oof[sel] = p[sel]
    pf = ndcg_per_feed(oof + rand_v, Dv['t_np'], Dv['off_np'], Dv['M_np']); d = pf - dot_pf
    print(f'[val-5fold {kind}] OOF NDCG {pf.mean():.4f} vs dot {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(nfv):.4f})', flush=True)
print('done', time.time() - t0)
