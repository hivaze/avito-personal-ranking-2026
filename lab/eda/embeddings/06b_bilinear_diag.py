"""Diagnostics for the bilinear map: held-out-train vs val NDCG over training (is the drop a train->val shift or overfitting?)."""
import os, sys, time
import numpy as np
import torch
sys.argv = [sys.argv[0], '220000']
exec(open('06_bilinear.py').read().split("t0 = time.time()")[0])  # reuse defs only

t0 = time.time()
Dva = load_split('val'); Dv = prep(Dva)
rand_v = np.random.default_rng(0).random(len(Dv['t_np'])) * 1e-6
fidx = np.sort(np.random.default_rng(42).choice(len(feeds('train')), 220000, replace=False))
Dt = prep(load_split('train', fidx))
nft = len(Dt['M_np'])
perm = np.random.default_rng(3).permutation(nft)
tr_f = torch.tensor(np.sort(perm[20000:]), device=dev); ho_f = np.sort(perm[:20000])
rand_t = np.random.default_rng(1).random(len(Dt['t_np'])) * 1e-6
ho_mask = np.zeros(nft, bool); ho_mask[ho_f] = True


def ho_ndcg(m):
    p = predict(m, Dt, torch.tensor(ho_f, device=dev))
    pf = ndcg_per_feed(p + rand_t, Dt['t_np'], Dt['off_np'], Dt['M_np'])
    return pf[ho_mask].mean()


def val_ndcg(m):
    return ndcg_per_feed(predict(m, Dv) + rand_v, Dv['t_np'], Dv['off_np'], Dv['M_np']).mean()


for kind, lr in [('bilinear', 1e-4), ('bilinear', 1e-3), ('bias', 1e-4), ('hist', 3e-4)]:
    torch.manual_seed(0)
    m = Model(kind).to(dev)
    print(f'[{kind} lr={lr}] init: heldout-train {ho_ndcg(m):.4f}  val {val_ndcg(m):.4f}', flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0)
    n = len(tr_f); step = 0
    for ep in range(3):
        pp = tr_f[torch.randperm(n, device=dev)]
        for i in range(0, n, 256):
            U, P, C, t, mask = batch(Dt, pp[i:i + 256])
            loss = loss_fn(m(U, P, C), t, mask)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if step in (50, 200, 800):
                print(f'   step {step}: heldout-train {ho_ndcg(m):.4f}  val {val_ndcg(m):.4f}', flush=True)
        print(f'   ep{ep}: heldout-train {ho_ndcg(m):.4f}  val {val_ndcg(m):.4f}', flush=True)
    if kind != 'hist':
        W = m.W.detach().cpu().numpy()
        ev = np.linalg.eigvals(W)
        print(f'   ||W-I||_F/||I||_F = {np.linalg.norm(W - np.eye(128)) / np.sqrt(128):.3f}, |b| = {m.b.norm().item():.3f}, eig real range [{ev.real.min():.2f},{ev.real.max():.2f}]', flush=True)
print('done', time.time() - t0)
