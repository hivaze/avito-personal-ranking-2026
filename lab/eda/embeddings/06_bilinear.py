"""Learned user-item / history-item scoring on GPU (listwise softmax loss), evaluated on val with exact NDCG.
Models: dot (no training), bilinear u^T W c, bilinear + item bias b^T c, history-query, deep query.
Training sources: (a) train subsample (position-biased production traffic), (b) 5-fold CV on val (unbiased)."""
import os
import sys
import time
import numpy as np
import numba as nb
import torch
import torch.nn as nn
from common import ndcg_per_feed, feeds
from emb_features import load_split, l2n

assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('4', '5', '6', '7'), 'use GPU only'
dev = 'cuda'
torch.manual_seed(0)
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000


@nb.njit(parallel=True)
def pooled(Hn, valid, hoff, H):
    # [last1, mean last5, mean last20, mean all, recency hl=3]
    nf = len(H); d = Hn.shape[1]
    out = np.zeros((nf, 5, d), np.float32)
    for f in nb.prange(nf):
        c = 0
        for j in range(H[f]):
            o = hoff[f] + j
            if valid[o]:
                if c < 1: out[f, 0] += Hn[o]
                if c < 5: out[f, 1] += Hn[o]
                if c < 20: out[f, 2] += Hn[o]
                out[f, 3] += Hn[o]
                out[f, 4] += np.exp(-c / 3.0) * Hn[o]
                c += 1
        for q in range(5):
            n = np.sqrt((out[f, q] ** 2).sum())
            if n > 0:
                out[f, q] /= n
    return out


def prep(D):
    Cn, _ = l2n(D['C_raw'])
    Hn, _ = l2n(D['H_raw'])
    P = pooled(Hn, D['h_emb_idx'] >= 0, D['hoff'], D['H'])
    Un, _ = l2n(D['user'])
    return dict(C=torch.tensor(Cn, dtype=torch.float16, device=dev),
                U=torch.tensor(Un, device=dev), P=torch.tensor(P, device=dev),
                t=torch.tensor(D['c_target'].astype(np.float32), device=dev),
                M=torch.tensor(D['M'], device=dev), off=torch.tensor(D['coff'], device=dev),
                M_np=D['M'], off_np=D['coff'], t_np=D['c_target'].astype(np.float64))


class Model(nn.Module):
    def __init__(self, kind, d=128):
        super().__init__()
        self.kind = kind
        self.W = nn.Parameter(torch.eye(d))
        self.b = nn.Parameter(torch.zeros(d))
        if kind in ('hist', 'deep'):
            self.A = nn.Parameter(torch.zeros(5, d, d))
        if kind == 'deep':
            self.q = nn.Sequential(nn.Linear(6 * d, 512), nn.GELU(), nn.Linear(512, d))
            self.it = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, d))
            nn.init.zeros_(self.q[2].weight); nn.init.zeros_(self.q[2].bias)
            nn.init.zeros_(self.it[2].weight); nn.init.zeros_(self.it[2].bias)

    def forward(self, U, P, C):  # U [B,d], P [B,5,d], C [B,L,d]
        q = U @ self.W
        if self.kind in ('hist', 'deep'):
            q = q + torch.einsum('bkd,kde->be', P, self.A)
        if self.kind == 'deep':
            q = q + self.q(torch.cat([U, P.flatten(1)], 1))
            C = C + self.it(C)
        s = torch.einsum('bd,bld->bl', q, C)
        if self.kind != 'bilinear':
            s = s + C @ self.b
        return s


def batch(Dt, fids):
    M = Dt['M'][fids]; L = int(M.max())
    ar = torch.arange(L, device=dev)
    mask = ar[None, :] < M[:, None]
    idx = (Dt['off'][fids][:, None] + ar[None, :]).clamp(max=len(Dt['t']) - 1)
    C = Dt['C'][idx].float()
    t = Dt['t'][idx] * mask
    return Dt['U'][fids], Dt['P'][fids], C, t, mask


def loss_fn(s, t, mask):
    s = s.masked_fill(~mask, -1e4)
    ls = torch.log_softmax(s, 1)
    npos = t.sum(1).clamp(min=1)
    return -((ls * t).sum(1) / npos).mean()


@torch.no_grad()
def predict(model, Dt, fids=None):
    nf = len(Dt['M']) if fids is None else len(fids)
    fids = torch.arange(len(Dt['M']), device=dev) if fids is None else fids
    out = torch.zeros(len(Dt['t']), device=dev)
    for i in range(0, nf, 2048):
        fb = fids[i:i + 2048]
        U, P, C, t, mask = batch(Dt, fb)
        s = model(U, P, C)
        idx = (Dt['off'][fb][:, None] + torch.arange(mask.shape[1], device=dev)[None, :])
        out[idx[mask]] = s[mask]
    return out.cpu().numpy()


def nd(scores, Dt, sel=None):
    pf = ndcg_per_feed(scores, Dt['t_np'], Dt['off_np'], Dt['M_np'])
    return pf


def train(model, Dt, fids, epochs, lr=1e-3, wd=0.0, bs=256, reg_W=0.0, Dv=None, log=''):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    I = torch.eye(model.W.shape[0], device=dev)
    n = len(fids)
    for ep in range(epochs):
        perm = fids[torch.randperm(n, device=dev)]
        tl = 0; nb_ = 0
        for i in range(0, n, bs):
            U, P, C, t, mask = batch(Dt, perm[i:i + bs])
            s = model(U, P, C)
            loss = loss_fn(s, t, mask)
            if reg_W > 0:
                loss = loss + reg_W * ((model.W - I) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb_ += 1
        msg = f'{log} ep{ep} loss {tl / nb_:.4f}'
        if Dv is not None:
            msg += f' val NDCG {nd(predict(model, Dv), Dv).mean():.4f}'
        print(msg, flush=True)
    return model


t0 = time.time()
Dva = load_split('val'); Dv = prep(Dva)
rand_t = np.random.default_rng(0).random(len(Dv['t_np'])) * 1e-6
dot_pf = nd((Dv['U'].cpu().numpy().repeat(Dv['M_np'], 0) * Dv['C'].float().cpu().numpy()).sum(1) + rand_t, Dv)
print(f'dot(user,item) val NDCG {dot_pf.mean():.4f}')
fidx = np.sort(np.random.default_rng(42).choice(len(feeds('train')), NTR, replace=False))
Dtr_ = load_split('train', fidx); Dt = prep(Dtr_)
del Dtr_
print('loaded', time.time() - t0, flush=True)
all_tr = torch.arange(len(Dt['M']), device=dev)
nfv = len(Dv['M_np'])
res = {}

# (a) train on train subsample
for kind, ep, lr in [('bilinear', 3, 1e-3), ('bias', 3, 1e-3), ('hist', 3, 1e-3), ('deep', 3, 5e-4)]:
    torch.manual_seed(0)
    m = Model(kind).to(dev)
    train(m, Dt, all_tr, ep, lr=lr, Dv=Dv, log=f'[train->{kind}]')
    pf = nd(predict(m, Dv) + rand_t, Dv)
    d = pf - dot_pf
    res[f'train:{kind}'] = pf
    print(f'== train-fit {kind}: val NDCG {pf.mean():.4f}  vs dot {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(nfv):.4f})', flush=True)
    np.save(f'<repo>/eda/embeddings/cache/val_pred_train_{kind}.npy', predict(m, Dv))

# (b) 5-fold CV on val
folds = np.random.default_rng(7).integers(0, 5, nfv)
for kind, ep, lr, reg in [('bilinear', 15, 1e-3, 1e-3), ('bias', 15, 1e-3, 1e-3), ('hist', 15, 1e-3, 1e-3)]:
    oof = np.zeros(len(Dv['t_np']))
    for k in range(5):
        torch.manual_seed(k)
        m = Model(kind).to(dev)
        trf = torch.tensor(np.where(folds != k)[0], device=dev)
        tef = torch.tensor(np.where(folds == k)[0], device=dev)
        train(m, Dv, trf, ep, lr=lr, reg_W=reg, wd=1e-4, bs=128)
        p = predict(m, Dv, tef)
        sel = np.repeat(folds == k, Dv['M_np'])
        oof[sel] = p[sel]
    pf = nd(oof + rand_t, Dv); d = pf - dot_pf
    res[f'valcv:{kind}'] = pf
    print(f'== val-5foldCV {kind}: OOF NDCG {pf.mean():.4f} vs dot {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(nfv):.4f})', flush=True)

# (c) train-fit model fine-tuned? skip. summary
print('\nSUMMARY')
print(f'dot: {dot_pf.mean():.4f}')
for k, v in res.items():
    print(f'{k}: {v.mean():.4f}')
print('done', time.time() - t0)
