"""Tiny interpretable history-attention scorers (few hundred params) on GPU, listwise softmax loss.
score_i = w_pool * c_i.(sum_j softmax_j(bias_j) h_j)          [target-independent learned pooling]
        + w_attn * sum_j softmax_j(beta*c_i.h_j + bias_j) * c_i.h_j   [target attention, DIN-like]
        + w_user * c_i.u
bias_j = a[event_type_j] + b[event_source_j] + r[hpos_j]  (all learnable). Train on train subsample and 5-fold CV on val."""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from common import ndcg_per_feed, feeds
from emb_features import load_split, l2n

assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('4', '5', '6', '7')
dev = 'cuda'
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
HL = 256


def prep(D):
    Cn, _ = l2n(D['C_raw']); Hn, _ = l2n(D['H_raw']); Un, _ = l2n(D['user'])
    nf = len(D['M'])
    # padded history
    hp = D['h_hpos'].astype(np.int64)
    fh = np.repeat(np.arange(nf), D['H'])
    Hp = np.zeros((nf, HL, 128), np.float16); Hp[fh, hp] = Hn
    Ty = np.zeros((nf, HL), np.int64); Ty[fh, hp] = D['h_event_type']
    So = np.zeros((nf, HL), np.int64); So[fh, hp] = D['h_event_source']
    Va = np.zeros((nf, HL), bool); Va[fh, hp] = D['h_emb_idx'] >= 0
    g = lambda x, dt=None: torch.tensor(x, device=dev) if dt is None else torch.tensor(x, device=dev, dtype=dt)
    return dict(C=g(Cn, torch.float16), U=g(Un), Hp=g(Hp), Ty=g(Ty), So=g(So), Va=g(Va),
                t=g(D['c_target'].astype(np.float32)), M=g(D['M']), off=g(D['coff']),
                M_np=D['M'], off_np=D['coff'], t_np=D['c_target'].astype(np.float64))


class Att(nn.Module):
    def __init__(self, use_type=True, use_src=True, use_rec=True, use_attn=True, use_pool=True, use_user=True):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(11)); self.b = nn.Parameter(torch.zeros(32)); self.r = nn.Parameter(torch.zeros(HL))
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.w = nn.Parameter(torch.tensor([10.0, 10.0, 10.0]))
        self.flags = dict(type=use_type, src=use_src, rec=use_rec, attn=use_attn, pool=use_pool, user=use_user)

    def bias(self, Ty, So):
        z = torch.zeros(Ty.shape, device=dev)
        if self.flags['type']: z = z + self.a[Ty]
        if self.flags['src']: z = z + self.b[So]
        if self.flags['rec']: z = z + self.r[None, :]
        return z

    def forward(self, U, Hp, Ty, So, Va, C):  # C [B,L,d]
        Hf = Hp.float()
        bj = self.bias(Ty, So).masked_fill(~Va, -1e4)  # [B,HL]
        anyv = Va.any(1, keepdim=True).float()  # [B,1]
        s = torch.zeros(C.shape[:2], device=dev)
        S = None
        if self.flags['pool']:
            w = torch.softmax(bj, 1)
            pv = torch.einsum('bh,bhd->bd', w, Hf)
            pv = pv / pv.norm(dim=1, keepdim=True).clamp(min=1e-6)
            s = s + self.w[0] * torch.einsum('bld,bd->bl', C, pv) * anyv
        if self.flags['attn']:
            S = torch.einsum('bld,bhd->blh', C, Hf)
            lg = self.beta * S + bj[:, None, :]
            att = torch.softmax(lg, 2)
            s = s + self.w[1] * (att * S).sum(2) * anyv
        if self.flags['user']:
            s = s + self.w[2] * torch.einsum('bld,bd->bl', C, U)
        return s


def batch(Dt, fids):
    M = Dt['M'][fids]; L = int(M.max()); ar = torch.arange(L, device=dev)
    mask = ar[None, :] < M[:, None]
    idx = (Dt['off'][fids][:, None] + ar[None, :]).clamp(max=len(Dt['t']) - 1)
    return (Dt['U'][fids], Dt['Hp'][fids], Dt['Ty'][fids], Dt['So'][fids], Dt['Va'][fids], Dt['C'][idx].float()), Dt['t'][idx] * mask, mask, idx


def loss_fn(s, t, mask):
    ls = torch.log_softmax(s.masked_fill(~mask, -1e4), 1)
    return -((ls * t).sum(1) / t.sum(1).clamp(min=1)).mean()


@torch.no_grad()
def predict(m, Dt, fids=None):
    fids = torch.arange(len(Dt['M']), device=dev) if fids is None else fids
    out = torch.zeros(len(Dt['t']), device=dev)
    for i in range(0, len(fids), 512):
        x, t, mask, idx = batch(Dt, fids[i:i + 512])
        out[idx[mask]] = m(*x)[mask]
    return out.cpu().numpy()


def fit(m, Dt, fids, epochs, lr=1e-2, bs=128):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        p = fids[torch.randperm(len(fids), device=dev)]
        for i in range(0, len(p), bs):
            x, t, mask, _ = batch(Dt, p[i:i + bs])
            loss = loss_fn(m(*x), t, mask)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


t0 = time.time()
Dv = prep(load_split('val'))
rv = np.random.default_rng(0).random(len(Dv['t_np'])) * 1e-6
nfv = len(Dv['M_np'])
fidx = np.sort(np.random.default_rng(42).choice(len(feeds('train')), NTR, replace=False))
Dt = prep(load_split('train', fidx))
print('loaded', time.time() - t0, flush=True)
alltr = torch.arange(len(Dt['M_np']), device=dev)
ref = None
variants = [
    ('user only', dict(use_attn=False, use_pool=False, use_type=False, use_src=False, use_rec=False)),
    ('pool: uniform mean', dict(use_attn=False, use_user=False, use_type=False, use_src=False, use_rec=False)),
    ('pool: +recency', dict(use_attn=False, use_user=False, use_type=False, use_src=False)),
    ('pool: +recency+type', dict(use_attn=False, use_user=False, use_src=False)),
    ('pool: +recency+type+source', dict(use_attn=False, use_user=False)),
    ('attn: +recency+type+source', dict(use_pool=False, use_user=False)),
    ('pool+attn (no user)', dict(use_user=False)),
    ('full: pool+attn+user', dict()),
]
fullm = None
for nm, kw in variants:
    torch.manual_seed(0)
    m = Att(**kw).to(dev)
    fit(m, Dt, alltr, 1)
    pf = ndcg_per_feed(predict(m, Dv) + rv, Dv['t_np'], Dv['off_np'], Dv['M_np'])
    if ref is None:
        ref = pf
    d = pf - ref
    print(f'[train-fit] {nm:30s} val NDCG {pf.mean():.4f}  vs user-only {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(nfv):.4f})  [{time.time() - t0:.0f}s]', flush=True)
    if nm.startswith('full'):
        fullm = m
        np.save('<repo>/eda/embeddings/cache/val_pred_attn_full.npy', predict(m, Dv))

m = fullm
print('\nlearned params (full model):')
print('  beta', round(m.beta.item(), 2), ' w(pool,attn,user)', np.round(m.w.detach().cpu().numpy(), 2))
a = m.a.detach().cpu().numpy(); print('  event_type bias (1..10, rel. to type1):', np.round(a[1:] - a[1], 2))
b = m.b.detach().cpu().numpy(); print('  event_source bias (0..31, rel. to mean):', np.round(b - b.mean(), 2))
r = m.r.detach().cpu().numpy(); print('  recency bias r[hpos] at 0,1,2,4,8,16,32,64,128,255 (rel. r0):', np.round((r - r[0])[[0, 1, 2, 4, 8, 16, 32, 64, 128, 255]], 2))

# 5-fold CV on val for full model (few params -> little overfitting)
folds = np.random.default_rng(7).integers(0, 5, nfv)
for nm, kw in [('full: pool+attn+user', dict()), ('user only', variants[0][1])]:
    oof = np.zeros(len(Dv['t_np']))
    for k in range(5):
        torch.manual_seed(k)
        mk = Att(**kw).to(dev)
        fit(mk, Dv, torch.tensor(np.where(folds != k)[0], device=dev), 4)
        p = predict(mk, Dv, torch.tensor(np.where(folds == k)[0], device=dev))
        sel = np.repeat(folds == k, Dv['M_np']); oof[sel] = p[sel]
    pf = ndcg_per_feed(oof + rv, Dv['t_np'], Dv['off_np'], Dv['M_np'])
    print(f'[val-5fold] {nm:30s} OOF NDCG {pf.mean():.4f}', flush=True)
print('done', time.time() - t0)
