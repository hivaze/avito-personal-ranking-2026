# Solutions review, 2026-11-11: sixteen ranking losses compared on one small proxy ranker (an MLP on 48
# pre-computed candidate features, 300,000 train feeds, 3 epochs, CPU only), scored on val and on the holdout.
# Run as (from lab/solvers/solvers/run_B.sh):
#   python loss_zoo.py --kinds listmle,ce_l2pp,ce_rw --seeds $s --ntr 300000 --epochs 3 --threads 5 --tag A
# It wrote per-feed NDCG and raw scores to solvers/out/solvers/zoo_<tag>_<loss>_s<seed>.npz.
# Result on record: nothing beat softmax cross-entropy; full ListMLE read -0.00516 (z -6.5) on val.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/solvers/solvers/loss_zoo.py
# Loss-zoo proxy (CPU only). Same harness as reports/review_ideas.md 1.5 (MLP on 48 precomputed per-candidate
# features + NaN flags, 300K production train feeds of Sep 9-19, 3 epochs), extended to many ranking losses.
# Anchors that the real NN has already measured: bce (NN: -0.001 vs softmax) and k9 (NN: +0.0024 vs no aux),
# so we can see whether the proxy is sensitive to the kind of changes we care about.
# Usage: CUDA_VISIBLE_DEVICES="" python loss_zoo.py --kinds ce,bce,... --seeds 1,2,3 --ntr 300000 --threads 8 --tag A
import sys, time, argparse, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '<repo>/solvers/solvers')
from proxy_lib import *

ap = argparse.ArgumentParser()
ap.add_argument('--kinds', default='ce')
ap.add_argument('--seeds', default='1,2,3')
ap.add_argument('--ntr', type=int, default=300000)
ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--tag', default='A')
ap.add_argument('--pool', default='all', choices=['all', 'random', 'prod', 'randplus', 'rand50', 'randshort'])  # training-feed pool
ap.add_argument('--nrand', type=int, default=300000)
ap.add_argument('--width', type=int, default=0)  # >0: wide MLP d_in-W-W-2 (memorisation-capable) instead of 128-64
ap.add_argument('--evals', type=int, default=1)  # >1: also evaluate val/holdout at this many evenly spaced points  # randplus: number of random-like feeds added to ntr production feeds
a = ap.parse_args()
torch.set_num_threads(a.threads)
OUT = '<repo>/solvers/out/solvers'
NAMES = CONTENT + COLLAB
KINDS = a.kinds.split(','); SEEDS = [int(s) for s in a.seeds.split(',')]

f, M, off = feed_meta('train'); day = day_of(f['timestamp'].to_numpy()); p = np.nan_to_num(prandom(), nan=0.0)
rng = np.random.default_rng(0)
hold_c = np.where((p > 0.9) & (M >= 50))[0]; hold = np.sort(rng.choice(hold_c, 20000, replace=False))
if a.pool == 'all':
    pool = np.setdiff1d(np.where(day >= 3)[0], hold)
elif a.pool == 'random':  # random-like exploration feeds only (p_random > 0.5), all M
    pool = np.setdiff1d(np.where((day >= 3) & (p > 0.5))[0], hold)
elif a.pool == 'rand50':  # random-like with M >= 50 (detector reliable; = the NN's cooldown pool definition)
    pool = np.setdiff1d(np.where((day >= 3) & (p > 0.5) & (M >= 50))[0], hold)
elif a.pool == 'randshort':  # 'random-like' with M < 50 (detector not identifiable there; label-driven selection)
    pool = np.setdiff1d(np.where((day >= 3) & (p > 0.5) & (M < 50))[0], hold)
elif a.pool == 'prod':  # production feeds only (p_random <= 0.1 or unknown)
    pool = np.setdiff1d(np.where((day >= 3) & (p <= 0.1))[0], hold)
if a.pool == 'randplus':  # nrand random-like feeds + ntr production feeds
    pr_ = np.setdiff1d(np.where((day >= 3) & (p > 0.5))[0], hold); pp_ = np.setdiff1d(np.where((day >= 3) & (p <= 0.1))[0], hold)
    tr = np.sort(np.concatenate([rng.choice(pr_, min(a.nrand, len(pr_)), replace=False), rng.choice(pp_, a.ntr, replace=False)]))
else:
    tr = np.sort(rng.choice(pool, min(a.ntr, len(pool)), replace=False))
t0 = time.time()
rtr = rows_of(off, M, tr); Xtr = load_X('train', rtr, NAMES)
ytr = np.asarray(np.load(FLAT / 'train/cand_target.npy', mmap_mode='r')[rtr], np.float32)
ptr = np.asarray(np.load(FLAT / 'train/cand_pos.npy', mmap_mode='r')[rtr], np.int64)
rh = rows_of(off, M, hold); Xh = load_X('train', rh, NAMES)
fv, Mv, ov = feed_meta('val'); Xv = load_X('val', np.arange(int(Mv.sum())), NAMES)
mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-6; hasnan = np.isnan(Xtr).any(0)
def prep(X):
    nanm = np.isnan(X).astype(np.float32); X = (X - mu) / sd; X = np.nan_to_num(X, nan=0.0); X = np.clip(X, -8, 8)
    return np.concatenate([X, nanm[:, hasnan]], 1).astype(np.float32)
Xtr, Xh, Xv = prep(Xtr), prep(Xh), prep(Xv)
print('loaded', Xtr.shape, Xh.shape, Xv.shape, f'{time.time()-t0:.0f}s', 'pool', a.pool, len(tr), flush=True)
Mtr = M[tr]; otr = np.concatenate([[0], np.cumsum(Mtr)[:-1]]); prtr = p[tr].astype(np.float32)

def batches(rng, B=256):
    order = rng.permutation(len(tr)); out = []
    for i in range(0, len(order), B * 20):
        ch = order[i:i + B * 20]; ch = ch[np.argsort(Mtr[ch])]
        for j in range(0, len(ch), B): out.append(ch[j:j + B])
    rng.shuffle(out); return out

def pad(ids):
    L = int(Mtr[ids].max()); idx = otr[ids][:, None] + np.arange(L)[None, :]; m = np.arange(L)[None, :] < Mtr[ids][:, None]
    idx = np.where(m, idx, 0)
    return (torch.from_numpy(Xtr[idx]), torch.from_numpy(ytr[idx] * m), torch.from_numpy(m),
            torch.from_numpy(ptr[idx] * m), torch.from_numpy(Mtr[ids]), torch.from_numpy(prtr[ids]))

LOG2 = np.log(2.0)
def D(r): return torch.log2(1.0 + r)
_IDCG = torch.cat([torch.zeros(1), torch.cumsum(1.0 / torch.log2(torch.arange(2, 600).float()), 0)])
def idcg(K): return _IDCG[K.long().clamp(0, 598)].clamp_min(1e-6)

def ranks(s, m):  # 1-based descending ranks under current scores (padding at the end)
    lg = s.detach().masked_fill(~m, -1e9)
    o = torch.argsort(lg, 1, descending=True); r = torch.empty_like(o); r.scatter_(1, o, torch.arange(1, s.shape[1] + 1).expand_as(o))
    return r.float()

def gather_pos(s, y):  # positive scores [B,Kmax] + mask + their column index
    Kmax = int(y.sum(1).max().item()); key = y + torch.rand_like(y) * 0.5  # positives first, random order among them
    idx = torch.topk(key, Kmax, 1).indices; pm = torch.gather(y, 1, idx) > 0
    return torch.gather(s, 1, idx), pm, idx

def feed_mean(l, K):
    v = (K > 0).float(); return (l * v).sum() / v.sum()

def loss_fn(kind, out, y, m, pos, Mi, pr):
    s = out[:, :, 0]; K = y.sum(1); lg = s.masked_fill(~m, -1e4); logp = torch.log_softmax(lg, 1)
    ce = -(y * logp).sum(1) / K.clamp_min(1)
    if kind in ('ce', 'pos'):
        return feed_mean(ce, K)
    if kind == 'bce':
        l = F.binary_cross_entropy_with_logits(s, y, reduction='none'); l = (l * m).sum(1) / m.sum(1)
        return feed_mean(l, K)
    if kind == 'ce_bce':
        l = F.binary_cross_entropy_with_logits(s, y, reduction='none'); l = (l * m).sum(1) / m.sum(1)
        return feed_mean(ce + 0.5 * l * 10, K)  # BCE per-feed mean is ~0.1: x10 puts it on the CE gradient scale
    if kind in ('ce_ls', 'ce_ls2'):
        eps = 0.1 if kind == 'ce_ls' else 0.2; t = (1 - eps) * y / K.clamp_min(1)[:, None] + eps * m.float() / m.sum(1, keepdim=True)
        return feed_mean(-(t * logp).sum(1), K)
    if kind == 'focal':
        pp = logp.exp(); l = -(y * (1 - pp).pow(2) * logp).sum(1) / K.clamp_min(1)
        return feed_mean(l, K)
    if kind == 'anchor':  # calibrated softmax: a virtual candidate with fixed logit 0 in every feed's denominator
        lga = torch.cat([lg, torch.zeros_like(lg[:, :1])], 1); lpa = torch.log_softmax(lga, 1)[:, :-1]
        return feed_mean(-(y * lpa).sum(1) / K.clamp_min(1), K)
    if kind == 'k9':  # ce + listwise production-order distillation (aux head), weight 0.5 * (1 - p_random)
        a_ = out[:, :, 1].masked_fill(~m, -1e4); tgt = torch.softmax((-pos.float() / 10.0).masked_fill(~m, -1e4), 1)
        aux = -(tgt * torch.log_softmax(a_, 1)).sum(1); w = (1 - pr)
        return feed_mean(ce, K) + 0.5 * (aux * w * (K > 0)).sum() / (K > 0).float().sum()
    if kind in ('ranknet', 'lambdarank', 'l2pp', 'approx', 'plk'):
        sp, pm, pidx = gather_pos(s, y)
        neg = (m & (y == 0)).float()
        diff = sp[:, :, None] - s[:, None, :]  # [B,Kmax,L] positive minus every candidate
        if kind == 'ranknet':
            P = pm[:, :, None].float() * neg[:, None, :]
            l = (F.softplus(-diff) * P).sum((1, 2)) / P.sum((1, 2)).clamp_min(1)
            return feed_mean(l, K)
        if kind in ('lambdarank', 'l2pp'):
            r = ranks(s, m); rp = torch.gather(r, 1, pidx)
            P = pm[:, :, None].float() * neg[:, None, :]
            rho = (1.0 / D(rp)[:, :, None] - 1.0 / D(r)[:, None, :]).abs()
            if kind == 'lambdarank':
                w = rho
            else:
                dd = (rp[:, :, None] - r[:, None, :]).abs().clamp_min(1.0)
                delta = (1.0 / D(dd) - 1.0 / D(dd + 1)).abs()
                w = rho + 5.0 * delta
            w = w / idcg(K)[:, None, None]
            l = (w.detach() * F.softplus(-diff) / LOG2 * P).sum((1, 2))
            return feed_mean(l, K)
        if kind == 'approx':  # ApproxNDCG (Qin et al. 2010), alpha = 1
            sig = torch.sigmoid(-diff) * m[:, None, :].float()  # P(candidate j above positive i)
            rhat = 1.0 + sig.sum(2) - 0.5  # subtract the self term sigmoid(0)=0.5
            g = (pm.float() / D(rhat)).sum(1)
            return feed_mean(1.0 - g / idcg(K), K)
        if kind == 'plk':  # Plackett-Luce over the positives (random order), negatives never ranked
            Km = sp.shape[1]
            # candidate j is removed from the denominator of positive k if j is an earlier positive (k' < k)
            earlier = torch.zeros_like(diff, dtype=torch.bool)
            for k in range(Km):
                for k2 in range(k):
                    earlier[:, k, :] |= (torch.arange(s.shape[1])[None, :] == pidx[:, k2:k2 + 1]) & pm[:, k2:k2 + 1]
            den = torch.logsumexp(lg[:, None, :].expand_as(diff).masked_fill(earlier, -1e4), 2)
            l = ((den - sp) * pm.float()).sum(1) / K.clamp_min(1)
            return feed_mean(l, K)
    if kind == 'listmle':  # full ListMLE (Xia et al. 2008; Avito production loss): PL likelihood of one label-consistent
        # permutation (positives first, random order within positives and within negatives), triangular-mask form
        key = (y * 2 + torch.rand_like(y)).masked_fill(~m, -1.0)
        o = torch.argsort(key, 1, descending=True); so = torch.gather(lg, 1, o); mo = torch.gather(m, 1, o)
        # log-sum-exp over the suffix j >= k, computed by a reversed cumulative logsumexp
        suf = torch.flip(torch.logcumsumexp(torch.flip(so.masked_fill(~mo, -1e4), [1]), 1), [1])
        l = ((suf - so) * mo.float()).sum(1) / mo.float().sum(1)
        return feed_mean(l, K)
    if kind in ('ce_l2pp', 'ce_rw'):
        if kind == 'ce_rw':  # softmax CE re-weighted per feed by the NDCG sensitivity at the current positive ranks
            r = ranks(s, m); rp = (r * y).sum(1) / K.clamp_min(1)  # mean rank of the positives (detached)
            w = (rp / (1 + rp)) / (np.log(2.0) * D(rp) ** 2)  # d NDCG / d log r for a single positive
            w = w / (w * (K > 0)).sum().clamp_min(1e-6) * (K > 0).sum()  # mean 1 over the batch
            return feed_mean(ce * w.detach(), K)
        return feed_mean(ce, K) + loss_fn('l2pp', out, y, m, pos, Mi, pr)
    if kind == 'neuralndcg':  # NeuralNDCG (Pobrotyn & Bialobrzeski 2021): NeuralSort tau=1 + Sinkhorn, full list
        B, L = s.shape; n = m.sum(1).float(); mf = m.float()
        sm = s.masked_fill(~m, 0.0)
        A = ((sm[:, :, None] - sm[:, None, :]).abs() * mf[:, None, :]).sum(2)  # [B,L] sum_k |s_j - s_k|
        i = torch.arange(1, L + 1, dtype=torch.float32)
        scal = (n[:, None] + 1 - 2 * i[None, :])  # [B,L] per rank row
        logits = scal[:, :, None] * sm[:, None, :] - A[:, None, :]  # [B,L(rank),L(item)]
        valid = (m[:, :, None] & m[:, None, :])  # rank rows < n and item columns < n
        lp = logits.masked_fill(~valid, -1e4)
        lp = lp - torch.logsumexp(lp, 2, keepdim=True)
        for _ in range(10):  # Sinkhorn in log space on the valid n x n block
            lp = (lp - torch.logsumexp(lp, 1, keepdim=True)).masked_fill(~valid, -1e4)
            lp = (lp - torch.logsumexp(lp, 2, keepdim=True)).masked_fill(~valid, -1e4)
        Pm = lp.exp() * valid.float()
        eg = (Pm * y[:, None, :]).sum(2)  # expected gain at each rank
        dcg = (eg / D(i)[None, :] * mf).sum(1)
        return feed_mean(1.0 - dcg / idcg(K), K)
    raise ValueError(kind)

class Net(torch.nn.Module):
    def __init__(self, d_in, pos_tower=False):
        super().__init__()
        W = a.width
        if W > 0:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(d_in, W), torch.nn.GELU(), torch.nn.Linear(W, W), torch.nn.GELU())
            self.head = torch.nn.Linear(W, 2)
        else:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(d_in, 128), torch.nn.GELU(), torch.nn.Linear(128, 64), torch.nn.GELU())
            self.head = torch.nn.Linear(64, 2)
        self.pos_tower = pos_tower
        if pos_tower:
            self.g = torch.nn.Embedding(400, 1); self.g2 = torch.nn.Embedding(400, 1)
            torch.nn.init.zeros_(self.g.weight); torch.nn.init.zeros_(self.g2.weight)
    def forward(self, X, pos=None, Mi=None, pr=None):
        o = self.head(self.mlp(X))
        if self.pos_tower and pos is not None:  # additive train-only position logit (PAL/shallow tower), off at inference
            rev = (Mi[:, None] - 1 - pos).clamp(0, 399)
            add = (self.g(pos.clamp(0, 399)) + self.g2(rev))[..., 0] * (1 - pr)[:, None]
            o = torch.cat([o[..., :1] + add[..., None], o[..., 1:]], -1)
        return o

def predict(net, X):
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(X[i:i + 500000]))[:, 0].numpy() for i in range(0, len(X), 500000)])

for seed in SEEDS:
    for kind in KINDS:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        net = Net(Xtr.shape[1], pos_tower=(kind == 'pos'))
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        bl = [b for e in range(a.epochs) for b in batches(rng)]; T = len(bl); t1 = time.time()
        for st, b in enumerate(bl):
            for g in opt.param_groups: g['lr'] = 2e-3 * 0.5 * (1 + np.cos(np.pi * st / T))
            X, y, m, pos, Mi, pr = pad(b)
            o = net(X, pos, Mi, pr)
            l = loss_fn(kind, o, y, m, pos, Mi, pr)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            if a.evals > 1 and (st + 1) % (T // a.evals) == 0 and st + 1 < T:
                print(seed, kind, 'step %d/%d val %.5f hold %.5f' % (st + 1, T, per_feed(predict(net, Xv), 'val').mean(),
                      per_feed(predict(net, Xh), 'train', hold).mean()), flush=True)
        sv = predict(net, Xv); sh = predict(net, Xh)
        pv = per_feed(sv, 'val'); ph = per_feed(sh, 'train', hold)
        np.savez(f'{OUT}/zoo_{a.tag}_{kind}_s{seed}.npz', val=pv.astype(np.float32), hold=ph.astype(np.float32),
                 sval=sv.astype(np.float32), shold=sh.astype(np.float32))
        print(seed, kind, 'val %.5f hold %.5f' % (pv.mean(), ph.mean()), f'{time.time()-t1:.0f}s', flush=True)
