"""[train2] PER-HALF SHARED (siamese) stacker: c(x_half) applied to ft and to halves; val score = z(spec) + 0.5*(c_ft + c_to);
holdout score = z(to pool) + c_to. Optional holdout pretraining (--pre) then nested val fine-tuning; val-only models are
also evaluated OOS on the holdout to-half (acceptance criterion iii).
Neural listwise residual stacker (LIN / MLP / DeepSets / set-transformer) on family z-scores.

score = base + corr(x); base = z(equal pool); corr head zero-initialised (starts at the equal pool).
--train val : nested on the FT folds (outer k: 4 inner models, each trained on 3 folds, early-stopped on the 4th);
              OOF = mean of the 4 inner per-feed z; test = mean over all 20 inner models.
--train hold: to-half inputs only; 5 models on a user-grouped 5-fold split of the 30K holdout (ES on the held fold),
              applied to val (ft half and to half separately) and test.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, '<repo>/src')
from common import feed_meta
from stack_val import feed_rank_z
from metrics import ndcg_per_feed
from sklearn.model_selection import GroupKFold

R = Path('<repo>'); D = R / 'nn_x/stack/nnstack/data'; O = R / 'nn_x/stack/nnstack/out'; O.mkdir(exist_ok=True)
POOLS = {'PF7x': ['F5cd', 'F6', 'F7x', 'F9', 'F10', 'F12'], 'G3': ['F10', 'F7x', 'F7x70']}
p = argparse.ArgumentParser()
p.add_argument('--pool', default='PF7x'); p.add_argument('--mode', default='set')
p.add_argument('--train', default='val'); p.add_argument('--h', type=int, default=64)
p.add_argument('--layers', type=int, default=2); p.add_argument('--drop', type=float, default=0.1)
p.add_argument('--wd', type=float, default=1e-2); p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--lam', type=float, default=0.0, help='L2 penalty on the correction')
p.add_argument('--epochs', type=int, default=40); p.add_argument('--patience', type=int, default=6)
p.add_argument('--bs', type=int, default=64); p.add_argument('--ctx', type=int, default=1, help='use context features')
p.add_argument('--fams', type=int, default=1, help='use family z inputs'); p.add_argument('--tag', default='')
p.add_argument('--seed', type=int, default=0); p.add_argument('--no_test', action='store_true')
p.add_argument('--pre', type=int, default=0, help='pretrain on holdout first'); p.add_argument('--pre_lr', type=float, default=3e-4)
p.add_argument('--nseeds', type=int, default=1); p.add_argument('--fixed_ep', type=int, default=0, help='no ES: train exactly N epochs')
a = p.parse_args()
tag = a.tag or f'{a.train}_{a.pool}_{a.mode}'
LOG = open(O / f'{tag}.log', 'w')
def log(*s):
    m = ' '.join(str(x) for x in s); print(m, flush=True); LOG.write(m + '\n'); LOG.flush()
log('args', json.dumps(vars(a)))
dev = torch.device('cuda:0')
Fs = POOLS[a.pool]


def zf(o, M, x): return feed_rank_z(o, M, np.asarray(x, np.float32))


def feats(split, half):
    """Per-candidate inputs. half: 'both' (val-trained), 'ft' or 'to' (holdout-trained stacker applied to one half)."""
    if split == 'hold':
        M, o = np.load(D / 'hold_meta.npy')
    else:
        _, M, o = feed_meta(split)
    pre = split
    cols = []
    if half == 'both':
        base = np.load(D / f'{pre}_pool_{a.pool}_spec.npy')
        if a.fams:
            cols += [np.load(D / f'{pre}_fam_{F}_{h}.npy') for h in ('ft', 'to') for F in Fs]
        cols += [np.load(D / f'{pre}_pool_{a.pool}_ft.npy'), np.load(D / f'{pre}_pool_{a.pool}_to.npy')]
    else:
        base = np.load(D / f'{pre}_pool_{a.pool}_{half}.npy')
        if a.fams:
            cols += [np.load(D / f'{pre}_fam_{F}_{half}.npy') for F in Fs]
    rk = zf(o, M, base)[:, 0]
    cols += [base, rk]
    if a.ctx:
        ctx = np.load(D / f'{pre}_feedctx.npy')  # M, H, u0 per feed
        cold = np.load(D / f'{pre}_cold.npy')
        cs = np.bincount(np.repeat(np.arange(len(M)), M), cold, len(M)) / M
        rep = lambda v: np.repeat(v.astype(np.float32), M)
        cols += [rep(np.log(ctx[:, 0]) / np.log(300)), rep(np.log1p(ctx[:, 1]) / 6.0), rep(ctx[:, 2]), cold, rep(cs)]
    X = np.stack(cols, 1).astype(np.float32)
    return X, base.astype(np.float32), M.astype(np.int64), o.astype(np.int64)


class Split:
    """Padded per-feed tensors on GPU."""
    def __init__(self, X, base, M, o, y=None):
        n, L = len(M), int(M.max())
        self.M, self.o, self.n, self.L = M, o, n, L
        pos = np.arange(len(base)) - np.repeat(o, M); fid = np.repeat(np.arange(n), M)
        self.fid, self.pos = fid, pos
        Xp = np.zeros((n, L, X.shape[1]), np.float32); Xp[fid, pos] = X
        bp = np.zeros((n, L), np.float32); bp[fid, pos] = base
        mk = np.zeros((n, L), bool); mk[fid, pos] = True
        self.X = torch.from_numpy(Xp).to(dev); self.b = torch.from_numpy(bp).to(dev); self.mk = torch.from_numpy(mk).to(dev)
        self.Mt = torch.from_numpy(M).to(dev)
        if y is not None:
            yp = np.zeros((n, L), np.float32); yp[fid, pos] = y
            self.y = torch.from_numpy(yp).to(dev); self.yflat = y
            self.haspos = np.bincount(fid, y, n) > 0

    def batch(self, idx):
        idx_t = torch.as_tensor(idx, device=dev)
        L = int(self.M[idx].max())
        return self.X[idx_t, :L], self.b[idx_t, :L], self.mk[idx_t, :L], (self.y[idx_t, :L] if hasattr(self, 'y') else None)


class Block(nn.Module):
    def __init__(s, h, heads, drop):
        super().__init__()
        s.ln1 = nn.LayerNorm(h); s.att = nn.MultiheadAttention(h, heads, dropout=drop, batch_first=True)
        s.ln2 = nn.LayerNorm(h); s.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * h, h))
        s.dp = nn.Dropout(drop)

    def forward(s, x, mk):
        q = s.ln1(x)
        x = x + s.dp(s.att(q, q, q, key_padding_mask=~mk, need_weights=False)[0])
        return x + s.dp(s.ff(s.ln2(x)))


class Stacker(nn.Module):
    def __init__(s, d, h, mode, layers, drop):
        super().__init__()
        s.mode = mode
        if mode == 'lin':
            s.out = nn.Linear(d, 1)
        else:
            s.inp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, h))
            if mode == 'set':
                s.blocks = nn.ModuleList([Block(h, 4, drop) for _ in range(layers)])
            hin = 3 * h if mode == 'ds' else h
            s.out = nn.Sequential(nn.LayerNorm(hin), nn.Linear(hin, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, 1))
        last = s.out if mode == 'lin' else s.out[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        s.logt = nn.Parameter(torch.tensor(np.log(3.0), dtype=torch.float32))

    def forward(s, X, mk):
        if getattr(s, 'nh', 1) == 2:
            d = X.shape[-1] // 2
            return 0.5 * (s.fwd1(X[..., :d], mk) + s.fwd1(X[..., d:], mk))
        return s.fwd1(X, mk)

    def fwd1(s, X, mk):
        if s.mode == 'lin':
            return s.out(X).squeeze(-1)
        h = s.inp(X)
        if s.mode == 'set':
            for b in s.blocks:
                h = b(h, mk)
        elif s.mode == 'ds':
            m = mk.unsqueeze(-1).float()
            mean = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
            mx = h.masked_fill(~mk.unsqueeze(-1), -1e4).max(1, keepdim=True)[0]
            h = torch.cat([h, mean.expand_as(h), mx.expand_as(h)], -1)
        return s.out(h).squeeze(-1)


def predict(model, S, idx=None):
    model.eval(); idx = np.arange(S.n) if idx is None else idx
    res = []
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            bi = idx[i:i + 512]
            X, b, mk, _ = S.batch(bi)
            c = model(X, mk)
            sc = (b + c).masked_fill(~mk, 0)
            pad = torch.zeros((len(bi), S.L), device=dev); pad[:, :sc.shape[1]] = sc
            res.append(pad.cpu().numpy())
    full = np.concatenate(res, 0)  # rows = idx order
    return full


def flat_of(S, idx, full):
    """flat scores (feed order of idx) + meta for ndcg."""
    M = S.M[idx]; sc = np.concatenate([full[i, :m] for i, m in enumerate(M)])
    return sc, M, np.concatenate([[0], np.cumsum(M)[:-1]])


def yflat(S, idx):
    return np.concatenate([S.yflat[S.o[i]:S.o[i] + S.M[i]] for i in idx])


def ndcg_idx(S, idx, full):
    sc, M, o = flat_of(S, idx, full)
    return ndcg_per_feed(sc, yflat(S, idx), o, M).mean()


def train_one(S, tr_idx, es_idx, seed, init=None, lr=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Stacker(S.X.shape[2] // S.nh, a.h, a.mode, a.layers, a.drop).to(dev); model.nh = S.nh
    if init is not None: model.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or a.lr, weight_decay=a.wd)
    tr_idx = tr_idx[S.haspos[tr_idx]]
    yes = yflat(S, es_idx); Mes = S.M[es_idx]; oes = np.concatenate([[0], np.cumsum(Mes)[:-1]])
    def es_score():
        full = predict(model, S, es_idx); sc, _, _ = flat_of(S, es_idx, full)
        return ndcg_per_feed(sc, yes, oes, Mes).mean()
    best = es_score(); base_es = best; best_state = {k: v.clone() for k, v in model.state_dict().items()}; best_ep = 0; bad = 0
    curve = [round(best, 5)]
    for ep in range(1, a.epochs + 1):
        model.train()
        # length-bucketed batches: shuffle, chunk 32 batches, sort chunk by M
        perm = rng.permutation(tr_idx); chunks = []
        for i in range(0, len(perm), a.bs * 32):
            c = perm[i:i + a.bs * 32]; c = c[np.argsort(S.M[c], kind='stable')]
            chunks += [c[j:j + a.bs] for j in range(0, len(c), a.bs)]
        rng.shuffle(chunks)
        for bi in chunks:
            X, b, mk, y = S.batch(bi)
            c = model(X, mk)
            s = (b + c) * model.logt.exp()
            s = s.masked_fill(~mk, -1e9)
            lp = Fn.log_softmax(s, 1)
            loss = -(lp * y).sum(1) / y.sum(1).clamp(min=1)
            loss = loss.mean()
            if a.lam > 0:
                loss = loss + a.lam * (c.pow(2) * mk).sum() / mk.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sc = es_score(); curve.append(round(sc, 5))
        if a.fixed_ep:
            if ep >= a.fixed_ep:
                best_state = {k: v.clone() for k, v in model.state_dict().items()}; best, best_ep = sc, ep
                break
            continue
        if sc > best:
            best, best_ep, bad = sc, ep, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, best_ep, best - base_es, curve


def zfull(S, idx, full):
    """per-feed z of padded predictions -> padded z"""
    sc, M, o = flat_of(S, idx, full)
    zz = zf(o, M, sc)[:, 1]
    out = np.zeros_like(full); pos = np.arange(len(sc)) - np.repeat(o, M); out[np.repeat(np.arange(len(idx)), M), pos] = zz
    return out


def slices_report(name, s_new, s_ref):
    _, M, o = feed_meta('val'); y = np.load(D / 'val_y.npy')
    d = ndcg_per_feed(s_new, y, o, M) - ndcg_per_feed(s_ref, y, o, M)
    tue = np.load(D / 'val_tue.npy'); cold = np.load(D / 'val_cold.npy')
    fid = np.repeat(np.arange(len(M)), M); npos = np.bincount(fid, y, len(M)); nc = np.bincount(fid, y * cold, len(M))
    sl = {'all': np.ones(len(M), bool), 'M<50': M < 50, 'M>=50': M >= 50, 'Tue': tue, 'Thu': ~tue,
          'coldpos': (npos > 0) & (nc == npos), 'warmpos': (npos > 0) & (nc == 0)}
    out = {}
    for k, m in sl.items():
        dd = d[m]; out[k] = (dd.mean(), dd.std() / np.sqrt(m.sum()), int(m.sum()))
    log(f'{name}: new {ndcg_per_feed(s_new, y, o, M).mean():.5f} ref {ndcg_per_feed(s_ref, y, o, M).mean():.5f} | ' +
        ' '.join(f'{k} {v[0]:+.5f}+-{v[1]:.5f}' for k, v in out.items()))
    return out


def to_flat(S, full):
    return flat_of(S, np.arange(S.n), full)[0]



def mk_split(split, y=None):
    if split == 'hold':
        X, b, M, o = feats('hold', 'to'); S = Split(X, b, M, o, y); S.nh = 1; return S
    Xf, _, M, o = feats(split, 'ft'); Xt_, _, _, _ = feats(split, 'to')
    b = np.load(D / f'{split}_pool_{a.pool}_spec.npy')
    S = Split(np.concatenate([Xf, Xt_], 1), b, M, o, y); S.nh = 2; return S


def as_hold(m):  # evaluate a model on the holdout (single half)
    m.nh = 1; return m


t0 = time.time()
yv = np.load(D / 'val_y.npy'); yh = np.load(D / 'hold_y.npy')
S = mk_split('val', yv); Sh = mk_split('hold', yh)
log('val X', tuple(S.X.shape), 'hold X', tuple(Sh.X.shape))
base_v = np.load(D / f'val_pool_{a.pool}_spec.npy'); base_h = np.load(D / f'hold_pool_{a.pool}_to.npy')
Mh, oh = Sh.M, Sh.o
pre_states = [None]
if a.pre:  # pretrain on holdout: 5 user-grouped folds, ES on the held fold; keep the 5 states
    users = np.load(D / 'hold_user.npy'); g = np.where(users == 0, -1 - np.arange(len(users)), users)
    hfold = np.zeros(Sh.n, np.int8)
    for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(Sh.n), groups=g)): hfold[ho] = k
    pre_states = []
    for k in range(5):
        m, ep, gain, curve = train_one(Sh, np.where(hfold != k)[0], np.where(hfold == k)[0], seed=77 + k, lr=a.pre_lr)
        pre_states.append({kk: v.clone() for kk, v in m.state_dict().items()})
        log(f'pretrain {k}: best_ep {ep} es_gain {gain:+.5f} curve {curve} t {time.time() - t0:.0f}s')
fold = np.load(D / 'val_fold.npy')
oof = np.zeros((S.n, S.L), np.float32); hacc = np.zeros((Sh.n, Sh.L), np.float32)
if not a.no_test:
    St = mk_split('test'); test = np.zeros((St.n, St.L), np.float32)
info = []; nmod = 0
for k in range(5):
    ho = np.where(fold == k)[0]; acc = np.zeros((len(ho), S.L), np.float32)
    for j in [j for j in range(5) if j != k]:
        tr = np.where((fold != k) & (fold != j))[0]; es = np.where(fold == j)[0]
        for sd in range(a.nseeds):
            init = pre_states[(k + j + sd) % len(pre_states)]
            m, ep, gain, curve = train_one(S, tr, es, seed=100 * k + j + 1000 * a.seed + 10000 * sd, init=init)
            acc += zfull(S, ho, predict(m, S, ho)) / (4 * a.nseeds)
            if not a.no_test:
                test += zfull(St, np.arange(St.n), predict(m, St)) / (20 * a.nseeds)
            m.nh = 1; hacc += zfull(Sh, np.arange(Sh.n), predict(m, Sh)); nmod += 1
            info.append((k, j, ep, gain)); log(f'outer {k} inner-es {j} seed {sd}: best_ep {ep} es_gain {gain:+.5f} curve {curve} t {time.time() - t0:.0f}s')
    oof[ho] = acc
s_oof = to_flat(S, oof); np.save(O / f'{tag}_val_oof.npy', s_oof)
log('mean inner es gain', np.mean([i[3] for i in info]), 'mean best_ep', np.mean([i[2] for i in info]))
slices_report(f'{tag} OOF vs equal {a.pool}', s_oof, base_v)
sh = to_flat(Sh, hacc / nmod); np.save(O / f'{tag}_hold.npy', sh)
dh = ndcg_per_feed(sh, yh, oh, Mh) - ndcg_per_feed(base_h, yh, oh, Mh)
log(f'{tag} HOLDOUT (val-trained models on to-half; OOS only if --pre 0) delta vs equal to-pool {dh.mean():+.5f} +- {dh.std() / np.sqrt(len(dh)):.5f}')
if not a.no_test:
    np.save(O / f'{tag}_test.npy', to_flat(St, test))
log(f'done {time.time() - t0:.0f}s')
