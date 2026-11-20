"""Neural listwise residual stacker (LIN / MLP / DeepSets / set-transformer) on family z-scores.

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
POOLS = {'PF7x': ['F5cd', 'F6', 'F7x', 'F9', 'F10', 'F12'], 'G3': ['F10', 'F7x', 'F7x70'], 'M2': ['F7x', 'F7x70']}
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
p.add_argument('--hold_eval', type=int, default=0, help='val-trained: also apply to holdout with to-half z copied into the ft slots');
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


def feats_hold_both():
    """holdout inputs in the val 'both' layout, the (missing) ft half replaced by the to half."""
    M, o = np.load(D / 'hold_meta.npy'); base = np.load(D / f'hold_pool_{a.pool}_to.npy')
    cols = [np.load(D / f'hold_fam_{F}_to.npy') for _ in ('ft', 'to') for F in Fs] if a.fams else []
    cols += [base, base, base, zf(o, M, base)[:, 0]]
    if a.ctx:
        ctx = np.load(D / 'hold_feedctx.npy'); cold = np.load(D / 'hold_cold.npy')
        cs = np.bincount(np.repeat(np.arange(len(M)), M), cold, len(M)) / M
        rep = lambda v: np.repeat(v.astype(np.float32), M)
        cols += [rep(np.log(ctx[:, 0]) / np.log(300)), rep(np.log1p(ctx[:, 1]) / 6.0), rep(ctx[:, 2]), cold, rep(cs)]
    return np.stack(cols, 1).astype(np.float32), base.astype(np.float32), M.astype(np.int64), o.astype(np.int64)


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
        elif mode == 'wft':  # 2 per-feed scalars from feed context: corr = a(feed)*(z_ft - z_to) + b(feed)*rank(base); needs --fams 0 --ctx 1
            s.out = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))
        else:
            s.inp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, h))
            if mode == 'set':
                s.blocks = nn.ModuleList([Block(h, 4, drop) for _ in range(layers)])
            hin = 3 * h if mode == 'ds' else h
            s.dg = d - (5 if a.ctx else 0)  # gate over score columns (family z's, pool z's, base, rank), not context
            s.nf = (2 if a.train == 'val' else 1) * len(Fs)  # family z columns come first
            if mode == 'cgate':  # feed-adaptive CONVEX re-weighting of family z's: corr = sum_f (w_f(feed) - 1/nf) z_f
                s.out = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, s.nf))
            elif mode == 'gate':  # feed-conditional linear re-weighting: corr_i = sum_c g_c(feed) * x_ic
                s.out = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, s.dg))
            else:
                s.out = nn.Sequential(nn.LayerNorm(hin), nn.Linear(hin, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, 1))
        last = s.out if mode == 'lin' else s.out[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        s.logt = nn.Parameter(torch.tensor(np.log(3.0), dtype=torch.float32))

    def forward(s, X, mk):
        if s.mode == 'wft':
            g = s.out(X[:, 0, [4, 5, 6, 8]])  # per-feed context (log M, log H, user0, cold share) from the first candidate
            return g[:, 0:1] * (X[..., 0] - X[..., 1]) + g[:, 1:2] * X[..., 3]
        if s.mode == 'lin':
            return s.out(X).squeeze(-1)
        h = s.inp(X)
        if s.mode == 'set':
            for b in s.blocks:
                h = b(h, mk)
        elif s.mode == 'cgate':
            m = mk.unsqueeze(-1).float()
            mean = (h * m).sum(1) / m.sum(1)
            mx = h.masked_fill(~mk.unsqueeze(-1), -1e4).max(1)[0]
            w = torch.softmax(s.out(torch.cat([mean, mx], -1)), -1) - 1.0 / s.nf  # [B, nf], zero at init
            return (X[..., :s.nf] * w.unsqueeze(1)).sum(-1)
        elif s.mode == 'gate':
            m = mk.unsqueeze(-1).float()
            mean = (h * m).sum(1) / m.sum(1)
            mx = h.masked_fill(~mk.unsqueeze(-1), -1e4).max(1)[0]
            g = s.out(torch.cat([mean, mx], -1))  # [B, dg]
            return (X[..., :s.dg] * g.unsqueeze(1)).sum(-1)
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


def train_one(S, tr_idx, es_idx, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Stacker(S.X.shape[2], a.h, a.mode, a.layers, a.drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
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


t0 = time.time()
if a.train == 'val':
    X, base, M, o = feats('val', 'both'); y = np.load(D / 'val_y.npy')
    log('val X', X.shape)
    S = Split(X, base, M, o, y)
    fold = np.load(D / 'val_fold.npy')
    oof = np.zeros((S.n, S.L), np.float32)
    if not a.no_test:
        Xt, bt, Mt, ot = feats('test', 'both'); St = Split(Xt, bt, Mt, ot); test = np.zeros((St.n, St.L), np.float32)
    info = []
    if a.hold_eval:
        Xh, bh, Mh, oh = feats_hold_both(); yh = np.load(D / 'hold_y.npy'); Sh = Split(Xh, bh, Mh, oh, yh)
        hacc = np.zeros((Sh.n, Sh.L), np.float32)
    for k in range(5):
        ho = np.where(fold == k)[0]; acc = np.zeros((len(ho), S.L), np.float32)
        for j in [j for j in range(5) if j != k]:
            tr = np.where((fold != k) & (fold != j))[0]; es = np.where(fold == j)[0]
            for sd in range(a.nseeds):
                m, ep, gain, curve = train_one(S, tr, es, seed=100 * k + j + 1000 * a.seed + 10000 * sd)
                acc += zfull(S, ho, predict(m, S, ho)) / (4 * a.nseeds)
                if not a.no_test:
                    test += zfull(St, np.arange(St.n), predict(m, St)) / (20 * a.nseeds)
                if a.hold_eval:
                    hacc += zfull(Sh, np.arange(Sh.n), predict(m, Sh)) / (20 * a.nseeds)
                info.append((k, j, ep, gain)); log(f'outer {k} inner-es {j} seed {sd}: best_ep {ep} es_gain {gain:+.5f} curve {curve} t {time.time() - t0:.0f}s')
        oof[ho] = acc
        log(f'outer {k}: fold ndcg new {ndcg_idx(S, ho, acc):.5f} base {ndcg_idx(S, ho, S.b[torch.as_tensor(ho, device=dev)].cpu().numpy()):.5f}')
    s_oof = to_flat(S, oof)
    np.save(O / f'{tag}_val_oof.npy', s_oof)
    log('mean inner es gain', np.mean([i[3] for i in info]), 'mean best_ep', np.mean([i[2] for i in info]))
    slices_report(f'{tag} OOF vs equal {a.pool}', s_oof, base)
    if a.hold_eval:
        sh = to_flat(Sh, hacc); np.save(O / f'{tag}_hold.npy', sh)
        dh = ndcg_per_feed(sh, yh, oh, Mh) - ndcg_per_feed(bh, yh, oh, Mh)
        log(f'{tag} HOLDOUT OOS (val-trained, ft slots <- to half) delta vs equal to-pool {dh.mean():+.5f} +- {dh.std() / np.sqrt(len(dh)):.5f}')
    if not a.no_test:
        np.save(O / f'{tag}_test.npy', to_flat(St, test))
        rho = np.mean([np.corrcoef(to_flat(St, test)[ot[i]:ot[i] + Mt[i]], bt[ot[i]:ot[i] + Mt[i]])[0, 1] for i in range(0, St.n, 7)])
        rv = np.mean([np.corrcoef(s_oof[o[i]:o[i] + M[i]], base[o[i]:o[i] + M[i]])[0, 1] for i in range(0, S.n, 7) if M[i] > 2])
        log(f'mean per-feed corr new-vs-base: val {rv:.4f} test {rho:.4f}')
else:  # holdout-trained, to-half inputs
    X, base, M, o = feats('hold', 'to'); y = np.load(D / 'hold_y.npy')
    log('hold X', X.shape)
    S = Split(X, base, M, o, y)
    users = np.load(D / 'hold_user.npy'); g = np.where(users == 0, -1 - np.arange(len(users)), users)
    hfold = np.zeros(S.n, np.int8)
    for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(S.n), groups=g)): hfold[ho] = k
    Sv = {}
    for half in ('ft', 'to'):
        Xv, bv, Mv, ov = feats('val', half); Sv[half] = Split(Xv, bv, Mv, ov)
    accv = {h: np.zeros((Sv[h].n, Sv[h].L), np.float32) for h in Sv}
    hoof = np.zeros((S.n, S.L), np.float32)
    if not a.no_test:
        St = {}; acct = {}
        for half in ('ft', 'to'):
            Xt, bt, Mt, ot = feats('test', half); St[half] = Split(Xt, bt, Mt, ot); acct[half] = np.zeros((St[half].n, St[half].L), np.float32)
    for k in range(5):
        tr = np.where(hfold != k)[0]; es = np.where(hfold == k)[0]
        m, ep, gain, curve = train_one(S, tr, es, seed=7 + k + 1000 * a.seed)
        hoof[es] = zfull(S, es, predict(m, S, es))
        for h in Sv:
            accv[h] += zfull(Sv[h], np.arange(Sv[h].n), predict(m, Sv[h])) / 5
            if not a.no_test:
                acct[h] += zfull(St[h], np.arange(St[h].n), predict(m, St[h])) / 5
        log(f'hold model {k}: best_ep {ep} es_gain {gain:+.5f} curve {curve} t {time.time() - t0:.0f}s')
    # holdout in-domain OOF (ES fold = eval fold, so slightly optimistic)
    sh = to_flat(S, hoof); dh = ndcg_per_feed(sh, y, o, M) - ndcg_per_feed(base, y, o, M)
    log(f'holdout in-domain (ES fold) delta {dh.mean():+.5f} +- {dh.std() / np.sqrt(len(dh)):.5f}')
    _, Mv, ov = feed_meta('val')
    sft, sto = to_flat(Sv['ft'], accv['ft']), to_flat(Sv['to'], accv['to'])
    np.save(O / f'{tag}_val_ft.npy', sft); np.save(O / f'{tag}_val_to.npy', sto)
    ref = np.load(D / f'val_pool_{a.pool}_spec.npy')
    zz = lambda x: zf(ov, Mv, x)[:, 1]
    both = zz(0.5 * zz(sft) + 0.5 * zz(sto))
    slices_report(f'{tag} hold->val both halves vs equal {a.pool}', both, ref)
    toonly = zz(0.5 * np.load(D / f'val_pool_{a.pool}_ft.npy') + 0.5 * zz(sto))
    slices_report(f'{tag} hold->val to-half only vs equal {a.pool}', toonly, ref)
    slices_report(f'{tag} hold->val TO-HALF ALONE vs equal to-pool', sto, np.load(D / f'val_pool_{a.pool}_to.npy'))
    np.save(O / f'{tag}_val_both.npy', both)
    if not a.no_test:
        _, Mt, ot = feed_meta('test'); zt = lambda x: zf(ot, Mt, x)[:, 1]
        np.save(O / f'{tag}_test_ft.npy', to_flat(St['ft'], acct['ft'])); np.save(O / f'{tag}_test_to.npy', to_flat(St['to'], acct['to']))
log(f'done {time.time() - t0:.0f}s')
