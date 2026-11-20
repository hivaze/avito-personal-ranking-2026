"""P2: label-conditional (co-click) information and what the in-sample teacher memorises.
 (A) HOLDOUT feeds (30K, never trained on; teacher_mean11 is honest there), feeds with >= 2 positives:
     hide one positive i, reveal the others R; candidates = feed minus R. score_j = t_j + b1*z(maxcos(j,R)) + b2*samecat(j,R).
     b fitted by conditional logit on a user-half, read on the other half. NDCG/MRR of the hidden positive with and without the reveal.
 (B) 30K exploration TRAIN feeds (not holdout): D_j = centred in-sample teacher logit - centred OOF teacher logit.
     Within-feed regression of D on y_j, z(maxcos(j, P\\{j})), samecat(j, P\\{j}) -> does the in-sample teacher put its
     label-dependent mass on the click only (label copy) or also on the click's neighbours (label-conditional smoothing)?
CPU, 2 threads."""
from lb_common import *
import polars as pl, json
T = ROOT / 'nn_x/w12/kd/out'
ftr = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['user_id', 'M', 'cand_off'])
uid = ftr['user_id'].to_numpy().astype(np.int64); M = ftr['M'].to_numpy().astype(np.int64); off = ftr['cand_off'].to_numpy().astype(np.int64)
pr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0); pr[M < 50] = 0
sel = np.where((pr > 0.9) & (M >= 50))[0]; hold = np.sort(np.random.default_rng(12345).choice(sel, 30000, replace=False))
ex = np.sort(np.random.default_rng(1).choice(np.setdiff1d(sel, hold), 30000, replace=False))
d = ROOT / 'data/competition/train/item_embeddings'; meta = json.loads((d / 'data.meta').read_text())
table = np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))
emb_idx = np.load(FLAT / 'train/cand_emb_idx.npy', mmap_mode='r'); cat = np.load(FLAT / 'train/cand_category_id.npy', mmap_mode='r')
ytr = np.load(FLAT / 'train/cand_target.npy', mmap_mode='r')
t_in = np.load(T / 'teacher_mean11.npy', mmap_mode='r'); t_oof = np.load(T / 'teacher_oof.npy', mmap_mode='r')

def feed(i):
    s = slice(off[i], off[i] + M[i]); e = np.asarray(emb_idx[s]); E = np.asarray(table[np.maximum(e, 0)], np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6); E[e < 0] = 0
    return E, np.asarray(cat[s]), np.asarray(ytr[s]).astype(bool), np.asarray(t_in[s], np.float64), np.asarray(t_oof[s], np.float64)
zs = lambda x: (x - x.mean()) / (x.std() + 1e-9)

# ---------- (A) leave-one-positive-out on the holdout ----------
rows = []  # per LOO task: (feed idx, user, t (cands), simz, samecat, target position)
for i in hold:
    E, c, y, ti, _ = feed(i)
    P_ = np.where(y)[0]
    if len(P_) < 2: continue
    S = E @ E[P_].T  # [M, |P|]
    for k, ip in enumerate(P_):
        R = np.delete(P_, k); keep = np.ones(len(y), bool); keep[R] = False
        Rk = [kk for kk in range(len(P_)) if kk != k]
        sim = S[:, Rk].max(1)[keep]; sc = np.isin(c, c[R])[keep].astype(np.float64)
        rows.append((i, uid[i], zs(ti[keep]) , zs(sim), sc - sc.mean(), int(np.where(np.where(keep)[0] == ip)[0][0])))
print(f'(A) holdout feeds with >=2 positives: {len(set(r[0] for r in rows))} of 30000; LOO tasks {len(rows)}')
def score(rs, b):
    nd = []; rr = []
    for _, _, t, s, sc, tp in rs:
        v = b[0] * t + b[1] * s + b[2] * sc; rk = int((v > v[tp]).sum()) + 1; nd.append(1 / np.log2(rk + 1)); rr.append(1 / rk)
    return np.array(nd), np.array(rr)
def fit(rs, iters=60, lr=0.5):
    b = np.array([1.0, 0.0, 0.0])  # conditional logit over [t_z, sim_z, samecat]
    for _ in range(iters):
        g = np.zeros(3)
        for _, _, t, s, sc, tp in rs:
            Xf = np.stack([t, s, sc], 1); v = Xf @ b; p = np.exp(v - v.max()); p /= p.sum(); g += Xf[tp] - p @ Xf
        b += lr * g / len(rs)
    return b
h = np.array([(r[1] * 2654435761 % 4294967296) % 2 for r in rows]); A = [r for r, k in zip(rows, h) if k == 0]; B = [r for r, k in zip(rows, h) if k == 1]
out = []
for tr_, te_ in ((A, B), (B, A)):
    b0 = fit([(a, u, t, np.zeros_like(s), np.zeros_like(sc), tp) for a, u, t, s, sc, tp in tr_]); b1 = fit(tr_)
    n0, r0 = score(te_, b0); n1, r1 = score(te_, b1); out.append((n0, n1, r0, r1, b1))
n0 = np.concatenate([o[0] for o in out]); n1 = np.concatenate([o[1] for o in out]); r0 = np.concatenate([o[2] for o in out]); r1 = np.concatenate([o[3] for o in out])
print(f'    hidden-positive NDCG: teacher only {n0.mean():.5f} -> + revealed co-clicks {n1.mean():.5f}  delta {f(n1 - n0)} | MRR {r0.mean():.5f} -> {r1.mean():.5f}')
print(f'    fitted coefficients [teacher_z, simz_to_revealed, samecat] per half: {[np.round(o[4], 3).tolist() for o in out]}')
bfit = np.mean([o[4] for o in out], 0)

# ---------- (B) what the in-sample teacher adds over the cross-fitted one ----------
XtX = np.zeros((4, 4)); Xty = np.zeros(4); n = 0; dpos = []; dneg_near = []; dneg_far = []; sd_in = []; sd_oof = []
nd_in = []; nd_oof = []
for i in ex:
    E, c, y, ti, to = feed(i)
    ti = ti - ti.mean(); to = to - to.mean(); D = ti - to
    P_ = np.where(y)[0]; S = E @ E[P_].T
    sim = S.max(1).astype(np.float64); sc = np.isin(c, c[P_]).astype(np.float64)
    for k, ip in enumerate(P_):
        Rk = [kk for kk in range(len(P_)) if kk != k]
        if Rk: sim[ip] = S[ip, Rk].max(); sc[ip] = float(c[ip] in set(c[P_[Rk]].tolist()))
        else: sim[ip] = np.nan; sc[ip] = 0.0
    ok = ~np.isnan(sim)
    if ok.sum() < 5: continue
    sz = np.zeros(len(y)); sz[ok] = zs(sim[ok])
    Xf = np.stack([np.ones(len(y)), y.astype(float), sz, sc], 1)[ok]; Xf[:, 1:] -= Xf[:, 1:].mean(0)
    XtX += Xf.T @ Xf; Xty += Xf.T @ (D[ok] - D[ok].mean()); n += ok.sum()
    dpos.append(D[y].mean()); neg = ok & ~y
    if neg.sum() > 10:
        q = np.quantile(sim[neg], [0.1, 0.9]); dneg_near.append(D[neg & (sim >= q[1])].mean()); dneg_far.append(D[neg & (sim <= q[0])].mean())
    sd_in.append(ti.std()); sd_oof.append(to.std())
beta = np.linalg.solve(XtX + 1e-6 * np.eye(4), Xty)
print(f'(B) 30K exploration train feeds: D = in-sample - OOF teacher logit (centred per feed); teacher logit sd in-sample {np.mean(sd_in):.3f} OOF {np.mean(sd_oof):.3f}')
print(f'    within-feed OLS of D on [y, z(maxcos to OTHER clicked), samecat with OTHER clicked]: {np.round(beta[1:], 4).tolist()}  (n rows {n})')
print(f'    mean D on clicked items {np.mean(dpos):+.4f} | on non-clicked nearest decile to a click {np.mean(dneg_near):+.4f} | farthest decile {np.mean(dneg_far):+.4f}')
json.dump(dict(bfit=bfit.tolist(), beta_B=beta.tolist(), loo_ndcg=(float(n0.mean()), float(n1.mean()))), open(ROOT / 'nn_x/w12/recheck/long_bets/out/p2_coclick.json', 'w'))
