"""How is the user embedding related to the history? + extra recency variants and simple fusions on val."""
import numpy as np
import numba as nb
from common import ndcg_per_feed
from emb_features import load_split, l2n

D = load_split('val')
M, H = D['M'], D['H']
nf = len(M)
U = D['user']; un = np.linalg.norm(U, axis=1)
print(f'val: user all-zero {np.mean(un == 0):.4f}; H==0 {np.mean(H == 0):.4f}; zero-user & H==0 {np.mean((un == 0) & (H == 0)):.4f}; '
      f'P(H==0 | zero user) {np.mean(H[un == 0] == 0):.3f}; P(zero user | H==0) {np.mean(un[H == 0] == 0):.3f}')
for s in ['train', 'test']:
    from common import feeds, arr
    f = feeds(s); Hs = f['H'].to_numpy()
    Us = arr(s, 'user_emb')
    idx = np.random.default_rng(0).choice(len(f), min(len(f), 300000), replace=False)
    z = np.abs(np.asarray(Us[np.sort(idx)])).sum(1) == 0
    Hh = Hs[np.sort(idx)]
    print(f'{s}: user all-zero {z.mean():.4f}; H==0 {np.mean(Hh == 0):.4f}; P(H==0|zero) {np.mean(Hh[z] == 0):.3f}; P(zero|H==0) {np.mean(z[Hh == 0]):.3f}')

Hn, _ = l2n(D['H_raw'])
valid = D['h_emb_idx'] >= 0
et = D['h_event_type']; es = D['h_event_source']
Un, _ = l2n(U)


@nb.njit(parallel=True)
def pooled_cos(Un, Hn, valid, hoff, H, kmax, typ_filter, et):
    nf = len(H)
    out = np.full(nf, np.nan)
    for f in nb.prange(nf):
        acc = np.zeros(Hn.shape[1], np.float32); c = 0
        for j in range(H[f]):
            o = hoff[f] + j
            if valid[o] and (typ_filter < 0 or et[o] == typ_filter):
                acc += Hn[o]; c += 1
                if c >= kmax:
                    break
        if c > 0 and (Un[f] ** 2).sum() > 0:
            out[f] = (acc * Un[f]).sum() / (np.sqrt((acc * acc).sum()) + 1e-9)
    return out


print('\ncos(user, mean of last-k valid history item vectors)   [random user-item cos baseline below]')
for k in [1, 2, 5, 10, 20, 50, 100, 256]:
    c = pooled_cos(Un, Hn, valid, D['hoff'], H, k, -1, et)
    print(f'  k={k:4d}: mean {np.nanmean(c):.4f} median {np.nanmedian(c):.4f}')
for t in range(1, 11):
    c = pooled_cos(Un, Hn, valid, D['hoff'], H, 256, t, et)
    print(f'  type {t:2d} all: mean {np.nanmean(c):.4f} (coverage {np.mean(np.isfinite(c)):.3f})')
# random baseline: user vs another feed's history
perm = np.random.default_rng(1).permutation(nf)
c = pooled_cos(Un[perm], Hn, valid, D['hoff'], H, 10, -1, et)
print(f'  shuffled-user baseline (k=10): {np.nanmean(c):.4f}')
Cn, _ = l2n(D['C_raw'])
uc = (np.repeat(Un, M, 0) * Cn).sum(1)
print(f'  mean cos(user, candidate) {uc[np.repeat(un > 0, M)].mean():.4f}')

# linear map: user ~ W [mean last5, mean last20, mean all]  (fit on half, R2 on other half)
@nb.njit(parallel=True)
def pooled_vecs(Hn, valid, hoff, H, ks):
    nf = len(H); d = Hn.shape[1]
    out = np.zeros((nf, len(ks), d), np.float32)
    for f in nb.prange(nf):
        c = 0
        for j in range(H[f]):
            o = hoff[f] + j
            if valid[o]:
                for q in range(len(ks)):
                    if c < ks[q]:
                        out[f, q] += Hn[o]
                c += 1
        for q in range(len(ks)):
            n = np.sqrt((out[f, q] ** 2).sum())
            if n > 0:
                out[f, q] /= n
    return out

ks = np.array([1, 5, 20, 256])
P = pooled_vecs(Hn, valid, D['hoff'], H, ks).reshape(nf, -1)
ok = (un > 0) & (H > 0)
X = np.c_[P[ok], np.ones(ok.sum())]; Y = Un[ok]
n = len(X); tr = np.arange(n) < n // 2
W = np.linalg.lstsq(X[tr], Y[tr], rcond=None)[0]
Yp = X[~tr] @ W
r2 = 1 - ((Y[~tr] - Yp) ** 2).sum() / ((Y[~tr] - Y[tr].mean(0)) ** 2).sum()
cs = (Yp * Y[~tr]).sum(1) / np.linalg.norm(Yp, axis=1)
print(f'\nlinear reconstruction of user vec from pooled history (last1/5/20/all): held-out R2={r2:.3f}, cos(pred,user) mean {cs.mean():.3f}')

# ---------- extra recency variants ----------
rng = np.random.default_rng(123)
fid = np.repeat(np.arange(nf), M)
tieb = rng.random(len(fid))
t = D['c_target'].astype(np.float64)


def nd(s):
    s = np.asarray(s, np.float64).copy(); s[~np.isfinite(s)] = -1e18
    order = np.lexsort((tieb, s, fid)); r = np.empty(len(s)); r[order] = np.arange(len(s))
    return ndcg_per_feed(r, t, D['coff'], M)


@nb.njit(parallel=True, fastmath=True)
def rec_sim(Cn, coff, M, Hn, valid, hoff, H, hl, center, mu):
    out = np.full(Cn.shape[0], np.nan, np.float32)
    for f in nb.prange(len(M)):
        nv = 0
        for j in range(H[f]):
            if valid[hoff[f] + j]:
                nv += 1
        if nv == 0:
            continue
        hv = np.empty((nv, Cn.shape[1]), np.float32); w = np.empty(nv, np.float32); k = 0
        for j in range(H[f]):
            if valid[hoff[f] + j]:
                hv[k] = Hn[hoff[f] + j] - mu
                w[k] = np.exp(-k / hl) if hl > 0 else 1.0
                k += 1
        for i in range(M[f]):
            c = Cn[coff[f] + i]
            num = 0.0; den = 0.0
            for j in range(nv):
                num += w[j] * (c * hv[j]).sum(); den += w[j]
            out[coff[f] + i] = num / den
    return out


mu = Hn[valid].mean(0).astype(np.float32)
base = {}
for hl in [1, 2, 3, 5, 8, 12]:
    s = rec_sim(Cn, D["coff"], M, Hn, valid, D["hoff"], H, float(hl), False, np.zeros_like(mu))
    pf = nd(s); base[hl] = s
    print(f'rec_sim hl={hl}: NDCG {pf.mean():.4f}')

# fusion: z-score within feed then sum
def zfeed(s):
    s = np.where(np.isfinite(s), s, np.nan).astype(np.float64)
    m = np.bincount(fid, np.nan_to_num(s), nf) / np.maximum(np.bincount(fid, np.isfinite(s), nf), 1)
    d = np.where(np.isfinite(s), s - m[fid], 0)
    sd = np.sqrt(np.bincount(fid, d * d, nf) / M)
    return d / np.maximum(sd[fid], 1e-9)

zu = zfeed(uc); zr = zfeed(base[3]); zf1 = zfeed(D['c_f1'])
f1pf = nd(D['c_f1'])
for nm, s in [('user_cos', zu), ('rec_hl3', zr), ('user+rec_hl3', zu + zr), ('f1', zf1), ('f1+user', zf1 + zu), ('f1+rec_hl3', zf1 + zr),
              ('f1+user+rec_hl3', zf1 + zu + zr), ('f1+0.5(user+rec)', zf1 + 0.5 * (zu + zr))]:
    pf = nd(s); d = pf - f1pf
    print(f'  {nm:20s} NDCG {pf.mean():.4f}  vs f1 {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(nf):.4f})')
