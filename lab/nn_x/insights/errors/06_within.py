"""Within-category errors of G3: what the same-category negatives that beat a positive look like; near-duplicate structure;
calibration (act/exp) of the TO and FT halves by positive class (residual position/recency bias check)."""
from lib import *
import json
import numba as nb

V = Val('val')
y, M, off, fid, n = V.y, V.M, V.off, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet')
g = lambda c: T[c].to_numpy()
keep = ~g('leak')
S = dict(np.load(OUT / 'out/cand_scores.npz'))
cat = V.cat
ck = fid * 64 + cat
minh = np.load(OUT / 'out/cand_cat_minh.npy')
cold = np.load(OUT / 'out/cold_luo.npy')
age = np.load(ROOT / 'eda/errors/synthesis/derived/val/ag_age.npy').astype(np.float64)
z = S['G3'].astype(np.float64)


def ld(name, src='feats'):
    p = (FEATS / 'val' / f'{name}.npy') if src == 'feats' else (FC / 'val' / f'{name}.npy')
    return np.nan_to_num(np.load(p).astype(np.float64), nan=0.0)


meta = json.load(open(ROOT / 'data/competition/val/item_embeddings/data.meta'))
E = np.memmap(ROOT / 'data/competition/val/item_embeddings/data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))
ei = np.load(FLAT / 'val/cand_emb_idx.npy').astype(np.int64)
miss = ei < 0
X = np.asarray(E[np.where(miss, 0, ei)], dtype=np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9; X[miss] = 0
log('emb loaded', X.shape, 'missing', miss.mean())

feat = {'cold': cold.astype(float), 'age_days': age, 'log1p_luo_pop': np.log1p(ld('it_hist_all_luo', 'fc')),
        'log1p_train_imp': np.log1p(ld('it_cand_cnt_train', 'fc')), 'f1': V.rk(ld('f1')), 'f2': V.rk(ld('f2')), 'f4': V.rk(ld('f4')), 'f5': V.rk(ld('f5')),
        'maxsim32': V.rk(ld('maxsim32')), 'pre2_fe': V.rk(ld('pre2_fe')), 'seq_cv_invd20': V.rk(ld('seq_cv_invd20')), 'ctx_nd_max': V.rk(ld('ctx_nd_max')),
        'cos_u': V.rk(ld('cos_u')), 'G3_to': V.rk(S['G3_to']), 'G3_ft': V.rk(S['G3_ft'])}
names = list(feat); FM = np.column_stack([feat[k] for k in names])


@nb.njit
def beaters(off, M, keep, y, z, ck, X, FM, miss):
    """for each positive: same-category negatives scored above it. Accumulate (pos - beater) feature diffs, cos(pos, beater),
    and the same for same-category negatives scored BELOW it (control)."""
    K = FM.shape[1]
    dA = np.zeros(K); nA = 0.0; dB = np.zeros(K); nB = 0.0
    cosA = 0.0; cosB = 0.0; ndA = 0.0; ndB = 0.0; ncA = 0.0; ncB = 0.0
    hist = np.zeros(10)  # cos histogram of beaters
    npos_beaten = 0.0; npos_tot = 0.0; nbeat_tot = 0.0
    for f in range(len(M)):
        if not keep[f]:
            continue
        o, m = off[f], M[f]
        for i in range(o, o + m):
            if y[i] <= 0:
                continue
            npos_tot += 1; nb_ = 0
            for j in range(o, o + m):
                if y[j] > 0 or ck[j] != ck[i]:
                    continue
                c = 0.0
                ok = (not miss[i]) and (not miss[j])
                if ok:
                    for t in range(X.shape[1]):
                        c += X[i, t] * X[j, t]
                if z[j] > z[i]:
                    nb_ += 1
                    for k in range(K):
                        dA[k] += FM[i, k] - FM[j, k]
                    nA += 1
                    if ok:
                        cosA += c; ncA += 1
                        if c > 0.9: ndA += 1
                        b = int(min(max(c, 0.0), 0.9999) * 10); hist[b] += 1
                else:
                    for k in range(K):
                        dB[k] += FM[i, k] - FM[j, k]
                    nB += 1
                    if ok:
                        cosB += c; ncB += 1
                        if c > 0.9: ndB += 1
            if nb_ > 0:
                npos_beaten += 1
            nbeat_tot += nb_
    return dA / nA, dB / nB, cosA / ncA, cosB / ncB, ndA / ncA, ndB / ncB, hist / hist.sum(), npos_beaten / npos_tot, nbeat_tot / npos_tot


dA, dB, cA, cB, ndA, ndB, hist, pb, nbt = beaters(off, M, keep, y, z, ck, X, FM, miss)
print('=== Positives vs same-category negatives scored ABOVE them by G3 ("beaters") vs BELOW them (control) ===')
print(f'share of positives with >= 1 same-category beater: {pb:.4f}; mean beaters per positive {nbt:.2f}')
print(f'mean cos(positive, beater) {cA:.4f} vs cos(positive, same-cat neg below) {cB:.4f}; share with cos > 0.9: {ndA:.4f} vs {ndB:.4f}')
print('cos histogram of beaters (0-.1 ... .9-1):', np.round(hist, 4))
print(f'{"feature":18s} {"pos - beater":>13s} {"pos - below":>12s}   (feature units: within-feed pct rank unless named)')
for k, a, b in zip(names, dA, dB):
    print(f'{k:18s} {a:+13.4f} {b:+12.4f}')

# ---- near-duplicate oracle (lift positive to the best score in its near-dup group, same category, cos > thr) ----
@nb.njit
def nd_oracle(off, M, y, z, ck, X, miss, thr):
    out = z.copy()
    for f in range(len(M)):
        o, m = off[f], M[f]
        for i in range(o, o + m):
            if y[i] <= 0 or miss[i]:
                continue
            best = z[i]
            for j in range(o, o + m):
                if j == i or ck[j] != ck[i] or miss[j]:
                    continue
                c = 0.0
                for t in range(X.shape[1]):
                    c += X[i, t] * X[j, t]
                if c > thr and z[j] > best:
                    best = z[j]
            out[i] = best + 1e-6
    return out

base = V.ndcg(z)
for thr in [0.9, 0.95]:
    s = V.ndcg(nd_oracle(off, M, y, z, ck, X, miss, thr))
    d = s - base
    print(f'near-dup oracle cos>{thr}: +{d[keep].mean():.5f} ± {se(d[keep]):.5f}')

# ---- calibration of TO vs FT halves by positive class ----
print('\n=== act/exp (within-feed softmax of z, temperature fitted per score) by candidate class ===')
cls = np.select([(minh < 8) & ~cold, (minh < 8) & cold, (minh >= 8) & (minh < 32), (minh >= 32) & (minh < 10 ** 6)], ['last8|warm', 'last8|cold', 'h8-31', 'h>=32'], 'never')
cls2 = np.where(cold, 'cold', 'warm')
for key in ['G3', 'G3_to', 'G3_ft', 'F7x_to', 'F7b_to', 'F10_to', 'nf2_tr', 'nf3_tr']:
    zz = S[key].astype(np.float64)
    best = None
    for tmp in [0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0]:
        e = np.exp(zz * tmp); p = e / np.bincount(fid, e, n)[fid] * V.npos[fid]
        ll = np.sum((y * np.log(p / V.npos[fid] + 1e-12))[keep[fid]])
        if best is None or ll > best[1]: best = (tmp, ll, p)
    p = best[2]
    out = []
    for c in ['last8|warm', 'last8|cold', 'h8-31', 'h>=32', 'never']:
        m = (cls == c) & keep[fid]
        a = y[m].sum(); ex = p[m].sum()
        # SE by feed-cluster bootstrap approx: sqrt(sum var)/ex using Poisson approx
        out.append(f'{c} {a/ex:.3f}±{np.sqrt(ex)/ex:.3f}')
    for c in ['cold', 'warm']:
        m = (cls2 == c) & keep[fid]
        out.append(f'{c} {y[m].sum()/p[m].sum():.3f}')
    print(f'{key:8s} T={best[0]:.1f}  ' + '  '.join(out))
