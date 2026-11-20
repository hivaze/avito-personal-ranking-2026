"""NDCG-weighted pairwise error mass of G3 (vs nf2_oof / nf3_oof): which (positive, negative) pair types carry the loss."""
from lib import *
import numba as nb

V = Val('val')
y, M, off, fid, n = V.y, V.M, V.off, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet')
keep = ~T['leak'].to_numpy()
S = dict(np.load(OUT / 'out/cand_scores.npz'))
cat = V.cat
minh = np.load(OUT / 'out/cand_cat_minh.npy')
cold_luo = np.load(OUT / 'out/cold_luo.npy')
cold_tr = np.load(ROOT / 'eda/errors/synthesis/derived/val/cold_item.npy').astype(bool)
npos = np.bincount(fid, y, n)
idcg_tab = np.array([np.sum(1 / np.log2(np.arange(int(k)) + 2)) for k in range(0, 400)])
ID = idcg_tab[npos.astype(int)]


@nb.njit(parallel=False)
def pair_mass(off, M, keep, y, rank, cp, cn, cat, ID, K):
    """W[a,b,s]: lambda-weighted mass of misordered pairs (neg above pos), C[a,b,s]: pair count, E[a,b,s]: misordered count.
    a = class of positive, b = class of negative, s = same category."""
    W = np.zeros((K, K, 2)); C = np.zeros((K, K, 2)); E = np.zeros((K, K, 2))
    for f in range(len(M)):
        if not keep[f]:
            continue
        o, m = off[f], M[f]
        for i in range(o, o + m):
            if y[i] <= 0:
                continue
            rp = rank[i]
            for j in range(o, o + m):
                if y[j] > 0:
                    continue
                s = 1 if cat[i] == cat[j] else 0
                a = cp[i]; b = cn[j]
                C[a, b, s] += 1
                if rank[j] < rp:
                    E[a, b, s] += 1
                    W[a, b, s] += (1.0 / np.log2(rank[j] + 2.0) - 1.0 / np.log2(rp + 2.0)) / ID[f]
    return W, C, E


def run(key, cls, names, label):
    r = V.prank(S[key].astype(np.float64))
    K = len(names)
    W, C, E = pair_mass(off, M, keep, y, r, cls, cls, cat, ID, K)
    tot = W.sum()
    print(f'\n[{key}] classes = {label}; total lambda mass {tot/keep.sum():.4f} per feed; overall pair AUC {1 - E.sum()/C.sum():.4f}')
    print(f'{"pos class":>10s} {"neg class":>10s} {"samecat":>7s} {"pair share":>10s} {"AUC":>7s} {"mass share":>10s} {"mass/pair (x1e4)":>16s}')
    for a in range(K):
        for b in range(K):
            for s in (1, 0):
                if C[a, b, s] < 100: continue
                print(f'{names[a]:>10s} {names[b]:>10s} {s:7d} {C[a,b,s]/C.sum():10.4f} {1 - E[a,b,s]/C[a,b,s]:7.4f} {W[a,b,s]/tot:10.4f} {1e4*W[a,b,s]/C[a,b,s]/keep.sum()*keep.sum()/1:16.3f}')
    # marginals
    for s in (1, 0):
        print(f'  samecat={s}: pair share {C[:,:,s].sum()/C.sum():.4f}  AUC {1 - E[:,:,s].sum()/C[:,:,s].sum():.4f}  mass share {W[:,:,s].sum()/tot:.4f}')
    for a in range(K):
        print(f'  pos class {names[a]:>8s}: pair share {C[a].sum()/C.sum():.4f}  AUC {1 - E[a].sum()/C[a].sum():.4f}  mass share {W[a].sum()/tot:.4f}')
    for b in range(K):
        print(f'  neg class {names[b]:>8s}: pair share {C[:, b].sum()/C.sum():.4f}  AUC {1 - E[:, b].sum()/C[:, b].sum():.4f}  mass share {W[:, b].sum()/tot:.4f}')
    return W, C, E


cls_cold_tr = cold_tr.astype(np.int64)  # 0 warm, 1 cold (train-bitmap definition, as in err_* reports)
cls_cold_luo = cold_luo.astype(np.int64)
nov = np.select([minh < 8, minh < 32, minh < 10 ** 6], [0, 1, 2], 3).astype(np.int64)
for key in ['G3', 'nf2_oof', 'nf3_oof']:
    run(key, cls_cold_tr, ['warm', 'cold'], 'coldness (train bitmap, as in err_* reports)')
for key in ['G3']:
    run(key, cls_cold_luo, ['warm', 'cold'], 'coldness (LUO history count == 0)')
    run(key, nov, ['last8', 'h8-31', 'h>=32', 'never'], 'category novelty (min hpos of the category in history)')
# negative is in a positive category? (only cross-category pairs can have neg in another positive category)
ck = fid * 64 + cat
negposcat = np.isin(ck, np.unique(ck[y > 0])).astype(np.int64)
run('G3', negposcat, ['otherCat', 'posCat'], 'candidate in a positive category (pos class always posCat)')
