"""Lever response by positive class (exact per-positive loss change), lambda mass by negative rank, private-mix lever deltas with SE."""
from lib import *
sys.path.insert(0, str(ROOT / 'src'))
from pmix import weights as pmix_weights
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
npos = V.npos
idcg_tab = np.array([np.sum(1 / np.log2(np.arange(int(k)) + 2)) for k in range(0, 400)])
ID = idcg_tab[npos.astype(int)]
pi_all = np.flatnonzero(y > 0)


def ppl(z):
    """per-positive loss aligned to pi_all."""
    r = V.prank(z)
    pi = pi_all
    o = np.lexsort((r[pi], fid[pi])); pis = pi[o]; fp = fid[pis]
    j = np.arange(len(pis)) - np.searchsorted(fp, fp)
    loss = np.empty(len(pi)); loss[o] = (1 / np.log2(j + 2) - 1 / np.log2(r[pis] + 2)) / ID[fp]
    return loss, r


fp = fid[pi_all]; kp = keep[fp]
zg = S['G3'].astype(np.float64)
uk, inv = np.unique(ck, return_inverse=True)
lse = np.log(np.bincount(inv, np.exp(zg))); kf = uk // 64
o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
pcr = cr[inv][pi_all]
nov = np.select([minh[pi_all] < 8, minh[pi_all] < 32], ['recent', 'mid'], 'novel')
cls = np.char.add(np.char.add(np.where(cold[pi_all], 'cold', 'warm'), '|'), nov.astype(str))
crk = np.where(pcr == 0, 'G3top', 'notTop')
LEV = {'G3 vs nf2_oof': ('G3', 'nf2_oof'), 'FT half (G3 vs G3_to)': ('G3', 'G3_to'), 'm02 (F7x_to vs F7b_to)': ('F7x_to', 'F7b_to'),
       'a4u (a4u_to vs F7s12_to)': ('a4u_to', 'F7_s12_to'), 'NEW (N1 vs G3)': ('N1', 'G3'), 'GRU fam (G3 vs M2)': ('G3', 'M2'),
       'cset (cset_to vs F7s12_to)': ('cset_to', 'F7_s12_to')}
L = {}
for k in set(sum([list(v) for v in LEV.values()], [])):
    L[k] = ppl(S[k].astype(np.float64))[0]
nf = keep.sum()
print('=== Lever response by positive class: change in summed per-positive loss (NDCG points per feed; + = lever helps) ===')
classes = ['warm|recent', 'warm|mid', 'warm|novel', 'cold|recent', 'cold|mid', 'cold|novel']
print(f'{"lever":28s} {"total":>8s} ' + ' '.join(f'{c:>12s}' for c in classes) + f' {"G3top":>8s} {"notTop":>8s}')
print(f'{"(positive share)":28s} {"":8s} ' + ' '.join(f'{np.mean(cls[kp] == c):12.3f}' for c in classes) + f' {np.mean(crk[kp] == "G3top"):8.3f} {np.mean(crk[kp] == "notTop"):8.3f}')
for name, (a, b) in LEV.items():
    d = L[b] - L[a]  # loss reduction
    tot = d[kp].sum() / nf
    print(f'{name:28s} {tot:+8.5f} ' + ' '.join(f'{d[kp & (cls == c)].sum()/nf:+12.5f}' for c in classes) +
          f' {d[kp & (crk == "G3top")].sum()/nf:+8.5f} {d[kp & (crk == "notTop")].sum()/nf:+8.5f}')
# SE by feed-level aggregation of the class contribution
print('\nfeed-clustered SE of the class contributions (same order):')
for name, (a, b) in LEV.items():
    d = L[b] - L[a]
    out = []
    for c in classes + ['G3top', 'notTop']:
        m = kp & ((cls == c) if c in classes else (crk == c))
        fsum = np.bincount(fp[m], d[m], n)[keep]
        out.append(f'{se(fsum):.5f}')
    print(f'{name:28s} ' + ' '.join(f'{x:>12s}' for x in out))

# ---- lambda mass by negative rank bucket ----
@nb.njit
def mass_by_negrank(off, M, keep, y, rank, ck, ID):
    W = np.zeros((5, 2))
    for f in range(len(M)):
        if not keep[f]:
            continue
        o, m = off[f], M[f]
        for i in range(o, o + m):
            if y[i] <= 0:
                continue
            for j in range(o, o + m):
                if y[j] > 0 or rank[j] >= rank[i]:
                    continue
                rj = rank[j]
                b = 0 if rj < 3 else (1 if rj < 10 else (2 if rj < 30 else (3 if rj < 100 else 4)))
                s = 1 if ck[i] == ck[j] else 0
                W[b, s] += (1.0 / np.log2(rj + 2.0) - 1.0 / np.log2(rank[i] + 2.0)) / ID[f]
    return W

r = V.prank(zg)
Wn = mass_by_negrank(off, M, keep, y, r, ck, ID)
print('\n=== lambda (swap) mass of misordered pairs by the NEGATIVE\'s rank (G3) ===')
for b, nm in enumerate(['neg rank 0-2', '3-9', '10-29', '30-99', '100+']):
    print(f'  {nm:12s} share {Wn[b].sum()/Wn.sum():.4f}  (same-cat {Wn[b,1]/Wn.sum():.4f}, cross-cat {Wn[b,0]/Wn.sum():.4f})')

# ---- private-mix lever deltas with weighted SE ----
W = pmix_weights(); wp = W['private'][keep]
print('\n=== feed-level lever deltas: val vs private-mix (pmix weights; report only) ===')
for name, (a, b) in LEV.items():
    d = (g('ndcg_' + a) - g('ndcg_' + b))[keep]
    m = np.average(d, weights=wp); neff = wp.sum() ** 2 / (wp ** 2).sum(); s_ = np.sqrt(np.average((d - m) ** 2, weights=wp) / neff)
    print(f'  {name:28s} val {d.mean():+.5f} ± {se(d):.5f}   priv-mix {m:+.5f} ± {s_:.5f}')
