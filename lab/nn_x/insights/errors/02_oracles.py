"""Headroom oracles on G3 (val, leak-free feeds unless noted)."""
from lib import *
import numba as nb

V = Val('val')
y, M, off, fid, n = V.y, V.M, V.off, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet')
g = lambda c: T[c].to_numpy()
keep = ~g('leak')
S = dict(np.load(OUT / 'out/cand_scores.npz'))
LV = dict(np.load(OUT / 'out/feed_label_vars.npz', allow_pickle=True))
cat = V.cat
ck = fid * 64 + cat
poscat = np.isin(ck, np.unique(ck[y > 0]))
minh = np.load(OUT / 'out/cand_cat_minh.npy')
cold = np.load(OUT / 'out/cold_luo.npy')


def rep(name, s, base):
    d = s - base
    print(f'{name:58s} {s[keep].mean():.5f}  gain {d[keep].mean():+.5f} ± {se(d[keep]):.5f}', flush=True)
    return s


def within_orc_blocks(z):
    """segments-lens definition: categories as blocks ordered by best model score, positives first inside each."""
    uk, inv = np.unique(ck, return_inverse=True)
    cmax = np.full(len(uk), -np.inf); np.maximum.at(cmax, inv, z)
    kf = uk // 64
    o = np.lexsort((-cmax, kf)); cr = np.empty(len(uk)); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
    zr = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return -cr[inv] * 100 + y * 10 + zr


def within_orc_reassign(z):
    """category_intent definition: keep each (feed, category) cell's multiset of scores, give positives the top values."""
    o1 = np.lexsort((-z, ck))            # cell, score desc
    o2 = np.lexsort((-z, -y, ck))        # cell, positives first, then score desc
    out = np.empty_like(z); out[o2] = z[o1]
    return out


def topk_orc(z, k):
    """perfect re-ordering of the model's top-k only (positives inside top-k moved to its top)."""
    r = V.prank(z)
    return np.where(r < k, 1e3 + y * 10 + z / 1e3, z)


base_all = {}
for key in ['G3', 'N1', 'nf2_oof']:
    z = S[key].astype(np.float64)
    base = V.ndcg(z); base_all[key] = base
    print(f'\n=== {key}: base {base[keep].mean():.5f} (leak-free) ===')
    rep('cat-orc (positive categories first, model order inside)', V.ndcg(z + 1e3 * poscat), base)
    rep('within-orc blocks (segments-lens def)', V.ndcg(within_orc_blocks(z)), base)
    rep('within-orc reassign (category_intent def)', V.ndcg(within_orc_reassign(z)), base)
    for k in [3, 10, 30]:
        rep(f'top-{k} re-order oracle', V.ndcg(topk_orc(z, k)), base)
    # exploration-intent oracle: in feeds whose positives are ALL in novel categories (not in last 32 events), boost novel candidates
    novc = minh >= 32
    expl_feed = np.bincount(fid, (y > 0) & ~novc, n) == 0
    best = None
    for dlt in [0.25, 0.5, 1.0, 2.0]:
        s = V.ndcg(z + dlt * (novc & expl_feed[fid]))
        if best is None or s[keep].mean() > best[1][keep].mean(): best = (dlt, s)
    rep(f'exploration-intent oracle (share {expl_feed[keep].mean():.3f}; boost novel by {best[0]})', best[1], base)
    # cold-intent oracle: feeds whose positives are all cold (luo): boost cold candidates by delta
    cf = g('coldpos')
    best = None
    for dlt in [0.25, 0.5, 1.0, 2.0]:
        s = V.ndcg(z + dlt * (cold & cf[fid]))
        if best is None or s[keep].mean() > best[1][keep].mean(): best = (dlt, s)
    rep(f'cold-intent oracle (share {cf[keep].mean():.3f}; boost cold by {best[0]})', best[1], base)
    wf = g('warmpos')
    best = None
    for dlt in [0.25, 0.5, 1.0, 2.0]:
        s = V.ndcg(z - dlt * (cold & wf[fid]))
        if best is None or s[keep].mean() > best[1][keep].mean(): best = (dlt, s)
    rep(f'warm-intent oracle (share {wf[keep].mean():.3f}; demote cold by {best[0]})', best[1], base)
    both = V.ndcg(z + 1.0 * (cold & cf[fid]) - 1.0 * (cold & wf[fid]))
    rep('cold+warm intent oracle (delta 1.0 both)', both, base)

# ---- per-positive "rank halving" marginal value, by current rank bucket (G3) ----
print('\n=== Marginal NDCG value of moving positives up (G3; each bucket alone; rank r -> floor(r/2)) ===')
z = S['G3'].astype(np.float64)
r = V.prank(z)
npos = np.bincount(fid, y, n)
# ideal DCG
idcg = np.array([np.sum(1 / np.log2(np.arange(int(k)) + 2)) for k in range(0, 400)])
ID = idcg[npos.astype(int)]
pos_i = np.flatnonzero(y > 0)
rp = r[pos_i]; fp = fid[pos_i]; Mp = M[fp]
gain_half = (1 / np.log2(rp // 2 + 2) - 1 / np.log2(rp + 2)) / ID[fp]
gain_top = (1 / np.log2(0 + 2) - 1 / np.log2(rp + 2)) / ID[fp]
rel = rp / np.maximum(Mp - 1, 1)
bk = np.select([rp == 0, rp <= 2, rp <= 9, rp <= 29, rel <= 0.5], ['r0', 'r1-2', 'r3-9', 'r10-29', 'r30..M/2'], 'r>M/2')
kp = keep[fp]
print(f'{"bucket":10s} {"pos share":>9s} {"halve-rank gain":>16s} {"to-top gain (upper)":>20s}')
for b in ['r0', 'r1-2', 'r3-9', 'r10-29', 'r30..M/2', 'r>M/2']:
    m = (bk == b) & kp
    print(f'{b:10s} {m.sum()/kp.sum():9.4f} {gain_half[m].sum()/keep.sum():16.5f} {gain_top[m].sum()/keep.sum():20.5f}')
print('(sums divided by number of feeds = NDCG points; halving ignores collisions between positives)')
# what a 1-rank improvement of every positive is worth by bucket (derivative)
g1 = (1 / np.log2(np.maximum(rp - 1, 0) + 2) - 1 / np.log2(rp + 2)) / ID[fp]
print('gain from moving every positive up by ONE rank:', f'{g1[kp].sum()/keep.sum():.5f}', ' by bucket:',
      ' '.join(f'{b} {g1[(bk == b) & kp].sum()/keep.sum():.5f}' for b in ['r1-2', 'r3-9', 'r10-29', 'r30..M/2', 'r>M/2']))
# fraction of pos ranked > M/2 vs old
for key in ['G3', 'N1', 'nf2_oof', 'nf3_oof']:
    rr = V.prank(S[key].astype(np.float64))[pos_i]
    print(f'{key:8s}: positives at rank > M/2 (MP def, rank >= M/2): {np.mean((rr >= Mp / 2)[kp]):.4f}; top-3 {np.mean((rr < 3)[kp]):.4f}; mean pct rank {np.mean((rr / np.maximum(Mp - 1, 1))[kp]):.4f}')

# ---- best-member-per-feed oracle and a noise calibration ----
print('\n=== Best-member-per-feed oracle ===')
D = np.load(OUT / 'out/member_ndcg.npz')
mem, MN, MNto, nG3 = D['members'], D['final'], D['to'], int(D['G3n'])
for name, rows in [('G3 members (20)', np.arange(nG3)), ('G3+NEW members (30)', np.arange(len(mem))),
                   ('F7x only (7 seeds)', np.array([i for i, m in enumerate(mem) if m.startswith('r6_cd_m02_r')])),
                   ('7 across families (F10 s501-503 + F7x r101-102 + F7x70 r101-102)', np.array([i for i, m in enumerate(mem) if m in
                     ['v23_gru_pre2s_s501', 'v23_gru_pre2s_s502', 'v23_gru_pre2s_s503', 'r6_cd_m02_r101', 'r6_cd_m02_r102', 'r6_cd_m02_p70k_r101', 'r6_cd_m02_p70k_r102']]))]:
    X = MN[rows][:, keep]
    print(f'{name:70s} mean member {X.mean():.5f}  best-member oracle {X.max(0).mean():.5f} (+{X.max(0).mean() - X.mean():.5f} over mean member; '
          f'+{X.max(0).mean() - base_all["G3"][keep].mean():.5f} over G3)')
# family-pool oracle
fams = ['F10', 'F7x', 'F7x70']
X = np.stack([g('ndcg_' + f) for f in fams])[:, keep]
print(f'best-family-pool oracle (F10/F7x/F7x70): {X.max(0).mean():.5f} (+{X.max(0).mean() - base_all["G3"][keep].mean():.5f} over G3); FT/TO half oracle '
      f'{np.maximum(g("ndcg_G3_ft"), g("ndcg_G3_to"))[keep].mean():.5f}')

# synthetic-noise calibration: k pseudo-members = G3 z + sigma*N(0,1), sigma matched to the mean within-feed Spearman(member, G3)
@nb.njit(parallel=True)
def feed_spear(off, M, a, b):
    out = np.empty(len(M))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        ra = np.argsort(np.argsort(a[o:o + m])).astype(np.float64); rb = np.argsort(np.argsort(b[o:o + m])).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        out[f] = (ra * rb).sum() / den if den > 0 else 1.0
    return out

zg = S['G3'].astype(np.float64)
P_ = ROOT / 'preds'
sp = []
for m in mem[:nG3]:
    zm = 0.5 * V.z(np.load(P_ / f'nn_{m}_n2cpv3_val_oof.npy')) + 0.5 * V.z(np.load(P_ / f'nn_{m}_trainonly_val.npy'))
    sp.append(feed_spear(off, M, zm, zg).mean())
sp_m = np.mean(sp)
print(f'mean within-feed Spearman(member final, G3) = {sp_m:.4f} (range {min(sp):.4f}-{max(sp):.4f})')
rng = np.random.default_rng(1)
best_sig = None
for sig in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    s_ = feed_spear(off, M, zg + sig * rng.standard_normal(len(zg)), zg).mean()
    if best_sig is None or abs(s_ - sp_m) < abs(best_sig[1] - sp_m): best_sig = (sig, s_)
sig = best_sig[0]
Xs = np.stack([V.ndcg(zg + sig * rng.standard_normal(len(zg))) for _ in range(nG3)])[:, keep]
print(f'synthetic: sigma {sig} (Spearman {best_sig[1]:.4f}); 20 noisy copies of G3: mean member {Xs.mean():.5f}, best-of-20 oracle {Xs.max(0).mean():.5f} '
      f'(+{Xs.max(0).mean() - Xs.mean():.5f} over mean member)')
X = MN[:nG3][:, keep]
print(f'real G3 members: best-of-20 over mean member +{X.max(0).mean() - X.mean():.5f}  => real/synthetic ratio {(X.max(0).mean() - X.mean())/(Xs.max(0).mean() - Xs.mean()):.2f}')
