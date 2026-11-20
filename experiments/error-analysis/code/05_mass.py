# Third pass (2026-11-12), on the 20-member pool that scored 0.392820 on the public board (0.40032 on val).
# Exact split of the loss over positives: 1 - NDCG of a feed is written as a sum of one term per positive, and
# the terms are grouped by cold or warm item, by how recently the user touched the category, by the rank the
# pool gives to the positive's category, and by the rank of the positive inside its category. Each table is
# repeated with val re-weighted to the private feed mix.
# Run by lab/nn_x/insights/errors/run_all.sh: python 05_mass.py > logs/05_mass.log (CPU, 12 threads); its
# loaders come from lab/nn_x/insights/errors/lib.py (not copied). Output: ../results/pass3_loss_mass.txt.
# Result: 53% of the loss is a positive whose category is not the pool's first choice; 37% is a positive in
# the first-choice category with 3 or more items of that category above it.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/insights/errors/05_mass.py
"""Where the remaining NDCG mass is: exact per-positive loss decomposition, regime x coldness map, category-level NDCG,
private-mix reweighting (report only), standardized worst-10% feeds."""
from lib import *
sys.path.insert(0, str(ROOT / 'src'))
from pmix import weights as pmix_weights

V = Val('val')
y, M, off, fid, n = V.y, V.M, V.off, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet')
g = lambda c: T[c].to_numpy()
keep = ~g('leak')
S = dict(np.load(OUT / 'out/cand_scores.npz'))
LV = dict(np.load(OUT / 'out/feed_label_vars.npz', allow_pickle=True))
cat = V.cat
ck = fid * 64 + cat
minh = np.load(OUT / 'out/cand_cat_minh.npy')
cold = np.load(OUT / 'out/cold_luo.npy')
W = pmix_weights(); wpriv = W['private']; wpub = W['public']
npos = V.npos
idcg_tab = np.array([np.sum(1 / np.log2(np.arange(int(k)) + 2)) for k in range(0, 400)])
ID = idcg_tab[npos.astype(int)]


def per_positive_loss(z):
    """exact additive split of (1 - NDCG_f) over the feed's positives: positive with the j-th best rank r_j owns
    (1/log2(j+2) - 1/log2(r_j+2)) / IDCG_f."""
    r = V.prank(z)
    pi = np.flatnonzero(y > 0)
    o = np.lexsort((r[pi], fid[pi])); pi = pi[o]
    fp = fid[pi]
    j = np.arange(len(pi)) - np.searchsorted(fp, fp)
    loss = (1 / np.log2(j + 2) - 1 / np.log2(r[pi] + 2)) / ID[fp]
    return pi, loss, r


z = S['G3'].astype(np.float64)
pi, pl_, r = per_positive_loss(z)
fp = fid[pi]
nd = g('ndcg_G3')
assert np.allclose(np.bincount(fp, pl_, n)[npos > 0], (1 - nd)[npos > 0], atol=1e-9)
kp = keep[fp]
tot = pl_[kp].sum()
totw = (pl_ * wpriv[fp])[kp].sum()
print(f'Total loss (leak-free) per feed {tot/keep.sum():.4f}; exact per-positive decomposition verified.')

# positive classes
nov = np.select([minh[pi] < 8, minh[pi] < 32, minh[pi] < 10 ** 6], ['last8', 'h8-31', 'h>=32'], 'never')
cl = np.where(cold[pi], 'cold', 'warm')
# G3 LSE category rank of the positive's category
uk, inv = np.unique(ck, return_inverse=True)
lse = np.log(np.bincount(inv, np.exp(z)))
kf = uk // 64
o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
pcr = cr[inv][pi]
crk = np.select([pcr == 0, pcr == 1, pcr <= 3], ['cat#1', 'cat#2', 'cat#3-4'], 'cat#5+')
# within-category rank of the positive under G3 (among its category's candidates)
o2 = np.lexsort((y, -z, ck)); wr = np.empty(len(z), np.int64); wr[o2] = np.arange(len(z)) - np.searchsorted(ck[o2], ck[o2])
wrk = np.select([wr[pi] == 0, wr[pi] <= 2, wr[pi] <= 9], ['w0', 'w1-2', 'w3-9'], 'w10+')


def table(lab, title, order=None):
    print(f'\n--- loss mass by {title} (positives; val leak-free | private-mix reweighted) ---')
    print(f'{"class":24s} {"pos share":>9s} {"loss share":>10s} {"loss/pos (x feeds)":>18s} {"priv pos share":>14s} {"priv loss share":>15s}')
    for c in (order or sorted(np.unique(lab))):
        m = (lab == c) & kp
        print(f'{c:24s} {m.sum()/kp.sum():9.4f} {pl_[m].sum()/tot:10.4f} {pl_[m].mean():18.4f} {(wpriv[fp]*m).sum()/(wpriv[fp]*kp).sum():14.4f} '
              f'{(pl_*wpriv[fp])[m].sum()/totw:15.4f}')


table(cl, 'coldness (LUO)')
table(nov, 'category novelty', ['last8', 'h8-31', 'h>=32', 'never'])
table(crk, 'G3 category rank of the positive (LSE)', ['cat#1', 'cat#2', 'cat#3-4', 'cat#5+'])
table(np.char.add(np.char.add(crk.astype(str), '|'), wrk.astype(str)), 'category rank x within-category rank')
table(np.char.add(np.char.add(nov.astype(str), '|'), cl.astype(str)), 'novelty x coldness')
# global rank bucket
rel = r[pi] / np.maximum(M[fp] - 1, 1)
rb = np.select([r[pi] == 0, r[pi] <= 2, r[pi] <= 9, r[pi] <= 29, rel < 0.5], ['r0', 'r1-2', 'r3-9', 'r10-29', 'r30..<M/2'], 'r>=M/2')
table(rb, 'positive rank bucket', ['r0', 'r1-2', 'r3-9', 'r10-29', 'r30..<M/2', 'r>=M/2'])

# ---- category-level NDCG (LSE), top-1 category hit ----
print('\n=== Category-level ranking (LSE of per-candidate z within (feed, category)); feeds with >= 2 categories ===')
ncat = np.bincount(kf, minlength=n)
m2 = (ncat >= 2) & keep
poscat_k = np.isin(uk, np.unique(ck[y > 0])).astype(float)
cm = np.repeat(np.arange(n), 1)  # placeholder
Mk = np.bincount(kf, minlength=n); offk = np.r_[0, np.cumsum(Mk)[:-1]]
res = {}
for key in ['G3', 'N1', 'G3_to', 'G3_ft', 'nf2_oof', 'nf2_tr', 'nf3_oof', 'nf3_tr']:
    zz = S[key].astype(np.float64)
    l = np.log(np.bincount(inv, np.exp(zz)))
    cn = ndcg_per_feed(l, poscat_k, offk, Mk)
    oo = np.lexsort((-l, kf)); first = np.r_[True, kf[oo][1:] != kf[oo][:-1]]
    top1 = np.zeros(n); top1[kf[oo][first]] = poscat_k[oo][first]
    res[key] = cn
    print(f'{key:8s} cat NDCG {cn[m2].mean():.4f}  top-1 category hit {top1[m2].mean():.4f}')
for a, b in [('G3', 'nf2_oof'), ('G3', 'nf3_oof'), ('G3', 'G3_to'), ('G3_to', 'nf3_tr')]:
    d = res[a] - res[b]
    print(f'  cat NDCG {a} - {b}: {d[m2].mean():+.5f} ± {se(d[m2]):.5f}')

# ---- regime x coldness map (segments-lens §2 analogue) ----
print('\n=== Error map: intent regime (positive in G3-top category / f3-top) x positive coldness (LUO) ===')
poscat = np.isin(ck, np.unique(ck[y > 0]))
catorc = V.ndcg(z + 1e3 * poscat) - nd
o1 = np.lexsort((-z, ck)); o3 = np.lexsort((-z, -y, ck)); zz = np.empty_like(z); zz[o3] = z[o1]
winorc = V.ndcg(zz) - nd
pc = np.select([g('coldpos'), g('warmpos')], ['cold', 'warm'], 'mixed')
D = {'G3-nf2oof': g('ndcg_G3') - g('ndcg_nf2_oof'), 'FT half': g('ndcg_G3') - g('ndcg_G3_to'), 'm02': g('ndcg_F7x_to') - g('ndcg_F7b_to'),
     'a4u': g('ndcg_a4u_to') - g('ndcg_F7_s12_to'), 'NEW': g('ndcg_N1') - g('ndcg_G3')}
L = (1 - nd)[keep].sum()
for regime_name, reg in [('G3-LSE top category', np.where(LV['pcr'] == 0, 'top', 'off')), ('f3 top category', np.where(g('pos_f3_catrank') == 0, 'top', 'off'))]:
    print(f'\n[{regime_name}]')
    print(f'{"cell":12s} {"share":>6s} {"G3":>7s} {"loss%":>6s} {"catorc%":>8s} {"winorc%":>8s} ' + ' '.join(f'{k:>17s}' for k in D))
    for rg in ['top', 'off']:
        for c in ['warm', 'mixed', 'cold', 'all']:
            m = (reg == rg) & keep & ((pc == c) if c != 'all' else True)
            print(f'{rg + "|" + c:12s} {m.sum()/keep.sum():6.3f} {nd[m].mean():7.4f} {100*(1-nd)[m].sum()/L:6.1f} {100*catorc[m].sum()/catorc[keep].sum():8.1f} '
                  f'{100*winorc[m].sum()/winorc[keep].sum():8.1f} ' + ' '.join(f'{d[m].mean():+.4f}({se(d[m]):.4f})' for d in D.values()))
print(f'\nheadroom totals: cat-orc {catorc[keep].mean():.4f}, within-orc (reassign) {winorc[keep].mean():.4f}')

# ---- private-mix and public-mix of label segments ----
print('\n=== Feed-level label segments: val share vs private-mix weighted share (pmix weights, report only) ===')
for nm, lab in [('pos coldness', pc), ('pos cat novelty', LV['nov']), ('pos cat G3 rank', np.select([LV['pcr'] == 0, LV['pcr'] == 1], ['#1', '#2'], '#3+'))]:
    for c in sorted(np.unique(lab)):
        m = (lab == c) & keep
        print(f'{nm:16s} {c:8s} val share {m.sum()/keep.sum():.4f}  priv-mix share {(wpriv*m).sum()/(wpriv*keep).sum():.4f}  pub-mix {(wpub*m).sum()/(wpub*keep).sum():.4f}'
              f'  G3 {nd[m].mean():.4f}')
print(f'G3 val {nd[keep].mean():.5f}; priv-mix {np.average(nd[keep], weights=wpriv[keep]):.5f}; pub-mix {np.average(nd[keep], weights=wpub[keep]):.5f} (pmix is level-miscalibrated; shares only)')
for k in D:
    d = D[k]
    print(f'  lever {k:10s}: val {d[keep].mean():+.5f}  priv-mix {np.average(d[keep], weights=wpriv[keep]):+.5f}')

# ---- standardized worst 10% ----
madj = LV['madj']
cell_M = np.digitize(M, np.quantile(M, np.linspace(0, 1, 21)[1:-1])); cell = cell_M * 4 + np.minimum(npos, 3).astype(int)
sd = np.sqrt(np.bincount(cell[keep], madj[keep] ** 2, cell.max() + 1) / np.maximum(np.bincount(cell[keep], minlength=cell.max() + 1), 1))
zres = madj / np.maximum(sd[cell], 1e-6)
w = zres <= np.quantile(zres[keep], 0.1)
print(f'\n=== Worst 10% by standardized M x npos residual: mean M {M[w & keep].mean():.1f} (all {M[keep].mean():.1f}); G3 {nd[w & keep].mean():.4f}; '
      f'mean best-positive pct rank {(g("bprank_G3")/np.maximum(M-1,1))[w & keep].mean():.3f} (all {(g("bprank_G3")/np.maximum(M-1,1))[keep].mean():.3f})')
for nm, x in [('coldpos', g('coldpos')), ('warmpos', g('warmpos')), ('pos cat novel (>=32/never)', np.isin(LV['nov'], ['h>=32', 'never'])),
              ('pos cat never in history', LV['nov'] == 'never'), ('pos cat not G3 top', LV['pcr'] > 0), ('pos cat G3 rank >= 3', LV['pcr'] >= 3),
              ('pos cat not f3 top', g('pos_f3_catrank') > 0), ('pos pop luo == 0', LV['ppop'] == 0), ('uid0', g('uid0')), ('H==256', g('H') == 256),
              ('n_train_feeds>0', g('n_train_feeds') > 0), ('npos==1', npos == 1), ('cold_share>=45%', g('cold_share_luo') >= 0.45)]:
    a = x[w & keep].mean(); b = x[keep].mean()
    print(f'  {nm:28s} worst10std {a:.4f}  all {b:.4f}  lift {a/b:.2f}')
np.save(OUT / 'out/per_positive_loss.npy', np.column_stack([pi, pl_]))
