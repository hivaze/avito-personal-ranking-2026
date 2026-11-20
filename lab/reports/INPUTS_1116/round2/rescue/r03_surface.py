"""RESCUE r03: P4 surface split in the view D5 (PREDECLARED.txt).
n_c(item) = number of distinct OTHER users with a type-1 history event of source c (c = 1, 2) on the item in the snapshot of a TRAIN feed of the window
(construction of nn_x/codes/exploit/02_s1s2contrast/run.py, variant V-lf: guest snapshot = own key, own user removed), X3_5 = log((n_s1 + 1) / (n_s2 + 1)).
Label-free: no label array is opened. Sources are history snapshots of train feeds of the window (holdout feeds included: they are other users'
earlier snapshots); the window ends before the cut, so no later snapshot of anyone can enter.
CPU only. Output: out/arr/<set>__{ns1_5,ns2_5,X3_5}.npy, log out/r03_surface.log."""
from rs_common import *
P = Log(OUT / 'r03_surface.log')
V = Views(P)
a = V.A['train']
ci = {s: np.load(FLAT / s / 'cand_item_id.npy', mmap_mode='r') for s in SPLITS}
R = {q: V.rows(q) for q in V.Q}
ITQ = {q: np.asarray(ci[V.Q[q][0]][R[q][0]]).astype(np.int64) for q in V.Q}
WK = {q: V.row_wkey(q) for q in V.Q}
hi = np.load(FLAT / 'train/hist_item_id.npy')
NITEM = int(max(int(hi.max()), max(int(x.max()) for x in ITQ.values()))) + 1
want = np.zeros(NITEM, bool)
for q in V.Q:
    want[ITQ[q]] = True
want_items = np.nonzero(want)[0]; iidx = np.full(NITEM, -1, np.int64); iidx[want_items] = np.arange(len(want_items)); nI = len(want_items)
t = np.load(FLAT / 'train/hist_event_type.npy'); so = np.load(FLAT / 'train/hist_event_source.npy')
keep = want[hi] & (t == 1) & ((so == 1) | (so == 2))
del t
hf = np.load(FLAT / 'train/hist_fidx.npy')
s12f = S12[a['slot']]                                   # every train feed of a dense slot is a source (label-free)
keep &= s12f[hf] >= 0
hi = hi[keep].astype(np.int64); so = so[keep]; hf = hf[keep].astype(np.int64)
uk = a['u'][hf]; uk = np.where(uk > 0, uk, -(hf + 1))    # guest snapshot = its own key
D = pl.DataFrame({'uk': uk, 'item': hi, 'c': so.astype(np.int8), 'bit': (np.int64(1) << s12f[hf]).astype(np.int32)})
P(f'train history rows kept (type 1, source 1|2, wanted item, dense slot): {len(D)}; wanted items {nI}')
del hi, so, hf, uk, keep
D = D.group_by(['uk', 'item', 'c']).agg(pl.col('bit').bitwise_or().alias('mask'))
P(f'distinct (user key, item, source): {len(D)}; by source {D["c"].value_counts().sort("c").to_dicts()}')
res = {q: {n: np.full(len(ITQ[q]), np.nan, np.float64) for n in ('ns1_5', 'ns2_5')} for q in V.Q}
for k in V.wkeys:
    g = D.filter((pl.col('mask') & k) != 0).group_by(['item', 'c']).len()
    N = np.zeros((3, nI), np.int64); N[g['c'].to_numpy().astype(np.int64), iidx[g['item'].to_numpy()]] = g['len'].to_numpy()
    P(f'window key {k} (slots {DENSE[[j for j in range(12) if (k >> j) & 1]].tolist()}): (user, item) pairs s1 {int(N[1].sum())}, s2 {int(N[2].sum())}; items with any {int(((N[1] + N[2]) > 0).sum())}')
    for q in V.Q:
        m = WK[q] == k
        if not m.any():
            continue
        ii = iidx[ITQ[q][m]]
        res[q]['ns1_5'][m] = N[1, ii]; res[q]['ns2_5'][m] = N[2, ii]
# own-user removal (leave-user-out): a (user, item, source) record of the row's own non-zero user inside the row's window is subtracted
for q, (sp, sel, cut) in V.Q.items():
    rows, fl, off, M = R[q]; u = V.A[sp]['u'][sel][fl]
    cu = pl.DataFrame({'row': np.arange(len(rows)), 'uk': u, 'item': ITQ[q], 'wk': WK[q].astype(np.int32)}).filter((pl.col('uk') > 0) & (pl.col('wk') > 0))
    own = cu.join(D, on=['uk', 'item'], how='inner').filter((pl.col('mask') & pl.col('wk')) != 0)
    for c in (1, 2):
        o = own.filter(pl.col('c') == c)
        r_ = o['row'].to_numpy(); assert len(np.unique(r_)) == len(r_)
        res[q][f'ns{c}_5'][r_] -= 1.0
        assert np.nanmin(res[q][f'ns{c}_5']) >= 0
    n1, n2 = res[q]['ns1_5'], res[q]['ns2_5']; v = ~np.isnan(n1)
    put(q, 'ns1_5', n1); put(q, 'ns2_5', n2); put(q, 'X3_5', np.log((n1 + 1) / (n2 + 1)))
    P(f'{q}: rows {len(v)}, with a view {int(v.sum())}; own-user removals {len(own)}; mean n_s1 {n1[v].mean():.3f} n_s2 {n2[v].mean():.3f}; zero share s1 {np.mean(n1[v] == 0):.3f} s2 {np.mean(n2[v] == 0):.3f}; any {np.mean((n1[v] + n2[v]) > 0):.4f}')
P('DONE')
