"""RESCUE r04: P6 same-user earlier feeds in the view D5 (PREDECLARED.txt).
  uc5      = (pos + 0.0159 * 20) / (imp + 20) of (user, category) over the user's labelled NON-HOLDOUT train feeds of the row's window (tte_uc_ctr formula);
             NaN for user 0 and for a user without a feed in the window.
  reshown5 = 1 if the item was a candidate of the same non-zero user in a non-holdout train feed of the window (label-free), else 0.
Leak rule: only feeds of the same user inside the window, i.e. strictly EARLIER than the cut and the cut is before the row; never a later snapshot,
never the row's own feed, user 0 never linked.
STEP 1 (label-free, decides whether anything is built): feed-level coverage = share of feeds whose non-zero user has a feed in the window, train side
(pool feeds, lag view) against val / public / private feeds. If the two sides differ by more than a factor of two the piece STOPS (PREDECLARED P6).
CPU only. Output: out/arr/<set>__{uc5,reshown5,uhas5}.npy, log out/r04_user.log."""
from rs_common import *
P = Log(OUT / 'r04_user.log')
V = Views(P)
a = V.A['train']
fs12 = np.where(V.ish, -1, S12[a['slot']]).astype(np.int64)          # source feeds: non-holdout, dense slot
src = np.nonzero((fs12 >= 0) & (a['u'] > 0))[0]
UMAX = int(max(V.A[s]['u'].max() for s in SPLITS)) + 1
# per (user, table column): number of source feeds
UF = pl.DataFrame({'u': a['u'][src], 's': fs12[src]}).group_by(['u', 's']).len()
wantu = np.zeros(UMAX, bool)
for q, (sp, sel, cut) in V.Q.items():
    wantu[V.A[sp]['u'][sel]] = True
wantu[0] = False
uu = np.nonzero(wantu)[0]; uidx = np.full(UMAX, -1, np.int64); uidx[uu] = np.arange(len(uu)); nU = len(uu)
UF = UF.filter(pl.col('u').is_in(uu))
NF = np.zeros((nU, 12), np.int64); NF[uidx[UF['u'].to_numpy()], UF['s'].to_numpy()] = UF['len'].to_numpy()
P(f'wanted non-zero users {nU}; of them with a non-holdout train feed in a dense slot {int((NF.sum(1) > 0).sum())}')


def feed_cov(q, fmask=None):
    sp, sel, cut = V.Q[q]; u = V.A[sp]['u'][sel]; out = np.zeros(len(sel), bool)
    for c in np.unique(cut):
        if c < 0:
            continue
        cols = S12[window(int(c))]; m = (cut == c) & (u > 0)
        out[m] = NF[uidx[u[m]]][:, cols].sum(1) > 0
    return out


COV = {q: feed_cov(q) for q in V.Q}
tsl = a['slot']; ts_e = V.A['test']['ts']; Me = V.A['test']['M']; Mv = V.A['val']['M']; epub = ts_e < hc.PUB_TS
ps = tsl[V.poolf]; hs = tsl[V.hold]
c_tl = COV['pool'][(ps >= 11) & (ps <= 13)].mean(); c_vl = COV['pool'][(ps >= 6) & (ps <= 8)].mean()
c_htl = COV['hold'][(hs >= 11) & (hs <= 13)].mean(); c_hvl = COV['hold'][(hs >= 6) & (hs <= 8)].mean()
c_priv = COV['test_nat'][~epub & (Me >= 50)].mean(); c_pub = COV['test_nat'][epub & (Me >= 50)].mean(); c_val = COV['val_nat'][Mv >= 50].mean(); c_tw = COV['val_twin'][Mv >= 50].mean()
nz = lambda q: float((V.A[V.Q[q][0]]['u'][V.Q[q][1]] > 0).mean())
P('STEP 1, feed-level coverage (user non-zero AND a non-holdout train feed of the same user inside the window):')
P(f'   train side, lag view: pool feeds slots 11-13 {c_tl:.4f}, slots 6-8 {c_vl:.4f} | holdout feeds slots 11-13 {c_htl:.4f}, slots 6-8 {c_hvl:.4f}')
P(f'   test side (feeds with M >= 50): private {c_priv:.4f}, public {c_pub:.4f} | val natural {c_val:.4f}, val twin {c_tw:.4f}')
P(f'   share of feeds with a non-zero user: pool {nz("pool"):.4f}, holdout {nz("hold"):.4f}, val {nz("val_nat"):.4f}, test {nz("test_nat"):.4f}')
r1 = max(c_priv, c_pub) / max(c_tl, 1e-9); r2 = c_val / max(c_vl, 1e-9)
r1 = max(r1, 1 / max(r1, 1e-9)); r2 = max(r2, 1 / max(r2, 1e-9))
P(f'   ratio test-like: {max(c_priv, c_pub):.4f} / {c_tl:.4f} -> factor {r1:.2f}; val-like: {c_val:.4f} / {c_vl:.4f} -> factor {r2:.2f}')
# for scale: the same with ALL dense slots below the cut (the nx3 lag view without the equal-depth rule) and with every train day for val / test
def feed_cov_all(q):
    sp, sel, cut = V.Q[q]; u = V.A[sp]['u'][sel]; out = np.zeros(len(sel), bool)
    for c in np.unique(cut):
        if c < 0:
            continue
        cols = S12[DENSE[DENSE < c]]; m = (cut == c) & (u > 0)
        out[m] = NF[uidx[u[m]]][:, cols].sum(1) > 0
    return out
ca = {q: feed_cov_all(q) for q in ('pool', 'test_nat', 'val_nat')}
P(f'   for scale, every dense slot below the cut instead of 5: pool slots 11-13 {ca["pool"][(ps >= 11) & (ps <= 13)].mean():.4f}, slots 6-8 {ca["pool"][(ps >= 6) & (ps <= 8)].mean():.4f} | '
  f'private {ca["test_nat"][~epub & (Me >= 50)].mean():.4f}, public {ca["test_nat"][epub & (Me >= 50)].mean():.4f}, val {ca["val_nat"][Mv >= 50].mean():.4f}')
STOP = (r1 > 2.0) or (r2 > 2.0)
P(f'STEP 1 VERDICT: {"STOP (coverage differs by more than a factor of two; nothing with labels is built)" if STOP else "coverage within a factor of two -> columns are built"}')
if STOP:
    P('DONE'); sys.exit(0)

# ---------------------------------------------------------------- STEP 2: the columns
item = np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r'); y = np.load(FLAT / 'train/cand_target.npy', mmap_mode='r'); cat = np.load(FLAT / 'train/cand_category_id.npy', mmap_mode='r')
sf = src[wantu[a['u'][src]]]                               # source feeds of wanted users
Ms = a['M'][sf]; os_ = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64); srows = np.repeat(a['co'][sf] - os_, Ms) + np.arange(int(Ms.sum()), dtype=np.int64)
s_it = np.asarray(item[srows]).astype(np.int64); s_y = np.asarray(y[srows]).astype(np.float64); s_c = np.asarray(cat[srows]).astype(np.int64)
s_u = uidx[np.repeat(a['u'][sf], Ms)]; s_s = np.repeat(fs12[sf], Ms)
NC = int(max(int(s_c.max()), max(int(np.load(FLAT / s / 'cand_category_id.npy', mmap_mode='r').max()) for s in SPLITS))) + 1
P(f'source feeds of wanted users {len(sf)}, rows {len(srows)}; categories {NC}')
POS = np.zeros((nU, NC, 12), np.float32); IMP = np.zeros((nU, NC, 12), np.float32)
np.add.at(POS, (s_u, s_c, s_s), s_y); np.add.at(IMP, (s_u, s_c, s_s), 1.0)
UI = pl.DataFrame({'u': s_u, 'item': s_it, 'bit': (np.int64(1) << s_s).astype(np.int32)}).group_by(['u', 'item']).agg(pl.col('bit').bitwise_or().alias('mask'))
P(f'distinct (user, item) candidate pairs {len(UI)}')
PRIOR = 0.0159
ci = {s: np.load(FLAT / s / 'cand_item_id.npy', mmap_mode='r') for s in SPLITS}; cc = {s: np.load(FLAT / s / 'cand_category_id.npy', mmap_mode='r') for s in SPLITS}
for q, (sp, sel, cut) in V.Q.items():
    rows, fl, off, M = V.rows(q); u = V.A[sp]['u'][sel]; n = len(rows)
    itq = np.asarray(ci[sp][rows]).astype(np.int64); cq = np.asarray(cc[sp][rows]).astype(np.int64); wk = V.row_wkey(q)
    uc = np.full(n, np.nan); uh = np.full(n, np.nan); rs = np.full(n, np.nan)
    rcut = np.repeat(cut, M); ru = np.repeat(u, M); rcov = np.repeat(COV[q], M)
    for c in np.unique(cut):
        if c < 0:
            continue
        cols = S12[window(int(c))]; m = rcut == c
        uh[m] = rcov[m]; rs[m] = 0.0
        mm = m & rcov
        ui_ = uidx[ru[mm]]; p_ = POS[ui_, cq[mm]][:, cols].sum(1); i_ = IMP[ui_, cq[mm]][:, cols].sum(1)
        uc[mm] = (p_ + PRIOR * 20) / (i_ + 20.0)
    cu = pl.DataFrame({'row': np.arange(n), 'u': uidx[np.where(ru < UMAX, ru, 0)], 'item': itq, 'wk': wk.astype(np.int32)}).filter((pl.col('u') >= 0) & (pl.col('wk') > 0))
    hit = cu.join(UI, on=['u', 'item'], how='inner').filter((pl.col('mask') & pl.col('wk')) != 0)
    rs[hit['row'].to_numpy()] = 1.0
    put(q, 'uc5', uc); put(q, 'uhas5', uh); put(q, 'reshown5', rs)
    v = ~np.isnan(rs)
    P(f'{q}: rows {n}, with a view {int(v.sum())}; rows with a user record {np.nanmean(uh[v]):.4f}; uc5 present {np.mean(~np.isnan(uc[v])):.4f}, mean {np.nanmean(uc[v]):.5f}; reshown {rs[v].mean():.5f}')
P('DONE')
