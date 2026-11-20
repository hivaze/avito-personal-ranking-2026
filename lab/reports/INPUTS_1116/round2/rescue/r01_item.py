"""RESCUE r01: the three pieces that read the TRAIN CANDIDATE LOG of the window (view D5 of PREDECLARED.txt):
  P1 bt5 / bt5_seen   Bradley-Terry click strength, fit of record c06_build.fit_bt (lambda 3) on the exploration feeds of the window
  P2 coec5 / ctr5     all-feed item clicks over expected / smoothed CTR of the window (formulas of src/features_tte.py)
  P5 gapf5 / never5   first-seen gap inside the window, never-seen flag;  imp5 = impressions in the window
Sources: train feeds of the window that are NOT holdout feeds. Query sets: hold (lag view), pool (lag view), val_nat, val_twin, test_nat.
Label statistics are causal by construction: window slots < cut <= slot of the row (asserted per query set below).
CPU only. Output: out/arr/<set>__<name>.npy (float32, aligned with Views.rows(set)), log out/r01_item.log."""
from rs_common import *
sys.path.insert(0, '<repo>/nn_x/insights/ccb_1113/scratch/tester3')
import c06_build as c6
P = Log(OUT / 'r01_item.log')
V = Views(P)
a = V.A['train']
# ---------------------------------------------------------------- causality of the view (label-free check on the feeds tables)
for q, (sp, sel, cut) in V.Q.items():
    sl = V.A[sp]['slot'][sel]; ok = cut >= 0
    wmax = np.array([window(int(c)).max() for c in cut[ok]]) if ok.any() else np.array([0])
    assert (wmax < cut[ok]).all() and (cut[ok] <= sl[ok] - 1).all() if sp == 'train' else (wmax < cut[ok]).all() and (cut[ok] <= sl[ok]).all()
    P(f'causality {q}: newest source slot < cut on all {int(ok.sum())} feeds with a view; min (slot - newest source slot) = {int((sl[ok] - wmax).min())} days')

# ---------------------------------------------------------------- per (item, slot) tables over non-holdout train feeds
item = np.load(FLAT / 'train/cand_item_id.npy'); pos = np.load(FLAT / 'train/cand_pos.npy'); y = np.load(FLAT / 'train/cand_target.npy'); fidx = np.load(FLAT / 'train/cand_fidx.npy')
QI = {q: None for q in V.Q}
ci = {s: np.load(FLAT / s / 'cand_item_id.npy') for s in ('val', 'test')}; ci['train'] = item
NITEM = int(max(int(item.max()), int(ci['val'].max()), int(ci['test'].max()))) + 1
fs12 = np.where(V.ish, -1, S12[a['slot']]).astype(np.int64)
P(f'train rows {len(item)}; NITEM {NITEM}; source feeds (non-holdout, dense slot) {int((fs12 >= 0).sum())}; by slot ' + str(np.bincount(a['slot'][fs12 >= 0], minlength=15).tolist()))


@nb.njit(cache=True)
def pos_prior(pos, y, fidx, fsel, clk, imp):
    for r in range(len(pos)):
        if fsel[fidx[r]]:
            imp[pos[r]] += 1.0; clk[pos[r]] += y[r]


@nb.njit(cache=True)
def accum(item, pos, y, fidx, fs12, pc, cnt, clk, ex):
    for r in range(len(item)):
        s = fs12[fidx[r]]
        if s < 0:
            continue
        it = item[r]
        cnt[it, s] += 1
        if y[r] > 0:
            clk[it, s] += 1
        ex[it, s] += pc[pos[r]]


@nb.njit(parallel=True, cache=True)
def win_tables(cnt, clk, ex, cols, o_imp, o_clk, o_ex, o_gap):
    n = cnt.shape[0]; k = len(cols)
    for i in nb.prange(n):
        a_ = 0; b_ = 0; e_ = 0.0; first = -1
        for j in range(k):
            c = cols[j]; v = cnt[i, c]
            if v > 0 and first < 0:
                first = j
            a_ += v; b_ += clk[i, c]; e_ += ex[i, c]
        o_imp[i] = a_; o_clk[i] = b_; o_ex[i] = e_
        o_gap[i] = np.float32(k - first) if first >= 0 else np.float32(np.nan)


npos_ = int(pos.max()) + 1
pclk = np.zeros(npos_); pimp = np.zeros(npos_)
pos_prior(pos, y, fidx, (fs12 >= 0) & (a['slot'] <= 4), pclk, pimp)
pc = (pclk / np.maximum(pimp, 1)).astype(np.float32)
P(f'position prior from non-holdout feeds of slots 0-4: rows {pimp.sum():.0f}, clicks {pclk.sum():.0f}, rate {pclk.sum() / pimp.sum():.5f}; positions {npos_}; prior at pos 0 / 10 / 50 / 100: {pc[0]:.4f} / {pc[min(10, npos_ - 1)]:.4f} / {pc[min(50, npos_ - 1)]:.4f} / {pc[min(100, npos_ - 1)]:.4f}')
cnt = np.zeros((NITEM, 12), np.int32); clk = np.zeros((NITEM, 12), np.int32); ex = np.zeros((NITEM, 12), np.float32)
accum(item, pos, y, fidx, fs12, pc, cnt, clk, ex)
P(f'tables: impressions {int(cnt.sum(dtype=np.int64))}, clicks {int(clk.sum(dtype=np.int64))}, expected {float(ex.sum(dtype=np.float64)):.0f}; items seen {int((cnt.sum(1) > 0).sum())}')
del pos, fidx

# ---------------------------------------------------------------- query rows
R = {q: V.rows(q) for q in V.Q}
ITQ = {q: np.asarray(ci[V.Q[q][0]][R[q][0]]).astype(np.int64) for q in V.Q}
WK = {q: V.row_wkey(q) for q in V.Q}
res = {q: {n: np.full(len(ITQ[q]), np.nan, np.float32) for n in ('imp5', 'coec5', 'ctr5', 'gapf5', 'never5', 'bt5', 'bt5_seen')} for q in V.Q}
PRIOR = 0.0159
for k in V.wkeys:
    cols = np.array([j for j in range(12) if (k >> j) & 1], np.int64); assert len(cols) == DEPTH
    o_imp = np.zeros(NITEM, np.int32); o_clk = np.zeros(NITEM, np.int32); o_ex = np.zeros(NITEM, np.float32); o_gap = np.zeros(NITEM, np.float32)
    win_tables(cnt, clk, ex, cols, o_imp, o_clk, o_ex, o_gap)
    P(f'window key {k} (slots {DENSE[cols].tolist()}): impressions {int(o_imp.sum(dtype=np.int64))}, clicks {int(o_clk.sum(dtype=np.int64))}, CTR {o_clk.sum(dtype=np.int64) / o_imp.sum(dtype=np.int64):.5f}, items seen {int((o_imp > 0).sum())}, clicks / expected {o_clk.sum(dtype=np.int64) / float(o_ex.sum(dtype=np.float64)):.4f}')
    for q in V.Q:
        m = WK[q] == k
        if not m.any():
            continue
        it = ITQ[q][m]; im = o_imp[it].astype(np.float64); cl = o_clk[it].astype(np.float64); e_ = o_ex[it].astype(np.float64)
        r = res[q]
        r['imp5'][m] = im; r['coec5'][m] = (cl + 1.0) / (e_ + 1.0); r['ctr5'][m] = (cl + PRIOR * 20) / (im + 20.0)
        r['gapf5'][m] = o_gap[it]; r['never5'][m] = (im == 0)
del cnt, clk, ex, o_imp, o_clk, o_ex, o_gap

# ---------------------------------------------------------------- P1: the Bradley-Terry fit of record on the exploration feeds of each window
# data selection = line-by-line the one of c06_build.main() / critic c02_lagbt.py: p_random > 0.5 (raw, NaN -> 0), not holdout, >= 1 positive
praw = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
cand = np.nonzero((praw > 0.5) & ~V.ish)[0]
Mt, ot = a['M'], a['co']
Mc = Mt[cand]; oc = np.concatenate([[0], np.cumsum(Mc)[:-1]]).astype(np.int64)
y_c = np.empty(int(Mc.sum()), np.int8); i_c = np.empty(int(Mc.sum()), np.int32)
c6.gather_rows(cand, ot, Mt, oc, y, y_c); c6.gather_rows(cand, ot, Mt, oc, item, i_c)
npos_c = np.add.reduceat(y_c.astype(np.int64), oc); keep = npos_c >= 1; src = cand[keep]
Ms = Mt[src]; rowsel = np.repeat(keep, Mc); y_s = y_c[rowsel].astype(np.float64); i_s = i_c[rowsel]; sslot = a['slot'][src]
P(f'bt source feeds (p_random > 0.5, not holdout, >= 1 positive): {len(src)}; by slot {dict(zip(*[x.tolist() for x in np.unique(sslot, return_counts=True)]))}')
del item, y, y_c, i_c


def fit_subset(fmask, lam=3.0):
    rmask = np.repeat(fmask, Ms); items = i_s[rmask]; uit, inv = np.unique(items, return_inverse=True)
    Msub = Ms[fmask]; osub = np.concatenate([[0], np.cumsum(Msub)[:-1]]).astype(np.int64); yy = y_s[rmask]; npos = np.add.reduceat(yy, osub)
    th, r_ = c6.fit_bt(inv.astype(np.int64), yy, osub, Msub, npos, len(uit), lam, None)
    P(f'   fit: {int(fmask.sum())} feeds, {int(rmask.sum())} rows, {len(uit)} items, nit {r_.nit}, conv {r_.success}, theta sd {th.std():.4f}')
    return uit, th


for k in V.wkeys:
    cols = np.array([j for j in range(12) if (k >> j) & 1], np.int64); wsl = DENSE[cols]
    P(f'bt fit for window slots {wsl.tolist()}')
    uit, th = fit_subset(np.isin(sslot, wsl))
    for q in V.Q:
        m = WK[q] == k
        if not m.any():
            continue
        it = ITQ[q][m]; j = np.minimum(np.searchsorted(uit, it), len(uit) - 1); seen = uit[j] == it
        res[q]['bt5'][m] = np.where(seen, th[j], 0.0); res[q]['bt5_seen'][m] = seen

# ---------------------------------------------------------------- write + summary
for q in V.Q:
    r = res[q]; v = ~np.isnan(r['imp5'])
    for n, x in r.items():
        put(q, n, x)
    P(f'{q}: rows {len(v)}, with a view {int(v.sum())}; impression in the window {float((r["imp5"][v] > 0).mean()):.4f}; bt seen {float(r["bt5_seen"][v].mean()):.4f}; '
      f'mean imp5 {float(r["imp5"][v].mean()):.1f}; coec5 mean {float(r["coec5"][v].mean()):.4f}; gapf5 mean (seen) {float(np.nanmean(r["gapf5"][v])):.3f}; bt5 sd {float(r["bt5"][v].std()):.4f}')
P('DONE')
