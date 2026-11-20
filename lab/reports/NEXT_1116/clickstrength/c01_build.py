"""Lane A c01: build of the two forms of RULE.txt section 3 on the WITNESS rows (hold, pool, val_nat, val_twin, test_nat) in the view P3.
Sources: non-holdout train feeds of the row's window. The only label array opened is data/flat/train/cand_target.npy (source labels).
Also written (for g5 only): a deliberately LEAKY variant for hold / pool rows whose window is the last three dense slots up to and INCLUDING the row's own day.
usage: c01_build.py     output out/arr/<set>__<name>.npy (float32, aligned with Views.rows(set)), log out/c01_build.log"""
from cs_common import *
lens, condlib = load_lens()
P = Log(OUT / 'c01_build.log')
P('RULE.txt md5 ' + md5(CS + '/RULE.txt'))
V = Views(P)
a = V.A['train']; n = a['n']
# ---------------------------------------------------------------- causality of the view (feeds tables only)
for q, (sp, sel, e) in V.Q.items():
    sl = V.A[sp]['slot'][sel]; ok = e >= 0
    P(f'causality {q}: feeds with a view {int(ok.sum())}; lag = slot - newest source slot: min {int((sl[ok] - e[ok]).min())} d, max {int((sl[ok] - e[ok]).max())} d; windows all three dense slots: True')
    assert (sl[ok] - e[ok] >= 3).all()
# ---------------------------------------------------------------- train arrays
item = np.load(FLAT / 'train/cand_item_id.npy'); pos = np.load(FLAT / 'train/cand_pos.npy'); y = np.load(FLAT / 'train/cand_target.npy')
ci = {s: np.load(FLAT / s / 'cand_item_id.npy') for s in ('val', 'test')}; ci['train'] = item
NITEM = int(max(int(item.max()), int(ci['val'].max()), int(ci['test'].max()))) + 1
praw = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
regX = praw > 0.5
fs12 = np.where(V.ish, -1, S12[a['slot']]).astype(np.int64)        # source column of a feed (-1: holdout feed or not a dense slot)
P(f'train rows {len(item)}; NITEM {NITEM}; pos dtype {pos.dtype} max {int(pos.max())}; source feeds (non-holdout, dense slot) {int((fs12 >= 0).sum())}; of them exploration (p_random > 0.5) {int(((fs12 >= 0) & regX).sum())}')
assert int((fs12[V.hold] >= 0).sum()) == 0
NPOS = int(pos.max()) + 1


@nb.njit(parallel=True, cache=True)
def pos_tables(pos, y, co, M, sel, regX, npos_):
    T = 24; n = len(M)
    imp = np.zeros((T, 2, npos_)); clk = np.zeros((T, 2, npos_))
    for c in nb.prange(T):
        for f in range((n * c) // T, (n * (c + 1)) // T):
            if not sel[f]:
                continue
            r = 1 if regX[f] else 0; o = co[f]
            for j in range(M[f]):
                imp[c, r, pos[o + j]] += 1.0; clk[c, r, pos[o + j]] += y[o + j]
    return imp.sum(0), clk.sum(0)


@nb.njit(parallel=True, cache=True)
def feed_stats(pos, y, co, M, regX, w):
    n = len(M); npos = np.zeros(n); sw = np.zeros(n)
    for f in nb.prange(n):
        o = co[f]; r = 1 if regX[f] else 0; s = 0.0; c = 0.0
        for j in range(M[f]):
            s += w[r, pos[o + j]]; c += y[o + j]
        npos[f] = c; sw[f] = s
    return npos, sw


@nb.njit(cache=True)
def accum(item, pos, y, co, M, fs12, regX, npos, sw, w, imp, ca, ea, cx, ex):
    for f in range(len(M)):
        s = fs12[f]
        if s < 0:
            continue
        o = co[f]; np_ = npos[f]; r = 1 if regX[f] else 0
        sc = np_ / sw[f] if np_ > 0 else 0.0
        for j in range(M[f]):
            it = item[o + j]
            imp[it, s] += 1
            if np_ > 0:
                q = sc * w[r, pos[o + j]]
                ea[it, s] += q
                if y[o + j] > 0:
                    ca[it, s] += 1
                if r == 1:
                    ex[it, s] += q
                    if y[o + j] > 0:
                        cx[it, s] += 1


pimp, pclk = pos_tables(pos, y, a['co'], a['M'], (fs12 >= 0) & (a['slot'] <= 4), regX, NPOS)
rate = pclk.sum(1) / pimp.sum(1)
W = (pclk + 20.0 * rate[:, None]) / (pimp + 20.0)
P(f'position weights from non-holdout feeds of slots 0-4: production rows {pimp[0].sum():.0f} rate {rate[0]:.5f}; exploration rows {pimp[1].sum():.0f} rate {rate[1]:.5f}')
for r, nm in ((0, 'production'), (1, 'exploration')):
    P(f'   {nm}: w at pos 0 / 1 / 5 / 10 / 20 / 50 / 100 / 200 / {NPOS - 1}: ' + ' / '.join(f'{W[r, min(k, NPOS - 1)]:.4f}' for k in (0, 1, 5, 10, 20, 50, 100, 200, NPOS - 1)))
np.save(open(OUT / 'c01_posweights.npy', 'xb'), W)
npos_f, sw_f = feed_stats(pos, y, a['co'], a['M'], regX, W)
P(f'feeds with >= 1 click among sources: {int(((npos_f > 0) & (fs12 >= 0)).sum())} of {int((fs12 >= 0).sum())}; exploration: {int(((npos_f > 0) & (fs12 >= 0) & regX).sum())} of {int(((fs12 >= 0) & regX).sum())}')
imp = np.zeros((NITEM, 12), np.int32); ca = np.zeros((NITEM, 12), np.int32); ea = np.zeros((NITEM, 12), np.float32); cx = np.zeros((NITEM, 12), np.int32); ex = np.zeros((NITEM, 12), np.float32)
accum(item, pos, y, a['co'], a['M'], fs12, regX, npos_f, sw_f, W, imp, ca, ea, cx, ex)
P(f'tables: impressions {int(imp.sum(dtype=np.int64))}; all-feed clicks {int(ca.sum(dtype=np.int64))} expected {float(ea.sum(dtype=np.float64)):.1f}; exploration clicks {int(cx.sum(dtype=np.int64))} expected {float(ex.sum(dtype=np.float64)):.1f}; items seen {int((imp.sum(1) > 0).sum())}')
del pos, y

# ---------------------------------------------------------------- shrinkage constants from window {0, 1, 2}
c0 = ca[:, :3].sum(1, dtype=np.int64); e0 = ea[:, :3].sum(1, dtype=np.float64); A_a, tau_a = eb_a(c0, e0)
P(f'F1 all feeds, window slots 0-2: items with e > 0 {int((e0 > 0).sum())}, clicks {int(c0.sum())}, expected {e0.sum():.1f}, max e {e0.max():.2f}; tau2 {tau_a:.5f} -> a = {A_a:.4f}')
c0 = cx[:, :3].sum(1, dtype=np.int64); e0 = ex[:, :3].sum(1, dtype=np.float64); A_x, tau_x = eb_a(c0, e0)
P(f'F2 exploration feeds, window slots 0-2: items with e > 0 {int((e0 > 0).sum())}, clicks {int(c0.sum())}, expected {e0.sum():.1f}, max e {e0.max():.2f}; tau2 {tau_x:.5f} -> a = {A_x:.4f}')
del c0, e0
emax_a = float(ea.sum(1, dtype=np.float64).max()); emax_x = float(ex.sum(1, dtype=np.float64).max())
MT = {'a': m_table(A_a, emax_a), 'x': m_table(A_x, emax_x)}; AA = {'a': A_a, 'x': A_x}
rng = np.random.default_rng(1)
for k in ('a', 'x'):
    ee = np.exp(rng.uniform(np.log(1e-5), np.log(MT[k][0][-1]), 300)); err = max(abs(np.interp(e_, MT[k][0], MT[k][1]) - m_exact(AA[k], e_)) for e_ in ee)
    P(f'm table {k}: a {AA[k]:.4f}, grid to {MT[k][0][-1]:.1f}; max interpolation error on 300 random e {err:.2e}; m at e = 0.02 / 0.1 / 1 / 10: ' + ' / '.join(f'{m_exact(AA[k], e_):+.5f}' for e_ in (0.02, 0.1, 1.0, 10.0)))
    assert err < 1e-4
import json
json.dump(dict(a_all=A_a, tau2_all=tau_a, a_expl=A_x, tau2_expl=tau_x, rate_prod=float(rate[0]), rate_expl=float(rate[1])), open(OUT / 'c01_constants.json', 'x'))

# ---------------------------------------------------------------- query rows
R = {q: V.rows(q) for q in QS}


def gather(tab, it, cols):
    return tab[it[:, None], cols[None, :]].sum(1, dtype=np.float64)


def build(q, e_row, tag=''):
    rows, fl, off, M = R[q]; sp = V.Q[q][0]
    it = np.asarray(ci[sp][rows]).astype(np.int64); N = len(it)
    o = {k: np.full(N, np.nan) for k in ('imp3', 'cs3a_k', 'cs3a_e', 'cs3x_k', 'cs3x_e')}
    for e in np.unique(e_row[e_row >= 0]):
        m = e_row == e; cols = S12[np.array(WIN[int(e)])]; assert (cols >= 0).all()
        i_ = it[m]
        o['imp3'][m] = gather(imp, i_, cols); o['cs3a_k'][m] = gather(ca, i_, cols); o['cs3a_e'][m] = gather(ea, i_, cols); o['cs3x_k'][m] = gather(cx, i_, cols); o['cs3x_e'][m] = gather(ex, i_, cols)
    view = e_row >= 0
    st = strata(o['imp3'], view, off, M)
    for k in ('a', 'x'):
        x = np.where(view, x_stat(np.nan_to_num(o[f'cs3{k}_k']), np.nan_to_num(o[f'cs3{k}_e']), AA[k], MT[k]), np.nan)
        o[f'cs3{k}_x'] = x; o[f'cs3{k}_c'] = centre_in_strata(x, st, off, M, NB)
        assert (np.abs(x[view & (np.nan_to_num(o[f"cs3{k}_e"]) == 0)]) == 0).all()          # unseen item -> exactly 0
    for nme, x in o.items():
        put(q, nme + tag, x)
    put(q, 'stratum' + tag, np.where(view, st, -1), np.int8)
    v = view
    P(f'{q}{tag}: rows {N}, with a view {int(v.sum())}; imp3 > 0 {float((o["imp3"][v] > 0).mean()):.4f}, mean imp3 {float(o["imp3"][v].mean()):.2f}; '
      f'F1: e > 0 {float((o["cs3a_e"][v] > 0).mean()):.4f}, clicked item {float((o["cs3a_k"][v] > 0).mean()):.4f}, x sd {float(o["cs3a_x"][v].std()):.4f}, entered sd {float(np.nanstd(o["cs3a_c"][v])):.4f}, entered != 0 {float((o["cs3a_c"][v] != 0).mean()):.4f} | '
      f'F2: e > 0 {float((o["cs3x_e"][v] > 0).mean()):.4f}, clicked item {float((o["cs3x_k"][v] > 0).mean()):.4f}, x sd {float(o["cs3x_x"][v].std()):.4f}, entered sd {float(np.nanstd(o["cs3x_c"][v])):.4f}, entered != 0 {float((o["cs3x_c"][v] != 0).mean()):.4f}')


WIN = {e: window(e).tolist() for e in V.es}
P('windows of the view: ' + str(WIN))
for q in QS:
    build(q, V.row_e(q))
    sp, sel, e = V.Q[q]
    put(q, 'lag_row', np.where(V.row_e(q) >= 0, V.row_slot(q) - V.row_e(q), -1), np.int8)

# ---------------------------------------------------------------- the deliberately leaky variant (g5 clause 5): window = last three dense slots up to and INCLUDING the row's own day
for q in ('hold', 'pool'):
    sl = V.row_slot(q); e_row = V.row_e(q)
    for s in np.unique(sl[e_row >= 0]):
        d = DENSE[DENSE <= s][-3:]; WIN[100 + int(s)] = d.tolist()
    build(q, np.where(e_row >= 0, 100 + sl, -1), tag='_LEAK')
P('leaky windows (key 100 + slot): ' + str({k: v for k, v in WIN.items() if k >= 100}))
P('DONE')
