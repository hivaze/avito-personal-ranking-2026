"""poplag: rebuild the recency-dependent popularity inputs of the 30K holdout feeds under the exposure-matched lag of DECLARATION.txt.
Label-free (no target file is opened). CPU only, 12 threads. Writes cache/lag_<view>_<column>.npy (raw feat_cache scale, holdout row order) for
view in {chk (lag-0 semantics: code check against the stored columns, not saved as a view), pub, priv} and cache/view_<view>_{L,D,A}.npy.

Usage: CUDA_VISIBLE_DEVICES="" python build_lag.py
"""
import sys, time
from pl_common import *
import numba as nb
nb.set_num_threads(12)

T0 = time.time()
lg = lambda *a: print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
BIG = 1 << 30
DAY0 = int(day_of(np.array([DATA0]))[0])      # 20702 = Sep 6
DAYC = int(day_of(np.array([CUT_VT]))[0])     # 20716 = Sep 20
PAT = np.zeros(10, np.bool_); PAT[list(SPARSE_PAT)] = True
H_SP = np.uint64(0xC2B2AE3D27D4EB4F); H_LAG = np.uint64(0xD6E8FEB86659FD93)


def sparse_flag(feed_id):   # feat_new/hz_common.py::sparse_flag (the shared 1/37 hash sample of train feeds)
    h = (np.asarray(feed_id).astype(np.uint64) * H_SP) >> np.uint64(40)
    return (h % np.uint64(37)) == 0


@nb.njit(cache=True)
def build_post(hi, nidx, n_items):
    """Counting sort of the history events whose item is needed: start[k]..start[k+1] = event indices of item k, ascending."""
    cnt = np.zeros(n_items + 1, np.int64)
    for e in range(len(hi)):
        k = nidx[hi[e]]
        if k >= 0:
            cnt[k + 1] += 1
    start = np.cumsum(cnt)
    fill = start[:-1].copy()
    p_ev = np.empty(start[-1], np.int64)
    for e in range(len(hi)):
        k = nidx[hi[e]]
        if k >= 0:
            p_ev[fill[k]] = e; fill[k] += 1
    return start, p_ev


@nb.njit(parallel=True, cache=True)
def luo_kernel(ck, cfeed, ct, cu, cD, cA, asbuilt, start, p_feed, p_et, fuser, fts, fday, fsp,
               vstart, v_user, v_ts, v_day, v_et, pat, out):
    """out[r, 0] = all event types, out[r, 1..4] = event types 1..4. asbuilt: every train / val / test snapshot (stored LUO rule)."""
    for r in nb.prange(len(ck)):
        k = ck[r]; u = cu[r]; t = ct[r]; D = cD[r]; A = cA[r]; own = cfeed[r]
        for e in range(start[k], start[k + 1]):
            s = p_feed[e]
            if u != 0 and fuser[s] == u and fts[s] > t:
                continue
            ok = asbuilt
            if not ok:
                ds = fday[s]
                if ds < D or s == own:
                    ok = True
                elif fsp[s] and ds >= A and ds - A <= 9 and pat[ds - A]:
                    ok = True
            if ok:
                out[r, 0] += 1
                et = p_et[e]
                if 1 <= et <= 4:
                    out[r, et] += 1
        for e in range(vstart[k], vstart[k + 1]):
            if u != 0 and v_user[e] == u and v_ts[e] > t:
                continue
            ok = asbuilt
            if not ok:
                ds = v_day[e]
                if ds >= A and ds - A <= 9 and pat[ds - A]:
                    ok = True
            if ok:
                out[r, 0] += 1
                et = v_et[e]
                if 1 <= et <= 4:
                    out[r, et] += 1


@nb.njit(parallel=True, cache=True)
def covis_kernel(qfeed, qoff, qM, qhoff, qH, qhitem, qhpos, ck, cD, cA, citem, start, p_feed, p_pos,
                 items, starts, lens, fuser, fts, fday, fnd, fsp, pat, out):
    """feats_v2/seq_/feat_covis.py::_covis (invd20 = out[:,0], pop = out[:,1]) with a per-candidate dense cutoff day D and sparse anchor A.
    Index membership of a source sequence s for a candidate: dense = fday[s] < D <= fnd[s] (latest feed of its user before D; user 0: every feed);
    sparse = fsp[s] and fday[s] in A + PAT. Scan = the last 3000 members in feed order (stored cap), then: same non-zero user skipped,
    sparse sequences with ts >= own ts skipped (stored 'constrained' rule)."""
    for q in nb.prange(len(qfeed)):
        fi = qfeed[q]
        h = qH[q]; ho = qhoff[q]; u = fuser[fi]; t = fts[fi]
        hs = np.empty(h, np.int32)
        for j in range(h):
            hs[j] = qhitem[ho + j]
        order = np.argsort(hs)
        nu = 0
        uh = np.empty(h, np.int32); uw = np.empty(h, np.float64)
        for jj in range(h):
            j = order[jj]
            w = 0.95 ** qhpos[ho + j]
            if nu > 0 and uh[nu - 1] == hs[j]:
                if w > uw[nu - 1]:
                    uw[nu - 1] = w
            else:
                uh[nu] = hs[j]; uw[nu] = w; nu += 1
        buf = np.empty(3000, np.int64); bsp = np.empty(3000, np.bool_)
        co = qoff[q]; m = qM[q]
        for i in range(m):
            r = co + i
            out[r, 0] = 0.0; out[r, 1] = 0.0
            k = ck[r]
            if nu == 0 or start[k + 1] == start[k]:
                continue
            D = cD[r]; A = cA[r]; c = citem[r]
            nb_ = 0
            e = start[k + 1] - 1
            while e >= start[k] and nb_ < 3000:
                s = p_feed[e]; ds = fday[s]
                if ds < D:
                    if fnd[s] >= D:
                        buf[nb_] = e; bsp[nb_] = False; nb_ += 1
                elif fsp[s] and ds >= A and ds - A <= 9 and pat[ds - A]:
                    buf[nb_] = e; bsp[nb_] = True; nb_ += 1
                e -= 1
            npop = 0.0
            for bi in range(nb_ - 1, -1, -1):
                e = buf[bi]; s = p_feed[e]
                if u != 0 and fuser[s] == u:
                    continue
                if bsp[bi] and fts[s] >= t:
                    continue
                npop += 1.0
                p = p_pos[e]; st = starts[s]; L = lens[s]
                lo = max(0, p - 20); hi_ = min(L, p + 21)
                for qq in range(lo, hi_):
                    if qq == p:
                        continue
                    x = items[st + qq]
                    if x == c:
                        continue
                    kk = np.searchsorted(uh[:nu], x)
                    if kk < nu and uh[kk] == x:
                        d = abs(qq - p)
                        out[r, 0] += uw[kk] / d
            out[r, 1] = npop


@nb.njit(cache=True)
def imp_by_day(citem, cfidx, fday, nidx, n_items, day0, nd):
    cnt = np.zeros((n_items, nd), np.int32)
    for r in range(len(citem)):
        k = nidx[citem[r]]
        if k >= 0:
            d = fday[cfidx[r]] - day0
            if 0 <= d < nd:
                cnt[k, d] += 1
    return cnt


def main():
    mini = CACHE / 'mini'
    L_ = lambda n: np.load(mini / f'{n}.npy')
    hold, crow, hrow, Mh, Hh = L_('hold'), L_('crow'), L_('hrow'), L_('M'), L_('H')
    n_rows = len(crow); feed_of_row = np.repeat(np.arange(len(hold)), Mh)
    c_item = L_('c_item').astype(np.int64); f4 = L_('c_f')[:, 3]
    ftr = feeds('train'); nF = len(ftr)
    fts = ftr['timestamp'].to_numpy().astype(np.int64); fuser = ftr['user_id'].to_numpy().astype(np.int64)
    fday = day_of(fts).astype(np.int32); fH = ftr['H'].to_numpy().astype(np.int64); fho = ftr['hist_off'].to_numpy().astype(np.int64)
    fsp = sparse_flag(ftr['feed_id'].to_numpy())
    assert (np.diff(fts) >= 0).all() and (fho[1:] == fho[:-1] + fH[:-1]).all()
    # next feed of the same non-zero user in (timestamp, fidx) order -> validity interval of a de-duplicated source sequence (sources.py)
    fnd = np.full(nF, BIG, np.int32)
    o = np.lexsort((np.arange(nF), fts, fuser))
    same = (fuser[o][1:] == fuser[o][:-1]) & (fuser[o][:-1] != 0)
    fnd[o[:-1][same]] = fday[o[1:][same]]
    lg(f'train feeds {nF}; sparse-sample feeds {int(fsp.sum())}; holdout rows {n_rows}; multi-feed links {int(same.sum())}')
    # ---------------- test lag distribution (label-free) and lag assignment
    fte = feeds('test'); tts = fte['timestamp'].to_numpy().astype(np.int64); tL = day_of(tts) - DAYC; tpub = tts < PUB_TS
    fva = feeds('val'); vL = day_of(fva['timestamp'].to_numpy()) - DAYC
    lg('val lag (days from CUT) counts', dict(zip(*[x.tolist() for x in np.unique(vL, return_counts=True)])))
    dist = {}
    for nm, sel in (('pub', tpub), ('priv', ~tpub)):
        u, c = np.unique(tL[sel], return_counts=True); dist[nm] = (u, c)
        lg(f'test {nm}: feeds {int(sel.sum())}; lag counts {dict(zip(u.tolist(), c.tolist()))}; mean t - CUT {float(((tts[sel] - CUT_VT) / DAY).mean()):.2f} d')
    hfid = ftr['feed_id'].to_numpy()[hold]; ht = fts[hold]; hu = fuser[hold]; hday = fday[hold].astype(np.int64)
    uni = ((hfid.astype(np.uint64) * H_LAG) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    kn = knots(); age_h = age_hours(f4, kn)                      # NaN = capped
    t_row = ht[feed_of_row]; u_row = hu[feed_of_row]; own_row = hold[feed_of_row].astype(np.int64)
    views = {}
    for nm in ('pub', 'priv'):
        u, c = dist[nm]; cdf = np.cumsum(c) / c.sum()
        Lf = u[np.minimum(np.searchsorted(cdf, uni, side='right'), len(u) - 1)].astype(np.int64)
        cf = (hday - Lf) * DAY - MSK
        E = cf[feed_of_row] - (t_row - 3600.0 * age_h)          # NaN for capped
        w1 = np.minimum(CUT_VT, np.maximum(cf[feed_of_row], DATA0 + np.clip(np.nan_to_num(E, nan=1e12), 0, 14 * DAY)))
        D = np.floor((w1 + MSK + DAY / 2) / DAY).astype(np.int32)
        A = np.maximum((hday - Lf)[feed_of_row], D).astype(np.int32)
        views[nm] = dict(L=Lf, D=D, A=A, asb=False)
        for k_, v_ in (('L', Lf), ('D', D), ('A', A)):
            np.save(CACHE / f'view_{nm}_{k_}.npy', v_)
        dd = np.bincount(D - DAY0, minlength=15)
        nsd = np.array([sum(1 for o_ in SPARSE_PAT if ((DAY0 <= a + o_ <= 20710) or (20713 <= a + o_ <= 20715) or (a + o_ in (20718, 20720, 20722, 20723, 20724, 20725)))) for a in (hday - Lf)])
        lg(f'view {nm}: holdout lag counts {dict(zip(*[x.tolist() for x in np.unique(Lf, return_counts=True)]))}; rows by dense cutoff day D (Sep 6 .. Sep 20) share '
           f'{np.round(dd / dd.sum(), 3).tolist()}; rows with D = c_f day {float((D == (hday - Lf)[feed_of_row]).mean()):.3f}, D = Sep 20 {float((D == DAYC).mean()):.3f}, '
           f'empty dense (D <= Sep 6) {float((D <= DAY0).mean()):.3f}; sparse days with data for young rows (of 6), feed mean {nsd.mean():.2f}, dist {np.bincount(nsd, minlength=7).tolist()}')
    views['chk'] = dict(L=np.zeros(len(hold), np.int64), D=hday[feed_of_row].astype(np.int32), A=np.full(n_rows, BIG, np.int32), asb=True)
    # ---------------- postings of the needed items
    hi = np.load(FLAT / 'train/hist_item_id.npy')
    NI = int(max(hi.max(), c_item.max())) + 1
    uit = np.unique(c_item); nidx = np.full(NI, -1, np.int32); nidx[uit] = np.arange(len(uit), dtype=np.int32)
    ck = nidx[c_item].astype(np.int64)
    start, p_ev = build_post(hi, nidx, len(uit))
    lg(f'needed items {len(uit)}; train postings {len(p_ev)} of {len(hi)} history events')
    ev2f = np.repeat(np.arange(nF, dtype=np.int32), fH)
    p_feed = ev2f[p_ev]; del ev2f
    p_pos = (p_ev - fho[p_feed]).astype(np.int16)
    het = np.load(FLAT / 'train/hist_event_type.npy', mmap_mode='r'); p_et = np.asarray(het)[p_ev].astype(np.int8); del het
    lg('posting arrays ready')
    vparts = []
    for sp in ('val', 'test'):
        f = feeds(sp); h = np.load(FLAT / sp / 'hist_item_id.npy'); H = f['H'].to_numpy().astype(np.int64)
        ff = np.repeat(np.arange(len(f)), H); kk = np.where(h < NI, nidx[np.minimum(h, NI - 1)], -1); m = kk >= 0
        vparts.append((kk[m], f['user_id'].to_numpy().astype(np.int64)[ff[m]], f['timestamp'].to_numpy().astype(np.int64)[ff[m]],
                       np.load(FLAT / sp / 'hist_event_type.npy')[m].astype(np.int8)))
    vk = np.concatenate([p[0] for p in vparts]); vo = np.argsort(vk, kind='stable')
    v_user = np.concatenate([p[1] for p in vparts])[vo]; v_ts = np.concatenate([p[2] for p in vparts])[vo]
    v_et = np.concatenate([p[3] for p in vparts])[vo]; v_day = day_of(v_ts).astype(np.int32)
    vstart = np.concatenate([[0], np.cumsum(np.bincount(vk, minlength=len(uit)))]).astype(np.int64)
    lg(f'val/test postings {len(vk)}')
    # ---------------- impressions by day (xp_imp_c50)
    ND = DAYC - DAY0 + 1
    cnt = imp_by_day(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r'), np.load(FLAT / 'train/cand_fidx.npy', mmap_mode='r'), fday, nidx, len(uit), DAY0, ND)
    cum = np.concatenate([np.zeros((len(uit), 1), np.int64), np.cumsum(cnt, 1, dtype=np.int64)], 1)   # cum[k, j] = impressions on days < DAY0 + j
    del cnt
    # own-feed multiplicity and same-user later impressions (holdout users' other train feeds)
    hu_nz = np.unique(hu[hu != 0])
    uf = np.flatnonzero(np.isin(fuser, hu_nz)); fco = ftr['cand_off'].to_numpy().astype(np.int64); fM = ftr['M'].to_numpy().astype(np.int64)
    urow = rows_of(fco[uf], fM[uf]); uff = np.repeat(uf, fM[uf])
    later = pl.DataFrame({'user': fuser[uff], 'item': np.asarray(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')[urow]).astype(np.int64),
                          'ts2': fts[uff], 'day2': fday[uff].astype(np.int64)})
    lg(f'impressions by day ready; holdout non-zero users {len(hu_nz)}, their train feeds {len(uf)}, cand rows {len(later)}')
    hq = dict(qfeed=hold.astype(np.int64), qoff=np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64), qM=Mh.astype(np.int64),
              qhoff=np.concatenate([[0], np.cumsum(Hh)[:-1]]).astype(np.int64), qH=Hh.astype(np.int64),
              qhitem=np.asarray(np.load(FLAT / 'train/hist_item_id.npy', mmap_mode='r')[hrow]).astype(np.int32),
              qhpos=np.asarray(np.load(FLAT / 'train/hist_hpos.npy', mmap_mode='r')[hrow]).astype(np.int64))
    stored = {n: np.asarray(np.load(FC / 'train' / f'{n}.npy', mmap_mode='r')[crow]) for n in LAGCOLS}
    for nm in ('chk', 'pub', 'priv'):
        v = views[nm]; D = v['D']; A = v['A']
        # --- it_hist LUO
        o5 = np.zeros((n_rows, 5), np.int64)
        luo_kernel(ck, own_row, t_row, u_row, D, A, v['asb'], start, p_feed, p_et, fuser, fts, fday, fsp, vstart, v_user, v_ts, v_day, v_et, PAT, o5)
        lg(f'[{nm}] it_hist counted')
        # --- seq_cv
        o2 = np.zeros((n_rows, 2), np.float32)
        covis_kernel(hq['qfeed'], hq['qoff'], hq['qM'], hq['qhoff'], hq['qH'], hq['qhitem'], hq['qhpos'], ck, D, A, c_item.astype(np.int32),
                     start, p_feed, p_pos, hi, fho, fH, fuser, fts, fday, fnd, fsp, PAT, o2)
        norm = (o2[:, 0] / np.sqrt(o2[:, 1] + np.float32(1.0))).astype(np.float32)
        lg(f'[{nm}] seq_cv done')
        # --- xp_imp_c50
        n_imp = cum[ck, np.clip(D.astype(np.int64) - DAY0, 0, ND)]
        ownday = fday[own_row] < D
        rows_df = pl.DataFrame({'row': np.arange(n_rows), 'user': u_row, 'item': c_item, 't': t_row, 'D': D.astype(np.int64), 'own': own_row})
        j = (rows_df.lazy().filter(pl.col('user') != 0).join(later.lazy(), on=['user', 'item'], how='inner')
             .filter((pl.col('day2') < pl.col('D')) & (pl.col('ts2') >= pl.col('t'))).group_by('row').agg(pl.len().alias('n')).collect())
        sub = np.zeros(n_rows, np.int64); sub[j['row'].to_numpy()] = j['n'].to_numpy()     # own feed (ts2 == t) + later feeds of the same non-zero user
        own0 = ownday & (u_row == 0)                                                        # user 0: only the own row can be identified
        n_adj = n_imp - sub - own0.astype(np.int64)
        assert n_adj.min() >= 0, int(n_adj.min())
        xp = np.log1p(np.minimum(n_adj, 50)).astype(np.float32)
        cols = dict(zip(LUO, [o5[:, i].astype(np.float32) for i in range(5)]))
        cols.update(seq_cv_invd20=o2[:, 0], seq_cv_pop=o2[:, 1], seq_cv_norm=norm, xp_imp_c50=xp)
        if nm == 'chk':
            for n in LAGCOLS:
                a, b = cols[n].astype(np.float64), stored[n].astype(np.float64)
                d = np.abs(a - b)
                lg(f'CODE CHECK lag-0 semantics vs stored {n}: exact rows {float((a == b).mean()):.6f}, max |diff| {d.max():.6g}, rows |diff| > 1e-4 rel {int((d > 1e-4 * (1 + np.abs(b))).sum())}')
        else:
            for n in LAGCOLS:
                np.save(CACHE / f'lag_{nm}_{n}.npy', cols[n])
                a, b = cols[n], stored[n]
                lg(f'[{nm}] {n}: zero share {float((a == 0).mean()):.4f} (stored {float((b == 0).mean()):.4f}); mean log1p {float(np.log1p(np.maximum(a, 0)).mean()):.4f} '
                   f'(stored {float(np.log1p(np.maximum(b, 0)).mean()):.4f}); rows changed {float((a != b).mean()):.4f}; rows above stored {float((a > b + 1e-6).mean()):.4f}')
    lg('done')


if __name__ == '__main__':
    main()
