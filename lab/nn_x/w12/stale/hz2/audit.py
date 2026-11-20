"""hz2 code check V3: CAUSALITY AUDIT + independent recount (plain numpy / python sets; uses NO cache file of prep.py and NO kernel of kernels.py).
Reads only data/flat/train, the per-feed lag file of the view and the view's column files. Label-free.
For a random sample of train feeds (equal numbers per day group) every counted (row, snapshot) pair of the hz2 view is listed from the rule text of
DECLARATION.txt; reported: share of pairs dated after the row's day (must be 0), pairs of the same non-zero user with a later timestamp (must be 0), distance of
the dense days from the row, and the agreement of the recount with the view files (all nine columns).
Usage: python audit.py <view dir (feats | feats_L)> <m> [feeds per day group = 60] [seq feeds per day group = 20]
"""
import sys
from hz2_common import *

VDIR = HERE / sys.argv[1]
MPAR = int(sys.argv[2])
NPER = int(sys.argv[3]) if len(sys.argv) > 3 else 60
NSEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 20
out_lines = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    out_lines.append(s)
    print(f'[{time.time() - T0:5.0f}s] ' + s, flush=True)


def main():
    ftr = feeds('train'); nF = len(ftr)
    ts = ftr['timestamp'].to_numpy().astype(np.int64); user = ftr['user_id'].to_numpy().astype(np.int64)
    M = ftr['M'].to_numpy().astype(np.int64); coff = ftr['cand_off'].to_numpy().astype(np.int64)
    H = ftr['H'].to_numpy().astype(np.int64); hoff = ftr['hist_off'].to_numpy().astype(np.int64)
    slot = (day_of(ts) - DAY0).astype(np.int64)
    spf = sparse_flag(ftr['feed_id'].to_numpy())
    L = np.load(VDIR / 'train' / 'hz2_lag.npy').astype(np.int64)
    part_pub, L2, _, _ = assign_lags(ftr['feed_id'].to_numpy())
    assert np.array_equal(L, L2) and np.array_equal(part_pub, np.load(VDIR / 'train' / 'hz2_part_pub.npy'))
    rng = np.random.default_rng(3)
    groups = [np.flatnonzero(slot <= 2), np.flatnonzero((slot >= 3) & (slot <= 6)), np.flatnonzero((slot >= 7) & (slot <= 9)), np.flatnonzero(slot >= 11)]
    sel = np.sort(np.concatenate([rng.choice(g, NPER, replace=False) for g in groups]))
    seq_sel = set(np.concatenate([rng.choice(sel[np.isin(sel, g)], NSEQ, replace=False) for g in groups]).tolist())
    crow = np.concatenate([np.arange(coff[f], coff[f] + M[f]) for f in sel])
    frow = np.repeat(sel, M[sel])
    citem = np.asarray(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')[crow]).astype(np.int64)
    f4 = np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[crow])
    age_h = age_hours(f4, knots())
    pday = np.where(np.isnan(age_h), -30000, np.floor((ts[frow] - 3600.0 * np.nan_to_num(age_h) + MSK) / DAY) - DAY0).astype(np.int64)
    P(f'sample: {len(sel)} train feeds ({NPER} per day group), {len(crow)} rows; seq recount on {len(seq_sel)} of them; view {VDIR.name}, m = {MPAR}')
    # ---- all train history events and impressions of the sampled items
    NI = 111319575 + 1
    need = np.zeros(NI, np.bool_); need[citem] = True
    hi = np.load(FLAT / 'train/hist_item_id.npy', mmap_mode='r')
    ev = np.flatnonzero(need[np.asarray(hi)])
    ev_item = np.asarray(hi[ev]).astype(np.int64)
    ev_feed = np.asarray(np.load(FLAT / 'train/hist_fidx.npy', mmap_mode='r')[ev]).astype(np.int64)
    ev_et = np.asarray(np.load(FLAT / 'train/hist_event_type.npy', mmap_mode='r')[ev]).astype(np.int64)
    ev_pos = ev - hoff[ev_feed]
    o = np.argsort(ev_item, kind='stable')
    ev_item, ev_feed, ev_et, ev_pos = ev_item[o], ev_feed[o], ev_et[o], ev_pos[o]
    P(f'history events of the sampled items: {len(ev)}')
    ci_all = np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')
    ir = np.flatnonzero(need[np.asarray(ci_all)])
    im_item = np.asarray(ci_all[ir]).astype(np.int64)
    im_slot = slot[np.asarray(np.load(FLAT / 'train/cand_fidx.npy', mmap_mode='r')[ir])]
    o = np.argsort(im_item, kind='stable'); im_item, im_slot = im_item[o], im_slot[o]
    P(f'train impressions of the sampled items: {len(ir)}')
    # same-user feed lists (train)
    ou = np.lexsort((np.arange(nF), ts, user))
    rank = np.empty(nF, np.int64); rank[ou] = np.arange(nF)
    hist_item = hi
    hpos_arr = np.load(FLAT / 'train/hist_hpos.npy', mmap_mode='r')
    dense_days = [int(x) for x in DENSE_SLOTS]

    def later_feeds(s):
        res = []
        if user[s] == 0:
            return res
        k = rank[s] + 1
        while k < nF and user[ou[k]] == user[s]:
            res.append(int(ou[k])); k += 1
        return res

    cols = {n: np.asarray(np.load(VDIR / 'train' / f'{HZ2_NAME[n]}.npy', mmap_mode='r')[crow]) for n in COLS9}
    rec = {n: np.zeros(len(crow)) for n in COLS9}
    seq_rows = np.zeros(len(crow), np.bool_)
    n_pairs = n_after_day = n_same_user_later = n_dense = n_sparse = n_own = n_sparse_later_ts = n_same_user_any = 0
    dist_hist = np.zeros(20, np.int64)
    n_trunc = n_exact = 0
    for r in range(len(crow)):
        f = int(frow[r]); d = int(slot[f]); Lf = int(L[f]); c = d - Lf; t = int(ts[f]); u = int(user[f]); p = int(pday[r]); it = int(citem[r])
        S = {c + int(o_) for o_ in PAT if 0 <= c + int(o_) <= d}
        T = [c + int(o_) for o_ in DPAT if c + int(o_) >= p]
        A = [s for s in T if s >= 0]
        if len(A) == len(T):
            W = [c + int(o_) for o_ in DPAT if c + int(o_) >= 0]; rf = 1.0; n_exact += 1
        else:
            cr = min(d, max(c, MPAR))
            cand = [s for s in dense_days if s < cr]
            W = cand[:min(len(T), len(cand))]
            rf = min(len(T) / len(W), 4.0) if W else 1.0
            n_trunc += 1
        Wset = set(W); Sx = S - Wset
        a, b = np.searchsorted(ev_item, it), np.searchsorted(ev_item, it, side='right')
        acc = np.zeros(5); own_counted = False
        for e in range(a, b):
            s = int(ev_feed[e]); ss = int(slot[s]); w = 0.0
            if ss in Wset:
                w = rf; kind = 'd'
            elif spf[s] and ss in Sx:
                w = 1.0; kind = 's'
            elif s == f:
                w = 1.0; kind = 'o'
            if w == 0.0:
                continue
            if u != 0 and user[s] == u and ts[s] > t:
                continue                                    # later snapshot of the same user: never counted
            if s == f:
                own_counted = True
            n_pairs += 1
            n_after_day += ss > d
            n_same_user_later += (u != 0 and user[s] == u and ts[s] > t)
            n_same_user_any += (u != 0 and user[s] == u)
            if kind == 'd':
                n_dense += 1; dist_hist[min(d - ss, 19)] += 1
            elif kind == 's':
                n_sparse += 1; n_sparse_later_ts += ts[s] > t
            else:
                n_own += 1
            acc[0] += w
            if 1 <= ev_et[e] <= 4:
                acc[ev_et[e]] += w
        for i_, n in enumerate(LUO):
            rec[n][r] = acc[i_]
        a2, b2 = np.searchsorted(im_item, it), np.searchsorted(im_item, it, side='right')
        sl = im_slot[a2:b2]
        nimp = float(np.isin(sl, W).sum()) * rf if W else 0.0
        assert not (len(sl) and W and max(W) >= d)
        rec['xp_imp_c50'][r] = np.log1p(min(nimp, 50.0))
        if f in seq_sel:
            seq_rows[r] = True
            hs = np.asarray(hist_item[hoff[f]:hoff[f] + H[f]]).astype(np.int64)
            hw = 0.95 ** np.asarray(hpos_arr[hoff[f]:hoff[f] + H[f]]).astype(np.float64)
            wmap = {}
            for x_, w_ in zip(hs.tolist(), hw.tolist()):
                if w_ > wmap.get(x_, -1.0):
                    wmap[x_] = w_
            mem = []
            if len(hs):
                for e in range(a, b):
                    s = int(ev_feed[e]); ss = int(slot[s])
                    if ss in Wset:
                        if any(int(slot[g]) in Wset for g in later_feeds(s)):
                            continue                        # not the user's latest feed among the dense days
                        mem.append((e, False))
                    elif spf[s] and ss in Sx:
                        mem.append((e, True))
            mem = mem[-3000:]
            dinv = dpop = sinv = spop = 0.0
            for e, is_sp in mem:
                s = int(ev_feed[e])
                if u != 0 and user[s] == u:
                    continue
                if is_sp and ts[s] >= t:
                    continue
                assert slot[s] <= d
                pp = int(ev_pos[e]); seq = np.asarray(hist_item[hoff[s]:hoff[s] + H[s]]).astype(np.int64)
                acc2 = 0.0
                for qq in range(max(0, pp - 20), min(len(seq), pp + 21)):
                    if qq == pp or seq[qq] == it:
                        continue
                    w_ = wmap.get(int(seq[qq]))
                    if w_ is not None:
                        acc2 += w_ / abs(qq - pp)
                if is_sp:
                    spop += 1; sinv += acc2
                else:
                    dpop += 1; dinv += acc2
            pop = min(rf * dpop + spop, 3000.0); invd = rf * dinv + sinv
            rec['seq_cv_pop'][r] = pop; rec['seq_cv_invd20'][r] = invd; rec['seq_cv_norm'][r] = invd / np.sqrt(pop + 1.0)
    P(f'rows: exact {n_exact}, truncated {n_trunc}')
    P(f'it_hist counted (row, snapshot) pairs: {n_pairs} = dense {n_dense} + sparse {n_sparse} + own snapshot {n_own}')
    P(f'  pairs with snapshot day AFTER the row day: {n_after_day} (share {n_after_day / max(n_pairs, 1):.6f})  [must be 0]')
    P(f'  pairs of the same non-zero user with a later timestamp: {n_same_user_later}  [must be 0]')
    P(f'  pairs from ANY snapshot of the row\'s own non-zero user (own snapshot or earlier ones): {n_same_user_any} (share {n_same_user_any / max(n_pairs, 1):.6f})')
    P(f'  sparse pairs of other users later on the SAME day (allowed by the rule, ts > row ts): {n_sparse_later_ts} ({n_sparse_later_ts / max(n_sparse, 1):.4f} of sparse pairs)')
    P('  dense pairs by distance row day - snapshot day (days 1..13): ' + ' '.join(f'{int(x)}' for x in dist_hist[1:14]) + f'; distance 0: {int(dist_hist[0])} [must be 0]')
    ok = n_after_day == 0 and n_same_user_later == 0 and dist_hist[0] == 0
    for n in COLS9:
        mk = seq_rows if n in SEQ else np.ones(len(crow), np.bool_)
        a_, b_ = rec[n][mk], cols[n][mk].astype(np.float64)
        rel = np.abs(a_ - b_) / (1 + np.abs(b_))
        good = bool(rel.max() <= 1e-4) if mk.any() else True
        ok = ok and good
        P(f'  recount vs view file {HZ2_NAME[n]:17s}: rows {int(mk.sum())}, exact {float((a_ == b_).mean()):.4f}, max rel diff {rel.max():.3g}, non-zero rows {int((b_ != 0).sum())} -> {"ok" if good else "MISMATCH"}')
    P(f'V3 {"PASS" if ok else "FAIL"}')
    open(LOGS / f'audit_{VDIR.name}.txt', 'w').write('\n'.join(out_lines) + '\n')


if __name__ == '__main__':
    main()
