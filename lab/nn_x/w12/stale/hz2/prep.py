"""hz2 PREP: global feed table, item compaction, chronological postings, per-day count matrices. Label-free (no target file is opened).
Writes cache/*.npy. CPU only, 12 threads.  Usage: CUDA_VISIBLE_DEVICES="" python prep.py
"""
from hz2_common import *
import numba as nb
nb.set_num_threads(12)


@nb.njit(cache=True)
def count_sort(hi, nidx, n_items):
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
            p_ev[fill[k]] = e
            fill[k] += 1
    return start, p_ev


@nb.njit(parallel=True, cache=True)
def day_mats(start, p_feed, p_et, g_slot, g_sp, is_tr, DM, SM, dcum, scnt, ovf):
    """DM[k, slot<15, e] over train snapshots; SM[k, slot<24, e] over sparse-source snapshots (train 1/37 sample, val, test); e = 0 all, 1..4 event type.
    dcum[k, s] = # train postings of item k on slots < s; scnt[k, s] = # sparse-source postings on slot s."""
    n = len(start) - 1
    for k in nb.prange(n):
        d = np.zeros((15, 5), np.int32)
        s = np.zeros((24, 5), np.int32)
        for e in range(start[k], start[k + 1]):
            f = p_feed[e]
            sl = g_slot[f]
            et = p_et[e]
            if is_tr[f]:
                d[sl, 0] += 1
                if 1 <= et <= 4:
                    d[sl, et] += 1
            if g_sp[f]:
                s[sl, 0] += 1
                if 1 <= et <= 4:
                    s[sl, et] += 1
        c = 0
        for i in range(15):
            dcum[k, i] = c
            c += d[i, 0]
            for j in range(5):
                if d[i, j] > 32767:
                    ovf[0] += 1
                DM[k, i, j] = d[i, j]
        dcum[k, 15] = c
        for i in range(24):
            scnt[k, i] = s[i, 0]
            for j in range(5):
                if s[i, j] > 32767:
                    ovf[1] += 1
                SM[k, i, j] = s[i, j]


@nb.njit(cache=True)
def imp_by_day(citem, cfidx, fslot, nidx, n_items):
    cnt = np.zeros((n_items, 15), np.int32)
    for r in range(len(citem)):
        cnt[nidx[citem[r]], fslot[cfidx[r]]] += 1
    return cnt


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    F = {s: feeds(s) for s in ('train', 'val', 'test')}
    nF = {s: len(F[s]) for s in F}
    # ---------------- global feed table (train, val, test in this order = chronological)
    ts = np.concatenate([F[s]['timestamp'].to_numpy().astype(np.int64) for s in F])
    user = np.concatenate([F[s]['user_id'].to_numpy().astype(np.int64) for s in F])
    H = np.concatenate([F[s]['H'].to_numpy().astype(np.int64) for s in F])
    nH = {s: int(F[s]['H'].sum()) for s in F}
    hoff = np.concatenate([F['train']['hist_off'].to_numpy().astype(np.int64), F['val']['hist_off'].to_numpy().astype(np.int64) + nH['train'],
                           F['test']['hist_off'].to_numpy().astype(np.int64) + nH['train'] + nH['val']])
    assert (hoff[1:] == hoff[:-1] + H[:-1]).all(), 'history offsets are not contiguous in feed order'
    slot = (day_of(ts) - DAY0).astype(np.int16)
    assert slot.min() >= 0 and slot.max() < NSLOT
    assert (np.diff(slot.astype(np.int64)) >= 0).all(), 'feeds are not in day order (the per-day posting ranges need it)'
    assert (np.diff(ts[:nF['train']]) >= 0).all()
    is_tr = np.zeros(len(ts), np.bool_); is_tr[:nF['train']] = True
    assert slot[:nF['train']].max() < NTR
    g_sp = np.ones(len(ts), np.bool_); g_sp[:nF['train']] = sparse_flag(F['train']['feed_id'].to_numpy())
    lg(f'global feeds {len(ts)} (train {nF["train"]}, val {nF["val"]}, test {nF["test"]}); sparse-source feeds {int(g_sp.sum())} (train sample {int(g_sp[:nF["train"]].sum())})')
    for n, a in (('g_ts', ts), ('g_user', user), ('g_H', H), ('g_hoff', hoff), ('g_slot', slot), ('g_istr', is_tr), ('g_sp', g_sp)):
        np.save(CACHE / f'{n}.npy', a)
    # ---------------- same-user structures
    nG = len(ts)
    o = np.lexsort((np.arange(nG), ts, user))
    uo, to = user[o], ts[o]
    # later_mask (train feeds only; dedup of the seq_cv source): slots of the later train feeds of the same non-zero user in (ts, fidx) order
    later_mask = np.zeros(nG, np.uint16)
    run_start = np.flatnonzero(np.r_[True, uo[1:] != uo[:-1]])
    run_len = np.diff(np.r_[run_start, nG])
    multi = np.flatnonzero((run_len >= 2) & (uo[run_start] != 0))
    lg(f'non-zero users with >= 2 feeds (all splits): {len(multi)}; feeds of such users {int(run_len[multi].sum())}')
    lat_cnt = np.zeros(nG, np.int64)
    lat_pairs_a, lat_pairs_b = [], []
    for rs, rl in zip(run_start[multi].tolist(), run_len[multi].tolist()):
        idx = o[rs:rs + rl]
        tt = to[rs:rs + rl]
        acc = 0
        for j in range(rl - 1, -1, -1):          # suffix OR of train slots
            later_mask[idx[j]] = acc
            if is_tr[idx[j]]:
                acc |= (1 << int(slot[idx[j]]))
        for j in range(rl - 1):                    # later = strictly larger timestamp (stored LUO rule th > tc)
            k = np.searchsorted(tt, tt[j], side='right')
            if k < rl:
                lat_pairs_a.append(np.full(rl - k, idx[j], np.int64)); lat_pairs_b.append(idx[k:])
    a = np.concatenate(lat_pairs_a) if lat_pairs_a else np.zeros(0, np.int64)
    b = np.concatenate(lat_pairs_b) if lat_pairs_b else np.zeros(0, np.int64)
    oo = np.argsort(a, kind='stable')
    a, b = a[oo], b[oo]
    lat_start = np.zeros(nG + 1, np.int64)
    lat_cnt = np.bincount(a, minlength=nG).astype(np.int64)
    lat_start[1:] = np.cumsum(lat_cnt)
    np.save(CACHE / 'later_mask.npy', later_mask); np.save(CACHE / 'lat_start.npy', lat_start); np.save(CACHE / 'lat_list.npy', b.astype(np.int32))
    lg(f'later same-user pairs {len(b)}; feeds with a later train feed of the same user {int((later_mask != 0).sum())}')
    del o, uo, to
    # ---------------- concatenated histories
    hitem = np.concatenate([np.load(FLAT / s / 'hist_item_id.npy') for s in F]).astype(np.int32)
    het = np.concatenate([np.load(FLAT / s / 'hist_event_type.npy') for s in F]).astype(np.int8)
    hpos = np.concatenate([np.load(FLAT / s / 'hist_hpos.npy') for s in F]).astype(np.int16)
    assert len(hitem) == int(H.sum())
    np.save(CACHE / 'hitem.npy', hitem); np.save(CACHE / 'het.npy', het); np.save(CACHE / 'hpos.npy', hpos)
    del hpos
    lg(f'history events {len(hitem)}')
    # ---------------- item compaction: every candidate item of train / val / test
    NI = int(max(int(hitem.max()), max(int(np.load(FLAT / s / 'cand_item_id.npy', mmap_mode='r').max()) for s in F))) + 1
    isc = np.zeros(NI, np.bool_)
    for s in F:
        isc[np.asarray(np.load(FLAT / s / 'cand_item_id.npy', mmap_mode='r'))] = True
    n_it = int(isc.sum())
    nidx = np.full(NI, -1, np.int32); nidx[isc] = np.arange(n_it, dtype=np.int32)
    np.save(CACHE / 'nidx.npy', nidx)
    lg(f'candidate items (train + val + test) {n_it}; id space {NI}')
    # ---------------- postings (chronological inside each item)
    start, p_ev = count_sort(hitem, nidx, n_it)
    lg(f'postings {len(p_ev)} of {len(hitem)} events')
    ev2f = np.repeat(np.arange(nG, dtype=np.int32), H)
    p_feed = ev2f[p_ev]; del ev2f
    p_pos = (p_ev - hoff[p_feed]).astype(np.int16)
    p_et = het[p_ev]
    del p_ev
    np.save(CACHE / 'p_start.npy', start); np.save(CACHE / 'p_feed.npy', p_feed); np.save(CACHE / 'p_pos.npy', p_pos)
    DM = np.zeros((n_it, 15, 5), np.int16); SM = np.zeros((n_it, 24, 5), np.int16)
    dcum = np.zeros((n_it, 16), np.int32); scnt = np.zeros((n_it, 24), np.int32); ovf = np.zeros(2, np.int64)
    day_mats(start, p_feed, p_et, slot, g_sp, is_tr, DM, SM, dcum, scnt, ovf)
    assert ovf.sum() == 0, f'int16 overflow in the day matrices: {ovf}'
    np.save(CACHE / 'DM.npy', DM); np.save(CACHE / 'SM.npy', SM); np.save(CACHE / 'dcum.npy', dcum)
    lg(f'day matrices: dense occurrences {int(DM[:, :, 0].sum(dtype=np.int64))}, sparse-source occurrences {int(SM[:, :, 0].sum(dtype=np.int64))}')
    # sparse-source postings (separate list) + per-day offsets
    msk = g_sp[p_feed]
    s_feed = p_feed[msk]; s_pos = p_pos[msk]
    scum = np.zeros((n_it, 25), np.int32); scum[:, 1:] = np.cumsum(scnt, 1)
    s_start = np.zeros(n_it + 1, np.int64); s_start[1:] = np.cumsum(scum[:, 24].astype(np.int64))
    assert s_start[-1] == len(s_feed)
    np.save(CACHE / 's_start.npy', s_start); np.save(CACHE / 's_feed.npy', s_feed); np.save(CACHE / 's_pos.npy', s_pos); np.save(CACHE / 'scum.npy', scum)
    lg(f'sparse-source postings {len(s_feed)}')
    del DM, SM, dcum, scnt, p_feed, p_pos, p_et, msk, s_feed, s_pos, scum
    # ---------------- train impressions by day (xp_imp_c50)
    IM = imp_by_day(np.asarray(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')), np.asarray(np.load(FLAT / 'train/cand_fidx.npy', mmap_mode='r')),
                    slot[:nF['train']], nidx, n_it)
    np.save(CACHE / 'IM.npy', IM)
    lg(f'impressions by day: {int(IM.sum(dtype=np.int64))} (train candidate rows {len(np.load(FLAT / "train/cand_fidx.npy", mmap_mode="r"))})')
    lg('PREP DONE')


if __name__ == '__main__':
    main()
