"""5) Co-visitation scores from train histories.
Source sequences: latest train feed per user (user 0: each feed separately) -> 2.58M sequences, 456M events.
For a target feed (user u, history items S with recency weight 0.95^hpos), candidate c:
  for each occurrence of c in a source sequence of a DIFFERENT user, scan neighbours within window W in that sequence,
  and accumulate when the neighbour item is in S.
Outputs several variants + candidate popularity in train histories.
"""
import sys
import time
import numpy as np
import polars as pl
import numba as nb
from common import feeds, arr, load_split, eval_score, ndcg_per_feed

nb.set_num_threads(40)
OUT = '<repo>/eda/history_leak/'
COLS = ['cv_raw5', 'cv_rec5', 'cv_fwd5', 'cv_invd20', 'cv_raw20', 'pop_hist', 'cv_users5', 'cv_rec5_norm',
        'cv_fwd5_t1', 'cv_bwd5']


def build_source(extra_splits=()):
    t0 = time.time()
    f = feeds('train')
    lat = f.filter(pl.col('user_id') != 0).sort(['timestamp', 'fidx']).group_by('user_id').agg(pl.col('fidx').last())
    u0 = f.filter(pl.col('user_id') == 0).select(['user_id', 'fidx'])
    sel = np.sort(np.concatenate([lat['fidx'].to_numpy(), u0['fidx'].to_numpy()]))
    F = f.sort('fidx')
    hoff = F['hist_off'].to_numpy()[sel]; H = F['H'].to_numpy()[sel]; users = F['user_id'].to_numpy()[sel]
    hi = np.asarray(arr('train', 'hist_item_id')); ht = np.asarray(arr('train', 'hist_event_type'))
    idx = np.concatenate([np.arange(o, o + h) for o, h in zip(hoff, H)])
    items = [hi[idx]]; types = [ht[idx]]; lens = [H]; us = [users]
    del hi, ht
    for s in extra_splits:
        fs = feeds(s).sort('fidx')
        items.append(np.asarray(arr(s, 'hist_item_id'))); types.append(np.asarray(arr(s, 'hist_event_type')))
        lens.append(fs['H'].to_numpy()); us.append(fs['user_id'].to_numpy())
    items = np.concatenate(items); types = np.concatenate(types); lens = np.concatenate(lens).astype(np.int64)
    users = np.concatenate(us)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    seq = np.repeat(np.arange(len(lens), dtype=np.int32), lens)
    pos = (np.arange(len(items)) - np.repeat(starts, lens)).astype(np.int16)
    print(f'source: {len(lens)} sequences, {len(items)} events ({time.time()-t0:.0f}s)', flush=True)
    order = np.argsort(items, kind='stable')
    sitems = items[order]
    uitems, ustart, ucnt = np.unique(sitems, return_index=True, return_counts=True)
    p_seq = seq[order]; p_pos = pos[order]
    del order, sitems, seq, pos
    print(f'postings built: {len(uitems)} unique items ({time.time()-t0:.0f}s)', flush=True)
    return dict(items=items, types=types, starts=starts, lens=lens, users=users, uitems=uitems,
                ustart=ustart.astype(np.int64), ucnt=ucnt.astype(np.int64), p_seq=p_seq, p_pos=p_pos)


@nb.njit(parallel=True, cache=True)
def covis_kernel(sel, coff, M, hoff, H, fuser, citem, hitem, hpos, items, types, starts, lens, susers,
                 uitems, ustart, ucnt, p_seq, p_pos, out_off, out):
    for q in nb.prange(len(sel)):
        fi = sel[q]
        h = H[fi]; ho = hoff[fi]; u = fuser[fi]
        # unique history items with recency weight of most recent occurrence
        hs = np.empty(h, np.int32); hw = np.empty(h, np.float64)
        for j in range(h):
            hs[j] = hitem[ho + j]
        order = np.argsort(hs)
        shs = hs[order]
        shw = np.empty(h, np.float64)
        for j in range(h):
            shw[j] = 0.95 ** hpos[ho + order[j]]
        # dedup keep max weight
        nu = 0
        uh = np.empty(h, np.int32); uw = np.empty(h, np.float64)
        for j in range(h):
            if nu > 0 and uh[nu - 1] == shs[j]:
                if shw[j] > uw[nu - 1]:
                    uw[nu - 1] = shw[j]
            else:
                uh[nu] = shs[j]; uw[nu] = shw[j]; nu += 1
        uh = uh[:nu]; uw = uw[:nu]
        co = coff[fi]; m = M[fi]; oo = out_off[q]
        for i in range(m):
            c = citem[co + i]
            r = oo + i
            k = np.searchsorted(uitems, c)
            if k >= len(uitems) or uitems[k] != c:
                continue
            ps = ustart[k]; pc = ucnt[k]
            out[r, 5] = pc
            nusers = 0.0
            for e in range(ps, ps + pc):
                s = p_seq[e]
                if u != 0 and susers[s] == u:
                    continue
                p = p_pos[e]; st = starts[s]; L = lens[s]
                lo = max(0, p - 20); hi_ = min(L, p + 21)
                hit = False
                for qq in range(lo, hi_):
                    if qq == p:
                        continue
                    x = items[st + qq]
                    if x == c:
                        continue
                    kk = np.searchsorted(uh, x)
                    if kk < nu and uh[kk] == x:
                        d = abs(qq - p)
                        w = uw[kk]
                        out[r, 3] += w / d
                        out[r, 4] += 1.0
                        if d <= 5:
                            hit = True
                            out[r, 0] += 1.0
                            out[r, 1] += w
                            if qq > p:   # neighbour older than c in source sequence -> c came after x
                                out[r, 2] += w
                                if types[st + p] == 1:
                                    out[r, 8] += w
                            else:
                                out[r, 9] += w
                if hit:
                    nusers += 1.0
            out[r, 6] = nusers
            out[r, 7] = out[r, 1] / (pc + 5.0)


def run(src, split, sel=None):
    f = feeds(split).sort('fidx')
    coff = f['cand_off'].to_numpy(); M = f['M'].to_numpy(); hoff = f['hist_off'].to_numpy(); H = f['H'].to_numpy()
    fuser = f['user_id'].to_numpy()
    if sel is None:
        sel = np.arange(len(f))
    sel = np.asarray(sel, np.int64)
    m = M[sel]
    out_off = np.concatenate([[0], np.cumsum(m)[:-1]]).astype(np.int64)
    out = np.zeros((m.sum(), len(COLS)), np.float32)
    citem = np.asarray(arr(split, 'cand_item_id')); hitem = np.asarray(arr(split, 'hist_item_id'))
    hpos = np.asarray(arr(split, 'hist_hpos'))
    t0 = time.time()
    covis_kernel(sel, coff, M, hoff, H, fuser, citem, hitem, hpos, src['items'], src['types'], src['starts'],
                 src['lens'], src['users'], src['uitems'], src['ustart'], src['ucnt'], src['p_seq'], src['p_pos'],
                 out_off, out)
    print(f'covis {split}: {len(sel)} feeds in {time.time()-t0:.0f}s', flush=True)
    return out


def evaluate(X, d, tag):
    t = d['c_target']; off = d['feeds']['cand_off'].to_numpy(); M = d['feeds']['M'].to_numpy()
    rng = np.random.default_rng(0)
    rnd = ndcg_per_feed(rng.random(len(t)), t, off, M)
    f1 = d['c_f1'].astype(np.float64)
    base = eval_score(f1, d, 'f1', verbose=True)
    print(f'--- {tag}: single features (random tie-break | pessimistic) ---')
    for j, n in enumerate(COLS):
        x = X[:, j].astype(np.float64)
        a = ndcg_per_feed(x + 1e-9 * rng.random(len(x)), t, off, M); b = ndcg_per_feed(x, t, off, M)
        print(f'{n:14s} {a.mean():.4f} | {b.mean():.4f}   nonzero frac {(x>0).mean():.3f}  '
              f'target rate nonzero {t[x>0].mean():.4f} zero {t[x==0].mean():.4f}')
    fidx = d['c_fidx']
    def prank(x):
        return pl.DataFrame({'f': fidx, 'x': x}).select((pl.col('x').rank('average').over('f') / pl.len().over('f')))['x'].to_numpy()
    rf1 = prank(f1)
    for n in ['cv_rec5', 'cv_fwd5', 'cv_invd20', 'cv_rec5_norm', 'pop_hist']:
        x = X[:, COLS.index(n)].astype(np.float64)
        rx = prank(x)
        for w in [0.25, 0.5, 1.0]:
            eval_score(rf1 + w * rx, d, f'rank(f1)+{w}*rank({n})', base)


def main():
    extra = tuple(sys.argv[1].split(',')) if len(sys.argv) > 1 and sys.argv[1] else ()
    src = build_source(extra)
    d = load_split('val', cand_cols=('item_id', 'category_id', 'f1', 'target'))
    X = run(src, 'val')
    tag = 'train' + ('+' + '+'.join(extra) if extra else '')
    np.save(OUT + f'out_covis_val_{tag}.npy', X)
    evaluate(X, d, tag)
    if not extra:
        Xt = run(src, 'test')
        np.save(OUT + 'out_covis_test_train.npy', Xt)
        print('test: frac candidates with cv_rec5>0', (Xt[:, 1] > 0).mean(), ' val:', (X[:, 1] > 0).mean())
        # train sample for LGB (same 200k sample as 04_affinity)
        ft = feeds('train')
        sel = np.sort(np.random.default_rng(1).choice(len(ft), 200_000, replace=False))
        Xtr = run(src, 'train', sel)
        np.save(OUT + 'out_covis_trainsample.npy', Xtr)


if __name__ == '__main__':
    main()
