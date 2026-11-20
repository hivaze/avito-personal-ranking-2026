"""Check: do target items of a feed appear in the history of a LATER feed of the same user (same split)?"""
from common import *
import json
res = {}
for split in ['val', 'train']:
    F = feeds(split).with_columns(pl.col('fidx').cast(pl.Int64))
    if split == 'train':
        F = F.filter(pl.col('timestamp') < F['timestamp'].min() + 3 * 86400)  # first 3 days only to keep it light
    F = F.sort(['user_id', 'timestamp'])
    F = F.with_columns(pl.col('fidx').shift(-1).over('user_id').alias('next_fidx'), pl.col('timestamp').shift(-1).over('user_id').alias('next_ts'))
    pairs = F.filter(pl.col('next_fidx').is_not_null() & (pl.col('next_ts') > pl.col('timestamp')))
    base = feeds(split)
    off = base['cand_off'].to_numpy(); M = base['M'].to_numpy(); hoff = base['hist_off'].to_numpy(); H = base['H'].to_numpy()
    item = cand(split, 'item_id'); t = cand(split, 'target'); hitem = np.load(f'{FLAT}/{split}/hist_item_id.npy', mmap_mode='r')
    hev = np.load(f'{FLAT}/{split}/hist_event_type.npy', mmap_mode='r')
    n_pos = n_pos_hit = n_neg = n_neg_hit = 0; feeds_with_hit = 0; ev_counts = {}
    dt = []
    for a, b, ta, tb in pairs.select('fidx', 'next_fidx', 'timestamp', 'next_ts').iter_rows():
        ci = np.asarray(item[off[a]:off[a] + M[a]]); ct = np.asarray(t[off[a]:off[a] + M[a]])
        hi = np.asarray(hitem[hoff[b]:hoff[b] + H[b]]); he = np.asarray(hev[hoff[b]:hoff[b] + H[b]])
        inh = np.isin(ci, hi)
        n_pos += ct.sum(); n_pos_hit += (inh & (ct == 1)).sum(); n_neg += (ct == 0).sum(); n_neg_hit += (inh & (ct == 0)).sum()
        if (inh & (ct == 1)).any(): feeds_with_hit += 1
        for it in ci[inh & (ct == 1)]:
            for e in he[hi == it]:
                ev_counts[int(e)] = ev_counts.get(int(e), 0) + 1
        dt.append(tb - ta)
    r = dict(n_feeds=len(base), n_feeds_with_later_feed=len(pairs), share=len(pairs) / len(base), P_in_later_hist_pos=n_pos_hit / max(n_pos, 1),
             P_in_later_hist_neg=n_neg_hit / max(n_neg, 1), feeds_with_pos_hit=feeds_with_hit, hit_event_types=ev_counts,
             median_gap_hours=float(np.median(dt) / 3600) if dt else None)
    res[split] = r; print(split, r, flush=True)
json.dump(res, open(f'{OUT}/09_future_history_leak.json', 'w'), indent=1, default=int)
