# Builds cross-feed flags for every val and test candidate (the item is in the history of a later
# or an earlier feed of the same user, or is a candidate again), then reads their target rates and
# NDCG effect on val and their coverage on test. User id 0 is skipped: it pools all sessions that
# were not logged in.
# How it was run: from lab/eda/history_leak on 2026-11-11 (CPU), stdout saved as 03_leak.log; the
# command line is not on record.
# What it produced: out_crossfeed_{val,test}.parquet and results/crossfeed_flags_val_test.txt
# (flag "in a later history": 184 val candidates, 93.5% positive). The script cuts test at 30% of
# its feeds; the exact boundary found later gives 8,562 public and 20,033 private feeds.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/eda/history_leak/03_leak.py
"""3) Cross-feed (transductive / leak-like) signals for val and test candidates.
For each candidate (item c of feed X of user u at time t) look at ALL other feeds of u in train+val+test:
  later_hist  : c appears in the history of a later feed of u (=> clicked after t)    [future info]
  later_cand  : c appears as a candidate of a later feed of u (=> NOT viewed before that feed)  [future info]
  earlier_hist: c appears in the history of an earlier feed of u
  earlier_cand: c appeared as a candidate of an earlier feed of u (with its label, when known)
user_id == 0 is an anonymous bucket -> excluded from linking.
"""
import numpy as np
import polars as pl
from common import feeds, arr, load_split, eval_score, ndcg_per_feed

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
OUT = '<repo>/eda/history_leak/'
SPLITS = ['train', 'val', 'test']
SID = {'train': 0, 'val': 1, 'test': 2}


def build():
    F = {s: feeds(s).with_columns(pl.lit(SID[s]).alias('sid')) for s in SPLITS}
    users = pl.concat([F['val']['user_id'], F['test']['user_id']]).unique()
    users = users.filter(users != 0)
    allf = pl.concat([F[s].select(['sid', 'fidx', 'feed_id', 'timestamp', 'user_id', 'M', 'H', 'cand_off', 'hist_off'])
                      for s in SPLITS])
    allf = allf.filter(pl.col('user_id').is_in(users.implode()))
    print('feeds of val/test users (non-zero) by split:', allf.group_by('sid').len().sort('sid').to_dicts())
    hist_parts, cand_parts = [], []
    for s in SPLITS:
        ff = allf.filter(pl.col('sid') == SID[s]).sort('fidx')
        if len(ff) == 0:
            continue
        hoff = ff['hist_off'].to_numpy(); H = ff['H'].to_numpy(); coff = ff['cand_off'].to_numpy(); M = ff['M'].to_numpy()
        hidx = np.concatenate([np.arange(o, o + h) for o, h in zip(hoff, H)])
        cidx = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, M)])
        hp = pl.DataFrame({'sid': np.full(len(hidx), SID[s], np.int8),
                           'fidx': np.repeat(ff['fidx'].to_numpy(), H),
                           'item_id': np.asarray(arr(s, 'hist_item_id')[hidx]),
                           'etype': np.asarray(arr(s, 'hist_event_type')[hidx]),
                           'esrc': np.asarray(arr(s, 'hist_event_source')[hidx]),
                           'hpos': np.asarray(arr(s, 'hist_hpos')[hidx])})
        tgt = np.asarray(arr(s, 'cand_target')[cidx]) if s != 'test' else np.full(len(cidx), -1, np.int8)
        cp = pl.DataFrame({'sid': np.full(len(cidx), SID[s], np.int8),
                           'fidx': np.repeat(ff['fidx'].to_numpy(), M),
                           'item_id': np.asarray(arr(s, 'cand_item_id')[cidx]),
                           'target': tgt})
        hist_parts.append(hp); cand_parts.append(cp)
    meta = allf.select(['sid', 'fidx', 'user_id', 'timestamp']).with_columns(pl.col('sid').cast(pl.Int8))
    hist = pl.concat(hist_parts).join(meta, on=['sid', 'fidx'])
    cand = pl.concat(cand_parts).join(meta, on=['sid', 'fidx'])
    print('linked hist rows', len(hist), 'cand rows', len(cand))
    return hist, cand, allf


def features_for(split, hist, cand):
    """per-candidate cross-feed features for all candidates of `split` (in flat order)."""
    s = SID[split]
    f = feeds(split)
    M = f['M'].to_numpy()
    base = pl.DataFrame({'fidx': np.repeat(f['fidx'].to_numpy(), M),
                         'item_id': np.asarray(arr(split, 'cand_item_id')),
                         'user_id': np.repeat(f['user_id'].to_numpy(), M),
                         'timestamp': np.repeat(f['timestamp'].to_numpy(), M)}).with_row_index('row')
    b = base.filter(pl.col('user_id') != 0)
    # --- history matches in OTHER feeds of the same user
    hm = b.join(hist.rename({'fidx': 'ofidx', 'timestamp': 'ots', 'sid': 'osid'}), on=['user_id', 'item_id'])
    hm = hm.filter(~((pl.col('osid') == s) & (pl.col('ofidx') == pl.col('fidx'))))
    hm = hm.with_columns((pl.col('ots') - pl.col('timestamp')).alias('dt'))
    # own-feed check
    own = b.join(hist.filter(pl.col('sid') == s).rename({'timestamp': 'ots'}), on=['fidx', 'user_id', 'item_id'])
    print(f'[{split}] candidates found in OWN history: {len(own)}')
    later = hm.filter(pl.col('dt') > 0)
    earlier = hm.filter(pl.col('dt') < 0)
    same_t = hm.filter(pl.col('dt') == 0)
    print(f'[{split}] hist matches: later {len(later)}, earlier {len(earlier)}, same-timestamp {len(same_t)}')
    # for later: take the earliest later feed containing it; type of the OLDEST event (max hpos) in that feed
    lagg = later.sort(['row', 'dt', 'hpos'], descending=[False, False, True]).group_by('row').agg(
        pl.col('dt').first().alias('later_hist_dt'), pl.col('etype').first().alias('later_hist_type'),
        pl.col('esrc').first().alias('later_hist_src'), pl.col('hpos').first().alias('later_hist_hpos'),
        pl.col('osid').first().alias('later_hist_sid'), pl.len().alias('later_hist_n'),
        pl.col('etype').unique().alias('later_hist_types'))
    eagg = earlier.sort(['row', 'dt'], descending=[False, True]).group_by('row').agg(
        pl.col('dt').first().alias('earlier_hist_dt'), pl.col('etype').first().alias('earlier_hist_type'),
        pl.col('osid').first().alias('earlier_hist_sid'), pl.len().alias('earlier_hist_n'))
    # --- candidate matches in other feeds of same user
    cm = b.join(cand.rename({'fidx': 'ofidx', 'timestamp': 'ots', 'sid': 'osid', 'target': 'otarget'}),
                on=['user_id', 'item_id'])
    cm = cm.filter(~((pl.col('osid') == s) & (pl.col('ofidx') == pl.col('fidx'))))
    cm = cm.with_columns((pl.col('ots') - pl.col('timestamp')).alias('dt'))
    lc = cm.filter(pl.col('dt') > 0).sort(['row', 'dt']).group_by('row').agg(
        pl.col('dt').first().alias('later_cand_dt'), pl.len().alias('later_cand_n'), pl.col('osid').first().alias('later_cand_sid'))
    ec = cm.filter(pl.col('dt') < 0).sort(['row', 'dt'], descending=[False, True]).group_by('row').agg(
        pl.col('dt').first().alias('earlier_cand_dt'), pl.len().alias('earlier_cand_n'),
        pl.col('otarget').max().alias('earlier_cand_maxtarget'), pl.col('osid').first().alias('earlier_cand_sid'))
    sc = cm.filter(pl.col('dt') == 0).group_by('row').agg(pl.len().alias('same_ts_cand_n'))
    # has any later / earlier feed at all (any split)
    allfeeds = pl.concat([hist.select(['user_id', 'timestamp', 'sid', 'fidx']).unique(),
                          cand.select(['user_id', 'timestamp', 'sid', 'fidx']).unique()]).unique()
    fu = f.select(['fidx', 'user_id', 'timestamp']).filter(pl.col('user_id') != 0).join(
        allfeeds.rename({'timestamp': 'ots', 'fidx': 'ofidx'}), on='user_id')
    fu = fu.filter(~((pl.col('sid') == s) & (pl.col('ofidx') == pl.col('fidx'))))
    fu = fu.with_columns((pl.col('ots') - pl.col('timestamp')).alias('dt'))
    fagg = fu.group_by('fidx').agg((pl.col('dt') > 0).sum().alias('n_later_feeds'),
                                   (pl.col('dt') < 0).sum().alias('n_earlier_feeds'),
                                   pl.col('dt').filter(pl.col('dt') > 0).min().alias('next_feed_dt'))
    out = base.join(lagg, on='row', how='left').join(eagg, on='row', how='left').join(lc, on='row', how='left') \
        .join(ec, on='row', how='left').join(sc, on='row', how='left').join(fagg, on='fidx', how='left').sort('row')
    out = out.with_columns(pl.col('n_later_feeds').fill_null(0), pl.col('n_earlier_feeds').fill_null(0))
    return out


def main():
    hist, cand, allf = build()
    res = {}
    for split in ['val', 'test']:
        X = features_for(split, hist, cand)
        X.drop('later_hist_types').write_parquet(OUT + f'out_crossfeed_{split}.parquet')
        res[split] = X
    # ---------------- analysis on val
    d = load_split('val')
    X = res['val']
    t = d['c_target']
    X = X.with_columns(pl.Series('target', t))
    print('\n=== VAL: coverage and target rates ===')
    nfe = len(d['feeds'])
    for name, cond in [('later_hist', pl.col('later_hist_dt').is_not_null()),
                       ('later_cand', pl.col('later_cand_dt').is_not_null()),
                       ('earlier_hist', pl.col('earlier_hist_dt').is_not_null()),
                       ('earlier_cand', pl.col('earlier_cand_dt').is_not_null()),
                       ('earlier_cand & earlier target=1', pl.col('earlier_cand_maxtarget') == 1),
                       ('earlier_cand & earlier target=0', pl.col('earlier_cand_maxtarget') == 0),
                       ('feed has later feed of user', pl.col('n_later_feeds') > 0),
                       ('feed has earlier feed of user', pl.col('n_earlier_feeds') > 0)]:
        sub = X.filter(cond)
        print(f'{name:40s} cands={len(sub):8d} feeds={sub["fidx"].n_unique():6d} ({sub["fidx"].n_unique()/nfe:.4f}) '
              f'target_rate={sub["target"].mean() if len(sub) else float("nan"):.4f}  n_pos={sub["target"].sum()}')
    print('base target rate', t.mean())
    print('\nlater_hist by event type (val):')
    print(X.filter(pl.col('later_hist_dt').is_not_null()).group_by('later_hist_type').agg(
        pl.len(), pl.col('target').mean().alias('target_rate')).sort('later_hist_type'))
    print('later_hist by source split:')
    print(X.filter(pl.col('later_hist_dt').is_not_null()).group_by('later_hist_sid').agg(
        pl.len(), pl.col('target').mean().alias('target_rate')).sort('later_hist_sid'))
    print('earlier_cand by source split / earlier target:')
    print(X.filter(pl.col('earlier_cand_dt').is_not_null()).group_by(['earlier_cand_sid', 'earlier_cand_maxtarget']).agg(
        pl.len(), pl.col('target').mean().alias('target_rate')).sort(['earlier_cand_sid', 'earlier_cand_maxtarget']))
    print('earlier_hist by type:')
    print(X.filter(pl.col('earlier_hist_dt').is_not_null()).group_by(['earlier_hist_type']).agg(
        pl.len(), pl.col('target').mean().alias('target_rate')).sort('earlier_hist_type'))
    # recall of positives in feeds that have a later feed
    sub = X.filter(pl.col('n_later_feeds') > 0)
    print(f'\nVal feeds with a later feed of the same user: {sub["fidx"].n_unique()}; positives there {sub["target"].sum()}, '
          f'of which in later history {sub.filter(pl.col("later_hist_dt").is_not_null())["target"].sum()}')
    # ---- NDCG
    f1 = d['c_f1'].astype(np.float64)
    off = d['feeds']['cand_off'].to_numpy(); M = d['feeds']['M'].to_numpy()
    base = eval_score(f1, d, 'f1')
    lh = X['later_hist_dt'].is_not_null().to_numpy().astype(float)
    lc = X['later_cand_dt'].is_not_null().to_numpy().astype(float)
    ec0 = (X['earlier_cand_maxtarget'] == 0).fill_null(False).to_numpy().astype(float)
    ec = X['earlier_cand_dt'].is_not_null().to_numpy().astype(float)
    s1 = f1 + 10 * lh
    eval_score(s1, d, 'f1 + 10*later_hist', base)
    s2 = f1 + 10 * lh - 10 * lc
    eval_score(s2, d, 'f1 + 10*later_hist - 10*later_cand', base)
    s3 = f1 - 10 * ec
    eval_score(s3, d, 'f1 - 10*earlier_cand', base)
    s3b = f1 + 10 * ec
    eval_score(s3b, d, 'f1 + 10*earlier_cand', base)
    s4 = s2 - 10 * ec
    eval_score(s4, d, 'f1 + later_hist - later_cand - earlier_cand', base)
    # restricted to affected feeds
    pf_b = ndcg_per_feed(f1, t, off, M); pf_4 = ndcg_per_feed(s2, t, off, M)
    aff = X.group_by('fidx').agg(((pl.col('later_hist_dt').is_not_null()) | (pl.col('later_cand_dt').is_not_null())).any().alias('a')).sort('fidx')['a'].to_numpy()
    print(f'affected val feeds (later info): {aff.sum()}  NDCG there f1={pf_b[aff].mean():.4f} -> leak={pf_4[aff].mean():.4f}')
    # ---------------- test coverage
    T = res['test']
    print('\n=== TEST coverage ===')
    ft = feeds('test')
    nft = len(ft)
    pub_cut = int(round(0.3 * nft))  # rows sorted chronologically -> first 30% public
    T = T.with_columns((pl.col('fidx') >= pub_cut).alias('private'))
    for name, cond in [('later_hist', pl.col('later_hist_dt').is_not_null()),
                       ('later_cand', pl.col('later_cand_dt').is_not_null()),
                       ('earlier_hist', pl.col('earlier_hist_dt').is_not_null()),
                       ('earlier_cand', pl.col('earlier_cand_dt').is_not_null()),
                       ('feed has later feed of user', pl.col('n_later_feeds') > 0),
                       ('feed has earlier feed of user', pl.col('n_earlier_feeds') > 0)]:
        sub = T.filter(cond)
        nf = sub['fidx'].n_unique()
        nfp = sub.filter(pl.col('private'))['fidx'].n_unique()
        print(f'{name:40s} cands={len(sub):8d} feeds={nf:6d} ({nf/nft:.4f}); private feeds={nfp} ({nfp/(nft-pub_cut):.4f})')
    print('test later_hist by type:')
    print(T.filter(pl.col('later_hist_dt').is_not_null()).group_by('later_hist_type').len().sort('later_hist_type'))
    print('test earlier_cand by split:')
    print(T.filter(pl.col('earlier_cand_dt').is_not_null()).group_by('earlier_cand_sid').len())


if __name__ == '__main__':
    main()
