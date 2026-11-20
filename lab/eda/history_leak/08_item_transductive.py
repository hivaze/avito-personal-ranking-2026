"""8) Item-level transductive signals from OTHER users' histories / candidate lists.
For a val candidate c in feed X (time t, user u):
  hist_other_{later,earlier}_{all,fc}: # of feeds of other users in val+test whose history contains c (any event / type1-src1 'feed click'),
                                       split by whether that feed is later or earlier than X.
  cand_other: # of other feeds in val+test where c is a candidate (exposure count in the exploration traffic)
  train_hist_fc: # train sequences (all train feeds, dedup by user) containing c as type1-src1
Evaluated alone and blended with f1 on val. For test the same features can be computed (test + val histories)."""
import numpy as np
import polars as pl
from common import feeds, arr, load_split, eval_score, ndcg_per_feed

pl.Config.set_tbl_rows(50)
OUT = '<repo>/eda/history_leak/'


def split_hist(s, sid):
    f = feeds(s)
    H = f['H'].to_numpy()
    df = pl.DataFrame({'sid': np.full(H.sum(), sid, np.int8), 'fidx': np.repeat(f['fidx'].to_numpy(), H),
                       'user_id': np.repeat(f['user_id'].to_numpy(), H), 'ots': np.repeat(f['timestamp'].to_numpy(), H),
                       'item_id': np.asarray(arr(s, 'hist_item_id')), 'etype': np.asarray(arr(s, 'hist_event_type')),
                       'esrc': np.asarray(arr(s, 'hist_event_source'))})
    df = df.with_columns(((pl.col('etype') == 1) & (pl.col('esrc') == 1)).alias('fc'))
    # one row per (feed, item): any event / any feed-click
    return df.group_by(['sid', 'fidx', 'user_id', 'ots', 'item_id']).agg(pl.col('fc').any())


def split_cand(s, sid):
    f = feeds(s)
    M = f['M'].to_numpy()
    return pl.DataFrame({'sid': np.full(M.sum(), sid, np.int8), 'fidx': np.repeat(f['fidx'].to_numpy(), M),
                         'user_id': np.repeat(f['user_id'].to_numpy(), M), 'ots': np.repeat(f['timestamp'].to_numpy(), M),
                         'item_id': np.asarray(arr(s, 'cand_item_id'))})


def feats(split, sid, H, C):
    f = feeds(split)
    M = f['M'].to_numpy()
    base = pl.DataFrame({'fidx': np.repeat(f['fidx'].to_numpy(), M), 'user_id': np.repeat(f['user_id'].to_numpy(), M),
                         'ts': np.repeat(f['timestamp'].to_numpy(), M),
                         'item_id': np.asarray(arr(split, 'cand_item_id'))}).with_row_index('row')
    j = base.join(H, on='item_id')
    j = j.filter(~((pl.col('sid') == sid) & (pl.col('fidx_right') == pl.col('fidx'))))
    j = j.filter((pl.col('user_id_right') != pl.col('user_id')) | (pl.col('user_id') == 0))
    agg = j.group_by('row').agg(
        (pl.col('ots') > pl.col('ts')).sum().alias('ho_later_all'),
        ((pl.col('ots') > pl.col('ts')) & pl.col('fc')).sum().alias('ho_later_fc'),
        (pl.col('ots') <= pl.col('ts')).sum().alias('ho_earlier_all'),
        ((pl.col('ots') <= pl.col('ts')) & pl.col('fc')).sum().alias('ho_earlier_fc'))
    jc = base.join(C, on='item_id')
    jc = jc.filter(~((pl.col('sid') == sid) & (pl.col('fidx_right') == pl.col('fidx'))))
    cagg = jc.group_by('row').agg(pl.len().alias('cand_other'),
                                  (pl.col('ots') > pl.col('ts')).sum().alias('cand_other_later'))
    out = base.join(agg, on='row', how='left').join(cagg, on='row', how='left').sort('row').fill_null(0)
    return out


def main():
    H = pl.concat([split_hist('val', 1), split_hist('test', 2)])
    C = pl.concat([split_cand('val', 1), split_cand('test', 2)])
    d = load_split('val')
    t = d['c_target']; off = d['feeds']['cand_off'].to_numpy(); M = d['feeds']['M'].to_numpy()
    X = feats('val', 1, H, C)
    X.write_parquet(OUT + 'out_item_trans_val.parquet')
    Xt = feats('test', 2, H, C)
    Xt.write_parquet(OUT + 'out_item_trans_test.parquet')
    rng = np.random.default_rng(0)
    f1 = d['c_f1'].astype(np.float64)
    base = eval_score(f1, d, 'f1')
    fidx = d['c_fidx']
    def prank(x):
        return pl.DataFrame({'f': fidx, 'x': x}).select((pl.col('x').rank('average').over('f') / pl.len().over('f')))['x'].to_numpy()
    rf1 = prank(f1)
    for c in ['ho_later_all', 'ho_later_fc', 'ho_earlier_all', 'ho_earlier_fc', 'cand_other', 'cand_other_later']:
        x = X[c].to_numpy().astype(np.float64)
        a = ndcg_per_feed(x + 1e-9 * rng.random(len(x)), t, off, M).mean()
        print(f'{c:18s} alone {a:.4f}  nonzero {(x>0).mean():.4f} test nonzero {(Xt[c].to_numpy()>0).mean():.4f}  '
              f'target rate nonzero {t[x>0].mean():.4f} zero {t[x==0].mean():.4f}')
        for w in [0.25, 0.5]:
            eval_score(rf1 + w * prank(x), d, f'  rank(f1)+{w}*rank({c})', base)
    df = pl.DataFrame({'t': t, 'x': X['ho_later_fc'].to_numpy()})
    print(df.with_columns(pl.col('x').clip(0, 5)).group_by('x').agg(pl.len(), pl.col('t').mean()).sort('x'))
    df = pl.DataFrame({'t': t, 'x': X['cand_other'].to_numpy()})
    print(df.with_columns(pl.col('x').clip(0, 5)).group_by('x').agg(pl.len(), pl.col('t').mean()).sort('x'))


if __name__ == '__main__':
    main()
