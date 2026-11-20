"""14) Leak post-processing on top of the strongest LGB val prediction + per-subset analysis + expected test gain."""
import numpy as np
import polars as pl
from common import load_split, eval_score, ndcg_per_feed, feeds

OUT = '<repo>/eda/history_leak/'


def main():
    d = load_split('val')
    t = d['c_target']; off = d['feeds']['cand_off'].to_numpy(); M = d['feeds']['M'].to_numpy()
    L = pl.read_parquet(OUT + 'leak_flags_val.parquet')
    lp = L['leak_pos'].to_numpy().astype(float); ln = L['leak_neg'].to_numpy().astype(float)
    eh = L['earlier_hist'].to_numpy().astype(float)
    eht = L['earlier_hist_event_type'].to_numpy()
    for name in ['base+emb+aff+covis', 'base+emb']:
        fn = OUT + f'out_lgb_val_{name.replace(" ", "_").replace("+", "_")}.npy'
        p = np.load(fn)
        # convert to within-feed rank-normalized score in [0,1] so that additive flags dominate
        fidx = d['c_fidx']
        r = pl.DataFrame({'f': fidx, 'x': p}).select(pl.col('x').rank('average').over('f') / pl.len().over('f'))['x'].to_numpy()
        print(f'=== base model: LGB {name} ===')
        b = eval_score(r, d, 'model')
        s1 = eval_score(r + 10 * lp, d, '+ leak_pos (later history)', b)
        s2 = eval_score(r + 10 * lp - 10 * ln, d, '+ leak_pos - leak_neg', b)
        for w in [0.1, 0.25, 0.5, 1.0]:
            eval_score(r + 10 * lp - 10 * ln + w * eh, d, f'+ leak_pos - leak_neg + {w}*earlier_hist', b)
        eh23 = eh * np.isin(eht, [2, 3, 4, 5, 6, 7]).astype(float)
        eval_score(r + 10 * lp - 10 * ln + 0.5 * eh23, d, '+ leak_pos - leak_neg + 0.5*earlier_hist(type!=1)', b)
        aff = (L.with_columns(pl.Series('f', fidx)).group_by('f', maintain_order=True)
               .agg(((pl.col('leak_pos') == 1) | (pl.col('leak_neg') == 1)).any().alias('a'))['a'].to_numpy())
        print(f'   affected feeds {aff.sum()}: model {b[aff].mean():.4f} -> with leak {s2[aff].mean():.4f} '
              f'(delta per affected feed {(s2[aff]-b[aff]).mean():.4f})')
        dpf = (s2[aff] - b[aff]).mean()
    # expected test gain
    ft = feeds('test')
    n = len(ft); cut = int(round(0.3 * n))
    T = pl.read_parquet(OUT + 'leak_flags_test.parquet').join(ft.select(['feed_id', 'fidx']), on='feed_id')
    g = T.group_by('fidx').agg(((pl.col('leak_pos') == 1) | (pl.col('leak_neg') == 1)).any().alias('a'))
    aff_t = g.filter(pl.col('a'))['fidx'].to_numpy()
    npub = (aff_t < cut).sum(); npriv = (aff_t >= cut).sum()
    print(f'\nTEST affected feeds: {len(aff_t)} (public {npub}/{cut}, private {npriv}/{n-cut})')
    print(f'expected gain with val per-affected-feed delta {dpf:.3f}: all {len(aff_t)*dpf/n:+.4f}, '
          f'public {npub*dpf/cut:+.4f}, private {npriv*dpf/(n-cut):+.4f}')


if __name__ == '__main__':
    main()
