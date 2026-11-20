"""e03: per-event-type profiles on val, test and a 200K-feed random train sample (seed 0).
share, hpos profile, repetition / bursts on one item, per-item funnel order (first-occurrence precedence), next-event
transitions on the same item, sources, categories, user_id 0, item popularity, missing embeddings, item-level f4/f5."""
import sys, os
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from common_ev import hist_frame, cand_frame, feeds

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_cols(40); pl.Config.set_float_precision(4)
OUT = '<repo>/nn_x/codes/events/out'; os.makedirs(OUT, exist_ok=True)
T = list(range(1, 11))


def profiles(df, name):
    print(f'\n################ {name}: {df.height:,} events, {df["fidx"].n_unique():,} feeds')
    n = df.height
    # newest-first index within the same (feed,item): rank of event among the item's events
    df = df.sort(['fidx', 'hpos'])
    g = df.group_by(['fidx', 'item_id']).agg(pl.len().alias('n_item'), pl.col('et').n_unique().alias('n_types_item'))
    df = df.join(g, on=['fidx', 'item_id'])
    gt = df.group_by(['fidx', 'item_id', 'et']).agg(pl.len().alias('n_same'))
    df = df.join(gt, on=['fidx', 'item_id', 'et'])
    # adjacency in history (hpos-1 = newer neighbour)
    df = df.with_columns(pl.col('item_id').shift(1).over('fidx').alias('item_newer'), pl.col('et').shift(1).over('fidx').alias('et_newer'))
    df = df.with_columns(((pl.col('item_newer') == pl.col('item_id')) & (pl.col('et_newer') == pl.col('et'))).fill_null(False).alias('adj_same'),
                         (pl.col('item_newer') == pl.col('item_id')).fill_null(False).alias('adj_item'))
    tab = df.group_by('et').agg(
        pl.len().alias('n'), pl.col('hpos').mean().alias('hpos_mean'), pl.col('hpos').median().alias('hpos_med'),
        (pl.col('hpos') < 32).mean().alias('in_hpos<32'), (pl.col('hpos') >= 192).mean().alias('in_hpos>=192'),
        pl.col('fidx').n_unique().alias('feeds_with'),
        (pl.col('n_same') > 1).mean().alias('repeat_share'), pl.col('n_same').mean().alias('ev_per_item_type'),
        (pl.col('n_types_item') > 1).mean().alias('item_has_other_type'),
        pl.col('adj_same').mean().alias('adj_newer_same_item_type'), pl.col('adj_item').mean().alias('adj_newer_same_item'),
        (pl.col('emb_idx') < 0).mean().alias('no_emb'), (pl.col('cat') == 0).mean().alias('cat0'),
        (pl.col('es') == 0).mean().alias('src0'), (pl.col('es') == 1).mean().alias('src1'), (pl.col('es') == 2).mean().alias('src2'),
        (pl.col('user_id') == 0).mean().alias('anon_share')).sort('et')
    nf = df['fidx'].n_unique()
    tab = tab.with_columns((pl.col('n') / n).alias('share'), (pl.col('feeds_with') / nf).alias('feeds_with_share'))
    print('[P1] per-type profile'); print(tab)
    tab.write_csv(f'{OUT}/e03_profile_{name}.csv')
    # hpos profile: share of each type within hpos bins
    hb = df.with_columns((pl.col('hpos') // 32).alias('hb')).group_by(['hb', 'et']).agg(pl.len()).pivot(on='et', index='hb', values='len').sort('hb')
    cols = [str(t) for t in T if str(t) in hb.columns]
    hb = hb.with_columns(pl.sum_horizontal(cols).alias('tot'))
    print('[P2] type mix by hpos bin of 32 (rows sum to 1)')
    print(hb.select('hb', *[(pl.col(c) / pl.col('tot')).alias('e' + c) for c in cols]))
    # same-item first-occurrence precedence: P(oldest a event older than oldest b event | item has both)
    fo = df.group_by(['fidx', 'item_id', 'et']).agg(pl.col('hpos').max().alias('first_hpos'), pl.col('hpos').min().alias('last_hpos'))
    P = np.full((11, 11), np.nan); N = np.zeros((11, 11), int)
    fo_w = fo.pivot(on='et', index=['fidx', 'item_id'], values='first_hpos')
    for a in T:
        for b in T:
            if a == b or str(a) not in fo_w.columns or str(b) not in fo_w.columns:
                continue
            x = fo_w.select(pl.col(str(a)), pl.col(str(b))).drop_nulls()
            if x.height < 30:
                continue
            N[a, b] = x.height
            P[a, b] = ((x[str(a)] > x[str(b)]).sum() + 0.5 * (x[str(a)] == x[str(b)]).sum()) / x.height
    print('[P3] P(first a-event is OLDER than first b-event | same feed & item has both)  rows a, cols b  (>0.5: a precedes b)')
    print('     ' + ' '.join(f'{b:>6d}' for b in T))
    for a in T:
        print(f'{a:>4d} ' + ' '.join(f'{P[a,b]:6.2f}' if N[a, b] else '     .' for b in T))
    print('     co-occurrence counts N (items with both):')
    for a in T:
        print(f'{a:>4d} ' + ' '.join(f'{N[a,b]:7d}' for b in T))
    # next event on the same item (newer), item-level transition matrix
    nxt = df.sort(['fidx', 'item_id', 'hpos'], descending=[False, False, True]).with_columns(
        pl.col('et').shift(-1).over(['fidx', 'item_id']).alias('et_next'))
    tr = nxt.drop_nulls('et_next').group_by(['et', 'et_next']).agg(pl.len()).pivot(on='et_next', index='et', values='len').sort('et').fill_null(0)
    cols = [str(t) for t in T if str(t) in tr.columns]
    tr = tr.with_columns(pl.sum_horizontal(cols).alias('tot'))
    ends = nxt.filter(pl.col('et_next').is_null()).group_by('et').agg(pl.len().alias('last_on_item'))
    tr = tr.join(ends, on='et').with_columns((pl.col('last_on_item') / (pl.col('tot') + pl.col('last_on_item'))).alias('P(no newer)'))
    print('[P4] next NEWER event on the same item given current type (row-normalised), and P(no newer event on item)')
    print(tr.select('et', *[(pl.col(c) / pl.col('tot')).round(3).alias('->' + c) for c in cols], 'tot', 'P(no newer)'))
    # item-level type-set patterns
    ts = df.group_by(['fidx', 'item_id']).agg(pl.col('et').unique().sort().alias('ts'))
    ts = ts.with_columns(pl.col('ts').cast(pl.List(pl.Utf8)).list.join(',').alias('typeset'))
    print('[P5] item-level type sets (share of (feed,item) pairs), top 30')
    print(ts.group_by('typeset').agg(pl.len()).with_columns((pl.col('len') / ts.height).alias('share')).sort('len', descending=True).head(30))
    # sources by type
    sc = df.with_columns(pl.when(pl.col('es') <= 8).then(pl.col('es')).otherwise(99).alias('es2')).group_by(['et', 'es2']).agg(pl.len()) \
        .pivot(on='es2', index='et', values='len').sort('et').fill_null(0)
    cols = [c for c in sc.columns if c != 'et']
    sc = sc.with_columns(pl.sum_horizontal(cols).alias('tot'))
    print('[P6] source distribution per type (rows; 99 = sources 9..30)')
    print(sc.select('et', *[(pl.col(c) / pl.col('tot')).round(3).alias('s' + c) for c in sorted(cols, key=int)]))
    ns = df.group_by('et').agg(pl.col('es').n_unique().alias('n_sources'))
    print(ns.sort('et').transpose(include_header=True))
    # sources: which types use each source (columns)
    st = df.filter(pl.col('es') > 0).group_by(['es', 'et']).agg(pl.len()).pivot(on='et', index='es', values='len').sort('es').fill_null(0)
    cols = [c for c in st.columns if c != 'es']
    st = st.with_columns(pl.sum_horizontal(cols).alias('tot'))
    print('[P6b] per source: type mix (rows)')
    print(st.select('es', 'tot', *[(pl.col(c) / pl.col('tot')).round(3).alias('e' + c) for c in sorted(cols, key=int)]))
    # anonymous users
    an = df.with_columns((pl.col('user_id') == 0).alias('anon')).group_by(['anon', 'et']).agg(pl.len()).pivot(on='et', index='anon', values='len').fill_null(0)
    cols = [c for c in an.columns if c != 'anon']
    an = an.with_columns(pl.sum_horizontal(cols).alias('tot'))
    print('[P7] type mix: user_id == 0 vs logged-in')
    print(an.select('anon', 'tot', *[(pl.col(c) / pl.col('tot')).round(4).alias('e' + c) for c in sorted(cols, key=int)]))
    fa = df.group_by('fidx').agg(pl.col('user_id').first().alias('uid'), pl.len().alias('H'), *[(pl.col('et') == t).sum().alias(f'n{t}') for t in T])
    fa = fa.with_columns((pl.col('uid') == 0).alias('anon'))
    print('[P7b] share of feeds having >=1 event of type t, anon vs logged-in; mean H')
    print(fa.group_by('anon').agg(pl.len(), pl.col('H').mean(), *[(pl.col(f'n{t}') > 0).mean().alias(f'has{t}') for t in T]))
    return df


def category_mix(df, name):
    # type x category: per category type mix, and lift
    cm = df.group_by(['cat', 'et']).agg(pl.len()).pivot(on='et', index='cat', values='len').fill_null(0).sort('cat')
    cols = [str(t) for t in T if str(t) in cm.columns]
    cm = cm.with_columns(pl.sum_horizontal(cols).alias('tot'))
    tot = cm.select(cols).sum()
    glob = {c: tot[c][0] / tot.sum_horizontal()[0] for c in cols}
    out = cm.select('cat', 'tot', *[(pl.col(c) / pl.col('tot')).round(3).alias('e' + c) for c in cols]).filter(pl.col('tot') > 2000)
    out = out.with_columns((pl.col('e2') + pl.col('e3')).alias('contact23'), (pl.col('e5') + pl.col('e6') + pl.col('e8')).alias('msg568'),
                           (pl.col('e7') + pl.col('e9')).alias('e79')).sort('tot', descending=True)
    print(f'[C1] {name}: type mix per history category (cats with > 2000 events), global:', {k: round(v, 3) for k, v in glob.items()})
    print(out)
    out.write_csv(f'{OUT}/e03_catmix_{name}.csv')
    # correlation structure of the per-category mixes across categories (which types co-vary)
    X = out.select([f'e{t}' for t in T]).to_numpy()
    w = out['tot'].to_numpy()
    C = np.corrcoef(X.T)
    print('[C2] across-category correlation of type shares (unweighted, categories with >2000 events)')
    print('      ' + ' '.join(f'{t:>6d}' for t in T))
    for i, t in enumerate(T):
        print(f'{t:>5d} ' + ' '.join(f'{C[i,j]:6.2f}' for j in range(10)))


def popularity(dfs, cand_all):
    # item popularity = number of distinct feeds whose history contains the item (val+test+train sample)
    allh = pl.concat([d.select('fidx', 'item_id').with_columns(pl.lit(k).alias('sp')) for k, d in dfs.items()])
    pop = allh.unique(['sp', 'fidx', 'item_id']).group_by('item_id').agg(pl.len().alias('pop_hist'))
    cpop = cand_all.group_by('item_id').agg(pl.len().alias('pop_cand'), pl.col('f4').mean().alias('f4'), pl.col('f5').mean().alias('f5'),
                                            pl.col('target').mean().alias('ctr'))
    for k, d in dfs.items():
        x = d.join(pop, on='item_id', how='left').join(cpop, on='item_id', how='left')
        x = x.with_columns(pl.col('pop_cand').fill_null(0))
        print(f'[I1] {k}: item popularity / item-level candidate attributes by event type (pop_hist = #feeds with item in history, val+test+train200K)')
        print(x.group_by('et').agg(pl.col('pop_hist').mean().alias('pop_mean'), pl.col('pop_hist').median().alias('pop_med'),
                                   (pl.col('pop_hist') == 1).mean().alias('pop_only_this_feed'),
                                   (pl.col('pop_cand') > 0).mean().alias('is_cand_somewhere'),
                                   pl.col('f4').mean().alias('f4_mean(if cand)'), pl.col('f5').mean().alias('f5_mean(if cand)'),
                                   pl.col('ctr').mean().alias('cand_ctr')).sort('et'))


def main():
    vh, _ = hist_frame('val'); th, _ = hist_frame('test')
    F = feeds('train')
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(F.height, 200_000, replace=False))
    np.save(f'{OUT}/train_sample200k_fidx.npy', samp)
    trh, _ = hist_frame('train', samp)
    dfs = {'val': vh, 'test': th, 'train200k': trh}
    for k, d in dfs.items():
        profiles(d, k)
        category_mix(d, k)
    vc, _ = cand_frame('val'); tc, _ = cand_frame('test')
    tc = tc.with_columns(pl.lit(None, dtype=pl.Int8).alias('target')) if 'target' not in tc.columns else tc
    cand_all = pl.concat([vc.select('item_id', 'f4', 'f5', pl.col('target').cast(pl.Float32)),
                          tc.select('item_id', 'f4', 'f5', pl.col('target').cast(pl.Float32))])
    popularity(dfs, cand_all)


if __name__ == '__main__':
    main()
