"""Item 1: coverage and distributions."""
import numpy as np, pandas as pd
ROOT = '<repo>'; OUT = f'{ROOT}/nn_x/w12/hindsight/out'
p = pd.read_parquet(f'{OUT}/pairs_all.parquet'); s = pd.read_parquet(f'{OUT}/feeds_sel.parquet')
f = pd.read_parquet(f'{ROOT}/data/flat/train/feeds.parquet'); y = np.load(f'{ROOT}/data/flat/train/cand_target.npy', mmap_mode='r')
NF = len(f); NPOS = int(np.add.reduceat(np.asarray(y, dtype=np.int64), f.cand_off.values).sum())
el = p[p.ok & (p.k > 0)]
print(f'train feeds {NF}, positives {NPOS}')
print(f'feeds with a later same-user feed (user!=0): {len(p)} = {len(p)/NF:.4f} of feeds, positives {p.npos.sum()} = {p.npos.sum()/NPOS:.4f}')
print(f'  aligned (>=50% of A\'s 24 newest events vote the modal shift, gap>0): {p.ok.sum()} = {p.ok.mean():.4f} of pairs')
print(f'  eligible = aligned & k>=1 raw new events: {len(el)} = {len(el)/NF:.4f} of feeds; positives {el.npos.sum()} = {el.npos.sum()/NPOS:.4f}')
print(f'  eligible in holdout: {el.hold.sum()} of 30000 = {el.hold.sum()/30000:.4f}; eligible with p_random>0.9&M>=50: {(el.p_random>0.9).sum()}; p_random<=0.5: {(el.p_random<=0.5).sum()}')
q = [.05, .25, .5, .75, .9, .95, .99]
print('raw new events k (all eligible pairs) quantiles', dict(zip(q, el.k.quantile(q).values)), 'mean', el.k.mean())
print('gap hours (all eligible) quantiles', dict(zip(q, (el.gap / 3600).quantile(q).round(1).values)))
for nm, lo, hi in (('<1d', 0, 86400), ('1-3d', 86400, 3 * 86400), ('>3d', 3 * 86400, 1e12)):
    m = (el.gap >= lo) & (el.gap < hi); print(f'  gap {nm}: {m.mean():.3f} of eligible, median k {el.k[m].median():.0f}')
for tag, d in (('sample non-holdout (120K)', s[~s.hold]), ('holdout eligible', s[s.hold])):
    print(f'--- {tag}: {len(d)} feeds')
    for c in ['n_LEAK', 'n_FUT', 'n_FUTNS1', 'n_FUTC', 'hops']:
        print(f'  {c:9s} mean {d[c].mean():6.2f} quantiles', dict(zip(q, d[c].quantile(q).values)), f'share ==0: {(d[c]==0).mean():.4f}')
    print(f'  events removed by the candidate exclusion: {(d.n_LEAK-d.n_FUT).sum()/d.n_LEAK.sum():.4f} of new events; share of FUT events with source 1: {1-d.n_FUTNS1.sum()/d.n_FUT.sum():.4f}')
    print(f'  new events on items already in A history: {d.n_new_item_in_Ahist.sum()/d.n_LEAK.sum():.4f}')
    print(f'  feeds with n_FUT>=1: {(d.n_FUT>=1).mean():.4f}; n_FUTNS1>=1: {(d.n_FUTNS1>=1).mean():.4f}; chain gap median h {np.median(d.chain_gap)/3600:.1f}')
r = len(el) / NF
print(f'coverage with n_FUT>=1 (est. from sample): {r*(s[~s.hold].n_FUT>=1).mean():.4f} of train feeds; with n_FUTNS1>=1: {r*(s[~s.hold].n_FUTNS1>=1).mean():.4f}')
