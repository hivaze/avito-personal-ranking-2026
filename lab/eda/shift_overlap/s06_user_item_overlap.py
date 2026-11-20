"""User-item overlap: re-impressions of the same (user,item) across splits; cross-feed label leakage via later-feed history."""
from common import *
from featlib import subset
import time
t0 = time.time()
rng = np.random.default_rng(1)
F = {s: feeds(s) for s in SPLITS}
ftr = F['train']
vt_users = np.union1d(F['val']['user_id'].unique().to_numpy(), F['test']['user_id'].unique().to_numpy())
vt_users = vt_users[vt_users != 0]
sel = np.where(np.isin(ftr['user_id'].to_numpy(), vt_users))[0]
print(f'train feeds of val/test users (excl user0): {len(sel)}', flush=True)
D = subset('train', sel)
trc = pl.DataFrame({'user_id': D['f']['user_id'].to_numpy()[D['cf']], 'item_id': D['C']['item_id'], 'tr_t': D['C']['target'].astype(np.int8)})
trc = trc.group_by(['user_id', 'item_id']).agg(pl.len().alias('tr_n'), pl.col('tr_t').max().alias('tr_pos'))
trh = pl.DataFrame({'user_id': D['f']['user_id'].to_numpy()[D['hf']], 'item_id': D['Hh']['item_id'], 'et': D['Hh']['event_type']}).unique(['user_id', 'item_id'])
print('built train user-item tables', round(time.time() - t0), flush=True)
for s in ['val', 'test']:
    f = F[s]; M = f['M'].to_numpy()
    df = pl.DataFrame({'fi': np.repeat(np.arange(len(f)), M), 'user_id': np.repeat(f['user_id'].to_numpy(), M), 'item_id': np.asarray(cand(s, 'item_id'))})
    if s == 'val':
        df = df.with_columns(pl.Series('t', np.asarray(cand(s, 'target')).astype(np.int8)))
    df = df.join(trc, on=['user_id', 'item_id'], how='left').join(trh.select('user_id', 'item_id', pl.lit(1).alias('in_tr_hist')), on=['user_id', 'item_id'], how='left')
    nz = df['user_id'] != 0
    seen = df['tr_n'].is_not_null()
    print(f'\n{s}: candidates (excl user0) shown to the SAME user in train: {seen.filter(nz).mean()*100:.2f}% of rows; '
          f'among users present in train: {df.filter(nz & pl.col("user_id").is_in(ftr["user_id"].unique().implode()))["tr_n"].is_not_null().mean()*100:.2f}%; '
          f'feeds with >=1 such candidate {df.filter(seen & nz)["fi"].n_unique()/len(f)*100:.2f}%')
    print(f'{s}: candidates present in the same user\'s TRAIN history: {df["in_tr_hist"].is_not_null().filter(nz).mean()*100:.3f}%')
    if s == 'val':
        g = df.filter(nz).with_columns(pl.when(pl.col('tr_n').is_null()).then(pl.lit('not shown in train')).when(pl.col('tr_pos') == 1).then(pl.lit('shown, target=1 in train'))
                                       .otherwise(pl.lit('shown, target=0 in train')).alias('grp'))
        print(g.group_by('grp').agg(pl.len().alias('n'), (pl.col('t').mean() * 100).alias('val_pos_rate%')).sort('grp'))
        # NDCG effect: f1 + bonus for re-impression
        Mv = M; off = f['cand_off'].to_numpy(); t = np.asarray(cand('val', 'target')).astype(float)
        df = df.sort('fi', maintain_order=True)
        f1 = np.asarray(cand('val', 'f1'), dtype=np.float64)
        base = ndcg_per_feed(f1 + rng.random(len(f1)) * 1e-7, t, off, Mv)
        for nm, w in [('re-shown (any)', df['tr_n'].is_not_null().to_numpy()), ('re-shown & tr_pos=1', (df['tr_pos'] == 1).fill_null(False).to_numpy()),
                      ('re-shown & tr_pos=0', (df['tr_pos'] == 0).fill_null(False).to_numpy())]:
            for bonus in [+1.0, -1.0]:
                sc = f1 + bonus * w + rng.random(len(f1)) * 1e-7
                d = ndcg_per_feed(sc, t, off, Mv) - base
                print(f'   f1 {"+" if bonus>0 else "-"} 1[{nm}] : NDCG delta {d.mean():+.4f} +- {d.std()/np.sqrt(len(d)):.4f}')

# --------- (b) train: earlier-feed candidates in later-feed history of the same user => which event types encode the target?
print('\n==== train: consecutive same-user feed pairs (A earlier, B later, gap>0, user!=0); does A-candidate appear in B history?')
f = ftr.filter(pl.col('user_id') != 0).sort(['user_id', 'timestamp', 'feed_id'])
f = f.with_columns(pl.col('fidx').shift(-1).over('user_id').alias('nxt'), (pl.col('timestamp').shift(-1).over('user_id') - pl.col('timestamp')).alias('gap'))
pairs = f.filter(pl.col('nxt').is_not_null() & (pl.col('gap') > 0)).select('fidx', 'nxt', 'gap')
pairs = pairs[rng.choice(len(pairs), min(150000, len(pairs)), replace=False)]
print('pairs sampled', len(pairs), 'gap median (h)', pairs['gap'].median() / 3600)
A = subset('train', pairs['fidx'].to_numpy())
pa = pairs.sort('fidx')
B = subset('train', pa['nxt'].to_numpy())
nxt_of = dict(zip(pa['fidx'].to_list(), pa['nxt'].to_list()))
fidxA = A['f']['fidx'].to_numpy(); fidxB = B['f']['fidx'].to_numpy()
ca = pl.DataFrame({'nxt': np.array([nxt_of[x] for x in fidxA])[A['cf']], 'item_id': A['C']['item_id'], 't': A['C']['target'].astype(np.int8),
                   'gap': pa['gap'].to_numpy()[A['cf']] if np.array_equal(pa['fidx'].to_numpy(), fidxA) else None})
hb = pl.DataFrame({'nxt': fidxB[B['hf']], 'item_id': B['Hh']['item_id'], 'et': B['Hh']['event_type'].astype(np.int8), 'hpos': B['Hh']['hpos']})
hb_agg = hb.group_by(['nxt', 'item_id']).agg(pl.col('et').unique().alias('ets'), pl.col('hpos').min().alias('hpos_min'))
j = ca.join(hb_agg, on=['nxt', 'item_id'], how='left')
j = j.with_columns(pl.col('ets').is_not_null().alias('inB'))
print(j.group_by('t').agg(pl.len().alias('n'), (pl.col('inB').mean() * 100).alias('in_B_history_%')).sort('t'))
jj = j.filter(pl.col('inB')).explode('ets')
tab = jj.group_by(['t', 'ets']).agg(pl.len().alias('n')).sort(['t', 'ets'])
npos = (ca['t'] == 1).sum(); nneg = (ca['t'] == 0).sum()
print('event_type of A-candidate in B-history: rate per 100 candidates (pos | neg) and lift')
for e in range(1, 11):
    a = tab.filter((pl.col('t') == 1) & (pl.col('ets') == e))['n'].sum(); b = tab.filter((pl.col('t') == 0) & (pl.col('ets') == e))['n'].sum()
    print(f'  et={e:2d}: pos {a/npos*100:6.2f}  neg {b/nneg*100:6.3f}  lift {(a/npos)/max(b/nneg,1e-9):7.1f}   P(target|et in B hist)={a/max(a+b,1)*100:5.1f}%')
# by gap bucket
for lo, hi in [(0, 3600), (3600, 86400), (86400, 10**9)]:
    m = (j['gap'] > lo) & (j['gap'] <= hi)
    jm = j.filter(m)
    print(f'  gap ({lo/3600:.0f}h,{hi/3600:.0f}h]: pos in B hist {jm.filter(pl.col("t")==1)["inB"].mean()*100:.1f}%  neg {jm.filter(pl.col("t")==0)["inB"].mean()*100:.2f}%  n_pos {jm.filter(pl.col("t")==1).height}')

# --------- (c) val/test: how many feeds could benefit (have a later feed from the same non-zero user)?
for s in ['val', 'test']:
    f = F[s].filter(pl.col('user_id') != 0).sort(['user_id', 'timestamp', 'feed_id'])
    f = f.with_columns(pl.col('fidx').shift(-1).over('user_id').alias('nxt'), (pl.col('timestamp').shift(-1).over('user_id') - pl.col('timestamp')).alias('gap'))
    p = f.filter(pl.col('nxt').is_not_null() & (pl.col('gap') > 0))
    print(f'\n{s}: feeds with a LATER feed of the same user: {len(p)} ({len(p)/len(F[s])*100:.2f}% of feeds); gap median {p["gap"].median()/3600 if len(p) else 0:.1f}h')
    if s == 'val' and len(p):
        A = subset('val', p['fidx'].to_numpy()); B = subset('val', p['nxt'].to_numpy())
        nmap = dict(zip(p['fidx'].to_list(), p['nxt'].to_list()))
        ca = pl.DataFrame({'nxt': np.array([nmap[x] for x in A['f']['fidx'].to_numpy()])[A['cf']], 'item_id': A['C']['item_id'], 't': A['C']['target'].astype(np.int8)})
        hb = pl.DataFrame({'nxt': B['f']['fidx'].to_numpy()[B['hf']], 'item_id': B['Hh']['item_id']}).unique()
        jv = ca.join(hb.with_columns(pl.lit(True).alias('inB')), on=['nxt', 'item_id'], how='left').with_columns(pl.col('inB').fill_null(False))
        print(jv.group_by('t').agg(pl.len().alias('n'), (pl.col('inB').mean() * 100).alias('in_later_hist_%')).sort('t'))
# val candidates appearing in the same user's TEST history (validation-only leak)
fv = F['val'].filter(pl.col('user_id') != 0); ftst = F['test'].filter(pl.col('user_id') != 0)
common_u = np.intersect1d(fv['user_id'].to_numpy(), ftst['user_id'].to_numpy())
print(f'\nval users (excl 0) that also have test feeds: {len(common_u)}; val feeds affected: {fv.filter(pl.col("user_id").is_in(common_u.tolist())).height}')
print('done', round(time.time() - t0))
