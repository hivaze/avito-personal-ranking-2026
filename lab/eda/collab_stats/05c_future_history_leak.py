"""Transductive 'future history' signal: candidate (u,i) of feed A appears in the history of a LATER feed B of the same user
(within val, or val->test cross split). Quantify on val; count availability on test."""
import numpy as np, polars as pl, time
from common import *
t0=time.time()
V=Val('val'); RP=V.rand_pf()
def tables(split):
    f=feeds(split)
    C=pl.DataFrame({'fidx':np.repeat(f['fidx'].to_numpy(),f['M'].to_numpy()),'u':np.repeat(f['user_id'].to_numpy(),f['M'].to_numpy()),
                    'ts':np.repeat(f['timestamp'].to_numpy(),f['M'].to_numpy()),'i':np.asarray(arr(split,'cand_item_id')),'row':np.arange(int(f['M'].sum()))})
    Hh=pl.DataFrame({'fidx2':np.repeat(f['fidx'].to_numpy(),f['H'].to_numpy()),'u':np.repeat(f['user_id'].to_numpy(),f['H'].to_numpy()),
                     'ts2':np.repeat(f['timestamp'].to_numpy(),f['H'].to_numpy()),'i':np.asarray(arr(split,'hist_item_id')),'et':np.asarray(arr(split,'hist_event_type')).astype(np.int8),'hpos':np.asarray(arr(split,'hist_hpos'))})
    return f,C,Hh
fv,Cv,Hv=tables('val'); ft,Ct,Ht=tables('test')
for nm,f in [('val',fv),('test',ft)]:
    g=f.group_by('user_id').agg(pl.len().alias('n'),pl.col('timestamp').max().alias('tmax'))
    j=f.join(g,on='user_id'); has_later=(j['timestamp']<j['tmax']).to_numpy()
    print(f'{nm}: feeds whose user has a LATER feed in same split: {has_later.mean():.4f} ({has_later.sum()})')
# val feeds whose user has a later test feed
uv_later_test = fv.join(ft.group_by('user_id').agg(pl.col('timestamp').max().alias('tt')),on='user_id',how='inner')
print('val feeds whose user has a test feed:', len(uv_later_test))
# feature on val: within-val later history + test history (test is later than val)
J=Cv.join(pl.concat([Hv, Ht]),on=['u','i']).filter(pl.col('ts2')>pl.col('ts'))
agg=J.group_by('row').agg(pl.len().alias('n_ev'), pl.col('et').min().alias('min_et'), pl.col('et').filter(pl.col('et')==1).len().alias('n_et1'))
x=np.zeros(len(V.target)); x[agg['row'].to_numpy()]=agg['n_ev'].to_numpy()
t=V.target
print('val cand rows found in later history:', (x>0).sum(), 'target rate among them', t[x>0].mean(), 'vs base', t.mean())
print('share of all val positives found:', (t*(x>0)).sum()/t.sum())
fm = np.bincount(V.fid,weights=(x>0))>0
a,d,se=V.cmp(x,RP); print(f'later-history feature alone: ndcg={a:.4f} lift={d:+.4f}+-{se:.4f}; feeds affected {fm.sum()}')
f1=np.asarray(arr('val','cand_f1'),dtype=np.float64)
F1=V.pf_rt(f1); comb=V.pf_rt(tiebreak(x>0, f1))
print(f'f1 tiebreak only {F1.mean():.4f} -> (later_hist>0, then f1) {comb.mean():.4f}; diff {comb.mean()-F1.mean():+.4f}')
np.save(f'{OUT}/val_future_hist.npy', x)
# test availability (within test only; nothing after test)
Jt=Ct.join(Ht,on=['u','i']).filter(pl.col('ts2')>pl.col('ts'))
print('test cand rows found in later test history:', Jt['row'].n_unique(), 'feeds:', Ct.filter(pl.col('row').is_in(Jt['row'].unique().to_list()))['fidx'].n_unique())
tlog(t0,'done')
