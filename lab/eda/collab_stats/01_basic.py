import numpy as np, polars as pl, time
from common import *
t0=time.time()
# item presence sets via bincount
tr_c = arr('train','cand_item_id')
cnt_tr_cand = np.bincount(tr_c, minlength=NITEM); tlog(t0,'train cand bincount')
cnt_tr_hist = np.bincount(arr('train','hist_item_id'), minlength=NITEM); tlog(t0,'train hist bincount')
res={}
for s in ['val','test']:
    c = np.asarray(arr(s,'cand_item_id'))
    h = np.asarray(arr(s,'hist_item_id'))
    cnt_c = np.bincount(c, minlength=NITEM); cnt_h=np.bincount(h, minlength=NITEM)
    print(f'--- {s}: cand rows {len(c)}, unique cand items {len(np.unique(c))}, hist rows {len(h)}')
    print(f' frac cand rows whose item seen as train candidate: {(cnt_tr_cand[c]>0).mean():.4f}')
    print(f' frac cand rows whose item seen in train history: {(cnt_tr_hist[c]>0).mean():.4f}')
    print(f' frac cand rows whose item seen in train cand OR hist: {((cnt_tr_hist[c]+cnt_tr_cand[c])>0).mean():.4f}')
    print(f' frac cand rows whose item appears in same-split history (any feed): {(cnt_h[c]>0).mean():.4f}')
    print(f' frac cand rows whose item appears as cand in >1 feeds same split: {(cnt_c[c]>1).mean():.4f}')
    u = np.unique(c); print(f' unique items in train cand: {(cnt_tr_cand[u]>0).mean():.4f}')
    q = np.quantile(cnt_tr_cand[c],[0.1,0.25,0.5,0.75,0.9]); print(' train impressions quantiles (per cand row):', q)
    if s=='test':
        vc = np.asarray(arr('val','cand_item_id')); vcnt=np.bincount(vc,minlength=NITEM)
        vh = np.bincount(np.asarray(arr('val','hist_item_id')),minlength=NITEM)
        print(f' test cand rows whose item is a val candidate: {(vcnt[c]>0).mean():.4f}; in val hist: {(vh[c]>0).mean():.4f}')
# users
ftr = feeds('train'); fv=feeds('val'); fte=feeds('test')
utr = set(ftr['user_id'].unique().to_list())
for nm,f in [('val',fv),('test',fte)]:
    u=f['user_id'].to_numpy()
    print(nm,'feeds whose user appears in train:', np.mean([x in utr for x in u]), ' unique users', len(set(u)), 'feeds/user', len(u)/len(set(u)))
print('val users in test:', np.mean([x in set(fte['user_id'].to_list()) for x in fv['user_id'].unique().to_list()]))
# train per-user feed counts
uc = ftr.group_by('user_id').len()['len'].to_numpy(); print('train feeds/user quantiles', np.quantile(uc,[.5,.9,.99,1]), 'mean',uc.mean())
# train position stats
pos = arr('train','cand_pos'); print('pos max', pos.max())
tr_t = arr('train','cand_target'); print('train pos rate', float(np.asarray(tr_t[:50_000_000]).mean()))
# timestamps
for nm,f in [('train',ftr),('val',fv),('test',fte)]:
    ts=f['timestamp'].to_numpy(); print(nm, 'ts range', ts.min(), ts.max(), (ts.max()-ts.min())/86400)
