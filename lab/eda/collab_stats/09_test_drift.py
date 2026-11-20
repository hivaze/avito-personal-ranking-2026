"""Coverage of train-derived item statistics / CF factors across test time (Public = first 30% feeds, Private = last 70%)."""
import numpy as np, polars as pl
from common import *
rel=np.load(f'{OUT}/rel_items.npy'); lut=np.full(NITEM,-1,np.int32); lut[rel]=np.arange(len(rel),dtype=np.int32)
A=dict(np.load(f'{OUT}/item_day_aggs.npz')); n=A['n'].sum(1)
HC=np.load(f'{OUT}/train_hist_counts.npy').sum(1)
als=np.load(f'{OUT}/alsgpu_f128_i12_evt_a5_test.npz'); i2v=np.load(f'{OUT}/i2v_d128_e10_nf0_test.npz')
vc=np.bincount(np.asarray(arr('val','cand_item_id')),minlength=NITEM)>0
for split in ['val','test']:
    f=feeds(split); M=f['M'].to_numpy(); ci=np.asarray(arr(split,'cand_item_id')); r=lut[ci]
    ts=np.repeat(f['timestamp'].to_numpy(),M); fi=np.repeat(np.arange(len(f)),M)
    day=(ts-ts.min())//86400
    if split=='test':
        pub=fi< int(0.3*len(f))
        for nm,m in [('Public(first30%)',pub),('Private(last70%)',~pub)]:
            print(f'test {nm}: train-cand cov {(n[r][m]>0).mean():.3f} train-hist cov {(HC[r][m]>0).mean():.3f} ALS item cov {als["lastk20"][m].__array__().__ne__(als["lastk20"][m]).__invert__().mean():.3f} i2v cov {i2v["cov"][m].mean():.3f} val-cand cov {vc[ci][m].mean():.3f}')
    for d in np.unique(day):
        m=day==d
        if m.sum()<1000: continue
        print(f'{split} day {d}: rows {m.sum():8d} train-cand cov {(n[r][m]>0).mean():.3f} train-hist cov {(HC[r][m]>0).mean():.3f}')
