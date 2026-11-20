"""C3-msplit: C3 (Q2a8s3) with the TRAIN-ONLY half alone on M<50 feeds, 0.5 z(FT) + 0.5 z(TO) on M>=50. Threshold 50 fixed a priori
(INSIGHTS_1112 item 2; never tuned here). Writes preds/candX_C3msplit_{val,test}.npy and (with 'build') subs/cand_C3_msplit_TOlt50.parquet."""
import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path; ROOT=Path('<repo>'); T='Q2a8s3'; P=ROOT/'preds'
fv,Mv,ov=feed_meta('val'); ft_,Mt,ot=feed_meta('test'); yv=np.load(FLAT/'val/cand_target.npy')
def blend(sp,M,o):
    zf=feed_rank_z(o,M,np.load(P/f'ens_{T}_ft_{sp}.npy').astype(np.float32))[:,1]; zt=feed_rank_z(o,M,np.load(P/f'ens_{T}_to_{sp}.npy').astype(np.float32))[:,1]
    small=np.repeat(M<50,M); return np.where(small,zt,0.5*zf+0.5*zt).astype(np.float32), (0.5*zf+0.5*zt).astype(np.float32)
bv,rv=blend('val',Mv,ov); bt,rt=blend('test',Mt,ot)
e=lambda s: ndcg_per_feed(s,yv,ov,Mv); d=e(bv)-e(rv)
print(f'C3 val {e(rv).mean():.5f} -> msplit {e(bv).mean():.5f}  delta {d.mean():+.5f}+-{d.std()/np.sqrt(len(d)):.5f} | M<50 {d[Mv<50].mean():+.5f}+-{d[Mv<50].std()/np.sqrt((Mv<50).sum()):.5f}')

import polars as pl
tv=pl.read_parquet(FLAT/'val/feeds.parquet')['timestamp'].to_numpy(); day=(tv+3*3600)//86400; 
for dd in np.unique(day):
    k=day==dd
    if k.sum()>100: print(' day',dd,k.sum(),f'{d[k].mean():+.5f}+-{d[k].std()/np.sqrt(k.sum()):.5f}')
print('test corr to C3:',np.corrcoef(bt,rt)[0,1],' test feeds M<50:',(Mt<50).sum())
np.save(P/'candX_C3msplit_val.npy',bv); np.save(P/'candX_C3msplit_test.npy',bt); np.save(P/'candX_C3ref_val.npy',rv); np.save(P/'candX_C3ref_test.npy',rt)
