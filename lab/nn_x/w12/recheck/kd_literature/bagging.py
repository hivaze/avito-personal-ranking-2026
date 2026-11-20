"""Data-level diversity (CSRec-D / codistillation on data shards): hard-label fold models w5_e10_xf<K>_s<seed> (each trained
without user fold K = 80% of users) vs full-data hard-label members w5_e10_base. val + holdout, train-only preds."""
import sys; sys.path.insert(0,'<repo>/src')
from pathlib import Path
import numpy as np, itertools
from common import FLAT, feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
W=Path('<repo>/nn_x/w5/preds')
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy')
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); eh=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zh=lambda x: feed_rank_z(oh,Mh,np.asarray(x,np.float32))[:,1]
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
ld=lambda v: (zv(np.load(W/f'nn_{v}_trainonly_val.npy')), zh(np.load(W/f'holdout/nn_{v}_holdout.npy')))
pool=lambda ms: (np.mean([m[0] for m in ms],0), np.mean([m[1] for m in ms],0))
B={s:ld(f'w5_e10_base_s{s}') for s in (951,952,953)}
X={(k,s):ld(f'w5_e10_xf{k}_s{s}') for k in range(5) for s in (951,952,953)}
mb=(np.mean([ev(m[0]) for m in B.values()],0), np.mean([eh(m[1]) for m in B.values()],0))
mx=(np.mean([ev(m[0]) for m in X.values()],0), np.mean([eh(m[1]) for m in X.values()],0))
print(f'member: base (100% data, 3 seeds) val {mb[0].mean():.5f} hold {mb[1].mean():.5f} | xf (80% of users, 15 models) val {mx[0].mean():.5f} hold {mx[1].mean():.5f} | xf - base: val {f(mx[0]-mb[0])} hold {f(mx[1]-mb[1])}')
c=lambda a,b: np.corrcoef(a[0],b[0])[0,1]
bb=np.mean([c(B[a],B[b]) for a,b in itertools.combinations(B,2)])
sf=np.mean([c(X[(k,a)],X[(k,b)]) for k in range(5) for a,b in itertools.combinations((951,952,953),2)])
ds=np.mean([c(X[(k,s)],X[(j,s)]) for s in (951,952,953) for k,j in itertools.combinations(range(5),2)])
dd=np.mean([c(X[(k,a)],X[(j,b)]) for k,j in itertools.combinations(range(5),2) for a in (951,952,953) for b in (951,952,953) if a!=b])
print(f'val rank-z corr: base x base (same data, different seed) {bb:.4f} | xf same fold, different seed {sf:.4f} | xf different fold, same seed {ds:.4f} | xf different fold, different seed {dd:.4f}')
# 3-member pools: same data vs different data
pb=pool(list(B.values()))
same=[pool([X[(k,s)] for s in (951,952,953)]) for k in range(5)]
diff=[pool([X[(ks[i],s)] for i,s in enumerate((951,952,953))]) for ks in ((0,1,2),(1,2,3),(2,3,4),(3,4,0),(4,0,1))]
sv=np.mean([ev(p[0]) for p in same],0); sh=np.mean([eh(p[1]) for p in same],0)
dv=np.mean([ev(p[0]) for p in diff],0); dh=np.mean([eh(p[1]) for p in diff],0)
print(f'3-pools: base (3 seeds) val {ev(pb[0]).mean():.5f} hold {eh(pb[1]).mean():.5f} | xf same fold x 3 seeds (mean of 5) val {sv.mean():.5f} hold {sh.mean():.5f} | xf 3 different folds x 3 seeds (mean of 5) val {dv.mean():.5f} hold {dh.mean():.5f}')
print(f' different-fold 3-pool minus same-fold 3-pool: val {f(dv-sv)} hold {f(dh-sh)} | different-fold xf 3-pool minus base 3-pool: val {f(dv-ev(pb[0]))} hold {f(dh-eh(pb[1]))}')
p15=pool(list(X.values())); print(f'all 15 xf models: val {ev(p15[0]).mean():.5f} hold {eh(p15[1]).mean():.5f}')
