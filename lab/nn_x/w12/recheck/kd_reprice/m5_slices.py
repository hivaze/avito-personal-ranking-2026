"""M5: where does KD / width gain live? N-matched pools (random 4-subsets averaged) of base6 (hard), narrow8, wide11 on val slices
(M bucket, H bucket, user0, weekday) and holdout slices (M bucket). Paired per-feed deltas of the mean 4-pool NDCG."""
from lib import *
import itertools, json
rng=np.random.default_rng(1)
fams={'base':fam('base',(951,952,953,954,955,956)),'narrow':fam('kd05')+fam('kd05t25'),'wide':fam('kd05w192',(951,952,953,954,955))+fam('kd05t25w192',(951,952,953,954,955,957))}
def p4(vs,k,e,nsub=40):
    subs=list(itertools.combinations(range(len(vs)),4)); subs=[subs[i] for i in rng.choice(len(subs),min(nsub,len(subs)),replace=False)]
    return np.mean([e(np.mean([ld(vs[i])[k] for i in s],0)) for s in subs],0)
V={n:p4(vs,0,ev) for n,vs in fams.items()}; H={n:p4(vs,1,eh) for n,vs in fams.items()}
Hv=fv['H'].to_numpy(); uv=fv['user_id'].to_numpy(); ts=fv['timestamp'].to_numpy(); day=(ts//86400); d0=np.unique(day)
npos=np.add.reduceat(yv,ov)
sl={'all':np.ones(len(Mv),bool),'M 5-16':Mv<=16,'M 17-49':(Mv>16)&(Mv<50),'M 50-99':(Mv>=50)&(Mv<100),'M 100-199':(Mv>=100)&(Mv<200),'M 200+':Mv>=200,
    'H 0-15':Hv<=15,'H 16-127':(Hv>15)&(Hv<128),'H 128-255':(Hv>=128)&(Hv<256),'H 256':Hv>=256,'user0':uv==0,'npos 1':npos==1,'npos 2+':npos>=2}
for d in d0: sl[f'day {d}']=day==d
print('VAL (mean 4-pool NDCG; paired deltas)')
print(f'{"slice":12s} {"n":>6s} {"share":>6s} | base4   narrow4 wide4   | narrow-base (KD)     | wide-narrow (width)   | wide-base')
out={}
for n,m in sl.items():
    b,na,w=V['base'][m],V['narrow'][m],V['wide'][m]
    print(f'{n:12s} {m.sum():6d} {m.mean():6.3f} | {b.mean():.5f} {na.mean():.5f} {w.mean():.5f} | {f(na-b)} | {f(w-na)} | {f(w-b)}')
    out['val '+n]=dict(n=int(m.sum()),kd=(na-b).mean(),kd_se=se(na-b),width=(w-na).mean(),width_se=se(w-na))
print('HOLDOUT')
slh={'all':np.ones(len(Mh),bool),'M 50-99':(Mh>=50)&(Mh<100),'M 100-199':(Mh>=100)&(Mh<200),'M 200+':Mh>=200}
for n,m in slh.items():
    b,na,w=H['base'][m],H['narrow'][m],H['wide'][m]
    print(f'{n:12s} {m.sum():6d} {m.mean():6.3f} | {b.mean():.5f} {na.mean():.5f} {w.mean():.5f} | {f(na-b)} | {f(w-na)} | {f(w-b)}')
    out['hold '+n]=dict(n=int(m.sum()),kd=(na-b).mean(),kd_se=se(na-b),width=(w-na).mean(),width_se=se(w-na))
json.dump(out,open(OUT/'m5_slices.json','w'),indent=1,default=float)
