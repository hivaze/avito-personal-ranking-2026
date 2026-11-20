from lib import *
import itertools
rng=np.random.default_rng(3)
wide=fam('kd05w192',(951,952,953,954,955))+fam('kd05t25w192',(951,952,953,954,955,957))
Z=[ld(v) for v in wide]
subs=list(itertools.combinations(range(11),8)); pick=[subs[i] for i in rng.choice(len(subs),80,replace=False)]
v=np.array([ev(np.mean([Z[i][0] for i in s],0)).mean() for s in pick]); h=np.array([eh(np.mean([Z[i][1] for i in s],0)).mean() for s in pick])
w8=[0,1,2,3,5,6,7,8]
print(f'random 8-subsets of the 11 wide students: val mean {v.mean():.5f} sd {v.std():.5f} | hold mean {h.mean():.5f} sd {h.std():.5f} (overlapping subsets: sd is a lower bound)')
print(f'actual W8 (seeds 951-954 x 2 arms): val {ev(np.mean([Z[i][0] for i in w8],0)).mean():.5f} hold {eh(np.mean([Z[i][1] for i in w8],0)).mean():.5f}')
all11=(np.mean([z[0] for z in Z],0),np.mean([z[1] for z in Z],0)); print(f'all 11: val {ev(all11[0]).mean():.5f} hold {eh(all11[1]).mean():.5f}')
# member sd
mv=np.array([ev(z[0]).mean() for z in Z]); mh=np.array([eh(z[1]).mean() for z in Z]); print(f'member val mean {mv.mean():.5f} sd {mv.std(ddof=1):.5f} | hold {mh.mean():.5f} sd {mh.std(ddof=1):.5f}')
print('members val:',np.round(mv,5)); print('members hold:',np.round(mh,5))
