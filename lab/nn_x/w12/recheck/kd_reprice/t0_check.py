from lib import *
import time
t=time.time()
v='w5_e10_kd05w192_s951'; z=ld(v)
a=ev(z[0]); b=ndcg_per_feed(z[0],yv,ov,Mv); print('val',a.mean(),b.mean(),np.abs(a-b).max())
a=eh(z[1]); b=ndcg_per_feed(z[1],yh,oh,Mh); print('hold',a.mean(),b.mean(),np.abs(a-b).max(), len(Mh), len(yh), time.time()-t)
t=time.time(); eh(z[1]); ev(z[0]); print('fast', time.time()-t)
print('val feeds w/o positive', int((np.add.reduceat(yv,ov)==0).sum()), 'M<50 share', (Mv<50).mean())
