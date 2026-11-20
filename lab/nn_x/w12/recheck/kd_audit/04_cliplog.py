"""K9 aux vs main gradient norms, hard-label base vs KD arms (from the saved CLIPLOG splits; CPU only)."""
import numpy as np
L='<repo>/nn_x/w5/logs/'
t_cool=int(round(0.8*98350))
def stats(arm,seeds=(951,952,953,954)):
    out=[]
    for s in seeds:
        try: z=np.load(L+f'w5_e10_{arm}_s{s}_cliplog.npz',allow_pickle=True)
        except Exception as e: continue
        r=z['split']; st=r[:,0]; pool=r[:,2]>0
        row=[]
        for nm,m in (('plateau 10-78K',(st>=10000)&(st<t_cool)&~pool),('cooldown',(st>=t_cool)&~pool),('last 5K',(st>=93350)&~pool)):
            row+=[np.median(r[m,3]),np.median(r[m,4]),np.median(r[m,4]/np.maximum(r[m,3],1e-12))]
        g=z['gn']; row+=[np.nanmedian(g[10000:t_cool]),np.nanmean(g[10000:t_cool]>1.0)]
        out.append(row)
    return np.array(out)
print('arm          n | plateau main / aux / ratio | cooldown main / aux / ratio | last5K main / aux / ratio | plateau total gn median, clipped share')
for a in ('base','kd03','kd05','kd08','kd10','kd05w192','kd05t25w192','kx05','kwT07','kwT15','kwsub5','ap1'):
    x=stats(a)
    if len(x)==0: print(f'{a:12s} no cliplog'); continue
    m=x.mean(0); print(f'{a:12s} {len(x)} | {m[0]:.3f} / {m[1]:.3f} / {m[2]:.3f} | {m[3]:.3f} / {m[4]:.3f} / {m[5]:.3f} | {m[6]:.3f} / {m[7]:.3f} / {m[8]:.3f} | {m[9]:.3f} {m[10]:.4f}')
