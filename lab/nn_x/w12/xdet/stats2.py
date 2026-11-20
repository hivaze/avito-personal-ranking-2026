import numpy as np
D='<repo>/data/flat/train/'
pos=np.load(D+'cand_pos.npy').astype(np.float64)
aux=sum(np.load(f'out/aux_s{s}.npy').astype(np.float64) for s in (101,102,103))/3
s=np.load('out/stats.npz'); M=s['M']; n=len(M); off=np.concatenate([[0],np.cumsum(M)[:-1]]); fid=np.repeat(np.arange(n),M)
def pz(t,x):  # exact permutation z of sum(t*x) per feed
    tm=np.bincount(fid,t,n)/M; xm=np.bincount(fid,x,n)/M
    tc=t-tm[fid]; xc=x-xm[fid]
    num=np.bincount(fid,tc*xc,n); vt=np.bincount(fid,tc*tc,n); vx=np.bincount(fid,xc*xc,n)
    return num/np.sqrt(np.maximum(vt*vx/(M-1),1e-30))
out={}
# log-softmax of aux within feed
mx=np.maximum.reduceat(aux,off); e=np.exp(aux-mx[fid]); lse=np.log(np.bincount(fid,e,n))+mx; lp=aux-lse[fid]
for tau in (3,10,30):
    t=np.exp(-pos/tau); out[f'zce{tau}']=pz(t,lp); out[f'zra{tau}']=pz(t,aux)
out['zlin']=pz(-pos,aux)
np.savez('out/stats2.npz',**{k:v.astype(np.float32) for k,v in out.items()})
p=np.load('<repo>/eda/labels_position/out/train_prandom.npz')['p_random']; z0=s['z']
for lo,hi in [(5,8),(8,12),(12,20),(20,35),(35,50),(50,100),(100,200),(200,301)]:
    m=(M>=lo)&(M<hi); line=f'M[{lo},{hi})'
    for k in ['z0']+list(out):
        zz=(z0 if k=='z0' else out[k])[m]; line+=f' | {k}: med {np.median(zz):5.2f} P<0 {np.mean(zz<0):.3f} P<-1 {np.mean(zz<-1):.3f}'
    print(line)
