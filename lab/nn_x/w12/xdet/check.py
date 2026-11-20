import numpy as np
D='<repo>/data/flat/train/'
pos=np.load(D+'cand_pos.npy'); y=np.load(D+'cand_target.npy'); st=np.load('out/stats.npz'); M=st['M']; gap=st['gap']; n=len(M); off=np.concatenate([[0],np.cumsum(M)[:-1]]); fid=np.repeat(np.arange(n),M)
v=np.load('out/prand_v2.npz'); new=v['p_random']; z=v['z']; old=np.load('<repo>/eda/labels_position/out/train_prandom.npz')['p_random']
rk=(np.arange(len(pos))-off[fid]); first=rk==0; top20=rk<np.maximum(1,M[fid]//5)
def rep(name,k):
    r=k[fid]; b=y[r].mean(); print(f'{name:38s} n {k.sum():7d} | z {z[k].mean():+.2f}±{z[k].std():.2f} | gap share {gap[k].mean():.3f} | pos-rate base {b:.4f} first {y[r&first].mean():.4f} (x{y[r&first].mean()/b:.2f}) top20% x{y[r&top20].mean()/b:.2f}')
big=M>=50
rep('M>=50 both (old>.9,new>.9)',big&(old>0.9)&(new>0.9))
rep('M>=50 dropped (old>.9,new<.1)',big&(old>0.9)&(new<0.1))
rep('M>=50 dropped mid (old>.9,.1<new<.5)',big&(old>0.9)&(new>=0.1)&(new<0.5))
rep('M>=50 newly found (old<.05,new>.9)',big&(old<0.05)&(new>0.9))
rep('M>=50 old nan, new>.9',big&np.isnan(old)&(new>0.9))
rep('M>=50 prod (old<.05,new<.02)',big&(old<0.05)&(new<0.02))
for lo,hi in [(5,12),(12,20),(20,35),(35,50)]:
    m=(M>=lo)&(M<hi)
    rep(f'M[{lo},{hi}) new>.9',m&(new>0.9)); rep(f'M[{lo},{hi}) .5<new<.9',m&(new>0.5)&(new<=0.9)); rep(f'M[{lo},{hi}) .1<new<.5',m&(new>0.1)&(new<=0.5)); rep(f'M[{lo},{hi}) new<.02',m&(new<0.02))
