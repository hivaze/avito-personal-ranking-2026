import numpy as np
D='<repo>/data/flat/train/'
pos=np.load(D+'cand_pos.npy'); y=np.load(D+'cand_target.npy'); st=np.load('out/stats.npz'); M=st['M']; gap=st['gap']; n=len(M); off=np.concatenate([[0],np.cumsum(M)[:-1]]); fid=np.repeat(np.arange(n),M)
v=np.load('out/prand_v2.npz'); new=v['p_random']; z=v['z']; old=np.load('<repo>/eda/labels_position/out/train_prandom.npz')['p_random']
o=np.lexsort((pos,fid)); rk=np.empty(len(pos),np.int32); rk[o]=np.arange(len(pos))-off[fid[o]]
first=rk==0; top=rk<np.maximum(1,M[fid]//5); last=rk==M[fid]-1
def rep(name,k):
    r=k[fid]; b=y[r&~last].mean(); print(f'{name:38s} n {k.sum():7d} | z {z[k].mean():+.2f} | gap {gap[k].mean():.3f} | base(excl last) {b:.4f} first x{y[r&first].mean()/b:.2f} top20% x{y[r&top&~last].mean()/b:.2f} last x{y[r&last].mean()/b:.1f}')
big=M>=50
rep('M>=50 both (old>.9,new>.9)',big&(old>0.9)&(new>0.9))
rep('M>=50 dropped (old>.9,new<.1)',big&(old>0.9)&(new<0.1))
rep('M>=50 dropped mid (old>.9,.1<new<.5)',big&(old>0.9)&(new>=0.1)&(new<0.5))
rep('M>=50 newly found (old<.05,new>.9)',big&(old<0.05)&(new>0.9))
rep('M>=50 old nan, new>.9',big&np.isnan(old)&(new>0.9))
rep('M>=50 prod (old<.05,new<.02)',big&(old<0.05)&(new<0.02))
for lo,hi in [(5,12),(12,20),(20,35),(35,50)]:
    m=(M>=lo)&(M<hi)
    for a,b in [(0.9,1.01),(0.5,0.9),(0.1,0.5),(0.02,0.1),(-1,0.02)]: rep(f'M[{lo},{hi}) {a}<new<={b}',m&(new>a)&(new<=b))
