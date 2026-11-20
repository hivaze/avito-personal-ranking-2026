# Second detector, step 3. Per feed-size bin it turns the classifier score of adv.py into a probability
# (density ratio of val + test to train) and multiplies the odds by an order likelihood ratio: how well the
# ranker's auxiliary head, which predicts the production order, agrees with the logged positions
# (out/stats2.npz, built by stats2.py from the dumps of dump_aux.py in the same lab folder).
# Run from lab/nn_x/w12/xdet. The command line is not on record.
# Produced out/prand_v2.npz, installed as lab/eda/labels_position/out/train_prandom_v2.npz.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/xdet/post.py
#
import numpy as np, polars as pl
Xt=np.load('out/setfeat_train.npy')[:,0]; Xp=np.concatenate([np.load('out/setfeat_val.npy')[:,0],np.load('out/setfeat_test.npy')[:,0]])
s=np.load('out/adv_train.npy').astype(np.float64); sp=np.load('out/adv_pos.npy').astype(np.float64); z=np.load('out/stats2.npz')['zce10'].astype(np.float64)
old=np.load('<repo>/eda/labels_position/out/train_prandom.npz')['p_random']
M=Xt.astype(int); Mp=Xp.astype(int); n=len(M)
edges=[5,6,7,8,10,12,15,20,25,30,35,42,50,65,80,100,130,160,200,250,301]
post_set=np.zeros(n); post=np.zeros(n); pis=[]
phi=lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(M>=lo)&(M<hi); mp=(Mp>=lo)&(Mp<hi)
    # density ratio f_pos/f_train from histogram of scores (robust, per M-bin), smoothed
    qs=np.quantile(np.r_[s[m][::7],sp[mp]],np.linspace(0,1,41)); qs[0]-=1; qs[-1]+=1; qs=np.unique(qs)
    ht=np.histogram(s[m],qs)[0]/m.sum(); hp=np.histogram(sp[mp],qs)[0]/mp.sum()
    t=np.median(sp[mp]); pi=np.mean(s[m]>t)/0.5   # production feeds assumed rare above the positives' median
    # refine: pi = min over upper bins of ht/hp (plateau), use top-half bins weighted
    up=qs[:-1]>=t; pi2=ht[up].sum()/hp[up].sum()
    r=np.clip(pi2*hp/np.maximum(ht,1e-9),0,1); b=np.clip(np.searchsorted(qs,s[m],'right')-1,0,len(ht)-1); ps=r[b]; post_set[m]=ps
    # order evidence: g_prod(z) from feeds with ps<0.02
    zz=z[m]; zb=np.linspace(-4,9,53); gp=np.histogram(zz[ps<0.02],zb,density=True)[0]+1e-4; zi=np.clip(np.searchsorted(zb,zz,'right')-1,0,len(gp)-1)
    lr=phi(zz)/gp[zi]; lr=np.clip(lr,1/50,50); od=ps/(1-ps+1e-9)*lr; pp=od/(1+od); post[m]=pp
    hi9=ps>0.9; pis.append((lo,hi,m.sum(),pi,pi2,ps.mean(),pp.mean()))
    print(f'M[{lo},{hi}) n {m.sum():7d} pi {pi:.3f} pi2 {pi2:.3f} mean post_set {ps.mean():.3f} final {pp.mean():.3f} | set>0.9: n {hi9.sum():6d} z mean {zz[hi9].mean():+.2f} sd {zz[hi9].std():.2f} | set<0.02: z mean {zz[ps<0.02].mean():.2f} | final>0.9 n {(pp>0.9).sum():6d} >0.5 n {(pp>0.5).sum():6d} sum {pp.sum():.0f}'
          +(f' | old>0.9: mean final {pp[old[m]>0.9].mean():.3f}; old<0.05: {pp[old[m]<0.05].mean():.3f}; old nan: n>0.5 {(pp[np.isnan(old[m])]>0.5).sum()}' if lo>=50 else ''),flush=True)
np.savez('out/prand_v2.npz',fidx=np.arange(n),p_random=post.astype(np.float32),p_set=post_set.astype(np.float32),z=z.astype(np.float32))
fd=pl.read_parquet('<repo>/data/flat/train/feeds.parquet'); day=((fd['timestamp'].to_numpy()+3*3600)//86400); 
for d in np.unique(day): 
    k=day==d
    if k.sum()>1000: print('day',d,k.sum(),'share post>0.5: M<50 %.3f M>=50 %.3f'%(np.mean(post[k&(M<50)]>0.5),np.mean(post[k&(M>=50)]>0.5)))
print('TOTAL >0.5:',(post>0.5).sum(),' M<50:',((post>0.5)&(M<50)).sum(),' M>=50:',((post>0.5)&(M>=50)).sum(),'| old pool (old>0.5,M>=50):',((np.nan_to_num(old)>0.5)&(M>=50)).sum(),' overlap:',((np.nan_to_num(old)>0.5)&(M>=50)&(post>0.5)).sum())
