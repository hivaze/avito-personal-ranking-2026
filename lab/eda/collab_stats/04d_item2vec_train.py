"""SGNS item2vec on GPU (CUDA_VISIBLE_DEVICES=N). Skip-gram windows over user histories + train next-item pairs."""
import os, sys, time
import numpy as np, torch, polars as pl
from common import *
t0=time.time()
D=int(sys.argv[1]) if len(sys.argv)>1 else 64
EPOCHS=float(sys.argv[2]) if len(sys.argv)>2 else 4
NEXT_FRAC=float(sys.argv[3]) if len(sys.argv)>3 else 0.15
W=5; NEG=5; B=1_000_000
TAG=f'i2v_d{D}_e{EPOCHS:g}_nf{NEXT_FRAC:g}'
dev='cuda'
tok=torch.from_numpy(np.load(f'{OUT}/i2v_tokens.npy')).to(dev)
sid=torch.from_numpy(np.load(f'{OUT}/i2v_sid.npy')).to(dev)
nxt=torch.from_numpy(np.load(f'{OUT}/i2v_nextpairs.npy')).to(dev)
cnt=torch.from_numpy(np.load(f'{OUT}/i2v_counts.npy')).to(dev).double()
V=len(cnt); N=len(tok); NP=nxt.shape[1]
cdf=torch.cumsum(cnt**0.75,0); cdf=(cdf/cdf[-1]).float()
emb_in=torch.nn.Embedding(V,D,sparse=True).to(dev); emb_out=torch.nn.Embedding(V,D,sparse=True).to(dev)
with torch.no_grad():
    emb_in.weight.uniform_(-0.5/D,0.5/D); emb_out.weight.zero_()
opt=torch.optim.SparseAdam(list(emb_in.parameters())+list(emb_out.parameters()), lr=0.005)
steps=int(EPOCHS*N*2/B)
print(TAG,'V',V,'N',N,'steps',steps, flush=True)
g=torch.Generator(device=dev); g.manual_seed(0)
for st in range(steps):
    nb_next=int(B*NEXT_FRAC); nb=B-nb_next
    c=torch.randint(0,N,(nb,),device=dev,generator=g)
    o=torch.randint(1,W+1,(nb,),device=dev,generator=g)*(torch.randint(0,2,(nb,),device=dev,generator=g)*2-1)
    x=(c+o).clamp(0,N-1)
    ok=sid[x]==sid[c]
    a=tok[c[ok]]; b=tok[x[ok]]
    j=torch.randint(0,NP,(nb_next,),device=dev,generator=g)
    a=torch.cat([a,nxt[1,j]]); b=torch.cat([b,nxt[0,j]])   # (history item -> clicked candidate)
    neg=torch.searchsorted(cdf, torch.rand(len(a),NEG,device=dev,generator=g)).clamp(max=V-1)
    ea=emb_in(a); eb=emb_out(b); en=emb_out(neg)
    pos_l=torch.nn.functional.logsigmoid((ea*eb).sum(-1))
    neg_l=torch.nn.functional.logsigmoid(-(en*ea.unsqueeze(1)).sum(-1)).sum(-1)
    loss=-(pos_l+neg_l).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if st%200==0: print(f'step {st}/{steps} loss {loss.item():.4f} t={time.time()-t0:.0f}s', flush=True)
Ein=emb_in.weight.detach(); Eout=emb_out.weight.detach()
vocab=np.load(f'{OUT}/i2v_vocab.npy'); vlut=np.full(NITEM,-1,np.int64); vlut[vocab]=np.arange(len(vocab))
Einn=Ein/Ein.norm(dim=1,keepdim=True).clamp(min=1e-6)
for split in ['val','test']:
    f=feeds(split); Mm=f['M'].to_numpy(); H=f['H'].to_numpy(); nf=len(f)
    ci=vlut[np.asarray(arr(split,'cand_item_id'))]; hi=vlut[np.asarray(arr(split,'hist_item_id'))]; hp=np.asarray(arr(split,'hist_hpos'))
    hf=np.repeat(np.arange(nf),H); cf=np.repeat(np.arange(nf),Mm)
    okc=torch.from_numpy(ci>=0).to(dev); cit=torch.from_numpy(np.maximum(ci,0)).to(dev); cft=torch.from_numpy(cf).to(dev)
    out={'cov':ci>=0}
    for K in [5,20,256]:
        mh=(hi>=0)&(hp<K)
        hft=torch.from_numpy(hf[mh]).to(dev); hit=torch.from_numpy(hi[mh]).to(dev)
        prof_in=torch.zeros(nf,D,device=dev).index_add_(0,hft,Einn[hit])
        prof_in_n=prof_in/prof_in.norm(dim=1,keepdim=True).clamp(min=1e-6)
        s_cos=(prof_in_n[cft]*Einn[cit]).sum(-1)       # in-in cosine with mean profile
        s_io=(prof_in_n[cft]*Eout[cit]).sum(-1)        # profile(in) . out(candidate): 'next item' score
        for nm,s in [('cos',s_cos),('io',s_io)]:
            s=torch.where(okc,s,torch.full_like(s,float('nan'))).cpu().numpy()
            out[f'{nm}_k{K}']=s
        # max-sim over last-K items (computed per history event then max per candidate) only for K=5/20 (memory)
        if K<=20:
            # for each candidate compute max over the feed's last-K history items: loop over hpos
            mx=torch.full((len(ci),),-2.0,device=dev)
            for p in range(K):
                mp=(hi>=0)&(hp==p)
                vec=torch.zeros(nf,D,device=dev); has=torch.zeros(nf,dtype=torch.bool,device=dev)
                ft=torch.from_numpy(hf[mp]).to(dev); vec[ft]=Einn[torch.from_numpy(hi[mp]).to(dev)]; has[ft]=True
                sp_=(vec[cft]*Einn[cit]).sum(-1); sp_=torch.where(has[cft],sp_,torch.full_like(sp_,-2.0))
                mx=torch.maximum(mx,sp_)
            mx=torch.where(okc&(mx>-2),mx,torch.full_like(mx,float('nan'))).cpu().numpy()
            out[f'maxcos_k{K}']=mx
    np.savez(f'{OUT}/{TAG}_{split}.npz',**out)
    print(split,'cand coverage', (ci>=0).mean(), flush=True)
need=np.unique(np.concatenate([vlut[np.asarray(arr(sp_,nm_))] for sp_ in ['val','test'] for nm_ in ['cand_item_id','hist_item_id']]))
need=need[need>=0]; nt=torch.from_numpy(need).to(dev)
np.savez(f'{OUT}/{TAG}_emb_sub.npz', vocab_idx=need, item_id=vocab[need], ein=Ein[nt].cpu().numpy(), eout=Eout[nt].cpu().numpy())
del tok,sid
Vv=Val('val'); RP=Vv.rand_pf()
d=dict(np.load(f'{OUT}/{TAG}_val.npz'))
for k,s in d.items():
    if k=='cov': continue
    x=np.where(np.isnan(s), np.nanmin(s)-1, s)
    a,dd,se=Vv.cmp(x,RP); print(f'{TAG} {k:10s} ndcg={a:.4f} lift={dd:+.4f}+-{se:.4f}', flush=True)
tlog(t0,'done')
