"""Is GPU inference (bf16 autocast trunk + TF32 matmul) costing NDCG vs fp32? CPU fp32 forward of the 8 wide students on ALL val feeds vs their saved
 GPU train-only val files: per-member paired NDCG, within-feed logit difference, and the 8-member pool. CPU only (12 threads)."""
import sys, os
sys.path.insert(0,'<repo>/nn_x/w5'); sys.path.insert(0,'<repo>/src')
import numpy as np, torch
torch.set_num_threads(12)
from data import SplitData
from model import Ranker
from metrics import ndcg_per_feed
R='<repo>'; dev=torch.device('cpu'); os.makedirs('out',exist_ok=True)
va=SplitData('val',dev); M=va.M_np; off=va.cand_off_np; y=va.target_np; fid=np.repeat(np.arange(va.n),M)
cen=lambda x: x-np.bincount(fid,weights=x)[fid]/M[fid]
zs=lambda x: cen(x)/np.sqrt(np.bincount(fid,weights=cen(x)**2)[fid]/M[fid]+1e-18)
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
mem=[f'w5_e10_{a}_s{s}' for a in ('kd05w192','kd05t25w192') for s in (951,952,953,954)]
xt=None; C=[];G=[];D=[]
for v in mem:
    p=f'out/fp32_{v}_val.npy'; g=np.load(f'{R}/nn_x/w5/preds/nn_{v}_trainonly_val.npy').astype(np.float64)
    if os.path.exists(p): c=np.load(p).astype(np.float64)
    else:
        ck=torch.load(f'{R}/nn_x/w5/ckpt/{v}.pt',map_location='cpu',weights_only=False)
        if ck['cfg']['xt_names']!=xt: va.attach_xt(ck['cfg']['xt_names']); xt=ck['cfg']['xt_names']
        m=Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval(); out=torch.zeros(va.n_cand)
        with torch.no_grad():
            for fb in va.batches(np.arange(va.n),65536,1024,shuffle=False):
                b=va.gather(fb); va.scatter(out,b,m(b,use_pos=False))
        c=out.numpy().astype(np.float64); np.save(p,c.astype(np.float32))
    ec,eg=ndcg_per_feed(c.astype(np.float32),y,off,M),ndcg_per_feed(g.astype(np.float32),y,off,M); d=cen(c)-cen(g)
    print(f'{v}: fp32 {ec.mean():.5f} GPU file {eg.mean():.5f} paired {f(ec-eg)} | centred diff sd {d.std():.4f} p99 {np.quantile(np.abs(d),0.99):.3f} max {np.abs(d).max():.2f} | mean |logit| {np.abs(g).mean():.0f}',flush=True)
    C.append(zs(c)); G.append(zs(g)); D.append(ec-eg)
pc,pg=np.mean(C,0),np.mean(G,0); ec,eg=ndcg_per_feed(pc.astype(np.float32),y,off,M),ndcg_per_feed(pg.astype(np.float32),y,off,M)
print(f'member mean paired fp32 - GPU: {f(np.mean(D,0))} | 8-member z-pool: fp32 {ec.mean():.5f} GPU {eg.mean():.5f} paired {f(ec-eg)}')
