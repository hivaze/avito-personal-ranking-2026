"""M14: are the 20 feed-context inputs (ctx_nd/hn/hs, computed over the FULL candidate set) M-dependent? If a feature's distribution
differs between short and long real feeds, then --aug_all --aug_cand sub-feeds (which keep the full-feed values: the R2 'stale' caveat in
cd_arms.py) are out of format for that input. Val (real shuffled feeds, all M). Effect size = (mean in M<=16 - mean in M>=100) / pooled sd,
per candidate; also positives only. Label use: slicing only (diagnostic), nothing enters a model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from lib import *
FC=ROOT/'nn/feat_cache/val'
names=['ctx_nd_hcent','ctx_nd_knn_f1','ctx_nd_knn_h','ctx_nd_knn_h_dev','ctx_nd_knn_u','ctx_nd_knn_u_dev','ctx_nd_max','ctx_nd_top5h','ctx_nd_top5u','ctx_nd_ucent',
       'ctx_hn_top1w','ctx_hn_rrw','ctx_hn_bestrk','ctx_hn_margin','ctx_hn_soft05','ctx_hn_e1_rrw','ctx_hs_mutual','ctx_hs_k128','ctx_hs_e1','ctx_hs_t10']
Mr=np.repeat(Mv,Mv); b=[Mr<=16,(Mr>16)&(Mr<50),(Mr>=50)&(Mr<100),Mr>=100]
print(f'{"feature":18s} mean by M bucket [5-16] [17-49] [50-99] [100+]            | effect size short-vs-long (all cands) | (positives)')
big=0
for n in names:
    x=np.load(FC/f'{n}.npy').astype(np.float64); ok=np.isfinite(x)
    mu=[x[m&ok].mean() for m in b]; sd=x[ok].std()
    es=(mu[0]-mu[3])/sd
    p=yv>0; esp=(x[b[0]&ok&p].mean()-x[b[3]&ok&p].mean())/sd
    big+=abs(es)>0.5
    print(f'{n:18s} {mu[0]:9.4f} {mu[1]:9.4f} {mu[2]:9.4f} {mu[3]:9.4f} | sd {sd:8.4f} | {es:+.2f} | {esp:+.2f}')
print('features with |effect| > 0.5 sd:',big,'of',len(names))
