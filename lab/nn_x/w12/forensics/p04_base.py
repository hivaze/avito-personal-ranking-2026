from fx import *
from stack_val import feed_rank_z
f,M,off,ft,to=load_w8("val"); y=np.load(FLAT/"val/cand_target.npy")
a=feed_rank_z(off,M,ft.astype(np.float32)); b=feed_rank_z(off,M,to.astype(np.float32))
for k in (0,1):
    s=0.5*a[:,k]+0.5*b[:,k]; print("col",k,ndcg_per_feed(s,y,off,M).mean())
print("ft",ndcg_per_feed(ft,y,off,M).mean(),"to",ndcg_per_feed(to,y,off,M).mean())
print(ft.shape, ft.dtype, ft[:5], to[:5])
