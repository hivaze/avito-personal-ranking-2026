import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
MSK=3*3600
CONTENT=['f1','f2','f3','f4','f5','f1_z','f2_z','f3_z','f4_z','f5_z','f1_rk','f2_rk','f3_rk','f4_rk','f5_rk',
 'cos_u','cos_u_z','maxsim32','maxsim32_z','meansim32','attn32','attn32_z','cos_h10','cos_hall','seq_max_new8','seq_max_all',
 'seq_top3','seq_dcos_hl2','cat_cnt_20','cat_cnt_all','cat_share_all','feed_cat_share','ctx_hs_t10','ctx_hn_soft05','ctx_hn_rrw',
 'ctx_nd_knn_u','ctx_nd_top5h','emb_missing','item_norm','M','H']
COLLAB=['it_hist_all_luo','it_hist_et1_luo','it_hist_all_luo_z','seq_cv_invd20','seq_cv_pop','seq_cv_norm','seq_cv_invd20_rk']
LOG={'cat_cnt_20','cat_cnt_all','M','H','it_hist_all_luo','it_hist_et1_luo','seq_cv_invd20','seq_cv_pop','seq_cv_norm'}

def rows_of(off,M,fids):
    return np.concatenate([np.arange(off[i],off[i]+M[i]) for i in fids])

def load_X(split,rows,names):
    X=np.empty((len(rows),len(names)),np.float32)
    for j,n in enumerate(names):
        x=np.asarray(np.load(FEATS/split/f'{n}.npy',mmap_mode='r')[rows],dtype=np.float32)
        if n in LOG: x=np.log1p(np.maximum(x,0))
        X[:,j]=x
    return X

def day_of(ts): return (np.asarray(ts,np.int64)+MSK)//86400-20702

def prandom():
    pr=np.load('<repo>/eda/labels_position/out/train_prandom.npz'); p=np.zeros(3151208,np.float32); p[pr['fidx']]=pr['p_random']; return p

def per_feed(scores,split,fids=None):
    f,M,off=feed_meta(split)
    y=np.load(FLAT/split/'cand_target.npy',mmap_mode='r')
    if fids is None:
        return ndcg_per_feed(scores,np.asarray(y),off,M)
    Ms=M[fids]; offs=np.concatenate([[0],np.cumsum(Ms)[:-1]])
    r=rows_of(off,M,fids)
    return ndcg_per_feed(scores,np.asarray(y[r]),offs,Ms)

def paired(a,b):
    d=a-b; return d.mean(), d.std()/np.sqrt(len(d))
