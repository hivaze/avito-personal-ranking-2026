import sys; sys.path.insert(0,"<repo>/src"); sys.path.insert(0,"<repo>/eda/errors/synthesis")
from common import *; from stack_val import feed_rank_z; from qt import leak_mask

SNAP = __import__("pathlib").Path("<repo>/eda/errors/synthesis/snap")
def pp(fn):
    import pathlib
    return SNAP / fn if (SNAP / fn).exists() else pathlib.Path("<repo>/preds") / fn

f,M,off=feed_meta("val"); y=np.load(FLAT/"val/cand_target.npy").astype(np.float32); kf=~leak_mask()
P=Path("<repo>/preds")
def z(n): return feed_rank_z(off,M,np.load(pp(n)).astype(np.float32))[:,1]
res={}
for n in ["nn_final3_trainonly_val.npy","nn_final3_val_oof.npy","nn_final2_trainonly_val.npy","nn_final2_val_oof.npy","stk2_res4_clean_val.npy","xgb_x300_luo_val.npy"]:
    per=ndcg_per_feed(np.load(pp(n)).astype(np.float32),y,off,M); res[n]=per; print(n, f"all {per.mean():.5f} leakfree {per[kf].mean():.5f}")
b=ndcg_per_feed(0.5*z("nn_final3_val_oof.npy")+0.5*z("nn_final3_trainonly_val.npy"),y,off,M); print("blend f3oof+f3tr", b[kf].mean())
d=res["nn_final3_val_oof.npy"]-res["nn_final3_trainonly_val.npy"]; print("f3 oof - tr", d[kf].mean(), d[kf].std()/np.sqrt(kf.sum()))
