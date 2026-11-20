import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft")
import numpy as np
from sklearn.model_selection import GroupKFold
import pilot_read as R
v = "w5_e10_kd05w192_s951"
z = np.load(R.PR / f"nn_ident_{v}_s3_partial_folds0.npz")
n = len(R.Mv); uid = R.fv["user_id"].to_numpy().astype(np.int64); g = np.where(uid == 0, -1 - np.arange(n), uid)
tr_f, ho_f = next(iter(GroupKFold(n_splits=5).split(np.arange(n), groups=g)))
rows = np.concatenate([np.arange(R.ov[f], R.ov[f] + R.Mv[f]) for f in ho_f]); Ms = R.Mv[ho_f]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64); y = R.yv[rows]
nd = lambda s: R._ndcg(np.ascontiguousarray(s[rows], np.float32), y, offs, Ms)
st = np.load(R.W5 / f"nn_{v}_n2cpv3s3_val_oof.npy"); to = np.load(R.W5 / f"nn_{v}_trainonly_val.npy")
d = nd(z["oof_fin"]) - nd(st); print(f"FT final: mine - stored {d.mean():+.6f} +- {d.std(ddof=1)/np.sqrt(len(d)):.6f}")
d0 = nd(z["oof0"]) - nd(to); print(f"epoch 0 (no training at all): mine - stored train-only {d0.mean():+.6f} +- {d0.std(ddof=1)/np.sqrt(len(d0)):.6f}; levels {nd(z['oof0']).mean():.5f} vs {nd(to).mean():.5f}")
c0, t0 = R.feed_corr_top1(offs, Ms, z["oof0"][rows], to[rows]); print(f"epoch 0 per-feed corr {c0.mean():.7f}, top-1 differs {t0.mean():.4f}")
for e in range(1, 5):
    pass
print("epoch-by-epoch seed-mean OOF ndcg on fold 0 (mine):", [round(float(nd(z['oof'][e]).mean()), 5) for e in range(5)])
