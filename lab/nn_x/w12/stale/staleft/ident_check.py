"""staleft identity check (DECLARATION 2): finetune_stale.py --stale_train 0 --seeds 3 --only_folds 0 against the STORED 3-seed
fine-tune of the same member, on the fold-0 rows. CPU; labels are used only to score the two out-of-fold predictions."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["NUMBA_CACHE_DIR"] = "<repo>/nn_x/w12/stale/staleft/out/numba_cache"
import numpy as np, polars as pl
from pathlib import Path
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft")
import pilot_read as R
v = sys.argv[1] if len(sys.argv) > 1 else "w5_e10_kd05w192_s951"
src = Path(sys.argv[2]) if len(sys.argv) > 2 else R.W5
z = np.load(R.PR / f"nn_ident_{v}_s3_partial_folds0.npz")
n = len(R.Mv)
uid = R.fv["user_id"].to_numpy().astype(np.int64)
g = np.where(uid == 0, -1 - np.arange(n), uid)
tr_f, ho_f = next(iter(GroupKFold(n_splits=5).split(np.arange(n), groups=g)))
rows = np.concatenate([np.arange(R.ov[f], R.ov[f] + R.Mv[f]) for f in ho_f])
Ms = R.Mv[ho_f]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64)
y = R.yv[rows]
st = np.load(src / f"nn_{v}_n2cpv3s3_val_oof.npy")
mine = z["oof_fin"]
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
P(f"identity check {v}: fold 0 = {len(ho_f)} feeds, {len(rows)} rows; rows outside fold 0 in my partial file are zero: {bool((np.delete(mine, rows) == 0).all())}")
a, b = mine[rows], st[rows]
P(f"final EMA OOF, mine vs stored 3-seed file: max|d| {np.abs(a - b).max():.3e}, mean|d| {np.abs(a - b).mean():.3e}, bit-identical {bool((a == b).all())}, pooled corr {np.corrcoef(a, b)[0, 1]:.7f}")
c, t1 = R.feed_corr_top1(offs, Ms, a, b)
na, nb_ = R._ndcg(np.ascontiguousarray(a, np.float32), y, offs, Ms), R._ndcg(np.ascontiguousarray(b, np.float32), y, offs, Ms)
d = na - nb_
P(f"per-feed corr mean {c.mean():.6f} (min {c.min():.4f}); top-1 differs on {t1.mean():.4f} of feeds; NDCG mine {na.mean():.5f} stored {nb_.mean():.5f} diff {d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(len(d)):.5f}")
e0 = z["oof0"][rows]; to = np.load(src / f"nn_{v}_trainonly_val.npy")[rows]
P(f"epoch-0 (starting model) vs stored train-only val on the same rows: max|d| {np.abs(e0 - to).max():.3e}, bit-identical {bool((e0 == to).all())}")
tf = src / f"nn_{v}_n2cpv3s3_test_folds.npy"
if not tf.exists():
    tf = Path("<repo>/preds") / tf.name
if tf.exists():
    s = np.load(tf)[0].astype(np.float32); m = z["test_fin_folds"][0].astype(np.float32)
    ct, tt = R.feed_corr_top1(R.ot, R.Mt, m, s)
    P(f"test, fold-0 models (per-feed z, fp16), mine vs stored: max|d| {np.abs(m - s).max():.3e}, per-feed corr mean {ct.mean():.6f}, top-1 differs {tt.mean():.4f}, bit-identical {bool((m == s).all())}")
ok = c.mean() >= 0.9995 and abs(d.mean()) <= 0.0003
P(f"BAR (per-feed corr >= 0.9995 and |NDCG diff| <= 0.0003): {'MET' if ok else 'MISSED'}")
open(R.HERE / f"out/ident_check_{v}.txt", "w").write("\n".join(out) + "\n")
