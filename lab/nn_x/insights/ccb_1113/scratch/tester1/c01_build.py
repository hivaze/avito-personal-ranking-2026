"""C01 build: LightGBM lambdarank on cb26 XM rows (E3 feature list) -> score + pred_leaf -> TruncatedSVD(64) on 2M train rows.
Outputs out/c01_{score,svd}_{val,hold}.npy. CPU 12 threads."""
import os, sys, json, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np, polars as pl
import lightgbm as lgb
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = Path("<repo>")
FEATS, FNEW, FLAT = ROOT / "data/feats", ROOT / "data/feats_new", ROOT / "data/flat"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


names = json.loads((ROOT / "nn_x/cb26/data/e3_xm_yeti.json").read_text())["features"]
log(f"E3 features: {len(names)} (file nn_x/cb26/data/e3_xm_yeti.json)")
ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
Mtr = ftr["M"].to_numpy().astype(np.int64); otr = ftr["cand_off"].to_numpy().astype(np.int64)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(len(ftr))).all()
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
sel = np.setdiff1d(np.where((p > 0.5) & (Mtr >= 50))[0], hold)
assert len(sel) == 134118, len(sel)
rows = np.repeat(otr[sel] - np.r_[0, np.cumsum(Mtr[sel])[:-1]], Mtr[sel]) + np.arange(Mtr[sel].sum())
hrows = np.concatenate([np.arange(otr[i], otr[i] + Mtr[i]) for i in hold])
log(f"train feeds {len(sel)} rows {len(rows)} holdout overlap {np.isin(sel, hold).sum()} | holdout rows {len(hrows)}")


def load(split, rr=None):
    cols = []
    for n in names:
        q = FEATS / split / f"{n}.npy"
        if not q.exists():
            q = FNEW / split / f"{n}.npy"
        x = np.load(q, mmap_mode="r")
        cols.append(np.asarray(x if rr is None else x[rr], dtype=np.float32))
    return np.stack(cols, 1)


Xtr = load("train", rows)
ytr = np.asarray(np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows]).astype(np.float32)
log(f"Xtr {Xtr.shape} pos rate {ytr.mean():.4f}")
cat_idx = [names.index("category_id")]
params = dict(objective="lambdarank", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, label_gain=[0, 1],
              lambdarank_truncation_level=300, seed=0, num_threads=12, verbose=-1, deterministic=True, force_row_wise=True)
ds = lgb.Dataset(Xtr, ytr, group=Mtr[sel], feature_name=[n.replace("-", "_") for n in names], categorical_feature=cat_idx,
                 free_raw_data=False)
bst = lgb.train(params, ds, 500, callbacks=[lgb.log_evaluation(50)])
bst.save_model(str(OUT / "c01_lgb.txt"))
log(f"trained 500 trees ({bst.num_trees()})")
rng = np.random.default_rng(0)
samp = np.sort(rng.choice(len(rows), 2_000_000, replace=False))
Ltr = bst.predict(Xtr[samp], pred_leaf=True).astype(np.int32)
del Xtr, ds
NL = 63; nT = Ltr.shape[1]
log(f"leaf matrix train sample {Ltr.shape}, max leaf {Ltr.max()}")


def onehot(L):
    n = L.shape[0]
    cols = (L + (np.arange(nT) * NL)[None, :]).ravel()
    return sp.csr_matrix((np.ones(len(cols), np.float32), cols, np.arange(0, n * nT + 1, nT)), shape=(n, nT * NL))


svd = TruncatedSVD(64, algorithm="randomized", random_state=0)
svd.fit(onehot(Ltr))
log(f"SVD fitted: explained var ratio sum {svd.explained_variance_ratio_.sum():.4f}")
del Ltr
for split, rr, tag in (("val", None, "val"), ("train", hrows, "hold")):
    X = load(split, rr)
    s = bst.predict(X).astype(np.float32)
    np.save(OUT / f"c01_score_{tag}.npy", s)
    Z = np.empty((len(X), 64), np.float32)
    for i in range(0, len(X), 500_000):
        L = bst.predict(X[i:i + 500_000], pred_leaf=True).astype(np.int32)
        Z[i:i + 500_000] = svd.transform(onehot(L))
    np.save(OUT / f"c01_svd_{tag}.npy", Z)
    log(f"{tag}: score + svd saved {Z.shape}")
    del X
log("DONE")
