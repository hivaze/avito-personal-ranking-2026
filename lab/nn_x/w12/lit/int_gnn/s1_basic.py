"""int_gnn s1: label-free coverage facts from the flat tables and the stored inputs (CPU, read-only).
(a) users: share of val / test feeds (and distinct users) whose user_id has a train feed.
(b) stored inputs on val / test rows: zero share of the LUO count (the project's 'cold'), co-visitation > 0, by val / public / private.
Writes nothing but stdout (redirected to s1_basic.log)."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, polars as pl
from pathlib import Path

FLAT = Path("<repo>/data/flat")
FEATS = Path("<repo>/data/feats")
BOUND = 1790497920  # src/calib.py: public = test feeds with timestamp < BOUND

ft = pl.read_parquet(FLAT / "train/feeds.parquet")
fv = pl.read_parquet(FLAT / "val/feeds.parquet")
fte = pl.read_parquet(FLAT / "test/feeds.parquet")
tr_users = np.unique(ft["user_id"].to_numpy())
tr_users_nz = tr_users[tr_users != 0]
print(f"train feeds {len(ft)}, distinct non-zero users {len(tr_users_nz)}, user-0 feeds {(ft['user_id'].to_numpy() == 0).sum()}")
print(f"train feeds per non-zero user: mean {(ft['user_id'].to_numpy() != 0).sum() / len(tr_users_nz):.3f}")


def users(name, f, mask=None):
    u = f["user_id"].to_numpy()
    H = f["H"].to_numpy()
    if mask is not None:
        u = u[mask]; H = H[mask]
    n = len(u)
    z = u == 0
    intr = np.isin(u, tr_users_nz) & ~z
    du = np.unique(u[~z])
    dintr = np.isin(du, tr_users_nz)
    print(f"[users] {name}: feeds {n} | user-0 {z.mean():.4f} | non-zero user WITH a train feed {intr.mean():.4f} | "
          f"non-zero user WITHOUT a train feed {(~intr & ~z).mean():.4f} | distinct non-zero users {len(du)}, in train {dintr.mean():.4f} | "
          f"H==0 {np.mean(H == 0):.4f}, median H {np.median(H):.0f}")


users("val", fv)
tts = fte["timestamp"].to_numpy()
users("test", fte)
users("public", fte, tts < BOUND)
users("private", fte, tts >= BOUND)


def rows_mask(f, feed_mask):
    M = f["M"].to_numpy().astype(np.int64)
    return np.repeat(feed_mask, M)


def feat(split, n):
    return np.load(FEATS / split / f"{n}.npy", mmap_mode="r")


for split, f, groups in [("val", fv, {"val": None}), ("test", fte, {"test": None, "public": tts < BOUND, "private": tts >= BOUND})]:
    luo = np.asarray(feat(split, "it_hist_all_luo"))
    raw = np.asarray(feat(split, "it_hist_all"))
    cvi = np.asarray(feat(split, "seq_cv_invd20"))
    cvp = np.asarray(feat(split, "seq_cv_pop"))
    M = f["M"].to_numpy().astype(np.int64)
    for g, fm in groups.items():
        rm = np.ones(len(luo), bool) if fm is None else rows_mask(f, fm)
        l = luo[rm]
        print(f"[stored] {g}: rows {rm.sum()} | LUO count ==0 {np.mean(l == 0):.4f} >=1 {np.mean(l >= 1):.4f} >=3 {np.mean(l >= 3):.4f} >=10 {np.mean(l >= 10):.4f} | "
              f"raw it_hist_all ==0 {np.mean(raw[rm] == 0):.4f} | seq_cv_pop>0 {np.mean(cvp[rm] > 0):.4f} | seq_cv_invd20>0 {np.mean(cvi[rm] > 0):.4f} | "
              f"seq_cv_invd20>0 given warm {np.mean(cvi[rm][l > 0] > 0):.4f} | feeds M>=50 {np.mean((M if fm is None else M[fm]) >= 50):.4f} mean M {(M if fm is None else M[fm]).mean():.1f}")
