"""Critic check 3 (CPU): is the fine-tuned (FT) branch's val advantage a same-period channel that test cannot get?
FT members are cross-fitted by USER, not by time: they learn from other users' clicks on the same days and often the same items.
Proxy: per feed, share of candidates whose item_id is a candidate in ANOTHER user's val feed (same-period exposure overlap).
Compare FT-minus-TO and the final-blend weight curve across overlap terciles, and the same overlap share on test (vs val items).
Label-free covariate; information only."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed

P = "<repo>/preds"
fv, M, off = feed_meta("val"); n = len(M); fid = np.repeat(np.arange(n), M)
ft_, Mt, offt = feed_meta("test"); nt = len(Mt); fidt = np.repeat(np.arange(nt), Mt)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
iv = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64); it = np.load(FLAT / "test/cand_item_id.npy").astype(np.int64)
uv = fv["user_id"].to_numpy()


def z(x, fid=fid, n=n, M=M):
    x = np.asarray(x, np.float64); mu = np.bincount(fid, x, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, n) / M - mu ** 2, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


zft = z(np.load(f"{P}/ens_FD16P20_m04_02kd_ft_val.npy")); zto = z(np.load(f"{P}/ens_FD16P20_m04_02kd_to_val.npy"))
ev = lambda s: ndcg_per_feed(s, y, off, M)
eFT, eTO = ev(zft), ev(zto)
# other-user overlap on val: item appears in a feed of a different user (count distinct users per item)
df = pl.DataFrame({"i": iv, "u": np.repeat(uv, M), "f": fid})
nu = df.group_by("i").agg(pl.col("u").n_unique().alias("nu"), pl.col("f").n_unique().alias("nf"))
d = df.join(nu, on="i", how="left")
other = (d["nf"].to_numpy() > 1)  # shown in at least one other val feed (approx: other feed; same-user repeats are rare)
ov_share = np.bincount(fid, other.astype(float), n) / M
# test overlap with val items (what the FT branch could carry over in item terms)
valset = np.unique(iv)
tin = np.isin(it, valset)
print(f"val rows whose item is a candidate in another val feed: {other.mean():.3f}; test rows whose item is a val candidate: {tin.mean():.3f}")
dtt = pl.DataFrame({"i": it, "f": fidt}).group_by("i").agg(pl.col("f").n_unique().alias("nf"))
tt = pl.DataFrame({"i": it}).join(dtt, on="i", how="left")["nf"].to_numpy() > 1
print(f"test rows whose item is a candidate in another test feed: {tt.mean():.3f}")
q = np.quantile(ov_share, [1 / 3, 2 / 3]); terc = np.digitize(ov_share, q)
for w in (0.3, 0.4, 0.5, 0.6, 0.7):
    e = ev(w * zft + (1 - w) * zto)
    print(f"FT weight {w}: " + " | ".join(f"terc{t} {e[terc == t].mean():.5f}" for t in range(3)))
dd = eFT - eTO
for t in range(3):
    m = terc == t
    print(f"overlap tercile {t} (share {ov_share[m].mean():.3f}): FT - TO {dd[m].mean():+.5f} +- {dd[m].std() / np.sqrt(m.sum()):.5f}  (n {m.sum()})")
