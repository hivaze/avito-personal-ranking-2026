"""int_gnn s3b: is the 2-hop walk score new information next to the collaborative columns already on disk?
Val only, CPU, read-only. Uses walk_{mode}_val.npy from s3_walk.py.
(1) feed-mean within-feed Spearman of the walk scores with: stored co-visitation, LUO count, ALS profile (als_exp5), the
    pre-trained encoder's scores (pre2_fe = with its item-ID table, pre2_fe_noid = content only, pre2_id_contrib, pre2_idmax32),
    the W16 pool; on warm rows (deg >= 1).
(2) the same residual lens as s3 for als_exp5 / pre2_id_contrib / pre2_idmax32 (calibration against the record)."""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s3_walk import feed_auc, feed_z  # noqa: E402
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
mode = sys.argv[1]
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64)
coff = f["cand_off"].to_numpy().astype(np.int64)
nF = len(M)
fid = np.repeat(np.arange(nF), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
w = np.load(HERE / f"walk_{mode}_val.npy").astype(np.float64)
s1, s2, deg = w[:, 0], w[:, 1], w[:, 2]
warm = deg >= 1
z = lambda v: feed_z(np.asarray(v, np.float64), fid, nF)  # noqa: E731
pool = z(0.5 * z(np.load(ROOT / "preds/ens_W16kd_ft_val.npy")) + 0.5 * z(np.load(ROOT / "preds/ens_W16kd_to_val.npy")))
ld = lambda n: np.asarray(np.load(FEATS / "val" / f"{n}.npy"), np.float64)  # noqa: E731
cols = {"s1": s1, "s2/deg^.5": s2 / np.maximum(deg, 1) ** 0.5, "s2/deg": s2 / np.maximum(deg, 1),
        "seq_cv_invd20": ld("seq_cv_invd20"), "LUO count": ld("it_hist_all_luo"), "als_exp5": ld("als_exp5"),
        "pre2_fe": ld("pre2_fe"), "pre2_fe_noid": ld("pre2_fe_noid"), "pre2_id_contrib": ld("pre2_id_contrib"),
        "pre2_idmax32": ld("pre2_idmax32"), "W16 pool": pool}
for n, v in cols.items():
    print(f"[col] {n}: NaN {np.mean(np.isnan(v)):.4f} | NaN on warm rows {np.mean(np.isnan(v[warm])):.4f} | zero on warm rows {np.mean(v[warm] == 0):.4f}", flush=True)


def spearman_feed(a, b, sub):
    out = []
    for ff in range(nF):
        o = coff[ff]
        m = sub[o:o + M[ff]] & ~np.isnan(a[o:o + M[ff]]) & ~np.isnan(b[o:o + M[ff]])
        if m.sum() < 8:
            continue
        ra = rankdata(a[o:o + M[ff]][m]); rb = rankdata(b[o:o + M[ff]][m])
        if ra.std() == 0 or rb.std() == 0:
            continue
        out.append(np.corrcoef(ra, rb)[0, 1])
    return np.mean(out), len(out)


for a in ["s1", "s2/deg^.5", "s2/deg"]:
    line = f"[spearman, warm rows, feed mean] {a} vs"
    for b in cols:
        if b == a:
            continue
        r, n = spearman_feed(cols[a], cols[b], warm)
        line += f" {b} {r:+.3f}"
    print(line + f" (feeds {n})", flush=True)

sub = warm
line = "[AUC warm rows]"
for n, v in cols.items():
    vv = np.where(np.isnan(v), -1e30, v)
    au = feed_auc(vv, y, sub, coff, M)
    line += f" {n} {np.nanmean(au):.4f}"
print(line, flush=True)
sub = warm & (s1 == 0)
line = "[AUC warm rows without a 1-hop link]"
for n, v in cols.items():
    vv = np.where(np.isnan(v), -1e30, v)
    au = feed_auc(vv, y, sub, coff, M)
    line += f" {n} {np.nanmean(au):.4f}"
print(line, flush=True)

# AUC after removing what two content-similarity inputs already say (pooled OLS on within-feed centred percentile ranks)
def crank(v, sub):
    r = np.zeros(len(y))
    for ff in range(nF):
        o = coff[ff]
        m = sub[o:o + M[ff]]
        k = int(m.sum())
        if k == 0:
            continue
        tmp = np.zeros(M[ff])
        tmp[m] = (rankdata(v[o:o + M[ff]][m]) - 0.5) / k - 0.5
        r[o:o + M[ff]] = tmp
    return r


for sname, sub in [("warm rows", warm), ("warm rows without a 1-hop link", warm & (s1 == 0))]:
    rows_ = np.nonzero(sub)[0]
    A = np.stack([crank(np.nan_to_num(ld("seq_max_new8"), nan=-9.0), sub)[rows_], crank(cols["pre2_fe_noid"], sub)[rows_]], 1)
    line = f"[AUC on {sname} AFTER rank-residualising on seq_max_new8 + pre2_fe_noid]"
    for n in ["s1", "s2/deg^.5", "s2/deg", "als_exp5", "seq_cv_invd20", "LUO count"]:
        rx = crank(np.where(np.isnan(cols[n]), -1e30, cols[n]), sub)
        beta, *_ = np.linalg.lstsq(A, rx[rows_], rcond=None)
        res = np.zeros(len(y))
        res[rows_] = rx[rows_] - A @ beta
        au = feed_auc(res, y, sub, coff, M)
        ok = ~np.isnan(au)
        line += f" | {n} {np.nanmean(au):.4f}±{np.nanstd(au) / np.sqrt(ok.sum()):.4f}"
    print(line, flush=True)

# residual lens calibration on record columns
user = f["user_id"].to_numpy().astype(np.int64)
grp = np.where(user == 0, -1 - np.arange(nF), user)
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=grp))
grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
nd = lambda s: ndcg_per_feed(s, y, coff, M)  # noqa: E731
base = nd(pool)
for n in ["als_exp5", "pre2_id_contrib", "pre2_idmax32", "pre2_fe", "seq_cv_invd20"]:
    v = cols[n].copy()
    nanm = np.isnan(v)
    cnt = np.maximum(np.bincount(fid, weights=(~nanm).astype(float), minlength=nF), 1)
    mu = np.bincount(fid, weights=np.where(nanm, 0, v), minlength=nF) / cnt
    d = np.where(nanm, 0.0, v - mu[fid])
    sd = np.sqrt(np.bincount(fid, weights=d * d, minlength=nF) / cnt)
    x = d / np.maximum(sd[fid], 1e-9)  # z among present rows, 0 where missing
    per = {ww: nd(pool + ww * x) for ww in grid}
    delta = np.zeros(nF); ws = []
    for trf, vaf in folds:
        best = max(grid, key=lambda ww: per[ww][trf].mean())
        ws.append(best)
        delta[vaf] = per[best][vaf] - base[vaf]
    se = delta.std() / np.sqrt(nF)
    print(f"[nested W16, calibration] {n} (z among present rows): {delta.mean():+.5f} ± {se:.5f} (z {delta.mean() / max(se, 1e-12):+.2f}) w {ws} | in-sample "
          + " ".join(f"{ww}:{(per[ww] - base).mean():+.5f}" for ww in grid[1:]), flush=True)
print("DONE")
