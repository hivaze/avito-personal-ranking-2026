"""EMB-USABILITY stage 2: corrected GBM variant (deviation from PREDECLARED, documented in DEVIATIONS.txt).

The pre-declared GBM (lambdarank on [zG3, logM, logH, user0] from scratch) produces a step function of zG3 with many
within-feed ties (pessimistic tie rule) -> control = base - 0.014, so its deltas measure tie-breaking, not information.
Fix (same for control and with-x, so still paired): boost FROM G3: init_score = k * zG3, k = 1-D pairwise-logistic MLE of
the scale on the training folds' leak-free pairs; final score = k zG3 + trees + 1e-6 zG3. Everything else as declared
(lr 0.05, 150 rounds, 15 leaves, min_data_in_leaf 500, fold seed 0; control [zG3, logM, logH, user0] vs + [x, z(x)]).
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
sys.path.insert(0, "<repo>/nn_x/emb/usability")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from qt import leak_mask
from pmix import weights as pmix_weights, wmean_se
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat/val"; FEATS = ROOT / "data/feats/val"
OUT = ROOT / "nn_x/emb/usability/out"
T0 = time.time()
LOG = open(OUT / "gbm.log", "a")
def log(*a):
    s = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

f = pl.read_parquet(FLAT / "feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64); n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "cand_target.npy").astype(np.float64)
uid = f["user_id"].to_numpy(); users = uid.copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
keep = ~leak_mask(); wpriv = pmix_weights()["private"]
def z(x): return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
zG3 = z(np.load(ROOT / "preds/ens_G3_final_val.npy"))
base_per = ndcg_per_feed(zG3, y, off, M)
from s1_eval_lib import feed_z_nan
folds = list(GroupKFold(5, shuffle=True, random_state=0).split(np.arange(n), groups=users))

# pairs for the scale fit
pi, ni = [], []
for g in np.nonzero(keep)[0]:
    s, m = off[g], M[g]; yy = y[s:s + m]
    p = np.nonzero(yy == 1)[0] + s; q = np.nonzero(yy == 0)[0] + s
    if len(p) and len(q):
        pi.append(np.repeat(p, len(q))); ni.append(np.tile(q, len(p)))
pi = np.concatenate(pi); ni = np.concatenate(ni); pf = fid[pi]
def fit_k(mask):
    d = (zG3[pi] - zG3[ni])[mask]; k = 1.0
    for _ in range(30):
        s = 1 / (1 + np.exp(-k * d)); g = ((1 - s) * d).sum(); h = -(s * (1 - s) * d * d).sum(); k -= g / h
    return k

u0 = (uid == 0).astype(np.float64)
CTRL = np.stack([zG3, np.log(M)[fid], np.log1p(H)[fid], u0[fid]], 1)
PRM = dict(objective="lambdarank", learning_rate=0.05, num_leaves=15, min_data_in_leaf=500, seed=0, verbose=-1,
           num_threads=12, lambdarank_truncation_level=300, deterministic=True)
FOLDROWS = []
for trn, va in folds:
    trk = np.sort(trn[keep[trn]]); vs = np.sort(va)
    rtr = np.concatenate([np.arange(off[g], off[g] + M[g]) for g in trk])
    rva = np.concatenate([np.arange(off[g], off[g] + M[g]) for g in vs])
    k = fit_k(np.isin(pf, trk))
    FOLDROWS.append((trk, rtr, rva, k))
log("scale k per fold", [round(r[3], 3) for r in FOLDROWS])

def gbm_oof(Xm):
    out = np.zeros(len(y))
    for trk, rtr, rva, k in FOLDROWS:
        ds = lgb.Dataset(Xm[rtr], y[rtr], group=M[trk], init_score=k * zG3[rtr], free_raw_data=True)
        b = lgb.train(PRM, ds, num_boost_round=150)
        out[rva] = k * zG3[rva] + b.predict(Xm[rva], num_threads=12) + 1e-6 * zG3[rva]
    return out
ctrl_per = ndcg_per_feed(gbm_oof(CTRL), y, off, M)
d0 = (ctrl_per - base_per)[keep]
log(f"GBM control (init k*zG3; zG3, logM, logH, user0) minus base G3: {d0.mean():+.5f} +- {d0.std()/np.sqrt(len(d0)):.5f}")

S0 = np.load(OUT / "sig_val.npz")
names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["chg_d", "chg_flag", "zca_maxcos", "proj_mu", "nd_act", "maxcos_act", "cos_u", "raw_maxcos"]
rows = []
for nm in names:
    x = np.load(FEATS / "cos_u.npy").astype(np.float64) if nm == "cos_u" else S0[nm].astype(np.float64)
    fin = np.isfinite(x); zx = feed_z_nan(off, M, np.where(fin, x, np.nan))
    per = ndcg_per_feed(gbm_oof(np.column_stack([CTRL, np.where(fin, x, np.nan), zx])), y, off, M)
    d = (per - ctrl_per)[keep]; pm, pse = wmean_se(d, wpriv[keep])
    rows.append(dict(signal=nm, gbm=d.mean(), gbm_se=d.std() / np.sqrt(len(d)), pmix=pm, pmix_se=pse))
    log(f"{nm:14s} GBM(with x) - GBM(control): {d.mean():+.5f} +- {d.std()/np.sqrt(len(d)):.5f}  pmix {pm:+.5f} +- {pse:.5f}")
pl.DataFrame(rows).write_csv(OUT / "gbm.csv")
log("done")
