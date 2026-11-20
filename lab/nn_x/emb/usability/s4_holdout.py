"""EMB-USABILITY stage 4: ADDENDUM A holdout replica (see PREDECLARED.txt). CPU only."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/nn_x/emb/usability")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from s1_eval_lib import feed_z_nan
from sklearn.model_selection import GroupKFold

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; COMP = ROOT / "data/competition"; P = ROOT / "preds"
OUT = ROOT / "nn_x/emb/usability/out"
LOG = open(OUT / "holdout.log", "a"); T0 = time.time()
def log(*a):
    s = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
G3_MEMBERS = ([f"v23_gru_pre2s_s{s}" for s in range(501, 507)] + [f"r6_cd_m02_r{r}" for r in range(101, 108)]
              + [f"r6_cd_m02_p70k_r{r}" for r in range(101, 108)])

def holdout_index():
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    Mt = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M"])["M"].to_numpy().astype(np.int64)
    p[Mt < 50] = 0.0
    cand = np.where((p > 0.9) & (Mt >= 50))[0]
    return np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))

meta = json.load(open(COMP / "train/item_embeddings/data.meta"))
TR = np.memmap(COMP / "train/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
f = pl.read_parquet(FLAT / "train/feeds.parquet")
hold = holdout_index()
M = f["M"].to_numpy().astype(np.int64)[hold]; H = f["H"].to_numpy().astype(np.int64)[hold]
ca = f["cand_off"].to_numpy().astype(np.int64)[hold]; ha = f["hist_off"].to_numpy().astype(np.int64)[hold]
uid = f["user_id"].to_numpy()[hold]
n = len(M); off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64); hoff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
crows = np.repeat(ca - off, M) + np.arange(int(M.sum())); hrows = np.repeat(ha - hoff, H) + np.arange(int(H.sum()))
hm = np.load(P / "holdout/holdout_meta.npy"); assert (hm[0] == M).all()
g = lambda nm, rows: np.asarray(np.load(FLAT / f"train/{nm}.npy", mmap_mode="r")[rows])
y = g("cand_target", crows).astype(np.float64); assert (y == np.load(P / "holdout/holdout_y.npy")).all()
ce = g("cand_emb_idx", crows).astype(np.int64); he = g("hist_emb_idx", hrows).astype(np.int64); het = g("hist_event_type", hrows)
UE = np.asarray(np.load(FLAT / "train/user_emb.npy", mmap_mode="r")[hold]).astype(np.float32)
log(f"holdout feeds {n}, rows {len(y):,}, hist {len(he):,}")
rows_needed = np.unique(np.concatenate([ce[ce >= 0], he[he >= 0]]))
E = np.asarray(TR[rows_needed]).astype(np.float32)
pos = lambda r: np.searchsorted(rows_needed, r)
Uraw = E / np.linalg.norm(E, axis=1, keepdims=True)
fit = np.load(OUT / "fit.npz"); mu = fit["mu"].astype(np.float32); W = fit["W"].astype(np.float32); V = fit["V"]
mud = mu / np.linalg.norm(mu); pc1 = V[:, 0].astype(np.float32)
def unit(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
Uz = unit((Uraw - mu) @ W)
fid = np.repeat(np.arange(n), M)
S = {k: np.full(len(y), np.nan) for k in ["proj_mu", "proj_pc1", "zca_maxcos", "raw_maxcos", "nd_act", "cos_u"]}
okc_all = ce >= 0
pc = pos(ce[okc_all])
S["proj_mu"][okc_all] = Uraw[pc] @ mud; S["proj_pc1"][okc_all] = (Uraw[pc] - mu) @ pc1
eraw = E[pc]
S["cos_u"][okc_all] = (eraw * UE[fid[okc_all]]).sum(1) / ((np.linalg.norm(eraw, axis=1) + 1e-6) * (np.linalg.norm(UE[fid[okc_all]], axis=1) + 1e-6))
S["nd_act"][:] = 0
for gi in range(n):
    s, m = off[gi], M[gi]; hs, hh = hoff[gi], H[gi]
    c = ce[s:s + m]; okc = c >= 0; h = he[hs:hs + hh]; okh = h >= 0
    if not okc.any() or not okh.any(): continue
    idx = np.nonzero(okc)[0] + s
    A = Uraw[pos(c[okc])]; B = Uraw[pos(h[okh])]; Sm = A @ B.T
    S["raw_maxcos"][idx] = Sm.max(1)
    S["zca_maxcos"][idx] = (Uz[pos(c[okc])] @ Uz[pos(h[okh])].T).max(1)
    et = het[hs:hs + hh][okh]; act = (et >= 2) & (et <= 10)
    if act.any(): S["nd_act"][idx] = (Sm[:, act].max(1) > 0.98).astype(float)
log("signals built")

def z(x): return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
zG = np.mean([z(np.load(P / "holdout" / f"nn_{m}_holdout.npy")) for m in G3_MEMBERS], 0)
base = ndcg_per_feed(zG, y, off, M)
log(f"G3 holdout pool NDCG {base.mean():.5f}")
users = uid.astype(np.int64).copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
folds = {sd: list(GroupKFold(5, shuffle=True, random_state=sd).split(np.arange(n), groups=users)) for sd in range(2)}
WG = [0, .02, -.02, .05, -.05, .1, -.1, .15, -.15, .2, -.2, .3, -.3, .5, -.5]
pi, ni = [], []
for gi in range(n):
    s, m = off[gi], M[gi]; yy = y[s:s + m]
    p = np.nonzero(yy == 1)[0] + s; q = np.nonzero(yy == 0)[0] + s
    if len(p) and len(q): pi.append(np.repeat(p, len(q))); ni.append(np.tile(q, len(p)))
pi = np.concatenate(pi); ni = np.concatenate(ni)
for nm, x in S.items():
    fin = np.isfinite(x); zx = feed_z_nan(off, M, np.where(fin, x, np.nan))
    d = x[pi] - x[ni]; b = fin[pi] & fin[ni]; a = float(((d[b] > 0) + 0.5 * (d[b] == 0)).mean())
    pers = np.stack([ndcg_per_feed(zG + w * zx, y, off, M) for w in WG])
    ds, ws = [], []
    for sd in range(2):
        o = np.zeros(n)
        for trn, va in folds[sd]:
            j = int(np.argmax(pers[:, trn].mean(1))); ws.append(WG[j]); o[va] = pers[j, va] - pers[0, va]
        ds.append(o)
    dd = np.mean(ds, 0)
    jf = int(np.argmax(pers.mean(1)))
    log(f"{nm:12s} cov {fin.mean():.3f} AUC {a:.4f} | holdout nested {dd.mean():+.5f} +- {dd.std()/np.sqrt(n):.5f} "
        f"(w_full {WG[jf]:+.2f} in-sample {pers[jf].mean()-pers[0].mean():+.5f}; folds {','.join(f'{w:+.2f}' for w in ws)}) "
        f"| fixed w=+0.02: {pers[1].mean()-pers[0].mean():+.5f} +- {(pers[1]-pers[0]).std()/np.sqrt(n):.5f}")
log("done")
