"""P1 post-hoc checks (see POSTHOC.txt). CPU only, 12 threads. Output: out/p1_posthoc.log, out/p1_posthoc.json."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from qt import leak_mask
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; P = ROOT / "preds"; OUT = ROOT / "nn_x/pretrain2/P1/out"
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
TAU = 1000.0
FAM = {"F10": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)], "F7x": [f"r6_cd_m02_r{s}" for s in range(101, 108)],
       "F7x70": [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)]}
FN = list(FAM); NF = np.array([len(FAM[f]) for f in FN], float)
GRID = np.array([tuple(NF / NF.sum())] + [(a / 10, b / 10, (10 - a - b) / 10) for a in range(11) for b in range(11 - a)])
C = np.load(OUT / "clusters.npz")
def se(d): return float(np.std(d, ddof=1) / np.sqrt(len(d)))
def zf(off, M, x): return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
def groups_of(u):
    g = u.astype(np.int64).copy(); z0 = g == 0; g[z0] = -np.arange(1, z0.sum() + 1); return g
def folds_of(groups, seed):
    u, inv = np.unique(groups, return_inverse=True); return (np.random.default_rng(seed).permutation(len(u)) % 5)[inv]
fv = pl.read_parquet(FLAT / "val/feeds.parquet"); Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy().astype(np.int64); nv = len(Mv)
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64); gv = groups_of(fv["user_id"].to_numpy()); keep = ~leak_mask()
FT = {f: np.mean([zf(ov, Mv, np.load(P / f"nn_{m}_n2cpv3_val_oof.npy")) for m in FAM[f]], 0) for f in FN}
TO = {f: np.mean([zf(ov, Mv, np.load(P / f"nn_{m}_trainonly_val.npy")) for m in FAM[f]], 0) for f in FN}
NDv = np.stack([ndcg_per_feed(0.5 * zf(ov, Mv, sum(wi * FT[f] for wi, f in zip(w, FN))) + 0.5 * zf(ov, Mv, sum(wi * TO[f] for wi, f in zip(w, FN))), yv, ov, Mv) for w in GRID])
hm = np.load(P / "holdout/holdout_meta.npy"); Mh = hm[0].astype(np.int64); oh = hm[1].astype(np.int64); nh = len(Mh)
yh = np.load(P / "holdout/holdout_y.npy").astype(np.float64)
gh = groups_of(pl.read_parquet(FLAT / "train/feeds.parquet", columns=["user_id"])["user_id"].to_numpy()[C["feeds_hold"]])
HF = {f: np.mean([zf(oh, Mh, np.load(P / f"holdout/nn_{m}_holdout.npy")) for m in FAM[f]], 0) for f in FN}
NDh = np.stack([ndcg_per_feed(sum(wi * HF[f] for wi, f in zip(w, FN)), yh, oh, Mh) for w in GRID])
log(f"rebuilt: val G3 {NDv[0][keep].mean():.5f}, holdout G3 {NDh[0].mean():.5f}")
np.savez_compressed(OUT / "nd_grid.npz", NDv=NDv.astype(np.float32), NDh=NDh.astype(np.float32), GRID=GRID)

def nested(ND, lab, K, sel, ev, groups, tau):
    n = ND.shape[1]; o_ = np.zeros((n, K)); o_[np.arange(n), lab] = 1.0; pc = np.zeros(n); gl = np.zeros(n)
    for seed in (0, 1):
        fold = folds_of(groups, seed)
        for k in range(5):
            tr = (fold != k) & sel; te = (fold == k) & ev
            S_all = ND[:, tr].sum(1); n_all = tr.sum(); gb = int(np.argmax(S_all))
            S_c = ND[:, tr] @ o_[tr]; n_c = o_[tr].sum(0)
            obj = np.where(n_c > 0, S_c / np.maximum(n_c, 1), -np.inf) if tau is None else (S_c + tau * (S_all / n_all)[:, None]) / (n_c + tau)
            best = np.where(n_c > 0, np.argmax(obj, 0), gb)
            gl[te] += ND[gb, te] / 2; pc[te] += ND[best[lab[te]], te] / 2
    return pc - gl

def fullfit(ND, lab, K, sel, tau):
    o_ = np.zeros((ND.shape[1], K)); o_[np.arange(ND.shape[1]), lab] = 1
    S_all = ND[:, sel].sum(1); n_all = sel.sum(); gb = int(np.argmax(S_all)); S_c = ND[:, sel] @ o_[sel]; n_c = o_[sel].sum(0)
    obj = np.where(n_c > 0, S_c / np.maximum(n_c, 1), -np.inf) if tau is None else (S_c + tau * (S_all / n_all)[:, None]) / (n_c + tau)
    return gb, np.where(n_c > 0, np.argmax(obj, 0), gb)

R = {}
ALL = np.ones(nh, bool)
real = json.load(open(OUT / "cluster_diag.json"))["test1"]
log("(a) random-label null, 50 labelings")
for K in (8, 32):
    for sel, tau in (("raw", None), ("shrunk", TAU)):
        dv, dh = [], []
        for s in range(100, 150):
            r = np.random.default_rng(s)
            dv.append(nested(NDv, r.integers(K, size=nv), K, keep, keep, gv, tau)[keep].mean())
            dh.append(nested(NDh, r.integers(K, size=nh), K, ALL, ALL, gh, tau).mean())
        dv, dh = np.array(dv), np.array(dh)
        R[f"null_K{K}_{sel}"] = dict(val=(float(dv.mean()), float(dv.std())), hold=(float(dh.mean()), float(dh.std())))
        line = f"null K{K} {sel:6s}: val {dv.mean():+.5f} sd {dv.std():.5f} | hold {dh.mean():+.5f} sd {dh.std():.5f}"
        for key in ("ue", "hp"):
            a = real[f"{key}{K}_{sel}"]
            rv = (dv >= a["val"][0]).mean(); rh = (dh >= a["hold"][0]).mean()
            line += f" || {key}{K}: val {a['val'][0]:+.5f} (null frac >= {rv:.2f}, z_null {(a['val'][0]-dv.mean())/dv.std():+.2f}) hold {a['hold'][0]:+.5f} (null frac >= {rh:.2f}, z_null {(a['hold'][0]-dh.mean())/dh.std():+.2f})"
        log(line)
log("(b) holdout -> val transfer and (c) weight agreement")
for key in ("ue", "hp"):
    for K in (8, 32):
        lv = C[f"{key}{K}_val"].astype(np.int64); lh = C[f"{key}{K}_hold"].astype(np.int64)
        for sel, tau in (("raw", None), ("shrunk", TAU)):
            gbh, bh = fullfit(NDh, lh, K, ALL, tau)
            d = (NDv[bh[lv], np.arange(nv)] - NDv[gbh])[keep]
            gbv, bv = fullfit(NDv, lv, K, keep, tau)
            cnt = np.minimum(np.bincount(lv[keep], minlength=K), np.bincount(lh, minlength=K)); ok = cnt >= 100
            cf10 = float(np.corrcoef(GRID[bv[ok], 0], GRID[bh[ok], 0])[0, 1]) if ok.sum() > 2 else float("nan")
            cf7 = float(np.corrcoef(GRID[bv[ok], 1], GRID[bh[ok], 1])[0, 1]) if ok.sum() > 2 else float("nan")
            R[f"h2v_{key}{K}_{sel}"] = dict(val=(float(d.mean()), se(d)), corr_wF10=cf10, corr_wF7x=cf7, hold_global=GRID[gbh].tolist())
            log(f"{key}{K:<3d}{sel:7s} holdout-fitted per-cluster - holdout-global, applied to val: {d.mean():+.5f} ± {se(d):.5f} (z {d.mean()/se(d):+.2f}); "
                f"holdout-global w {GRID[gbh].round(2).tolist()}; corr over {ok.sum()} clusters of full-fit w(F10) val vs hold {cf10:+.2f}, w(F7x) {cf7:+.2f}")
json.dump(R, open(OUT / "p1_posthoc.json", "w"), indent=1)
log("done")
