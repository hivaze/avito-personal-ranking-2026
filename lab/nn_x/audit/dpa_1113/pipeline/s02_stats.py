"""Input pipeline audit, step 2: per-input statistics across trn / hold / val / pub / priv.

For each of the 104 dense inputs (pre-standardisation X from s01) and the 34 raw feat_cache columns:
  scale, tails, zero mass, NaN, clip saturation (attach_xt clip, set_norm clamp +-6 with the checkpoint stats),
  PSI (10 train-quantile bins + mass points), within-feed variance share, within-feed label AUC (val, hold),
  fp16 tie creation for the xt columns.  CPU only.  Outputs out/stats_X.csv, out/stats_raw.csv, out/s02.log
"""
import os, sys, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, numba as nb, polars as pl, torch

HERE = Path(__file__).resolve().parent; OUT = HERE / "out"
ROOT = Path("<repo>")
BOUND = 1790497920
LOG = open(OUT / "s02.log", "w")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


CK = torch.load(ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt", map_location="cpu", weights_only=False)
CFG = CK["cfg"]; FM = CK["state_dict"]["f_mean"].numpy().astype(np.float64); FS = CK["state_dict"]["f_std"].numpy().astype(np.float64)
CK2 = torch.load(ROOT / "nn/ckpt/v23_gru_pre2s_s501.pt", map_location="cpu", weights_only=False)
FM2 = CK2["state_dict"]["f_mean"].numpy(); FS2 = CK2["state_dict"]["f_std"].numpy()
XT = CFG["xt_names"]
base = []
for k in range(1, 6):
    base += [f"f{k}", f"f{k}_zf", f"f{k}_rkf"]
base += ["cmiss", "log1p_norm", "cos_u", "dot_u", "smax", "smax8", "smax32", "smean", "lse", "top3"]
base += [f"smax_et{t}" for t in range(1, 11)] + ["log1p_ccnt", "ccnt_share"] + [f"log1p_ccnt_et{t}" for t in range(1, 11)]
base += ["first_inv", "fcs", "logM", "log1pH"]
for nme in ["cos_u", "smax", "lse", "top3", "log1p_ccnt"]:
    base += [f"{nme}_zf", f"{nme}_rkf"]
base += ["user0", "user0_log1pH"]
xt_keep = [n for i, n in enumerate(XT) if i not in set(CFG["xt_relonly"])]
rel = []
for j in CFG["xt_rel"]:
    rel += [f"{XT[j]}_rkf", f"{XT[j]}_zf"]
NAMES = base + ["xt:" + n for n in xt_keep] + ["xt:" + n for n in rel]
assert len(NAMES) == CFG["n_dense"] == 104, len(NAMES)


def load(sub):
    X = np.load(OUT / f"{sub}_X.npy", mmap_mode="r"); R = np.load(OUT / f"{sub}_raw.npy", mmap_mode="r")
    mt = dict(np.load(OUT / f"{sub}_meta.npz"))
    return X, R, mt


@nb.njit(parallel=True, cache=True)
def within_var(x, off, M):
    nF = len(M); w = np.zeros(nF); n = np.zeros(nF)
    for i in nb.prange(nF):
        o = off[i]; m = M[i]; s = 0.0
        for k in range(m):
            s += x[o + k]
        mu = s / m; v = 0.0
        for k in range(m):
            v += (x[o + k] - mu) ** 2
        w[i] = v; n[i] = m
    return w.sum() / n.sum()


@nb.njit(parallel=True, cache=True)
def wauc(x, y, off, M):
    """mean over feeds (>=1 pos, >=1 neg) of the within-feed AUC with ties = 0.5."""
    nF = len(M); a = np.full(nF, np.nan)
    for i in nb.prange(nF):
        o = off[i]; m = M[i]; npos = 0
        for k in range(m):
            npos += y[o + k] > 0
        if npos == 0 or npos == m:
            continue
        s = 0.0
        for p in range(m):
            if y[o + p] <= 0:
                continue
            for q in range(m):
                if y[o + q] > 0:
                    continue
                if x[o + p] > x[o + q]:
                    s += 1.0
                elif x[o + p] == x[o + q]:
                    s += 0.5
        a[i] = s / (npos * (m - npos))
    return a


@nb.njit(parallel=True, cache=True)
def tie_pairs(x32, x16, off, M):
    """share of within-feed pairs distinct in fp32 but tied in fp16; and share of pairs tied in fp16."""
    nF = len(M); lost = np.zeros(nF); tied = np.zeros(nF); tot = np.zeros(nF)
    for i in nb.prange(nF):
        o = off[i]; m = M[i]
        for p in range(m):
            for q in range(p + 1, m):
                tot[i] += 1
                if x16[o + p] == x16[o + q]:
                    tied[i] += 1
                    if x32[o + p] != x32[o + q]:
                        lost[i] += 1
    return lost.sum() / max(tot.sum(), 1), tied.sum() / max(tot.sum(), 1)


def psi(ref, x, nb_=10):
    ref = ref[np.isfinite(ref)]; x = x[np.isfinite(x)]
    if len(ref) == 0 or len(x) == 0:
        return np.nan
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, nb_ + 1)[1:-1]))
    # mass points: values holding > 5% of ref get their own bin
    vals, cnt = np.unique(ref[:: max(1, len(ref) // 200000)], return_counts=True)
    mp = vals[cnt / cnt.sum() > 0.05]
    edges = np.unique(np.concatenate([qs, mp - 1e-9, mp + 1e-9]))
    br = np.bincount(np.searchsorted(edges, ref), minlength=len(edges) + 1) / len(ref)
    bx = np.bincount(np.searchsorted(edges, x), minlength=len(edges) + 1) / len(x)
    br = np.clip(br, 1e-4, None); bx = np.clip(bx, 1e-4, None)
    return float(((bx - br) * np.log(bx / br)).sum())


def main():
    t0 = time.time()
    D = {}
    for sub in ["trn", "hold", "val", "test"]:
        X, R, mt = load(sub)
        if sub == "test":
            pubf = mt["ts"] < BOUND
            rows_pub = np.repeat(pubf, mt["M"])
            for nm, sel in [("pub", pubf), ("priv", ~pubf)]:
                rows = np.repeat(sel, mt["M"])
                Ms = mt["M"][sel]
                D[nm] = dict(X=np.asarray(X[rows]), R=np.asarray(R[rows]), M=Ms, off=np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64), y=None)
        else:
            D[sub] = dict(X=np.asarray(X), R=np.asarray(R), M=mt["M"], off=mt["off"], y=mt["y"] if len(mt["y"]) else None)
    log(f"loaded {time.time()-t0:.0f}s; rows " + ", ".join(f"{k} {len(v['X'])}" for k, v in D.items()))
    # 1. set_norm parity: checkpoint stats vs trn sample (row-weighted) ------------------------------------------
    Xt = D["trn"]["X"].astype(np.float64)
    mu, sdv = Xt.mean(0), Xt.std(0)
    log("\n== set_norm parity (checkpoint r101 f_mean/f_std vs 20K-feed trn sample, row-weighted) ==")
    log(f"max |mean diff|/std {np.max(np.abs(mu - FM) / np.maximum(FS, 1e-3)):.3f}; std ratio range {np.min(sdv / FS):.3f}..{np.max(sdv / FS):.3f}")
    log(f"r101 vs s501 stats: max |dmean|/std {np.max(np.abs(FM - FM2) / np.maximum(FS, 1e-3)):.3f}, std ratio {np.min(FS2 / FS):.3f}..{np.max(FS2 / FS):.3f}")
    bad = np.where((np.abs(mu - FM) / np.maximum(FS, 1e-3) > 0.1) | (np.abs(sdv / FS - 1) > 0.1))[0]
    for j in bad:
        log(f"   {NAMES[j]:28s} ck mean {FM[j]:+.4f} std {FS[j]:.4f} | sample mean {mu[j]:+.4f} std {sdv[j]:.4f}")
    # 2. per-feature table ----------------------------------------------------------------------------------------
    rows = []
    for j, nm in enumerate(NAMES):
        r = dict(j=j, name=nm, ck_mean=FM[j], ck_std=FS[j])
        for s, d in D.items():
            x = d["X"][:, j].astype(np.float64)
            q = np.quantile(x, [0.001, 0.01, 0.5, 0.99, 0.999])
            xs = (x - FM[j]) / max(FS[j], 1e-3)
            r[f"{s}_mean"] = x.mean(); r[f"{s}_std"] = x.std(); r[f"{s}_p001"] = q[0]; r[f"{s}_p50"] = q[2]
            r[f"{s}_p999"] = q[4]; r[f"{s}_max"] = x.max(); r[f"{s}_min"] = x.min()
            r[f"{s}_zero"] = float((x == 0).mean())
            r[f"{s}_sat6"] = float((np.abs(xs) >= 6).mean())
            r[f"{s}_zmean"] = float(xs.mean()); r[f"{s}_zstd"] = float(xs.std())
        ref = D["trn"]["X"][:, j].astype(np.float64)
        for s in ["hold", "val", "pub", "priv"]:
            r[f"psi_trn_{s}"] = psi(ref, D[s]["X"][:, j].astype(np.float64))
        r["psi_val_pub"] = psi(D["val"]["X"][:, j].astype(np.float64), D["pub"]["X"][:, j].astype(np.float64))
        r["psi_val_priv"] = psi(D["val"]["X"][:, j].astype(np.float64), D["priv"]["X"][:, j].astype(np.float64))
        r["psi_hold_val"] = psi(D["hold"]["X"][:, j].astype(np.float64), D["val"]["X"][:, j].astype(np.float64))
        xv = D["val"]["X"][:, j].astype(np.float64)
        tv = xv.var()
        r["val_within_share"] = within_var(xv, D["val"]["off"], D["val"]["M"]) / tv if tv > 0 else np.nan
        for s in ["val", "hold"]:
            a = wauc(D[s]["X"][:, j].astype(np.float64), D[s]["y"].astype(np.float64), D[s]["off"], D[s]["M"])
            r[f"{s}_wauc"] = float(np.nanmean(a))
        rows.append(r)
        if j % 10 == 0:
            log(f"  feat {j} {nm} {time.time()-t0:.0f}s")
    df = pl.DataFrame(rows); df.write_csv(OUT / "stats_X.csv")
    # 3. raw xt columns ----------------------------------------------------------------------------------------------
    rr = []
    for j, nm in enumerate(XT):
        r = dict(j=j, name=nm)
        clip = 5 if nm.endswith("_z") else (None if (nm.startswith(("has:", "seq_cv", "it_hist")) or nm.endswith("_rk")) else 10)
        for s, d in D.items():
            x = d["R"][:, j].astype(np.float64); fin = np.isfinite(x)
            r[f"{s}_nan"] = float((~fin).mean())
            xf = x[fin]
            q = np.quantile(xf, [0.5, 0.9, 0.99, 0.999]) if len(xf) else [np.nan] * 4
            r[f"{s}_p50"] = q[0]; r[f"{s}_p90"] = q[1]; r[f"{s}_p99"] = q[2]; r[f"{s}_p999"] = q[3]
            r[f"{s}_max"] = xf.max() if len(xf) else np.nan; r[f"{s}_min"] = xf.min() if len(xf) else np.nan
            r[f"{s}_zero"] = float((xf == 0).mean()) if len(xf) else np.nan
            r[f"{s}_neg"] = float((xf < 0).mean()) if len(xf) else np.nan
            r[f"{s}_clip"] = float((np.abs(xf) >= clip).mean()) if (clip and len(xf)) else 0.0
        rr.append(r)
    pl.DataFrame(rr).write_csv(OUT / "stats_raw.csv")
    # 4. fp16 ties on val (xt columns after the attach_xt transform; raw rel inputs) ----------------------------------
    log("\n== fp16 storage: within-feed pairs distinct in fp32 but tied in fp16 (val) ==")
    d = D["val"]
    for j, nm in enumerate(XT):
        x = d["R"][:, j].astype(np.float32)
        if nm.startswith("has:"):
            continue
        if nm.endswith("_z"):
            x = np.clip(x, -5, 5)
        elif nm.startswith(("seq_cv", "it_hist")):
            x = np.log1p(np.maximum(x, 0))
        elif not nm.endswith("_rk"):
            x = np.clip(x, -10, 10)
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        x16 = x.astype(np.float16).astype(np.float32)
        lost, tied = tie_pairs(x, x16, d["off"], d["M"])
        log(f"  {nm:20s} pairs tied fp16 {tied:.4f}  of which created by fp16 {lost:.4f}  ({100*lost/max(tied,1e-12):.1f}% of ties)")
    log(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
