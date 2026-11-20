"""C03 Latent geometry (PREDECLARED_ccb.txt C03). Build only -> cache/c03_{val,hold}.npz and cache/c03_W.npy. Label-free.
loc16  = cos(Pf' c~, sum_{h: hpos<20} Pf' h~) (s08 e): ~ = raw int8 vector minus the split's category mean (val: val_cat_means.npy;
         holdout: category means of the train candidate rows of the 30K holdout feeds); NaN if no vector / no such history.
mode6, mode96 = 1[candidate on the same side of dim 6 (>12) / dim 96 (>40) as the history's dominant side] (s09); NaN unless
         >= 80% of the feed's history vectors are on one side (or no candidate vector).
W (16x128): ridge (sum of squared errors + 1.0 ||W||_F^2), torch CPU 12 threads, L-BFGS, of within-feed z(f2) on (W e_c).(W h32)
         (e = L2-normalised int8 vectors, h32 = mean of e_h over hpos < 32), on a 2M-row uniform sample of train candidate rows of
         non-holdout feeds (rows lacking a candidate vector or any last-32 history vector dropped after sampling).
g_shift  = cos(W e_c, p8) - cos(W e_c, pall) (p = mean of W e_h over hpos < 8 / all history);
g_f2res  = zf(f2) - zf(f2_hat), f2_hat = (W e_c).p32;
g_cmatch = share of distinct contacted (event type 2/3) history items (with a vector) with cos(W e_h, W e_c) >= 0.9; NaN if none."""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import time
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib_av import feed_z_nanmin  # noqa: E402
ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
ES = ROOT / "nn_x/emb/structure/cache"
T0 = time.time()
torch.set_num_threads(12)
nb.set_num_threads(12)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def emb_table(split):
    meta = json.load(open(ROOT / f"data/competition/{split}/item_embeddings/data.meta"))
    return np.memmap(ROOT / f"data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def fetch(E, idx):
    """raw int8 -> float32 rows (zeros where idx < 0), sorted access."""
    miss = idx < 0
    ii = np.where(miss, 0, idx)
    o = np.argsort(ii, kind="stable")
    X = np.empty((len(ii), 128), np.float32)
    X[o] = np.asarray(E[ii[o]], dtype=np.float32)
    X[miss] = 0.0
    return X, miss


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def seg_sum(V, seg, n, mask):
    return np.stack([np.bincount(seg[mask], weights=V[mask, d], minlength=n) for d in range(V.shape[1])], 1)


def cosr(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


@nb.njit(parallel=True, cache=True)
def cmatch(coff, M, hstart, hcnt, hid_sorted, WHn, WCn, cmiss, out):
    # hid_sorted: per feed, contiguous block [hstart, hstart+hcnt) of distinct contacted-history rows (indices into WHn)
    for f in nb.prange(len(M)):
        o = coff[f]
        n = hcnt[f]
        for i in range(M[f]):
            r = o + i
            if n == 0 or cmiss[r]:
                out[r] = np.nan
                continue
            c = 0
            for j in range(n):
                h = hid_sorted[hstart[f] + j]
                s = 0.0
                for d in range(WCn.shape[1]):
                    s += WCn[r, d] * WHn[h, d]
                if s >= 0.9:
                    c += 1
            out[r] = c / n


def build(split_tag, E, ce, cc, f2, coff, M, he, hc, hp, ht, hf, CM, Pf, W):
    """All arrays local to the query set. hf = local feed index of each history row."""
    F = len(M)
    fidc = np.repeat(np.arange(F), M)
    K = CM.shape[0]
    Xc, cmiss = fetch(E, ce)
    Xh, hmiss = fetch(E, he)
    hok = ~hmiss
    # loc16
    Rc = (Xc - CM[np.minimum(cc, K - 1)]) @ Pf
    Rh = (Xh - CM[np.minimum(hc, K - 1)]) @ Pf
    m20 = hok & (hp < 20)
    HR20 = seg_sum(Rh, hf, F, m20)
    n20 = np.bincount(hf[m20], minlength=F)
    loc16 = cosr(Rc, HR20[fidc])
    loc16[cmiss | (n20[fidc] == 0)] = np.nan
    # mode6 / mode96
    nh = np.bincount(hf[hok], minlength=F)
    modes = {}
    for d, thr in ((6, 12), (96, 40)):
        hin = np.zeros(len(he), bool)
        hin[hok] = Xh[hok, d] > thr
        sh = np.bincount(hf[hok], weights=hin[hok], minlength=F) / np.maximum(nh, 1)
        dom = np.where(sh >= 0.8, 1.0, np.where(sh <= 0.2, 0.0, np.nan))
        dom[nh == 0] = np.nan
        cin = (Xc[:, d] > thr).astype(np.float64)
        mm = (cin == dom[fidc]).astype(np.float64)
        mm[np.isnan(dom[fidc]) | cmiss] = np.nan
        modes[d] = mm
    # W features
    WC = unit(Xc) @ W.T
    WH = unit(Xh) @ W.T
    def pmean(mask):
        s = seg_sum(WH, hf, F, mask)
        n = np.bincount(hf[mask], minlength=F)
        return s / np.maximum(n, 1)[:, None], n
    p8, n8 = pmean(hok & (hp < 8))
    p32, n32 = pmean(hok & (hp < 32))
    pall, nall = pmean(hok)
    g_shift = cosr(WC, p8[fidc]) - cosr(WC, pall[fidc])
    g_shift[cmiss | (n8[fidc] == 0)] = np.nan
    f2hat = (WC * p32[fidc]).sum(1)
    f2hat[cmiss | (n32[fidc] == 0)] = np.nan
    g_f2res = feed_z_nanmin(f2.astype(np.float64), coff, M) - feed_z_nanmin(f2hat.astype(np.float64), coff, M)
    # g_cmatch: distinct contacted items (types 2/3) with a vector
    mc = hok & ((ht == 2) | (ht == 3))
    rows = np.nonzero(mc)[0]
    key = pl.DataFrame({"f": hf[rows], "e": he[rows], "r": rows}).unique(["f", "e"], keep="first").sort(["f", "r"])
    hsel = key["r"].to_numpy().astype(np.int64)
    hcnt = np.bincount(key["f"].to_numpy(), minlength=F).astype(np.int64)
    hstart = np.concatenate([[0], np.cumsum(hcnt)[:-1]]).astype(np.int64)
    gcm = np.empty(len(ce))
    cmatch(coff, M, hstart, hcnt, hsel, unit(WH).astype(np.float64), unit(WC).astype(np.float64), cmiss, gcm)
    out = dict(loc16=loc16, mode6=modes[6], mode96=modes[96], g_shift=g_shift, g_f2res=g_f2res, g_cmatch=gcm)
    for k, v in out.items():
        log(f"{split_tag} {k}: non-NaN {np.isfinite(v).mean():.4f} mean {np.nanmean(v):.4f} sd {np.nanstd(v):.4f}")
    return out


def main():
    ft = pl.read_parquet(FLAT / "train/feeds.parquet")
    Mt = ft["M"].to_numpy().astype(np.int64)
    ot = ft["cand_off"].to_numpy().astype(np.int64)
    Ht = ft["H"].to_numpy().astype(np.int64)
    hot = ft["hist_off"].to_numpy().astype(np.int64)
    p = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    p[Mt < 50] = 0.0
    HOLD = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (Mt >= 50))[0], 30000, replace=False))
    is_hold = np.zeros(len(Mt), bool)
    is_hold[HOLD] = True
    ET = emb_table("train")
    # ---------- W fit (label-free) ----------
    rng = np.random.default_rng(0)
    N = int(Mt.sum())
    fid_row_sample = None
    cand_rows = np.sort(rng.choice(N, 2_400_000, replace=False))
    frow = np.searchsorted(ot, cand_rows, side="right") - 1
    keep = ~is_hold[frow]
    cand_rows, frow = cand_rows[keep][:2_000_000 + 400_000], frow[keep]
    # take exactly a 2M uniform sample of non-holdout rows
    sub = np.sort(rng.choice(len(cand_rows), 2_000_000, replace=False))
    cand_rows, frow = cand_rows[sub], frow[sub]
    f2t = np.load(FLAT / "train/cand_f2.npy")
    s1 = np.add.reduceat(f2t.astype(np.float64), ot)
    s2 = np.add.reduceat(f2t.astype(np.float64) ** 2, ot)
    mu = s1 / Mt
    sd = np.sqrt(np.maximum(s2 / Mt - mu ** 2, 0))
    zf2 = np.where(sd[frow] > 1e-9, (f2t[cand_rows] - mu[frow]) / np.maximum(sd[frow], 1e-9), 0.0)
    del f2t
    ce_t = np.load(FLAT / "train/cand_emb_idx.npy", mmap_mode="r")
    ec, ecm = fetch(ET, np.asarray(ce_t[cand_rows]))
    ec = unit(ec)
    uf, finv = np.unique(frow, return_inverse=True)
    hrows = np.concatenate([np.arange(hot[f], hot[f] + Ht[f]) for f in uf])
    hseg = np.repeat(np.arange(len(uf)), Ht[uf])
    hp_t = np.load(FLAT / "train/hist_hpos.npy", mmap_mode="r")
    he_t = np.load(FLAT / "train/hist_emb_idx.npy", mmap_mode="r")
    hpp = np.asarray(hp_t[hrows])
    m32 = hpp < 32
    hrows, hseg = hrows[m32], hseg[m32]
    hee = np.asarray(he_t[hrows])
    eh, ehm = fetch(ET, hee)
    eh = unit(eh)
    okh = ~ehm
    h32 = seg_sum(eh, hseg, len(uf), okh)
    n32 = np.bincount(hseg[okh], minlength=len(uf))
    h32 = h32 / np.maximum(n32, 1)[:, None]
    del eh
    use = (~ecm) & (n32[finv] > 0)
    log(f"W fit sample: 2,000,000 uniform non-holdout rows from {len(uf)} feeds; usable {use.sum()} (cand vector and >= 1 last-32 history vector)")
    Ec = torch.from_numpy(ec[use].astype(np.float32))
    Hc = torch.from_numpy(h32[finv][use].astype(np.float32))
    Y = torch.from_numpy(zf2[use].astype(np.float32))
    torch.manual_seed(0)
    Wt = (0.1 * torch.randn(16, 128)).requires_grad_(True)
    opt = torch.optim.LBFGS([Wt], lr=1.0, max_iter=300, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                            tolerance_change=1e-12)
    it = [0]

    def closure():
        opt.zero_grad()
        pred = ((Ec @ Wt.T) * (Hc @ Wt.T)).sum(1)
        loss = ((Y - pred) ** 2).sum() + 1.0 * (Wt ** 2).sum()
        loss.backward()
        it[0] += 1
        return loss

    for rnd in range(2):
        loss = opt.step(closure)
        with torch.no_grad():
            pred = ((Ec @ Wt.T) * (Hc @ Wt.T)).sum(1)
            r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
        log(f"W fit round {rnd}: evals {it[0]}, loss {float(loss):.1f}, R2 {float(r2):.4f}, corr {float(np.corrcoef(pred.numpy(), Y.numpy())[0, 1]):.4f}")
    W = Wt.detach().numpy().astype(np.float32)
    np.save(HERE / "cache/c03_W.npy", W)
    del Ec, Hc, Y, ec
    Pf = np.load(ES / "feedlevel_subspace16.npy").astype(np.float32)
    # ---------- val ----------
    fv = pl.read_parquet(FLAT / "val/feeds.parquet")
    V = lambda n: np.load(FLAT / f"val/{n}.npy")  # noqa: E731
    CMv = np.load(ES / "val_cat_means.npy").astype(np.float32)
    outv = build("val", emb_table("val"), V("cand_emb_idx"), V("cand_category_id").astype(np.int64), V("cand_f2"),
                 fv["cand_off"].to_numpy().astype(np.int64), fv["M"].to_numpy().astype(np.int64), V("hist_emb_idx"),
                 V("hist_category_id").astype(np.int64), V("hist_hpos"), V("hist_event_type"), V("hist_fidx").astype(np.int64),
                 CMv, Pf, W)
    np.savez(HERE / "cache/c03_val.npz", **outv)
    # ---------- holdout ----------
    Mh = Mt[HOLD]
    oh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64)
    crow = np.concatenate([np.arange(ot[f], ot[f] + Mt[f]) for f in HOLD])
    hrow = np.concatenate([np.arange(hot[f], hot[f] + Ht[f]) for f in HOLD])
    hfl = np.repeat(np.arange(len(HOLD)), Ht[HOLD])
    T = lambda n, r: np.asarray(np.load(FLAT / f"train/{n}.npy", mmap_mode="r")[r])  # noqa: E731
    ce_h = T("cand_emb_idx", crow)
    cc_h = T("cand_category_id", crow).astype(np.int64)
    Xc_h, cm_h = fetch(ET, ce_h)
    K = CMv.shape[0]
    CMh = np.zeros_like(CMv)
    nk = np.bincount(cc_h[~cm_h], minlength=K).astype(np.float64)
    for d in range(128):
        CMh[:, d] = np.bincount(cc_h[~cm_h], weights=Xc_h[~cm_h, d], minlength=K) / np.maximum(nk, 1)
    log(f"holdout category means from {(~cm_h).sum()} holdout candidate rows; corr with val CM {np.corrcoef(CMh.ravel(), CMv.ravel())[0, 1]:.4f}")
    del Xc_h
    outh = build("hold", ET, ce_h, cc_h, T("cand_f2", crow), oh, Mh, T("hist_emb_idx", hrow), T("hist_category_id", hrow).astype(np.int64),
                 T("hist_hpos", hrow), T("hist_event_type", hrow), hfl, CMh, Pf, W)
    np.savez(HERE / "cache/c03_hold.npz", **outh)
    log("DONE")


if __name__ == "__main__":
    main()
