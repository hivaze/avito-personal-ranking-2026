"""P2 step A (PRETRAIN_PLAN_1113 §4 P2): f(history) = cross-fitted ridge prediction of the L2-normalised user_emb from
the feed's own history; r = u/||u|| - f(h); per-candidate cos_u and cos(c, r) for val and the 30K holdout feeds.
Rule: nn_x/pretrain2/P2/PREDECLARED.txt. CPU only, 12 threads. Writes only nn_x/pretrain2/P2/out/.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path
import numpy as np, polars as pl, numba as nb
from sklearn.model_selection import GroupKFold

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; COMP = ROOT / "data/competition"
OUT = ROOT / "nn_x/pretrain2/P2/out"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
LOG = open(OUT / "uresid.log", "w")
def log(*a):
    s = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

D = 128; NB = 11; NCAT = 54
NF = NB * D + NCAT + NCAT + 10 + 2 + NB


def table(split):
    meta = json.load(open(COMP / split / "item_embeddings/data.meta"))
    return np.memmap(COMP / split / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


@nb.njit(parallel=True)
def hist_feats(feeds, hoff, H, eidx, et, cat, tab, out):
    for j in nb.prange(len(feeds)):
        f = feeds[j]; o = hoff[f]; h = H[f]
        acc = np.zeros((11, 128)); cnt = np.zeros(11)
        cat64 = np.zeros(54); catall = np.zeros(54); ets = np.zeros(10); miss = 0.0
        v = np.zeros(128)
        for p in range(h):
            e = np.int64(eidx[o + p]); t = np.int64(et[o + p]); c = np.int64(cat[o + p])
            if c >= 0 and c < 54:
                catall[c] += 1.0
                if p < 64:
                    cat64[c] += 1.0
            if t >= 1 and t <= 10:
                ets[t - 1] += 1.0
            if e < 0:
                miss += 1.0; continue
            nrm = 0.0
            for d in range(128):
                x = float(tab[e, d]); v[d] = x; nrm += x * x
            if nrm <= 0:
                miss += 1.0; continue
            inv = 1.0 / np.sqrt(nrm)
            for d in range(128):
                v[d] *= inv
            wrec = 0.5 ** (p / 8.0)
            g = 6 + (t - 1 if t <= 4 else 4)
            for b in range(11):
                w = 0.0
                if b == 0:
                    w = 1.0
                elif b == 1:
                    w = wrec
                elif b == 2:
                    w = 1.0 if cnt[2] == 0 else 0.0
                elif b == 3:
                    w = 1.0 if p < 5 else 0.0
                elif b == 4:
                    w = 1.0 if p < 20 else 0.0
                elif b == 5:
                    w = 1.0 if p < 32 else 0.0
                else:
                    w = 1.0 if (p < 64 and b == g) else 0.0
                if w > 0:
                    cnt[b] += w
                    for d in range(128):
                        acc[b, d] += w * v[d]
        for b in range(11):
            if cnt[b] > 0:
                for d in range(128):
                    out[j, b * 128 + d] = acc[b, d] / cnt[b]
        k = 11 * 128
        h64 = max(min(h, 64), 1); ha = max(h, 1)
        for c in range(54):
            out[j, k + c] = cat64[c] / h64
            out[j, k + 54 + c] = catall[c] / ha
        k += 108
        for t in range(10):
            out[j, k + t] = ets[t] / ha
        k += 10
        out[j, k] = np.log1p(h); out[j, k + 1] = miss / ha
        k += 2
        for b in range(11):
            out[j, k + b] = 1.0 if cnt[b] == 0 else 0.0


def feats_for(split, feeds):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    hoff = f["hist_off"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
    ei = np.load(FLAT / split / "hist_emb_idx.npy", mmap_mode="r")
    et = np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r")
    ca = np.load(FLAT / split / "hist_category_id.npy", mmap_mode="r")
    out = np.zeros((len(feeds), NF), np.float32)
    hist_feats(np.asarray(feeds, np.int64), hoff, H, ei, et, ca, table(split), out)
    return out


def unit_u(split, idx=None):
    u = np.load(FLAT / split / "user_emb.npy", mmap_mode="r")
    u = np.asarray(u if idx is None else u[idx], np.float64)
    n = np.linalg.norm(u, axis=1)
    return u / np.maximum(n, 1e-12)[:, None], n


# ---------------- fit corpus ----------------
ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
Mtr = ftr["M"].to_numpy().astype(np.int64); Htr = ftr["H"].to_numpy().astype(np.int64); utr = ftr["user_id"].to_numpy()
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz"); assert (pr["fidx"] == np.arange(len(Mtr))).all()
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
cand = np.where((p > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
hm = np.load(ROOT / "preds/holdout/holdout_meta.npy"); assert (Mtr[hold] == hm[0]).all()
unorm_tr = np.linalg.norm(np.load(FLAT / "train/user_emb.npy", mmap_mode="r").astype(np.float32), axis=1)
ok_tr = (utr != 0) & (Htr >= 1) & (unorm_tr > 0)
pool = np.setdiff1d(np.where(ok_tr)[0], hold)
samp = np.sort(np.random.default_rng(7).choice(pool, 200000, replace=False))
hold_ok = hold[ok_tr[hold]]
fitf = np.concatenate([samp, hold_ok]); is_hold = np.r_[np.zeros(len(samp), bool), np.ones(len(hold_ok), bool)]
log(f"fit corpus: {len(samp)} sampled train feeds + {len(hold_ok)}/{len(hold)} holdout feeds "
    f"(holdout excluded: user0 {(utr[hold]==0).sum()}, H0 {(Htr[hold]==0).sum()}, |u|0 {(unorm_tr[hold]==0).sum()})")
t = time.time(); Xf = feats_for("train", fitf); log(f"train history features {Xf.shape} in {time.time()-t:.0f}s")
Yf, _ = unit_u("train", fitf)
gk = GroupKFold(5, shuffle=True, random_state=0)
folds = list(gk.split(np.arange(len(fitf)), groups=utr[fitf]))

LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
models = []; oof = {l: np.zeros_like(Yf) for l in LAMS}; r2 = {l: [] for l in LAMS}
for k, (tr, te) in enumerate(folds):
    t = time.time()
    X = Xf[tr].astype(np.float64); mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd; ym = Yf[tr].mean(0); Y = Yf[tr] - ym
    G = X.T @ X; B0 = X.T @ Y; del X
    s, V = np.linalg.eigh(G); VB = V.T @ B0
    Xte = (Xf[te].astype(np.float64) - mu) / sd
    mk = dict(mu=mu, sd=sd, ym=ym, B={})
    for l in LAMS:
        Bl = V @ (VB / (s + l)[:, None]); mk["B"][l] = Bl
        pte = Xte @ Bl + ym; oof[l][te] = pte
        yt = Yf[te]; r2[l].append(1 - ((yt - pte) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())
    models.append(mk)
    log(f"fold {k}: train {len(tr)} test {len(te)}  OOF R2 " + " ".join(f"lam{l:g} {r2[l][-1]:.4f}" for l in LAMS) + f"  {time.time()-t:.0f}s")
lam = max(LAMS, key=lambda l: np.mean(r2[l]))
log(f"chosen lambda {lam:g}: OOF R2 by fold {[round(float(x), 4) for x in r2[lam]]} mean {np.mean(r2[lam]):.4f}")
R = {"lambda": lam, "r2_by_fold": {str(l): [float(x) for x in r2[l]] for l in LAMS}}
# R2 on holdout feeds only and sampled only (OOF)
for nm, m in [("sampled", ~is_hold), ("holdout", is_hold)]:
    yt = Yf[m]; pt = oof[lam][m]
    R[f"r2_{nm}"] = float(1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())
    log(f"OOF R2 on {nm} feeds: {R[f'r2_{nm}']:.4f}")
r_fit = Yf - oof[lam]; nr_fit = np.linalg.norm(r_fit, axis=1)


def predict(Xs):
    acc = 0
    for mk in models:
        acc = acc + ((Xs.astype(np.float64) - mk["mu"]) / mk["sd"]) @ mk["B"][lam] + mk["ym"]
    return acc / len(models)


def qs(x):
    return {q: round(float(np.percentile(x, q)), 4) for q in (1, 10, 25, 50, 75, 90, 99)}


# ---------------- per-candidate features ----------------
@nb.njit(parallel=True)
def cand_cos(rows_emb, rows_feed, tab, U, Rr, ok, out_u, out_r):
    for i in nb.prange(len(rows_emb)):
        e = rows_emb[i]; g = rows_feed[i]
        out_u[i] = np.nan; out_r[i] = np.nan
        if e < 0 or not ok[g]:
            continue
        nc = 0.0; du = 0.0; dr = 0.0; nu = 0.0; nrr = 0.0
        for d in range(128):
            c = float(tab[e, d]); nc += c * c; du += c * U[g, d]; dr += c * Rr[g, d]
            nu += U[g, d] * U[g, d]; nrr += Rr[g, d] * Rr[g, d]
        if nc <= 0 or nu <= 0:
            continue
        out_u[i] = du / np.sqrt(nc * nu)
        if nrr > 0:
            out_r[i] = dr / np.sqrt(nc * nrr)


# val (and test: nr only, label-free)
for split in ("val", "test"):
    f = pl.read_parquet(FLAT / split / "feeds.parquet"); n = len(f)
    uid = f["user_id"].to_numpy(); Hs = f["H"].to_numpy()
    U, un = unit_u(split)
    ok = (uid != 0) & (Hs >= 1) & (un > 0)
    t = time.time(); Xs = feats_for(split, np.arange(n)); P = predict(Xs)
    Rr = U - P; nr = np.linalg.norm(Rr, axis=1); nr[~ok] = np.nan
    r2s = 1 - ((U[ok] - P[ok]) ** 2).sum() / ((U[ok] - U[ok].mean(0)) ** 2).sum()
    R[f"r2_{split}"] = float(r2s); R[f"nr_{split}"] = qs(nr[ok]); R[f"n_ok_{split}"] = int(ok.sum())
    log(f"{split}: {n} feeds, ok {ok.sum()} (user0 {(uid==0).sum()}, H0 {(Hs==0).sum()}, |u|0 {(un==0).sum()}); "
        f"R2 (fit on train) {r2s:.4f}; nr quantiles {R[f'nr_{split}']}  {time.time()-t:.0f}s")
    np.save(OUT / f"{split}_nr.npy", nr)
    if split == "test":
        ts = f["timestamp"].to_numpy().astype(np.int64); pub = ts < 1790497920
        R["nr_test_public"] = qs(nr[ok & pub]); R["nr_test_private"] = qs(nr[ok & ~pub])
        log(f"test nr public {R['nr_test_public']}  private {R['nr_test_private']}")
        continue
    M = f["M"].to_numpy().astype(np.int64)
    ce = np.load(FLAT / split / "cand_emb_idx.npy"); fid = np.repeat(np.arange(n), M)
    cu = np.empty(len(ce)); cr = np.empty(len(ce))
    cand_cos(ce.astype(np.int64), fid, table(split), U, Rr, ok, cu, cr)
    ref = np.load(ROOT / f"data/feats/{split}/cos_u.npy")
    mm = np.isfinite(cu) & np.isfinite(ref)
    log(f"{split} cos_u check vs data/feats: rows {mm.sum()}/{len(cu)}, max |diff| {np.abs(cu[mm]-ref[mm]).max():.2e}, corr {np.corrcoef(cu[mm], ref[mm])[0,1]:.6f}")
    np.save(OUT / f"{split}_cosu.npy", cu.astype(np.float32)); np.save(OUT / f"{split}_coscr.npy", cr.astype(np.float32))

# holdout: OOF residuals from the cross-fit
Uh, unh = unit_u("train", hold)
okh = ok_tr[hold]
Rh = np.zeros_like(Uh); pos = {f_: i for i, f_ in enumerate(fitf)}
ih = np.array([pos[f_] for f_ in hold_ok])
Rh[okh] = r_fit[ih]
nrh = np.full(len(hold), np.nan); nrh[okh] = nr_fit[ih]
np.save(OUT / "hold_nr.npy", nrh)
R["nr_holdout"] = qs(nrh[okh]); R["nr_trainsample"] = qs(nr_fit[~is_hold]); R["n_ok_holdout"] = int(okh.sum())
log(f"holdout nr quantiles (OOF) {R['nr_holdout']}; sampled-train nr {R['nr_trainsample']}")
toff = ftr["cand_off"].to_numpy().astype(np.int64)
hrows = np.concatenate([np.arange(o, o + m) for o, m in zip(toff[hold], Mtr[hold])])
ce = np.asarray(np.load(FLAT / "train/cand_emb_idx.npy", mmap_mode="r")[hrows]).astype(np.int64)
fidh = np.repeat(np.arange(len(hold)), Mtr[hold])
cu = np.empty(len(ce)); cr = np.empty(len(ce))
cand_cos(ce, fidh, table("train"), Uh, Rh, okh, cu, cr)
ref = np.asarray(np.load(ROOT / "data/feats/train/cos_u.npy", mmap_mode="r")[hrows])
mm = np.isfinite(cu) & np.isfinite(ref)
log(f"holdout cos_u check vs data/feats: rows {mm.sum()}/{len(cu)}, max |diff| {np.abs(cu[mm]-ref[mm]).max():.2e}, corr {np.corrcoef(cu[mm], ref[mm])[0,1]:.6f}")
np.save(OUT / "hold_cosu.npy", cu.astype(np.float32)); np.save(OUT / "hold_coscr.npy", cr.astype(np.float32))
np.save(OUT / "hold_fidx.npy", hold)
# parity: median nr val vs holdout (train random-like)
R["parity_ratio_val_over_holdout_median"] = float(R["nr_val"][50] / R["nr_holdout"][50])
log(f"parity: median nr val / holdout = {R['parity_ratio_val_over_holdout_median']:.3f} (pre-declared band 0.9-1.1)")
json.dump(R, open(OUT / "uresid.json", "w"), indent=1)
log("done")
