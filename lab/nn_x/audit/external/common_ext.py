"""Shared helpers for the external-data audit (CPU only, read-only on project data).
Re-uses the nested-additive harness of nn_x/audit/avito/common_av.py (copied, not imported, so this directory is self-contained)."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
PREDS = ROOT / "preds"
OUT = ROOT / "nn_x/audit/external/out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402

MSK = 3 * 3600
PUBLIC_TS = 1790497920  # first private timestamp (eda_shift_overlap)
DAY0 = (1788642048 + MSK) // 86400  # Sep 6 2026 MSK = day index 0
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    d = dict(fidx=f["fidx"].to_numpy(), ts=f["timestamp"].to_numpy().astype(np.int64),
             user=f["user_id"].to_numpy().astype(np.int64), M=f["M"].to_numpy().astype(np.int64),
             H=f["H"].to_numpy().astype(np.int64), off=f["cand_off"].to_numpy().astype(np.int64),
             hoff=f["hist_off"].to_numpy().astype(np.int64))
    t = d["ts"] + MSK
    d["hour"] = (t // 3600) % 24
    d["day"] = t // 86400 - DAY0  # 0 = Sep 6
    d["dow"] = (d["day"] + 6) % 7  # Sep 6 2026 is a Sunday -> 6; Monday = 0
    d["groups"] = np.where(d["user"] == 0, -1 - np.arange(len(d["user"])), d["user"])
    d["n"] = len(d["M"])
    return d


def day_label(day):
    import datetime as dt
    return (dt.date(2026, 9, 6) + dt.timedelta(days=int(day))).strftime("%a %d")


def cand(split, name):
    return np.load(FLAT / split / f"cand_{name}.npy", mmap_mode="r")


def hist(split, name):
    return np.load(FLAT / split / f"hist_{name}.npy", mmap_mode="r")


def feat(split, name):
    return np.load(FEATS / split / f"{name}.npy")


def g3(split):
    return np.load(PREDS / f"ens_G3_final_{split}.npy").astype(np.float64)


def hour_bucket(h):
    return np.digitize(h, [7, 12, 18])  # 0: 0-6, 1: 7-11, 2: 12-17, 3: 18-23


def feed_z(x, fidx, n):
    x = np.asarray(x, dtype=np.float64)
    c = np.bincount(fidx, minlength=n).astype(np.float64)
    mu = np.bincount(fidx, weights=x, minlength=n) / np.maximum(c, 1)
    v = np.bincount(fidx, weights=(x - mu[fidx]) ** 2, minlength=n) / np.maximum(c, 1)
    sd = np.sqrt(v)
    sd[sd < 1e-9] = 1.0
    return (x - mu[fidx]) / sd[fidx]


def paired(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def fmt(a, b):
    m, se = paired(a, b)
    return f"{m:+.5f} ± {se:.5f} (z {m / se if se > 0 else 0:+.1f})"


def make_folds(groups, nfold=5, seed=0):
    u, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    f = rng.permutation(len(u)) % nfold
    return f[inv]


WGRID = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5])


def nested_additive(base, extra, y, off, M, fold, wgrid=WGRID, allow_neg=True):
    """score = base + w * extra; w chosen on the other folds by mean per-feed NDCG. Returns (oof per-feed NDCG, base per-feed NDCG, chosen ws, full-fit curve)."""
    if allow_neg:
        wgrid = np.concatenate([-wgrid[1:][::-1], wgrid])
    n = len(M)
    nd = np.zeros((len(wgrid), n))
    for i, w in enumerate(wgrid):
        nd[i] = ndcg_per_feed(base + w * extra, y, off, M)
    base_nd = ndcg_per_feed(base, y, off, M)
    oof = np.zeros(n)
    ws = []
    for k in np.unique(fold):
        tr = fold != k
        te = fold == k
        i = int(np.argmax(nd[:, tr].mean(1)))
        oof[te] = nd[i, te]
        ws.append(float(wgrid[i]))
    return oof, base_nd, ws, nd.mean(1) - base_nd.mean(), wgrid


def nested_report(name, base, extra, y, off, M, fold, slices=None):
    oof, bnd, ws, curve, wg = nested_additive(base, extra, y, off, M, fold)
    s = f"{name:60s} OOF {fmt(oof, bnd)}; w {ws}; best full-fit {curve.max():+.5f} at w={wg[int(np.argmax(curve))]:+.2f}"
    if slices:
        for sn, sm in slices.items():
            s += f"; {sn} {fmt(oof[sm], bnd[sm])}"
    log(s)
    return oof, bnd


def holdout_index():
    """Reproduce nn/train.py's 30K random-like holdout feed list (seed 12345)."""
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    f = pl.read_parquet(FLAT / "train" / "feeds.parquet", columns=["M"])
    Mt = f["M"].to_numpy().astype(np.int64)
    p[Mt < 50] = 0.0
    cand_ = np.where((p > 0.9) & (Mt >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand_, min(30000, len(cand_)), replace=False))
    return hold, p


def g3_holdout_pool():
    """Member-equal z-mean of the G3 train-only members' holdout predictions."""
    names = [x.strip() for x in open(ROOT / "nn_x/pubdiag/build/g3_members.txt").read().replace("\n", ",").split(",") if x.strip()]
    hm = np.load(PREDS / "holdout/holdout_meta.npy")
    M = hm[0].astype(np.int64)
    fidx = np.repeat(np.arange(len(M)), M)
    acc = None
    used = []
    for nm in names:
        p = PREDS / "holdout" / f"nn_{nm}_holdout.npy"
        if not p.exists():
            continue
        z = feed_z(np.load(p), fidx, len(M))
        acc = z if acc is None else acc + z
        used.append(nm)
    return acc / len(used), used, M


def rows_of(off, M, feeds):
    return np.concatenate([np.arange(o, o + m) for o, m in zip(off[feeds], M[feeds])])


def smoothed_logratio(cat, grp, ngrp, y, NC=64, K=300.0):
    """log( smoothed rate(cat, grp) / rate(cat) ), chi2 per category, counts."""
    pos_c = np.bincount(cat, weights=y, minlength=NC)
    cnt_c = np.bincount(cat, minlength=NC).astype(np.float64)
    rate_c = pos_c / np.maximum(cnt_c, 1)
    key = cat * ngrp + grp
    pos = np.bincount(key, weights=y, minlength=NC * ngrp).reshape(NC, ngrp)
    cnt = np.bincount(key, minlength=NC * ngrp).reshape(NC, ngrp).astype(np.float64)
    sm = (pos + K * rate_c[:, None]) / (cnt + K)
    lr = np.log(np.maximum(sm, 1e-6)) - np.log(np.maximum(rate_c, 1e-6))[:, None]
    lr[cnt_c == 0] = 0
    exp = cnt * rate_c[:, None]
    chi2 = ((pos - exp) ** 2 / np.maximum(exp, 1e-9) + (cnt - pos - (cnt - exp)) ** 2 / np.maximum(cnt - exp, 1e-9)).sum(1)
    return lr, chi2, cnt_c, rate_c, pos, cnt
