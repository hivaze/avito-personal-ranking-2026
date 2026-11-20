"""ap04 (post-hoc diagnostic, NOT part of the gate): does a plain within-feed score shift by listing age help the proxy
on val at all? (a) s - w * L_age[M bucket, age bin] with the train-derived production/exploration lifts (no fitting on
val); (b) s + w * log(act/exp)[age bin] cross-fitted on 5 user-grouped folds (a07-style). Applied to the seed-mean r200
val predictions of base, age1, tprod1. Paired per-feed deltas vs the unshifted score."""
import json
from pathlib import Path

import numpy as np

ROOT = Path("<repo>")
import sys
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402

G = ROOT / "nn_x/insights/data_usage/out/gb"
AP = ROOT / "nn_x/audit/ageprod/out"
OUT = AP / "gb"
SEEDS = [11, 12, 13, 14]
mv = dict(np.load(G / "val_meta.npz")); M = mv["M"].astype(np.int64); y = mv["y"].astype(np.float64)
off = np.concatenate([[0], np.cumsum(M)[:-1]]); n = len(M); fid = np.repeat(np.arange(n), M)
b = np.load(AP / "val_agebin.npy").astype(np.int64)
LA = json.load(open(AP / "age_lifts.json"))["lifts"]
L_age = np.array([LA["prod_M<50"]["L"], LA["prod_M>=50"]["L"]])[(M >= 50).astype(int)[fid], b]
npos = np.bincount(fid, weights=y, minlength=n)
users = mv["user_id"].astype(np.int64)
grp = np.where(users == 0, -1 - np.arange(n), users)
u, inv = np.unique(grp, return_inverse=True)
fold = (np.random.default_rng(0).permutation(len(u)) % 5)[inv]
rfold = fold[fid]
lines = []


def log(s):
    print(s, flush=True); lines.append(s)


def pm(d):
    return f"{d.mean():+.5f}±{d.std(ddof=1) / np.sqrt(len(d)):.5f}"


def actexp(s, rows):
    z = s - np.repeat(np.maximum.reduceat(s, off), M); e = np.exp(z); p = e / np.repeat(np.add.reduceat(e, off), M)
    ex = p * npos[fid]
    return np.bincount(b[rows], weights=y[rows], minlength=8) / np.maximum(np.bincount(b[rows], weights=ex[rows], minlength=8), 1e-9)


for arm in ["base", "age1", "tprod1", "tprod1+age1"]:
    fs = [OUT / f"res_{arm.replace('+', '_')}_s{s}.npz" for s in SEEDS]
    if not all(f.exists() for f in fs):
        continue
    s0 = np.mean([np.load(f)["pred_val200"] for f in fs], 0).astype(np.float64)
    nd0 = ndcg_per_feed(s0, y, off, M)
    log(f"== {arm}: r200 seed-mean val {nd0.mean():.5f}; act/exp by age bin {np.round(actexp(s0, np.ones(len(y), bool)), 3).tolist()} ==")
    for w in [-0.5, -0.25, 0.25, 0.5, 0.75, 1.0]:
        nd = ndcg_per_feed(s0 - w * L_age, y, off, M)
        d = nd - nd0
        log(f"  (a) s - {w:+.2f} x L_age(train prod/expl): {pm(d)} | M<50 {pm(d[M < 50])} M>=50 {pm(d[M >= 50])}")
    for w in [0.5, 1.0]:
        oof = np.zeros(n)
        offs_all = []
        for k in range(5):
            tr = rfold != k
            ae = actexp(s0, tr)
            offs = np.clip(np.log(np.maximum(ae, 1e-3)), -1.0, 1.0)
            offs_all.append(offs)
            nd = ndcg_per_feed(s0 + w * offs[b], y, off, M)
            oof[fold == k] = nd[fold == k]
        d = oof - nd0
        log(f"  (b) cross-fitted log(act/exp) offset w={w}: {pm(d)} | M<50 {pm(d[M < 50])} M>=50 {pm(d[M >= 50])} | mean offsets {np.round(np.mean(offs_all, 0), 3).tolist()}")
open(AP / "ap04_posthoc.txt", "w").write("\n".join(lines) + "\n")
