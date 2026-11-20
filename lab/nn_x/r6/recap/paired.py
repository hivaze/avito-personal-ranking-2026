"""Matched family comparisons on stored train-only predictions (CPU only, read-only on preds/).

Per comparison: (1) run-level: mean of member scores, SE from the seed spread (Welch);
(2) per-feed paired: mean over members of per-feed NDCG for each side, delta per feed, SE over feeds (ignores seed
variance, so it is a lower bound on the true SE); slices M<50 / M>=50 / Tue / Thu.
Ensemble level: per-feed z-mean of the first k seeds; within-feed seed correlation of z-scores.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

R = "<repo>"
f = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)
day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
day = np.where(day <= day.min() + 1, 0, 1)
fid = np.repeat(np.arange(len(M)), M)
SL = {"M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1}

HM, HO = np.load(f"{R}/preds/holdout/holdout_meta.npy")
HY = np.load(f"{R}/preds/holdout/holdout_y.npy").astype(np.float64)
hfid = np.repeat(np.arange(len(HM)), HM)


def fz(s, fid_, M_):
    s = s.astype(np.float64)
    mu = np.bincount(fid_, s, len(M_)) / M_
    sd = np.sqrt(np.maximum(np.bincount(fid_, s * s, len(M_)) / M_ - mu ** 2, 0)) + 1e-6
    return (s - mu[fid_]) / sd[fid_]


_cache = {}


def pf(v, split="val"):
    k = (v, split)
    if k not in _cache:
        if split == "val":
            s = np.load(f"{R}/preds/nn_{v}_trainonly_val.npy")
            _cache[k] = (ndcg_per_feed(s, y, off, M), fz(s, fid, M))
        else:
            p = f"{R}/preds/holdout/nn_{v}_holdout.npy"
            if not os.path.exists(p):
                _cache[k] = (None, None)
            else:
                s = np.load(p)
                _cache[k] = (ndcg_per_feed(s, HY, HO, HM), fz(s, hfid, HM))
    return _cache[k]


FAM = {
    "F5": ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"],
    "F6": ["v14_s92", "v14_s93", "v14_s94", "v14_s95", "v14_s96", "v13_lr1_mix1"],
    "F7": [f"v17_pre2s_s{s}" for s in range(101, 108)],
    "F8": [f"v20_pre2s_e2_s{s}" for s in range(301, 313)],
    "F9": [f"v22_pre2s_lr1_s{s}" for s in range(401, 407)],
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84",
             "v15_cd_allprew_s85"],
    "F6cd": ["v21_cd_v14_s92", "v21_cd_v14_s94", "v21_cd_v14_s96"],
    "F6cd_base": ["v14_s92", "v14_s94", "v14_s96"],
    "sx2": ["sx_base_s23", "sx_base_s24"],
}
out = []
P = out.append


def single(name, A, B):
    """B - A at the single-member level."""
    for split in ("val", "hold"):
        a = [pf(v, split)[0] for v in FAM.get(A, [A])]
        b = [pf(v, split)[0] for v in FAM.get(B, [B])]
        a = [x for x in a if x is not None]
        b = [x for x in b if x is not None]
        if not a or not b:
            P(f"  {split}: missing preds")
            continue
        ma = np.array([x.mean() for x in a]); mb = np.array([x.mean() for x in b])
        se_run = np.sqrt((ma.var(ddof=1) / len(ma) if len(ma) > 1 else 0) + (mb.var(ddof=1) / len(mb) if len(mb) > 1 else 0))
        d = np.mean(b, 0) - np.mean(a, 0)
        se_f = d.std() / np.sqrt(len(d))
        line = (f"  {split:4s} single: {B} {mb.mean():.5f} (n={len(mb)}, sd {mb.std(ddof=1) if len(mb) > 1 else 0:.5f}) vs {A} "
                f"{ma.mean():.5f} (n={len(ma)}, sd {ma.std(ddof=1) if len(ma) > 1 else 0:.5f}): {d.mean():+.5f}  "
                f"run-SE {se_run:.5f}  feed-SE {se_f:.5f}  z_run {d.mean() / se_run if se_run > 0 else float('nan'):.2f}")
        if split == "val":
            line += "  | " + " ".join(f"{k} {d[m].mean():+.5f}" for k, m in SL.items())
        P(line)


def ens(names, split, k):
    zs = [pf(v, split)[1] for v in names[:k]]
    zs = [z for z in zs if z is not None]
    if not zs:
        return float("nan")
    s = np.mean(zs, 0)
    if split == "val":
        return ndcg_per_feed(s, y, off, M).mean()
    return ndcg_per_feed(s, HY, HO, HM).mean()


def seedcorr(names, split="val"):
    zs = [pf(v, split)[1] for v in names]
    zs = [z for z in zs if z is not None]
    c = [np.corrcoef(zs[i], zs[j])[0, 1] for i in range(len(zs)) for j in range(i + 1, len(zs))]
    return np.mean(c) if c else float("nan")


P("## Matched single-member comparisons (train-only, holdout-selected checkpoints)")
for nm, A, B in [("lr 2e-3 vs 1e-3 (pre2s, mix 0.1, 10 ep)", "F9", "F7"),
                 ("10 vs 2 epochs (pre2s, lr 1e-3, mix 0.1)", "F8", "F9"),
                 ("pre2s vs cvnd (lr 1e-3, mix 0.1, 10 ep)", "F6", "F9"),
                 ("lr 2e-3 vs 1e-3, seed 23, cvnd, mix 0.1", "v13_lr1_mix1", "v13_lr2_mix1"),
                 ("mix 0 vs mix 0.1, seed 23, cvnd, lr 1e-3", "v13_lr1_mix1", "v13_lr1_mix0"),
                 ("mix 0.1 vs mix 0.3, seed 23, cvnd, lr 2e-3", "v12_all_e10", "v13_lr2_mix1"),
                 ("1-ep cooldown continuation vs plateau ckpt (F5)", "F5", "F5cd"),
                 ("1-ep cooldown continuation vs plateau ckpt (F6 s92/94/96)", "F6cd_base", "F6cd"),
                 ("F7 (lr2 mix0.1 pre2s) vs F5 (lr2 mix0.3 cvnd/prew)", "F5", "F7"),
                 ("Muon+WSD 10ep cvnd seed23 (lr1 mix0) vs AdamW cos 10ep (v9_e10, no retr/gate)", "v9_e10", "v13_lr1_mix0")]:
    P(f"\n### {nm}")
    single(nm, A, B)

P("\n## Ensemble value (per-feed z-mean of the first k seeds, train-only)")
P("family  k=1(mean single)  k=2  k=3  k=6  gain1->6 | holdout k=1 k=6 | within-feed seed corr (val z)")
for fam in ("F5", "F6", "F7", "F8", "F9", "F5cd"):
    names = FAM[fam]
    v1 = np.mean([pf(v)[0].mean() for v in names[:6]])
    h1s = [pf(v, "hold")[0] for v in names[:6]]
    h1 = np.mean([h.mean() for h in h1s if h is not None]) if any(h is not None for h in h1s) else float("nan")
    e2, e3, e6 = ens(names, "val", 2), ens(names, "val", 3), ens(names, "val", 6)
    h6 = ens(names, "hold", 6)
    P(f"{fam:5s} {v1:.5f} {e2:.5f} {e3:.5f} {e6:.5f} {e6 - v1:+.5f} | {h1:.5f} {h6:.5f} {h6 - h1:+.5f} | {seedcorr(names[:6]):.4f}")
P("\n## Cross-family correlation of 6-seed TO ensembles (val z)")
Z = {fam: np.mean([pf(v)[1] for v in FAM[fam][:6]], 0) for fam in ("F5", "F6", "F7", "F8", "F9")}
ks = list(Z)
for i in ks:
    P(i + " " + " ".join(f"{np.corrcoef(Z[i], Z[j])[0, 1]:.4f}" for j in ks))
P("\n## Cross-family pairs: member-equal 12-member pools vs 6-seed families (val / holdout)")
for A, B in [("F7", "F9"), ("F6", "F9"), ("F7", "F6"), ("F8", "F9")]:
    na, nb = FAM[A][:6], FAM[B][:6]
    pv = ens(na + nb, "val", 12); ph = ens(na + nb, "hold", 12)
    P(f"{A}+{B}: val {pv:.5f} (vs {A} {ens(na, 'val', 6):.5f}, {B} {ens(nb, 'val', 6):.5f}); hold {ph:.5f} "
      f"(vs {ens(na, 'hold', 6):.5f}, {ens(nb, 'hold', 6):.5f})")
txt = "\n".join(out)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paired_report.txt"), "w").write(txt)
print(txt)
