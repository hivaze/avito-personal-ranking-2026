"""Critic re-check of the training stream (CPU only, stored predictions only).
Independent code: own per-feed z, own NDCG (binary gain, log2, pessimistic ties, no-positive feeds -> 1), pmix weights
from src/pmix.py (the only shared helper), own SEs.
 A. rebuild FD16P20@m0.4_0.2 from members and compare with the stored ens files (val NDCG, test corr)
 B. item 1 (pop-light part of FT branch = train-only pop-light preds on M>=thr) + per-M-bucket contributions
 C. selection-aware read: nested threshold/direction choice by user folds, and a sign-flip null of the max over the menu
 D. replication independence: per-feed corr of the item-1 delta on FD16P20 vs on W16+PD4w
 E. item 2 analog already on disk: click-label pop-light abl_e10_pop1_s951-954 fine-tuned with the nopop preset (25 extras)
    vs the standard preset (42 extras), single-seed n2cpv3, as the pop-light part on W16 and FD16 bases
 F. popularity-tercile O/E of the FINAL file (stored vs item 1) on M>=50
"""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from pathlib import Path
import pmix

P = Path("<repo>/preds"); FL = Path("<repo>/data/flat")
fv = pl.read_parquet(FL / "val/feeds.parquet"); Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy()
yv = np.load(FL / "val/cand_target.npy").astype(np.float64)
N = len(Mv); fid = np.repeat(np.arange(N), Mv); MROW = Mv[fid]; big = Mv >= 50
assert ov[0] == 0 and (np.diff(ov) == Mv[:-1]).all() and len(yv) == Mv.sum()
W = pmix.weights()

def zf(x, fid=fid, M=Mv):
    x = np.asarray(x, np.float64)
    mu = np.bincount(fid, x, len(M)) / M
    var = np.maximum(np.bincount(fid, x * x, len(M)) / M - mu ** 2, 0)
    return (x - mu[fid]) / (np.sqrt(var)[fid] + 1e-9)

def ndcg(s, y=yv, fid=fid, M=Mv, off=ov):
    s = np.asarray(s, np.float64)
    order = np.lexsort((y, -s, fid))          # within feed: score desc, ties -> negatives first (pessimistic)
    pos = np.arange(len(s)) - off[fid]        # rank position inside feed (fid sorted already)
    d = 1 / np.log2(pos + 2)
    dcg = np.bincount(fid, y[order] * d, len(M))
    k = np.bincount(fid, y, len(M))
    idcg = np.bincount(fid, (pos < k[fid]) * d, len(M))
    out = np.ones(len(M)); m = idcg > 0; out[m] = dcg[m] / idcg[m]
    return out

def se(d): return d.std(ddof=1) / np.sqrt(len(d))
def fmt(d, w=None):
    if w is None: return f"{d.mean():+.5f}+-{se(d):.5f}"
    m, s = pmix.wmean_se(d, w); return f"{m:+.5f}+-{s:.5f}"
def line(d): return (f"val {fmt(d)} | pub {fmt(d, W['public'])} | priv {fmt(d, W['private'])} | "
                     f"M<50 {fmt(d[~big])} | M>=50 {fmt(d[big])}")

_c = {}
def mem(m, suf):
    k = (m, suf)
    if k not in _c: _c[k] = zf(np.load(P / f"nn_{m}_{suf}_val{'_oof' if suf != 'trainonly' else ''}.npy"))
    return _c[k]
def pool(ms, suf): return np.mean([mem(m, suf) for m in ms], 0)

S4, S8 = range(951, 955), range(951, 959)
FD16 = [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
W16 = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
FDP4 = [f"abl_fd10_pop1kd05w192_s{s}" for s in S4]
PD4w = [f"abl_e10_pop1kd05w192_s{s}" for s in S4]
FT, TO = "n2cpv3s3", "trainonly"
wM = np.where(MROW >= 50, 0.2, 0.4)

def final(Fb, Fp, Tb, Tp, w=wM):  # as build_r4.py: pools re-z'd, branch = (1-w) z(base) + w z(pop), final = 0.5 z(FT) + 0.5 z(TO)
    return 0.5 * zf((1 - w) * zf(Fb) + w * zf(Fp)) + 0.5 * zf((1 - w) * zf(Tb) + w * zf(Tp))

