"""kd_reprice recheck helpers (CPU only, read-only on preds). Mirrors nn_x/w12/kd/pool_read.py loaders."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(k, "12")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, "<repo>/src")
from pathlib import Path
import numpy as np
import numba as nb
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path("<repo>")
W = ROOT / "nn_x/w5/preds"; AB = ROOT / "nn_x/abl/preds"; P = ROOT / "preds"; R6 = ROOT / "nn_x/r6/preds"
OUT = ROOT / "nn_x/w12/recheck/kd_reprice/out"

fv, Mv, ov = feed_meta("val"); ov = ov.astype(np.int64)
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
Mh, oh = np.load(W / "holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(W / "holdout/holdout_y.npy").astype(np.float32)


@nb.njit(parallel=True, cache=True)
def _ndcg(s, y, off, M):
    out = np.ones(len(M))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        yy = y[o:o + m]; ss = s[o:o + m]
        npos = 0
        for k in range(m):
            if yy[k] > 0:
                npos += 1
        if npos == 0:
            continue
        dcg = 0.0
        for i in range(m):
            if yy[i] > 0:
                r = 0
                for j in range(m):
                    if ss[j] > ss[i] or (ss[j] == ss[i] and (yy[j] < yy[i] or (yy[j] == yy[i] and j < i))):
                        r += 1
                dcg += 1.0 / np.log2(r + 2.0)
        idcg = 0.0
        for k in range(npos):
            idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


def ev(s): return _ndcg(np.ascontiguousarray(s, np.float32), yv, ov, Mv)
def eh(s): return _ndcg(np.ascontiguousarray(s, np.float32), yh, oh, Mh)
def zv(x): return feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
def zh(x): return feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]
def se(d): return d.std() / np.sqrt(len(d))
def f(d): return f"{d.mean():+.5f}+-{se(d):.5f}"


def _dir(v):
    if v.startswith("abl_"): return AB
    if v.startswith("w5_"): return W
    if v.startswith("r6_"): return R6
    return P


_cache = {}
def ld(v, suf=""):
    """(z val TO, z holdout) of a variant; None where a file is missing."""
    k = (v, suf)
    if k not in _cache:
        D = _dir(v)
        a = D / f"nn_{v}{suf}_trainonly_val.npy"; h = D / f"holdout/nn_{v}{suf}_holdout.npy"
        _cache[k] = (zv(np.load(a)) if a.exists() else None, zh(np.load(h)) if h.exists() else None)
    return _cache[k]


def ldft(v, tag="n2cpv3s3"):
    for D in (P, W, AB):
        a = D / f"nn_{v}_{tag}_val_oof.npy"
        if a.exists(): return zv(np.load(a))
    return None


def fam(arm, seeds=(951, 952, 953, 954), pre="w5_e10_"): return [f"{pre}{arm}_s{s}" for s in seeds]
def pool(vs):
    m = [ld(v) for v in vs]
    return np.mean([x[0] for x in m], 0), (np.mean([x[1] for x in m], 0) if all(x[1] is not None for x in m) else None)
