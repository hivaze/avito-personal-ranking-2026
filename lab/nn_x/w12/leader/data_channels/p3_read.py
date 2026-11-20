"""P3 read: validation labels of OTHER users as a content near-neighbour prior (cross-fitted by user halves).

For halving k, a row q of half X reads the labels of rows r of the other half (other users) with
  as-of   : ts(r) < ts(q)                         (earlier labels only; the direction a test row has)
  forward : Moscow day(r) < day(q), i.e. Tuesday labels -> Thursday rows only (1.5-2.5 days old, nearer the test gap)
  any     : any time (includes later labels; optimistic)
and the same category, cos >= tau (pairs from p3_build.py). Columns per tau in {exact item, 0.98, 0.95, 0.90}:
  shrunk click rate (p + a b) / (n + a) - b (a = 20, b = base rate), log1p(p), log1p(n); plus the highest cosine to a
  clicked source row and to a shown-not-clicked one. Read: nested clogit (dc_nest), placebo = labels of the source rows
  permuted within their feed (keeps every count, destroys which row was clicked).
"""
import sys, json
sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import *  # noqa
from dc_nest import Nested, fmt

v = Split("val", hist=False); s = v.best(); y = v.y
Z = np.load(f"{OUT}/out/p3_pairs_val.npz"); q = Z["q"].astype(np.int64); r = Z["r"].astype(np.int64); c = Z["c"].astype(np.float32)
same_item = v.item[q] == v.item[r]
log("pairs", len(q), "exact-item pairs", same_item.sum())
N = Nested(v, s)
base = y.mean(); A_ = 20.0
rng = np.random.default_rng(5)
yperm = y[np.lexsort((rng.random(v.N), v.fid))]       # labels permuted within each feed (placebo)
tsq, tsr = v.ts[v.fid[q]], v.ts[v.fid[r]]
dq, dr = v.day[v.fid[q]], v.day[v.fid[r]]


def cols_for(k, mode, yy):
    h = N.halves[k]
    hq, hr = h[v.fid[q]], h[v.fid[r]]
    m = hq != hr
    if mode == "asof":
        m &= tsr < tsq
    elif mode == "forward":
        m &= dr < dq
    out = []
    for tag, mm in (("ex", m & same_item), ("98", m & (c >= 0.98)), ("95", m & (c >= 0.95)), ("90", m)):
        n = np.bincount(q[mm], minlength=v.N).astype(np.float64)
        p = np.bincount(q[mm], weights=yy[r[mm]], minlength=v.N)
        out += [(p + A_ * base) / (n + A_) - base, np.log1p(p), np.log1p(n)]
    mp = m & (yy[r] > 0)
    mxp = np.zeros(v.N); np.maximum.at(mxp, q[mp], c[mp])
    out.append(mxp)
    return out


res = []
for mode in ("asof", "forward", "any"):
    for lab, yy in (("real", y), ("placebo", yperm)):
        name = f"P3 val-label kNN {mode} [{lab}]"
        r_ = N.read(name, (lambda k, mode=mode, yy=yy: cols_for(k, mode, yy)))
        print(fmt(r_), flush=True); res.append(r_)
# coverage of the forward form on Thursday rows (share of rows with >= 1 source row at each tau)
h = N.halves[0]; m = (h[v.fid[q]] != h[v.fid[r]]) & (dr < dq)
for tag, mm in (("exact", m & same_item), (">=0.98", m & (c >= 0.98)), (">=0.95", m & (c >= 0.95)), (">=0.90", m)):
    has = np.bincount(q[mm], minlength=v.N) > 0
    thu = v.day[v.fid] == 20720
    print(f"forward coverage on Thursday rows, {tag}: {has[thu].mean():.4f}; positive rate covered {y[has & thu].mean():.4f} vs uncovered {y[~has & thu].mean():.4f}")
json.dump(res, open(f"{OUT}/out/p3_read.json", "w"), indent=1, default=float)
log("done")
