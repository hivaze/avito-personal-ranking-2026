"""B3 pre-screen, step (i): per-column conditional logit on TRAIN exploration feeds (xm; train labels only; no val / holdout label).
For each new column c: nats gain of [z_oof, c~] over [z_oof] per feed, c~ = within-feed z of the column (NaN -> feed mean), fitted and read on
the same feeds (2 parameters; a random 40,000-feed subset of xm). z_oof = cross-fitted teacher (cbx.nn_input: the only out-of-sample NN score on train feeds).
Also the same statistic for a few of the 177 features (scale reference).  usage: python -u b3_uni.py a b [c]"""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB"); sys.path.insert(0, "<repo>/nn_x/w12/cb")
from fb_common import *
from nats import clogit

tr = cbx.load("xm")
NFEED = 40000
keep = np.zeros(tr.n_feeds, bool); keep[np.random.default_rng(3).choice(tr.n_feeds, NFEED, replace=False)] = True
rm = np.repeat(keep, tr.M); M = tr.M[keep]; off = np.r_[0, np.cumsum(M)[:-1]].astype(np.int64); gid = np.repeat(np.arange(NFEED), M)
y = np.asarray(tr.y, np.float64)[rm]
zo = cbx.feed_z(off, M, np.ascontiguousarray(np.asarray(cbx.nn_input(tr))[rm], np.float32)).astype(np.float64)
_, _, l0 = clogit(zo[:, None], y, off, M)
nf = int((l0 != 0).sum())
print(f"xm feeds used {NFEED} (random subset), with a positive {nf}; base loss {l0.sum() / nf:.5f}")


def one(col):
    x = np.asarray(col, np.float32)[rm]; nanm = ~np.isfinite(x)
    cnt = np.add.reduceat((~nanm).astype(np.float64), off); sm = np.add.reduceat(np.where(nanm, 0, x).astype(np.float64), off)
    mean = np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0)
    x[nanm] = mean[gid][nanm]
    z = cbx.feed_z(off, M, np.ascontiguousarray(x)).astype(np.float64)
    varies = np.add.reduceat(np.abs(z), off) > 0
    if not varies.any():
        return dict(nats=0.0, se=0.0, beta=0.0, bz=0.0, feeds_var=0.0, nats_cov=0.0)   # feed-constant column: no ranking information alone
    b, s, l1 = clogit(np.stack([zo, z], 1), y, off, M); d = l0 - l1
    return dict(nats=float(d.sum() / nf), se=float(np.std(d) / np.sqrt(len(d)) * len(d) / nf), beta=float(b[1]), bz=float(b[1] / s[1]), feeds_var=float(varies.mean()),
                nats_cov=float(d[varies].sum() / max(int((varies & (l0 != 0)).sum()), 1)))


res = {}
for ref in ("tte_uc_ctr", "tte_uc_imp", "tte_ui_imp", "tte_it_coec", "tte_c_ctr", "clu_pmi1024_mean", "f3", "cos_u"):
    r = one(tr.X[:, tr.names.index(ref)]); res["ref:" + ref] = r
    print(f"ref {ref:22s} nats {r['nats']:+.5f} +- {r['se']:.5f}  beta z {r['bz']:+6.1f}  feeds where it varies {r['feeds_var']:.3f}  nats on those {r['nats_cov']:+.5f}", flush=True)
for blk in sys.argv[1:]:
    names = json.load(open(FEAT / f"{blk}_names.json")); A = np.load(FEAT / f"{blk}_xm.npy", mmap_mode="r")
    for j, n in enumerate(names):
        r = one(A[:, j]); res[n] = r
        print(f"{blk} {n:22s} nats {r['nats']:+.5f} +- {r['se']:.5f}  beta z {r['bz']:+6.1f}  feeds where it varies {r['feeds_var']:.3f}  nats on those {r['nats_cov']:+.5f}", flush=True)
json.dump(res, open(HERE / f"out/b3_uni_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
