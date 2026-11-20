"""EVALUATION ONLY (no model, no selection): how much pool-independent information do the hand-made columns carry on the two witnesses?
Same statistic as nats.py (conditional logit of the label inside the feed), with the column(s) in the place of the tree score:
  per column c   : nats([z_W16, z_c]) - nats([z_W16])                  (2 parameters fitted on the witness, as nats.py does)
  per block      : a LINEAR composite of the block's columns whose coefficients are fitted on TRAIN feeds (40,000 xm feeds, conditional logit over
                   z_oof); the fixed composite is then read like a model score: nats([z_W16, comp]) - nats([z_W16]) and the same over [z_W16, z_B0].
                   (A first version cross-fitted 37-68 coefficients on halves of the witness: its estimation penalty p / 2n = 0.003 swamps the read.)
Nothing here is used to choose columns or arms (B3's composition was fixed before this was run).   usage: python -u b3_witness_info.py"""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB"); sys.path.insert(0, "<repo>/nn_x/w12/cb")
from fb_common import *
import nats
from nats import clogit

m = nats._meta()


def fit(X, y, off, M, iters=30):
    """conditional logit, Newton, matrix form. Returns beta."""
    n, p = X.shape; fid = np.repeat(np.arange(len(M)), M)
    npos = np.add.reduceat(y, off); keep = (npos > 0)[fid]; t = np.where(keep, y / np.maximum(npos[fid], 1), 0.0); b = np.zeros(p)
    for _ in range(iters):
        s = X @ b; mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[fid]); pr = np.where(keep, e / np.add.reduceat(e, off)[fid], 0.0)
        g = X.T @ (pr - t)
        mu = np.add.reduceat(X * pr[:, None], off, axis=0)
        H = (X * pr[:, None]).T @ X - mu.T @ mu + 1e-6 * np.eye(p)
        step = np.linalg.solve(H, g); b = b - step
        if np.abs(step).max() < 1e-8:
            break
    return b


def loss(X, b, y, off, M):
    fid = np.repeat(np.arange(len(M)), M); npos = np.add.reduceat(y, off); t = np.where((npos > 0)[fid], y / np.maximum(npos[fid], 1), 0.0)
    s = X @ b; mx = np.maximum.reduceat(s, off); lse = mx + np.log(np.add.reduceat(np.exp(s - mx[fid]), off))
    return -(np.add.reduceat(t * s, off) - np.where(npos > 0, lse, 0.0))


def zcol(x, off, M, gid):
    x = np.asarray(x, np.float32).copy(); nanm = ~np.isfinite(x)
    cnt = np.add.reduceat((~nanm).astype(np.float64), off); sm = np.add.reduceat(np.where(nanm, 0, x).astype(np.float64), off)
    x[nanm] = (np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0))[gid][nanm]
    sd = np.sqrt(np.maximum(np.add.reduceat(x.astype(np.float64) ** 2, off) / M - (np.add.reduceat(x.astype(np.float64), off) / M) ** 2, 0))
    z = cbx.feed_z(off, M, np.ascontiguousarray(x)).astype(np.float64)
    mean_abs = np.abs(np.add.reduceat(x.astype(np.float64), off) / M)
    z[(sd < 1e-5 * (mean_abs + 1.0))[gid]] = 0.0           # constant inside the feed (up to float32 noise): no ranking information
    return np.clip(z, -8, 8)


# ---- train-side linear composites: coefficients fitted on TRAIN feeds (xm, random 40,000) over z_oof; no witness label used for them ----
tr = cbx.load("xm"); NF = 40000
keep = np.zeros(tr.n_feeds, bool); keep[np.random.default_rng(3).choice(tr.n_feeds, NF, replace=False)] = True
rm = np.repeat(keep, tr.M); Mt = tr.M[keep]; ot = np.r_[0, np.cumsum(Mt)[:-1]].astype(np.int64); gt = np.repeat(np.arange(NF), Mt)
yt = np.asarray(tr.y, np.float64)[rm]; zo = cbx.feed_z(ot, Mt, np.ascontiguousarray(np.asarray(cbx.nn_input(tr))[rm], np.float32)).astype(np.float64)
names = {b: json.load(open(FEAT / f"{b}_names.json")) for b in ("a", "b", "c")}
SKIP = {"ua_ns", "ua_npos", "ua_dt_last"}                 # feed-constant columns: no ranking information in a linear score
beta = {}
for blk in ("a", "b", "c", "abc"):
    Z = np.stack([zcol(np.load(FEAT / f"{b}_xm.npy", mmap_mode="r")[:, j][rm], ot, Mt, gt) for b in blk for j, n in enumerate(names[b]) if n not in SKIP], 1)
    bt = fit(np.concatenate([zo[:, None], Z], 1), yt, ot, Mt)
    l0 = loss(zo[:, None], fit(zo[:, None], yt, ot, Mt), yt, ot, Mt); l1 = loss(np.concatenate([zo[:, None], Z], 1), bt, yt, ot, Mt)
    beta[blk] = bt[1:]
    print(f"[train] block {blk:3s}: {Z.shape[1]} columns, in-sample nats over z_oof on {NF} xm feeds {np.mean(l0 - l1):+.5f} +- {np.std(l0 - l1) / np.sqrt(NF):.5f}", flush=True)
del Z

res = {}
for sp, b0f in (("val", B0V), ("hold", B0H)):
    y, off, M = m.y[sp].astype(np.float64), m.off[sp], m.M[sp]; gid = np.repeat(np.arange(len(M)), M); nfeed = len(M)
    zw = m.zw[sp].astype(np.float64); zb = m.z(sp, np.load(b0f)).astype(np.float64)
    _, _, l_w = clogit(zw[:, None], y, off, M); nf = int((l_w != 0).sum())
    _, _, l_wb = clogit(np.stack([zw, zb], 1), y, off, M)
    cols = {}
    for blk in ("a", "b", "c"):
        A = np.load(FEAT / f"{blk}_{sp}.npy")
        for j, n in enumerate(names[blk]):
            if n not in SKIP:
                cols[(blk, n)] = zcol(A[:, j], off, M, gid)
    res[sp] = {"per_column": {}, "composite": {}}
    for (blk, n), z in cols.items():
        if z.std() < 1e-6:
            continue
        try:
            b, s_, l1 = clogit(np.stack([zw, z], 1), y, off, M); d = l_w - l1
        except np.linalg.LinAlgError:
            print(f"   [{sp}] {n}: singular (no usable within-feed variation), skipped"); continue
        res[sp]["per_column"][n] = dict(block=blk, nats=float(d.sum() / nf), se=float(np.std(d) / np.sqrt(len(d)) * len(d) / nf), bz=float(b[1] / s_[1]))
    top = sorted(res[sp]["per_column"].items(), key=lambda kv: -kv[1]["nats"])[:10]
    print(f"\n[{sp}] feeds {nfeed}; per-column nats over z_W16 (top 10 of {len(res[sp]['per_column'])}):")
    for n, r in top:
        print(f"   {r['block']} {n:20s} {r['nats']:+.5f} +- {r['se']:.5f} (beta z {r['bz']:+.1f})")
    for blk in ("a", "b", "c", "abc"):
        Z = np.stack([z for (b_, n), z in cols.items() if b_ in blk], 1); comp = Z @ beta[blk]
        zc = cbx.feed_z(off, M, np.ascontiguousarray(comp, np.float32)).astype(np.float64)
        b, s_, l1 = clogit(np.stack([zw, zc], 1), y, off, M); d = l_w - l1
        b2, s2, l2 = clogit(np.stack([zw, zb, zc], 1), y, off, M); d2 = l_wb - l2
        f_ = lambda d_: (float(d_.sum() / nf), float(np.std(d_) / np.sqrt(len(d_)) * len(d_) / nf))
        res[sp]["composite"][blk] = dict(over_w=f_(d), bz=float(b[1] / s_[1]), over_w_b0=f_(d2), bz2=float(b2[2] / s2[2]))
        print(f"[{sp}] block {blk:3s} train-fitted linear composite: nats over z_W16 {f_(d)[0]:+.5f} +- {f_(d)[1]:.5f} (beta z {b[1] / s_[1]:+.1f}) | "
              f"over [z_W16, z_B0] {f_(d2)[0]:+.5f} +- {f_(d2)[1]:.5f} (beta z {b2[2] / s2[2]:+.1f})", flush=True)
    d = l_w - l_wb
    print(f"[{sp}] B0 (ts_lr020_s0) over z_W16: {d.sum() / nf:+.5f} +- {np.std(d) / np.sqrt(len(d)) * len(d) / nf:.5f}")
json.dump(res, open(HERE / "out/b3_witness_info.json", "w"), indent=1)
print("WITNESS INFO DONE")
