"""EVALUATION ONLY. Ceiling of the LABEL-FREE catalogue-density signal: blocks c + d density columns, read over z_W16, [z_W16, z_B0] and [z_W16, z_B3].
(derived from b3_witness_split.py)  Split of the linear information of blocks a + b + c into LABEL-FREE density columns and the rest (same method as b3_witness_info.py:
coefficients fitted on 40,000 TRAIN feeds over z_oof, the fixed composite read on the witnesses over z_W16 and over [z_W16, z_B0]).
--- docstring of b3_witness_info.py ---
EVALUATION ONLY (no model, no selection): how much pool-independent information do the hand-made columns carry on the two witnesses?
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


tr = cbx.load("xm"); NF = 40000
keep = np.zeros(tr.n_feeds, bool); keep[np.random.default_rng(3).choice(tr.n_feeds, NF, replace=False)] = True
rm = np.repeat(keep, tr.M); Mt = tr.M[keep]; ot = np.r_[0, np.cumsum(Mt)[:-1]].astype(np.int64); gt = np.repeat(np.arange(NF), Mt)
yt = np.asarray(tr.y, np.float64)[rm]; zo = cbx.feed_z(ot, Mt, np.ascontiguousarray(np.asarray(cbx.nn_input(tr))[rm], np.float32)).astype(np.float64)
names = {b: json.load(open(FEAT / f"{b}_names.json")) for b in ("c", "d")}
allc = [(b, j, n) for b in "cd" for j, n in enumerate(names[b])]
DENS = {"kn_cos1", "kn_cos20", "kn_cos2", "kn_cos5", "kn_cos10", "dup_n99", "dup_n97", "dup_n95", "dup_n90"}
groups = {"density, block c (kn_cos1, kn_cos20)": [c for c in allc if c[2] in ("kn_cos1", "kn_cos20")],
          "density, blocks c + d (9 label-free columns)": [c for c in allc if c[2] in DENS],
          "near-duplicate click statistics (dup_coec97, dup_coec95)": [c for c in allc if c[2] in ("dup_coec97", "dup_coec95")],
          "all 17 columns of c + d": allc}
blk = {b: np.load(FEAT / f"{b}_xm.npy", mmap_mode="r") for b in "cd"}
Zt = {c: zcol(blk[c[0]][:, c[1]][rm], ot, Mt, gt) for c in allc}
beta = {g: fit(np.concatenate([zo[:, None], np.stack([Zt[c] for c in cs], 1)], 1), yt, ot, Mt)[1:] for g, cs in groups.items()}
del Zt
res = {}
for sp, b0f in (("val", B0V), ("hold", B0H)):
    y, off, M = m.y[sp].astype(np.float64), m.off[sp], m.M[sp]; gid = np.repeat(np.arange(len(M)), M)
    zw = m.zw[sp].astype(np.float64); zb = m.z(sp, np.load(b0f)).astype(np.float64); z3 = m.z(sp, np.load(PREDS / f"b3_abc_s0_{sp}.npy")).astype(np.float64)
    _, _, l_w = clogit(zw[:, None], y, off, M); nf = int((l_w != 0).sum()); _, _, l_wb = clogit(np.stack([zw, zb], 1), y, off, M); _, _, l_w3 = clogit(np.stack([zw, z3], 1), y, off, M)
    A = {b: np.load(FEAT / f"{b}_{sp}.npy") for b in "cd"}
    Zs = {c: zcol(A[c[0]][:, c[1]], off, M, gid) for c in allc}
    res[sp] = {}
    f_ = lambda d_: (float(d_.sum() / nf), float(np.std(d_) / np.sqrt(len(d_)) * len(d_) / nf))
    for g, cs in groups.items():
        comp = np.stack([Zs[c] for c in cs], 1) @ beta[g]
        zc = cbx.feed_z(off, M, np.ascontiguousarray(comp, np.float32)).astype(np.float64)
        _, _, l1 = clogit(np.stack([zw, zc], 1), y, off, M); b2, s2, l2 = clogit(np.stack([zw, zb, zc], 1), y, off, M); b3, s3, l3 = clogit(np.stack([zw, z3, zc], 1), y, off, M)
        res[sp][g] = dict(n=len(cs), over_w=f_(l_w - l1), over_w_b0=f_(l_wb - l2), over_w_b3=f_(l_w3 - l3))
        print(f"[{sp}] {g:58s}: over z_W16 {f_(l_w - l1)[0]:+.5f} +- {f_(l_w - l1)[1]:.5f} | over [z_W16, z_B0] {f_(l_wb - l2)[0]:+.5f} +- {f_(l_wb - l2)[1]:.5f} (beta z {b2[2] / s2[2]:+.1f}) | "
              f"over [z_W16, z_B3] {f_(l_w3 - l3)[0]:+.5f} +- {f_(l_w3 - l3)[1]:.5f} (beta z {b3[2] / s3[2]:+.1f})", flush=True)
json.dump(res, open(HERE / "out/b3_witness_density.json", "w"), indent=1)
print("WITNESS DENSITY DONE")
