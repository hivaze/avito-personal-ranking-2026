"""Lead's notice (2), linear form: do the hand-made statistics carry more on val when the TRAIN rows are built with the test rows' lag?
Coefficients of a linear composite are fitted on 40,000 xm feeds over z_oof, once from the FRESH train columns (feat/<blk>_xm.npy: a row reads everything up to
the day before / the second before) and once from the LAGGED train columns (feat/lag_<blk>_xm.npy: sources 6 to 9 days old, FB_LAG=1 builds). Each fixed composite is
read over [z_W16, z_B0] on val (columns with val's natural lag of 2 to 5 days), on the holdout with FRESH columns and on the holdout with LAGGED columns
(a stale-matched in-period witness). Evaluation only; no arm was trained on lagged columns.   usage: python -u lag_read.py"""
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
names = {b: json.load(open(FEAT / f"{b}_names.json")) for b in ("a", "b", "c")}
SKIP = {"ua_ns", "ua_npos", "ua_dt_last"}
allc = [(b, j, n) for b in "abc" for j, n in enumerate(names[b]) if n not in SKIP]
FREE = {"kn_cos1", "kn_cos20"}
EXPO = FREE | {"kn_ex20"} | {n for _, _, n in allc if n.endswith("_imp") or n.endswith("_imprw") or n.endswith("_share") or n in ("ue_k05_all", "ue_k20_all", "ue_top5_cos", "ua_item_dt")}
groups = {"all 67 columns": allc, "click statistics (51)": [c for c in allc if c[2] not in EXPO], "click-free columns (16)": [c for c in allc if c[2] in EXPO],
          "user block a, click statistics (23)": [c for c in allc if c[0] == "a" and c[2] not in EXPO]}
beta = {}; cover = {}
for var, pre in (("fresh", ""), ("lag", "lag_")):
    blk = {b: np.load(FEAT / f"{pre}{b}_xm.npy", mmap_mode="r") for b in "abc"}
    Zt = {c: zcol(blk[c[0]][:, c[1]][rm], ot, Mt, gt) for c in allc}
    cover[var] = {b: float(np.mean([np.add.reduceat(np.abs(Zt[c]), ot) > 0 for c in allc if c[0] == b and c[2] in ("ua_cat_coec", "tr_cat_l20m", "kn_coec20")][0])) for b in "abc"}
    for g, cs in groups.items():
        beta[(var, g)] = fit(np.concatenate([zo[:, None], np.stack([Zt[c] for c in cs], 1)], 1), yt, ot, Mt)[1:]
    print(f"[train {var}] share of the 40,000 xm feeds where the block has a within-feed-varying click statistic: {cover[var]}", flush=True)
    del Zt
res = {"cover": cover}
for sp, b0f, wvars in (("val", B0V, (("natural lag 2-5 d", ""),)), ("hold", B0H, (("FRESH columns", ""), ("LAGGED columns 6-9 d", "lag_")))):
    y, off, M = m.y[sp].astype(np.float64), m.off[sp], m.M[sp]; gid = np.repeat(np.arange(len(M)), M)
    zw = m.zw[sp].astype(np.float64); zb = m.z(sp, np.load(b0f)).astype(np.float64)
    _, _, l_w = clogit(zw[:, None], y, off, M); nf = int((l_w != 0).sum()); _, _, l_wb = clogit(np.stack([zw, zb], 1), y, off, M)
    f_ = lambda d_: (float(d_.sum() / nf), float(np.std(d_) / np.sqrt(len(d_)) * len(d_) / nf))
    for wname, wpre in wvars:
        A = {b: np.load(FEAT / f"{wpre}{b}_{sp}.npy") for b in "abc"}
        Zs = {c: zcol(A[c[0]][:, c[1]], off, M, gid) for c in allc}
        for g, cs in groups.items():
            Zg = np.stack([Zs[c] for c in cs], 1); line = []
            for var in ("fresh", "lag"):
                zc = cbx.feed_z(off, M, np.ascontiguousarray(Zg @ beta[(var, g)], np.float32)).astype(np.float64)
                b2, s2, l2 = clogit(np.stack([zw, zb, zc], 1), y, off, M); r = f_(l_wb - l2)
                res[f"{sp}|{wname}|{g}|{var}"] = dict(nats=r[0], se=r[1], bz=float(b2[2] / s2[2]))
                line.append(f"coefficients from {var.upper():5s} train rows {r[0]:+.5f} +- {r[1]:.5f} (beta z {b2[2] / s2[2]:+.1f})")
            print(f"[{sp}, {wname}] {g:36s} over [z_W16, z_B0]: " + " | ".join(line), flush=True)
json.dump(res, open(HERE / "out/lag_read.json", "w"), indent=1)
print("LAG READ DONE")
