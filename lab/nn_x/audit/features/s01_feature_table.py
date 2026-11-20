"""Per-feature adversarial-validation + label-relation + private-support table (CPU, label-free where stated).

For every feature file on disk (data/feats, data/feats_new; val+test required, train optional) plus the derived
`has:pre2_fe`, computes
  A. candidate-level univariate adversarial AUC (NaN -> below min) for the pairs
     trRL~val50, trProd~val50, trRL~priv50, trProd~trRL, val~pub, val~priv, pub~priv   (250K rows per side)
  B. feed-level AUC of 4 per-feed aggregates (nanmean, nan share, zero share, nanmax) for trRL~val50, val~pub,
     val~priv, pub~priv
  C. label relation: mean within-feed AUC (feeds with both classes) and the 1-D conditional-logit slope on the
     within-feed z, on trProd (p_random<0.05, M>=50), trRL (the exact 30K holdout), trSmall (M<50), valAll, val50,
     valTue, valThu
  D. private support: share of public/private rows outside val [min,max] and [q0.1%,q99.9%], overall and on
     LUO-cold rows; mean within-feed z on LUO-cold rows per population.
Outputs: out/feature_table.csv (one row per feature), out/s01.log.
"""
import os, sys, time, json
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, polars as pl, numba as nb
from scipy.stats import rankdata

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
FEATS_NEW = ROOT / "data/feats_new"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
BOUND = 1790497920
T0 = time.time()
LOG = open(OUT / "s01.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def feat_file(split, name):
    for d in (FEATS, FEATS_NEW):
        p = d / split / f"{name}.npy"
        if p.exists():
            return p
    return None


def meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    return f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)


# ---------------------------------------------------------------- row sets
fv, Mv, offv = meta("val")
ft, Mt, offt = meta("test")
ftr, Mtr, offtr = meta("train")
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
ytr_mm = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
tsv = fv["timestamp"].to_numpy().astype(np.int64)
dayv = (tsv + 10800) // 86400
fidv = np.repeat(np.arange(len(Mv)), Mv)
fidt = np.repeat(np.arange(len(Mt)), Mt)
pub_feed = ft["timestamp"].to_numpy() < BOUND
pub_row = pub_feed[fidt]
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(len(ftr))).all()
p_rand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
p_rand[Mtr < 50] = 0.0
cand = np.where((p_rand > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
hm = np.load(ROOT / "preds/holdout/holdout_meta.npy")
assert (Mtr[hold] == hm[0]).all(), "holdout reconstruction mismatch"
rng = np.random.default_rng(2026)
prod_pool = np.where((p_rand < 0.05) & (Mtr >= 50) & ~np.isin(np.arange(len(ftr)), hold))[0]
prod = np.sort(rng.choice(prod_pool, 12000, replace=False))
small_pool = np.where(Mtr < 50)[0]
small = np.sort(rng.choice(small_pool, 8000, replace=False))


def rows_of(off, M, fidx):
    return np.concatenate([np.arange(o, o + m) for o, m in zip(off[fidx], M[fidx])])


def local_off(M):
    return np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)


TR = {}
for nm, fx in (("trRL", hold), ("trProd", prod), ("trSmall", small)):
    r = rows_of(offtr, Mtr, fx)
    TR[nm] = dict(feeds=fx, rows=r, M=Mtr[fx], off=local_off(Mtr[fx]), y=np.asarray(ytr_mm[r], np.float64))
    log(f"{nm}: {len(fx)} feeds, {len(r)} rows, pos rate {TR[nm]['y'].mean():.4f}")
tr_all_rows = np.concatenate([TR[k]["rows"] for k in ("trRL", "trProd", "trSmall")])
tr_order = np.argsort(tr_all_rows, kind="stable")
tr_sorted = tr_all_rows[tr_order]
tr_slices = {}
i0 = 0
for k in ("trRL", "trProd", "trSmall"):
    n = len(TR[k]["rows"]); tr_slices[k] = (i0, i0 + n); i0 += n


def load_train(name):
    p = feat_file("train", name)
    if p is None:
        return None
    x = np.load(p, mmap_mode="r")
    v = np.empty(len(tr_sorted), np.float32)
    v[tr_order] = np.asarray(x[tr_sorted], np.float32)  # sorted gather, then unpermute
    return {k: v[a:b] for k, (a, b) in tr_slices.items()}


# val populations (feed index sets)
val50 = np.where(Mv >= 50)[0]
valTue = np.where(dayv <= dayv.min() + 1)[0]
valThu = np.where(dayv >= dayv.min() + 2)[0]
VALPOP = {"valAll": np.arange(len(Mv)), "val50": val50, "valTue": valTue, "valThu": valThu}
VALROWS = {k: rows_of(offv, Mv, v) for k, v in VALPOP.items()}
priv50_feeds = np.where(~pub_feed & (Mt >= 50))[0]
TESTROWS = {"pub": np.where(pub_row)[0], "priv": np.where(~pub_row)[0], "priv50": rows_of(offt, Mt, priv50_feeds)}

# cold flags (LUO-cold)
luo_v = np.load(feat_file("val", "it_hist_all_luo")); luo_t = np.load(feat_file("test", "it_hist_all_luo"))
cold_v, cold_t = luo_v == 0, luo_t == 0
luo_tr = load_train("it_hist_all_luo"); cold_tr = {k: v == 0 for k, v in luo_tr.items()}
log(f"cold share: val {cold_v.mean():.3f} pub {cold_t[pub_row].mean():.3f} priv {cold_t[~pub_row].mean():.3f} "
    f"trRL {cold_tr['trRL'].mean():.3f} trProd {cold_tr['trProd'].mean():.3f}")

SUB = 250_000
SEED = 7


def sub(x, n=SUB, seed=SEED):
    if len(x) <= n:
        return x
    return x[np.random.default_rng(seed).choice(len(x), n, replace=False)]


def auc2(a, b):
    """AUC of 'b' vs 'a' (P(b > a)); NaN -> -inf (below every value); returns |AUC-0.5| signed as AUC."""
    a = np.where(np.isnan(a), -np.inf, a); b = np.where(np.isnan(b), -np.inf, b)
    x = np.concatenate([a, b]); r = rankdata(x)
    nb_, na_ = len(b), len(a)
    return float((r[na_:].sum() - nb_ * (nb_ + 1) / 2) / (na_ * nb_))


@nb.njit(cache=True)
def feed_agg(x, off, M):
    nF = len(M)
    mean = np.full(nF, np.nan); nan = np.zeros(nF); zero = np.zeros(nF); mx = np.full(nF, np.nan)
    for i in range(nF):
        o = off[i]; m = M[i]; s = 0.0; c = 0; z = 0; best = -1e30
        for k in range(m):
            v = x[o + k]
            if v != v:
                continue
            s += v; c += 1
            if v == 0.0:
                z += 1
            if v > best:
                best = v
        nan[i] = 1.0 - c / m
        if c > 0:
            mean[i] = s / c; zero[i] = z / c; mx[i] = best
    return mean, nan, zero, mx


@nb.njit(cache=True)
def feed_auc(x, y, off, M):
    tot = 0.0; n = 0
    for i in range(len(M)):
        o = off[i]; m = M[i]; npos = 0
        for k in range(m):
            if y[o + k] > 0:
                npos += 1
        if npos == 0 or npos == m:
            continue
        v = np.empty(m)
        for k in range(m):
            t = x[o + k]
            v[k] = -1e30 if t != t else t
        order = np.argsort(v)
        ranks = np.empty(m); i0 = 0
        while i0 < m:
            j = i0
            while j + 1 < m and v[order[j + 1]] == v[order[i0]]:
                j += 1
            avg = 0.5 * (i0 + j) + 1.0
            for k in range(i0, j + 1):
                ranks[order[k]] = avg
            i0 = j + 1
        s = 0.0
        for k in range(m):
            if y[o + k] > 0:
                s += ranks[k]
        tot += (s - npos * (npos + 1) / 2) / (npos * (m - npos)); n += 1
    return tot / max(n, 1), n


@nb.njit(cache=True)
def feed_z_nanmin(x, off, M):
    """within-feed z with NaN set to the feed minimum (below the present values by one sd-step is not needed for a
    slope: NaN -> min keeps monotone order)"""
    out = np.zeros(len(x))
    for i in range(len(M)):
        o = off[i]; m = M[i]; mn = 1e30; c = 0
        for k in range(m):
            v = x[o + k]
            if v == v:
                c += 1
                if v < mn:
                    mn = v
        if c == 0:
            continue
        s = 0.0; s2 = 0.0
        for k in range(m):
            v = x[o + k]
            if v != v:
                v = mn
            s += v; s2 += v * v
        mu = s / m; sd = np.sqrt(max(s2 / m - mu * mu, 0.0))
        if sd < 1e-9:
            continue
        for k in range(m):
            v = x[o + k]
            if v != v:
                v = mn
            out[o + k] = (v - mu) / sd
    return out


@nb.njit(cache=True)
def clogit_slope(z, y, off, M, iters=12):
    """1-D conditional logit s = theta * z, listwise softmax per feed, Newton steps; returns theta, LL gain/feed."""
    th = 0.0
    ll0 = 0.0
    for it in range(iters + 1):
        g = 0.0; h = 0.0; ll = 0.0
        for i in range(len(M)):
            o = off[i]; m = M[i]; npos = 0.0
            for k in range(m):
                npos += y[o + k]
            if npos == 0:
                continue
            mx = -1e30
            for k in range(m):
                v = th * z[o + k]
                if v > mx:
                    mx = v
            Z = 0.0; ez = 0.0; ezz = 0.0
            for k in range(m):
                e = np.exp(th * z[o + k] - mx)
                Z += e; ez += e * z[o + k]; ezz += e * z[o + k] * z[o + k]
            Ez = ez / Z; Vz = max(ezz / Z - Ez * Ez, 0.0)
            spos = 0.0
            for k in range(m):
                if y[o + k] > 0:
                    spos += z[o + k]; ll += th * z[o + k] - (np.log(Z) + mx)
            g += -spos + npos * Ez; h += npos * Vz
        if it == 0:
            ll0 = ll
        if it == iters:
            return th, (ll - ll0) / len(M)
        if h <= 0:
            return th, (ll - ll0) / len(M)
        step = -g / (h + 1e-9)
        if step > 1.0:
            step = 1.0
        if step < -1.0:
            step = -1.0
        th += step
    return th, 0.0


def q(x, p):
    x = x[~np.isnan(x)]
    return float(np.quantile(x, p)) if len(x) else np.nan


# ---------------------------------------------------------------- feature list
names = sorted({p.stem for d in (FEATS, FEATS_NEW) for p in (d / "val").glob("*.npy")}
               & {p.stem for d in (FEATS, FEATS_NEW) for p in (d / "test").glob("*.npy")})
names.append("has:pre2_fe")
log(f"{len(names)} features; {time.time()-T0:.0f}s")

done = set()
csv = OUT / "feature_table.csv"
if csv.exists():
    done = {l.split(",")[0] for l in csv.read_text().splitlines()[1:]}
cols = None
rows_out = []


def load_split(split, name):
    if name.startswith("has:"):
        x = np.load(feat_file(split, name[4:]))
        return (~np.isnan(x)).astype(np.float32)
    return np.load(feat_file(split, name)).astype(np.float32)


def load_tr(name):
    if name.startswith("has:"):
        d = load_train(name[4:])
        return None if d is None else {k: (~np.isnan(v)).astype(np.float32) for k, v in d.items()}
    return load_train(name)


for name in names:
    if name in done:
        continue
    t1 = time.time()
    xv = load_split("val", name); xt = load_split("test", name); xtr = load_tr(name)
    r = {"feature": name, "on_train": int(xtr is not None)}
    r["nan_val"] = float(np.isnan(xv).mean()); r["nan_pub"] = float(np.isnan(xt[pub_row]).mean())
    r["nan_priv"] = float(np.isnan(xt[~pub_row]).mean())
    r["zero_val"] = float((xv == 0).mean()); r["zero_priv"] = float((xt[~pub_row] == 0).mean())
    r["mean_val"] = float(np.nanmean(xv)); r["sd_val"] = float(np.nanstd(xv))
    r["mean_pub"] = float(np.nanmean(xt[pub_row])); r["mean_priv"] = float(np.nanmean(xt[~pub_row]))
    r["smd_val_priv"] = (r["mean_priv"] - r["mean_val"]) / (r["sd_val"] + 1e-12)
    if xtr is not None:
        r["nan_trRL"] = float(np.isnan(xtr["trRL"]).mean()); r["nan_trProd"] = float(np.isnan(xtr["trProd"]).mean())
        r["zero_trRL"] = float((xtr["trRL"] == 0).mean())
        r["mean_trRL"] = float(np.nanmean(xtr["trRL"])); r["mean_trProd"] = float(np.nanmean(xtr["trProd"]))
        r["smd_trRL_val"] = (r["mean_val"] - r["mean_trRL"]) / (r["sd_val"] + 1e-12)
    # A. candidate-level adversarial AUC
    v50 = xv[VALROWS["val50"]]
    pairs = {"val~pub": (xv, xt[pub_row]), "val~priv": (xv, xt[~pub_row]), "pub~priv": (xt[pub_row], xt[~pub_row])}
    if xtr is not None:
        pairs.update({"trRL~val50": (xtr["trRL"], v50), "trProd~val50": (xtr["trProd"], v50),
                      "trRL~priv50": (xtr["trRL"], xt[TESTROWS["priv50"]]), "trProd~trRL": (xtr["trProd"], xtr["trRL"])})
    for k, (a, b) in pairs.items():
        r[f"cAUC_{k}"] = auc2(sub(a), sub(b))
    # B. feed-level aggregates
    agg_v = feed_agg(xv, offv, Mv); agg_t = feed_agg(xt, offt, Mt)
    aggn = ("mean", "nanshare", "zeroshare", "max")
    for j, an in enumerate(aggn):
        av, at = agg_v[j], agg_t[j]
        r[f"fAUC_{an}_val~pub"] = auc2(av, at[pub_feed]); r[f"fAUC_{an}_val~priv"] = auc2(av, at[~pub_feed])
        r[f"fAUC_{an}_pub~priv"] = auc2(at[pub_feed], at[~pub_feed])
        if xtr is not None:
            ah = feed_agg(xtr["trRL"], TR["trRL"]["off"], TR["trRL"]["M"])[j]
            r[f"fAUC_{an}_trRL~val50"] = auc2(ah, av[val50])
    # C. label relation
    pops = {}
    for k, fx in VALPOP.items():
        rr = VALROWS[k]
        pops[k] = (xv[rr], yv[rr], local_off(Mv[fx]), Mv[fx])
    if xtr is not None:
        for k in ("trRL", "trProd", "trSmall"):
            pops[k] = (xtr[k], TR[k]["y"], TR[k]["off"], TR[k]["M"])
    for k, (x_, y_, o_, m_) in pops.items():
        a, n = feed_auc(x_.astype(np.float64), y_, o_, m_)
        r[f"wAUC_{k}"] = a
        z = feed_z_nanmin(x_.astype(np.float64), o_, m_)
        th, llg = clogit_slope(z, y_, o_, m_)
        r[f"slope_{k}"] = th; r[f"llgain_{k}"] = llg
    # D. private support
    lo, hi = np.nanmin(xv), np.nanmax(xv); q1, q99 = q(xv, 0.001), q(xv, 0.999)
    for k, rr in (("pub", TESTROWS["pub"]), ("priv", TESTROWS["priv"])):
        xx = xt[rr]; ok = ~np.isnan(xx)
        r[f"out_minmax_{k}"] = float(((xx < lo) | (xx > hi))[ok].mean()) if ok.any() else np.nan
        r[f"out_q_{k}"] = float(((xx < q1) | (xx > q99))[ok].mean()) if ok.any() else np.nan
        cc = cold_t[rr] & ok
        r[f"out_q_cold_{k}"] = float(((xx < q1) | (xx > q99))[cc].mean()) if cc.any() else np.nan
    zv = feed_z_nanmin(xv.astype(np.float64), offv, Mv); zt = feed_z_nanmin(xt.astype(np.float64), offt, Mt)
    r["zcold_val"] = float(zv[cold_v].mean()) if cold_v.any() else np.nan
    r["zcold_pub"] = float(zt[pub_row & cold_t].mean()); r["zcold_priv"] = float(zt[~pub_row & cold_t].mean())
    if xtr is not None:
        zh = feed_z_nanmin(xtr["trRL"].astype(np.float64), TR["trRL"]["off"], TR["trRL"]["M"])
        r["zcold_trRL"] = float(zh[cold_tr["trRL"]].mean()) if cold_tr["trRL"].any() else np.nan
    # write
    if cols is None:
        cols = list(r.keys())
        if not csv.exists():
            csv.write_text(",".join(cols) + "\n")
        else:
            head = csv.read_text().splitlines()[0].split(",")
            if head != cols:  # column set changed: restart the file
                csv.write_text(",".join(cols) + "\n"); done = set()
    with open(csv, "a") as fh:
        fh.write(",".join("" if (isinstance(r.get(c), float) and np.isnan(r.get(c))) else str(r.get(c, "")) for c in cols) + "\n")
    log(f"{name:28s} tr={r['on_train']} cAUC val~priv {r['cAUC_val~priv']:.3f} trRL~val50 {r.get('cAUC_trRL~val50', np.nan):.3f} "
        f"wAUC val {r['wAUC_valAll']:.3f} trRL {r.get('wAUC_trRL', np.nan):.3f} trProd {r.get('wAUC_trProd', np.nan):.3f} "
        f"slope val {r['slope_valAll']:+.3f} trProd {r.get('slope_trProd', np.nan):+.3f} | {time.time()-t1:.1f}s total {time.time()-T0:.0f}s")
log("DONE")
