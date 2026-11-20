"""hzcv parity + leak detectors (pre-declared; labels read only to MEASURE, never to build).

Populations: train random-like feeds (raw p_random > 0.5, nothing excluded) vs all val feeds; label-free
statistics also for test public (ts < 1790497920) / private.
Per feature: within-feed AUC (feeds with >= 1 pos and >= 1 neg, average ties), mean per-feed z(pos) - z(neg),
zero share and lift P(y | x == 0)/P(y), NaN share and lift P(y | NaN)/P(y).
Flags: |AUC_train - AUC_val| > 0.03, or missingness (NaN) label-lift ratio > 1.5x between train and val.
Parity: censored share by lag L; seq_cvlag_pop == 0 share; lift of (cv = 0 & censored), (cv = 0 & exposed 0-72 h),
(cv = 0 & exposed > 72 h), (cv > 0) - the RESEARCH-2 table - for the as-built (seq_cv, L = 0) and the lagged rows.
Usage: CUDA_VISIBLE_DEVICES="" python feat_new/hz_detect.py [rl05|strict]
  rl05   (pre-declared default) train feeds with raw p_random > 0.5 vs all val feeds
  strict (supplementary) train feeds with p_random > 0.9 and M >= 50 (the holdout definition), holdout feeds
         EXCLUDED (they are all L = 0), vs val feeds with M >= 50
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hz_common as hz  # noqa: E402
sys.path.insert(0, "/tmp/scratchpad/review/leakage")
from common_rows import feed_stats, rows_of  # noqa: E402

SCR = Path("/tmp/scratchpad/featrev/hzcv")
PUB_BOUND = 1790497920
NEW = ["seq_cvlag_invd20", "seq_cvlag_pop", "seq_cvlag_norm", "hzcv_expo_slog", "hzcv_censored", "hzcv_stale_d"]
REF = {"seq_cvlag_invd20": "seq_cv_invd20", "seq_cvlag_pop": "seq_cv_pop", "seq_cvlag_norm": "seq_cv_norm",
       "hzcv_expo_slog": "src_expo_slog", "hzcv_censored": "src_censored"}


def load(split, name, rows):
    p = hz.FEATS_NEW / split / f"{name}.npy"
    if not p.exists():
        p = hz.FEATS / split / f"{name}.npy"
    return np.asarray(np.load(p, mmap_mode="r")[rows], np.float64)


def stats(x, y, off, M):
    auc, dz = feed_stats(x, y, off, M)
    py = y.mean()
    z = x == 0
    nan = np.isnan(x)
    r = dict(auc=float(np.mean(auc)), auc_se=float(np.std(auc) / np.sqrt(len(auc))), dz=float(np.mean(dz)),
             n_feeds=int(len(auc)), zero=float(z.mean()), nan=float(nan.mean()),
             lift0=float(y[z].mean() / py) if z.any() else None,
             liftnan=float(y[nan].mean() / py) if nan.any() else None)
    return r


def sub(rows_off_M, mask_feeds):
    """Restrict (rows, off, M) to a boolean mask over the sampled feeds."""
    rows, off, M = rows_off_M
    keep = np.repeat(mask_feeds, M)
    M2 = M[mask_feeds]
    off2 = np.concatenate([[0], np.cumsum(M2)[:-1]]).astype(np.int64)
    return keep, off2, M2


def lift_table(cv0, cens, E, y):
    py = y.mean()
    g = {"cv0&censored": cv0 & cens, "cv0&exposed0-72h": cv0 & ~cens & (E <= 72), "cv0&exposed>72h": cv0 & (E > 72),
         "cv>0": ~cv0}
    return {k: dict(share=float(m.mean()), lift=float(y[m].mean() / py) if m.any() else None) for k, m in g.items()}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rl05"
    res = {"mode": mode}
    # ---------------- train random-like ----------------
    f = hz.feeds("train")
    pr = hz.p_random_raw()
    L_feed = hz.lag_per_feed("train")
    ts = f["timestamp"].to_numpy().astype(np.int64)
    d = hz.day_of(ts)
    Mtr = f["M"].to_numpy()
    if mode == "rl05":
        fids = np.nonzero(pr > 0.5)[0]
    else:
        fids = np.setdiff1d(np.nonzero((pr > 0.9) & (Mtr >= 50))[0], hz.holdout_feeds())
    rows, off, M = rows_of("train", fids)
    y = np.asarray(np.load(hz.FLAT / "train/cand_target.npy", mmap_mode="r")[rows], np.float64)
    Lr = np.repeat(L_feed[fids], M)
    print("train random-like feeds", len(fids), "rows", len(rows), "P(y)", y.mean(), flush=True)
    X = {n: load("train", n, rows) for n in NEW + list(REF.values()) + ["src_expo_h"]}
    lag_rows = load("train", "hz_lag", rows)
    assert np.array_equal(lag_rows, Lr), "hz_lag file != hz_common lag"
    # offsets in hours for the lagged rows: E_lag = src_expo_h - 24 L
    E0 = X["src_expo_h"]
    E_lag = np.where(E0 >= 1e5 - 1, 1e5, E0 - 24.0 * Lr)
    tr = {}
    groups = {"all": np.ones(len(fids), bool), "L0": L_feed[fids] == 0, "L4": L_feed[fids] == 4,
              "L8": L_feed[fids] == 8, "from_sep9": d[fids] >= hz.train_days()[3]}
    for gname, gm in groups.items():
        keep, o2, M2 = sub((rows, off, M), gm)
        yy = y[keep]
        tr[gname] = {}
        for n in NEW + list(REF.values()):
            if n == "hzcv_stale_d":
                tr[gname][n] = dict(mean=float(X[n][keep].mean()), min=float(X[n][keep].min()),
                                    max=float(X[n][keep].max()))
                continue
            tr[gname][n] = stats(X[n][keep], yy, o2, M2)
        tr[gname]["parity_lagged"] = lift_table(X["seq_cvlag_pop"][keep] == 0, X["hzcv_censored"][keep] > 0,
                                                E_lag[keep], yy)
        tr[gname]["parity_asbuilt"] = lift_table(X["seq_cv_pop"][keep] == 0, X["src_censored"][keep] > 0,
                                                 E0[keep], yy)
        tr[gname]["censored_lagged"] = float((X["hzcv_censored"][keep] > 0).mean())
        tr[gname]["censored_asbuilt"] = float((X["src_censored"][keep] > 0).mean())
        tr[gname]["cvpop0_lagged"] = float((X["seq_cvlag_pop"][keep] == 0).mean())
        tr[gname]["cvpop0_asbuilt"] = float((X["seq_cv_pop"][keep] == 0).mean())
        tr[gname]["n_feeds"] = int(gm.sum())
        print(gname, json.dumps({k: tr[gname][k] for k in ("n_feeds", "censored_lagged", "censored_asbuilt",
                                                           "cvpop0_lagged", "cvpop0_asbuilt")}), flush=True)
        print("   lagged ", json.dumps(tr[gname]["parity_lagged"]), flush=True)
        print("   asbuilt", json.dumps(tr[gname]["parity_asbuilt"]), flush=True)
    res["train_rl"] = tr
    # censored share by L over ALL train rows (label-free)
    lag_all = np.load(hz.FEATS_NEW / "train/hz_lag.npy", mmap_mode="r")
    cen_all = np.load(hz.FEATS_NEW / "train/hzcv_censored.npy", mmap_mode="r")
    pop_all = np.load(hz.FEATS_NEW / "train/seq_cvlag_pop.npy", mmap_mode="r")
    res["train_all_rows"] = {}
    step = 25_000_000
    acc = {l: [0, 0, 0] for l in (0, 4, 8)}
    for s in range(0, len(lag_all), step):
        la = np.asarray(lag_all[s:s + step]); ce = np.asarray(cen_all[s:s + step]); po = np.asarray(pop_all[s:s + step])
        for l in (0, 4, 8):
            m = la == l
            acc[l][0] += int(m.sum()); acc[l][1] += int((ce[m] > 0).sum()); acc[l][2] += int((po[m] == 0).sum())
    for l in (0, 4, 8):
        res["train_all_rows"][f"L{l}"] = dict(rows=acc[l][0], censored=acc[l][1] / acc[l][0], cvpop0=acc[l][2] / acc[l][0])
    print("train all rows", json.dumps(res["train_all_rows"]), flush=True)
    # ---------------- val ----------------
    fv = hz.feeds("val")
    fidv = np.arange(len(fv)) if mode == "rl05" else np.nonzero(fv["M"].to_numpy() >= 50)[0]
    rv, ov, Mv = rows_of("val", fidv)
    yv = np.asarray(np.load(hz.FLAT / "val/cand_target.npy", mmap_mode="r")[rv], np.float64)
    Xv = {n: load("val", n, rv) for n in NEW + list(REF.values()) + ["src_expo_h"]}
    for a, b in REF.items():
        assert np.array_equal(Xv[a], Xv[b]), f"val {a} != {b}"
    dv = hz.day_of(fv["timestamp"].to_numpy())
    va = {}
    dv = dv[fidv]
    for gname, gm in {"all": np.ones(len(fidv), bool), "tue": dv <= dv.min() + 1, "thu": dv >= dv.min() + 2}.items():
        keep, o2, M2 = sub((rv, ov, Mv), gm)
        yy = yv[keep]
        va[gname] = {n: (stats(Xv[n][keep], yy, o2, M2) if n != "hzcv_stale_d" else
                         dict(mean=float(Xv[n][keep].mean()), min=float(Xv[n][keep].min()), max=float(Xv[n][keep].max())))
                     for n in NEW}
        va[gname]["parity"] = lift_table(Xv["seq_cvlag_pop"][keep] == 0, Xv["hzcv_censored"][keep] > 0,
                                         Xv["src_expo_h"][keep], yy)
        va[gname]["censored"] = float((Xv["hzcv_censored"][keep] > 0).mean())
        va[gname]["cvpop0"] = float((Xv["seq_cvlag_pop"][keep] == 0).mean())
        print("val", gname, json.dumps(va[gname]["parity"]), "censored", va[gname]["censored"], "cvpop0",
              va[gname]["cvpop0"], flush=True)
    res["val"] = va
    # ---------------- test (label-free) ----------------
    ft = hz.feeds("test")
    fi_t = np.load(hz.FLAT / "test/cand_fidx.npy")
    pub = (ft["timestamp"].to_numpy() < PUB_BOUND)[fi_t]
    te = {}
    for n in ("seq_cvlag_pop", "hzcv_censored", "hzcv_stale_d", "hzcv_expo_slog"):
        x = np.load(hz.FEATS_NEW / "test" / f"{n}.npy")
        for part, m in (("public", pub), ("private", ~pub)):
            te.setdefault(part, {})[n] = dict(zero=float((x[m] == 0).mean()), mean=float(x[m].mean()),
                                              min=float(x[m].min()), max=float(x[m].max()))
    res["test"] = te
    print("test", json.dumps(te), flush=True)
    # ---------------- flags ----------------
    flags = []
    for n in NEW:
        if n == "hzcv_stale_d":
            continue
        a_tr, a_va = res["train_rl"]["all"][n]["auc"], res["val"]["all"][n]["auc"]
        if abs(a_tr - a_va) > 0.03:
            flags.append(f"{n}: AUC train {a_tr:.4f} vs val {a_va:.4f}")
        ln_tr, ln_va = res["train_rl"]["all"][n]["liftnan"], res["val"]["all"][n]["liftnan"]
        if ln_tr is not None and ln_va is not None and max(ln_tr / ln_va, ln_va / ln_tr) > 1.5:
            flags.append(f"{n}: NaN lift train {ln_tr:.3f} vs val {ln_va:.3f}")
        if (ln_tr is None) != (ln_va is None):
            flags.append(f"{n}: NaN present on one side only")
    res["flags"] = flags
    print("FLAGS", flags, flush=True)
    print("\nfeature table (train random-like all / L0 / L4 / L8 | val all / tue / thu):")
    for n in NEW[:-1] + list(REF.values()):
        r = res["train_rl"]
        line = f"{n:18s} AUC " + " ".join(f"{r[g][n]['auc']:.4f}" for g in ("all", "L0", "L4", "L8"))
        if n in NEW:
            line += " | " + " ".join(f"{res['val'][g][n]['auc']:.4f}" for g in ("all", "tue", "thu"))
        line += "  dz " + " ".join(f"{r[g][n]['dz']:+.3f}" for g in ("all", "L0", "L4", "L8"))
        if n in NEW:
            line += " | " + " ".join(f"{res['val'][g][n]['dz']:+.3f}" for g in ("all", "tue", "thu"))
        line += f"  zero {r['all'][n]['zero']:.3f}" + (f"/{res['val']['all'][n]['zero']:.3f}" if n in NEW else "")
        l0t = r["all"][n]["lift0"]
        l0v = res["val"]["all"][n]["lift0"] if n in NEW else None
        line += f"  lift0 {l0t if l0t is None else round(l0t, 3)}/{l0v if l0v is None else round(l0v, 3)}"
        print(line, flush=True)
    (SCR / f"detect_{mode}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
