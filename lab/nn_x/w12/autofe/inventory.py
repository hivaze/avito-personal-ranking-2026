"""Step 1: inventory of the ranker's per-candidate / per-feed inputs: distribution, NN scaling, drift (PSI)."""
import numpy as np, json
from lib import *
from lib import _z
load_stats()
DSS = {k: DS(k) for k in ("trs", "val", "test")}
REL_IN_MODEL = {"f1", "f2", "f3", "f4", "f5", "cos_u", "seq_max_all", "seq_top3", "cat_cnt_all", "seq_cv_invd20",
                "it_hist_all_luo", "pre2_fe_s", "pre2_new_s", "pre2_hmax32"}


def scaling(n):
    if n in XT34:
        if n.startswith("has:"): return "0/1"
        if n.endswith("_z"): return "clip±5"
        if n.startswith(("seq_cv", "it_hist")): return "log1p"
        return "clip±10"
    if n.startswith("cat_cnt"): return "log1p"
    return "raw"


def mom(x):
    x = x.astype(np.float64); m = x.mean(); s = x.std() + 1e-12
    z = (x - m) / s
    return float((z ** 3).mean()), float((z ** 4).mean() - 3)


def psi(a, b, ed):
    pa = np.bincount(np.searchsorted(ed, a, side="right"), minlength=len(ed) + 1) / len(a)
    pb = np.bincount(np.searchsorted(ed, b, side="right"), minlength=len(ed) + 1) / len(b)
    pa = np.maximum(pa, 1e-4); pb = np.maximum(pb, 1e-4)
    return float(((pa - pb) * np.log(pa / pb)).sum())


rows = []
feedlvl = {"M": lambda D: D.M.astype(np.float32), "H": lambda D: D.H.astype(np.float32),
           "user0": lambda D: (D.uid == 0).astype(np.float32),
           "cold_share": lambda D: np.bincount(D.fid, weights=(D.get("it_hist_all_luo") == 0), minlength=D.nf) / D.M}
for n in BASE + list(feedlvl):
    r = {"name": n, "level": "feed" if n in feedlvl else "cand", "nn_scaling": scaling(n) if n not in feedlvl else ("log" if n in ("M", "H") else "raw/none")}
    raw = {}; tf = {}
    for k, D in DSS.items():
        if n in feedlvl:
            raw[k] = np.asarray(feedlvl[n](D), dtype=np.float32); tf[k] = np.log1p(raw[k]) if n in ("M", "H") else raw[k]
        else:
            raw[k] = D.raw(n); tf[k] = D.get(n)
    v = raw["val"]
    nan = np.isnan(v)
    vv = v[~nan]
    r["nan_val"] = float(nan.mean()); r["nan_trs"] = float(np.isnan(raw["trs"]).mean()); r["nan_test"] = float(np.isnan(raw["test"]).mean())
    r["zero"] = float((vv == 0).mean()) if len(vv) else 1.0
    u, c = np.unique(vv[:: max(1, len(vv) // 400000)], return_counts=True)
    r["card"] = int(len(u)); r["mode_mass"] = float(c.max() / c.sum())
    r["min"], r["p50"], r["p99"], r["max"] = [float(q) for q in (vv.min(), np.median(vv), np.quantile(vv, 0.99), vv.max())]
    r["skew_raw"], r["kurt_raw"] = mom(vv)
    t = tf["val"]
    r["skew_nn"], r["kurt_nn"] = mom(t)
    mu, sd = tf["trs"].mean(), tf["trs"].std() + 1e-6
    for k in ("val", "test"):
        z = (tf[k] - mu) / sd
        r[f"clamp6_{k}"] = float((np.abs(z) > 6).mean()); r[f"maxabsz_{k}"] = float(np.abs(z).max())
    ed = np.unique(np.quantile(tf["val"], np.linspace(0.1, 0.9, 9)))
    r["psi_trs_val"] = psi(tf["trs"], tf["val"], ed); r["psi_val_test"] = psi(tf["val"], tf["test"], ed)
    r["psi_trs_test"] = psi(tf["trs"], tf["test"], ed)
    if n not in feedlvl:
        D = DSS["val"]
        z = _z(D.off, D.M, t)
        const = np.bincount(D.fid, weights=np.abs(z), minlength=D.nf) == 0
        r["feed_const"] = float(const.mean())
        # share of total variance that is between feeds
        fm = np.bincount(D.fid, weights=t, minlength=D.nf) / D.M
        r["between_var"] = float(np.average((fm - t.mean()) ** 2, weights=D.M) / (t.var() + 1e-12))
        r["has_rel_in_model"] = n in REL_IN_MODEL
    rows.append(r)
    print(json.dumps(r), flush=True)
pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "inventory.csv")
# categoricals
for nm in ("cand_category_id", "hist_category_id", "hist_event_type", "hist_event_source"):
    d = {}
    for sp in ("val", "test"):
        a = np.load(FLAT / sp / f"{nm}.npy", mmap_mode="r"); d[sp] = np.bincount(np.asarray(a).astype(np.int64) + 1, minlength=130) / len(a)
    a = np.load(FLAT / "train" / f"{nm}.npy", mmap_mode="r"); d["trs"] = np.bincount(np.asarray(a[:20_000_000]).astype(np.int64) + 1, minlength=130) / 20_000_000
    f = lambda p, q: float(((np.maximum(p, 1e-5) - np.maximum(q, 1e-5)) * np.log(np.maximum(p, 1e-5) / np.maximum(q, 1e-5))).sum())
    print("CATEG", nm, "card", int((d["val"] > 0).sum()), "psi trn(first 20M rows)->val", round(f(d["trs"], d["val"]), 4), "val->test", round(f(d["val"], d["test"]), 4), flush=True)
