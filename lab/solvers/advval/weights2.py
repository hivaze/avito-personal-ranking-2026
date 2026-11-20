"""(i) intrinsic + hour feed-level classifiers for val->private / val->public; (ii) label-permutation null for the
importance-weighted level shift (is the 'intrinsic' shift beyond M-only distinguishable from classifier noise?);
(iii) SE of the level difference adversarial-weighted minus M-only."""
import json, sys
from common_av import *
from adv_lib import *
sys.path.insert(0, f"{ROOT}/src")
import calib  # noqa: E402

v = pl.read_parquet(f"{OUT}/feed_val.parquet")
t = pl.read_parquet(f"{OUT}/feed_test.parquet")
pub, pri = t.filter(pl.col("ts") < 0), t.filter(pl.col("ts") >= 0)
fa = json.load(open(f"{OUT}/feed_adv.json"))
INTR = list(fa["a_val_vs_private"]["intrinsic"]["imp"].keys())

fv, Mv, offv = meta("val")
y = np.load(f"{FLAT}/val/cand_target.npy")
z = lambda n: feed_z(offv, Mv, load_pred(n))
nd = lambda s: ndcg_per_feed(s, y, offv, Mv)
S = {"v10": nd(0.5 * z("nn_final4_val_oof") + 0.5 * z("nn_final4_trainonly_val") + 0.1 * z("pre_final_val")),
     "v11": nd(0.5 * z("nn_final5_val_oof") + 0.5 * z("nn_final5_trainonly_val")),
     "f5_ft": nd(load_pred("nn_final5_val_oof")), "f5_to": nd(load_pred("nn_final5_trainonly_val")),
     "cd12": nd(0.5 * z("nn_final5cd12_val_oof") + 0.5 * z("nn_final5cd12_trainonly_val"))}
D_ = {"v11-v10": S["v11"] - S["v10"], "ft-to": S["f5_ft"] - S["f5_to"], "cd12-v11": S["cd12"] - S["v11"]}


def fit_w(A, B, fl, perm_seed=None, clip=(0.1, 10.0)):
    D = pl.concat([A.select(["fidx", "user_id"] + fl).with_columns(pl.lit(0).alias("_y")),
                   B.select(["fidx", "user_id"] + fl).with_columns(pl.lit(1).alias("_y"))], how="vertical_relaxed")
    yy = D["_y"].to_numpy()
    g = groups_of(D["user_id"].to_numpy(), np.arange(len(D)))
    if perm_seed is not None:  # permute labels across user groups (keeps group structure)
        rng = np.random.default_rng(perm_seed)
        ug, inv = np.unique(g, return_inverse=True)
        lab_g = np.zeros(len(ug)); np.maximum.at(lab_g, inv, yy)  # (users in both splits get label 1)
        lab_g = rng.permutation(lab_g)
        yy = lab_g[inv].astype(int)
    X = D.select(fl).to_numpy().astype(np.float32)
    r = run_adv(X, yy, g, fl, P_FEED, 400, shap_rows=2000)
    va = D["fidx"].to_numpy() if perm_seed is None else None
    # val rows are the first len(A) rows
    p = r["oof"][:len(A)]
    nA, nB = (yy == 0).sum(), (yy == 1).sum()
    w = p / np.maximum(1 - p, 1e-6) * nA / nB; w /= w.mean()
    if clip:
        w = np.clip(w, *clip); w /= w.mean()
    order = np.argsort(A["fidx"].to_numpy())
    assert (A["fidx"].to_numpy()[order] == np.arange(len(A))).all()
    return w[order], r["auc"]


def shift(w, wref, x):
    """level difference sum((w - wref) x)/n with SE (feeds independent)"""
    d = (w - wref) * (x - x.mean())
    return float(((w - wref) * x).mean()), float(d.std(ddof=1) / np.sqrt(len(x)))


out = {}
for tgt, B in [("private", pri), ("public", pub)]:
    wM = calib.weights(tgt)
    for sname, fl in [("intrinsic", INTR), ("intrinsic_hour", INTR + ["hour"])]:
        w, auc = fit_w(v, B, fl)
        o = dict(auc=auc, ess=float(w.sum() ** 2 / (w ** 2).sum()),
                 level={k: wmean_se(x, w) for k, x in S.items()},
                 shift_vs_M={k: shift(w, wM, x) for k, x in S.items()},
                 delta={k: wmean_se(x, w) for k, x in D_.items()})
        nulls = []
        for ps in range(5):
            wn, aucn = fit_w(v, B, fl, perm_seed=100 + ps)
            nulls.append(dict(auc=aucn, shift_v11=float(((wn - 1) * S["v11"]).mean()),
                              delta_v11_v10=float((wn * D_["v11-v10"]).mean())))
        o["null"] = nulls
        out[f"{tgt}_{sname}"] = o
        print(tgt, sname, "AUC", round(auc, 4), "ESS", round(o["ess"]), "v11", round(o["level"]["v11"][0], 5),
              "shift vs M (v11)", np.round(o["shift_vs_M"]["v11"], 5), "v11-v10", np.round(o["delta"]["v11-v10"], 5),
              "ft-to", np.round(o["delta"]["ft-to"], 5), "cd12-v11", np.round(o["delta"]["cd12-v11"], 5), flush=True)
        print("   null AUCs", [round(n["auc"], 4) for n in nulls], "null shift(v11 vs uniform)",
              [round(n["shift_v11"], 5) for n in nulls], " real shift vs uniform", round(o["level"]["v11"][0] - S["v11"].mean(), 5))
    # M-only shift reference
    print(tgt, "M-only v11", round(float((wM * S["v11"]).mean()), 5), "shift vs uniform", round(float(((wM - 1) * S["v11"]).mean()), 5))
json.dump(out, open(f"{OUT}/weights2.json", "w"), indent=1, default=float)
