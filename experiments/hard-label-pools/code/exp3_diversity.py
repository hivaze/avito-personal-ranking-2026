# Measures how different the model families are: per-feed correlation of family scores on val and
# test, the effective number of models in one family, and curves of ensemble NDCG against the
# number of members on val and on the holdout. It also defines the family member lists that
# exp7_pairs.py imports.
# How it was run: CPU, 20 min, 2026-11-11, from lab/solvers/ensemble; the command line is not on
# record.
# What it produced: solvers/out/ensemble/exp3.{log,json}; the log is results/diversity.txt
# (train-only members of one recipe correlate 0.936 per feed; the 12 predictions of the family
# count as 1.1 independent models).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/solvers/ensemble/exp3_diversity.py
"""(3) Diversity: per-feed correlations of family scores on val and test, effective number of models,
and diminishing-returns curves (ensemble NDCG vs number of members) on val and on the train holdout.

Writes solvers/out/ensemble/exp3.json and prints tables.
"""
import itertools
import json
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Holdout, Split  # noqa: E402

F5 = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]
F5CD = ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"]
F4 = ["v9_e10", "v9_e10_s61", "v9_e10_s62", "v9_e10_s63"]
V13 = ["v13_lr1_mix1", "v13_lr2_mix1", "v13_lr1_mix0"]
OLD = ["v9_gru_e5_s52", "v9_gru_e5_s54", "v9_din_e5_s56", "v9_deep_e5_s55", "v9_e5_s51", "v9_e5_s53", "v9_e8_s57", "v9_e8_s58"]


def p(name):
    return np.load(f"{ROOT}/preds/{name}.npy").astype(np.float64)


def fam_sources(split):
    v = split == "val"
    s_ft, s_to = ("val_oof", "trainonly_val") if v else ("test", "trainonly_test")
    s_x = "val" if v else "test"
    src = {
        "F5_ft": [f"nn_{m}_n2cpf_{s_ft}" for m in F5],
        "F5_to": [f"nn_{m}_{s_to}" for m in F5],
        "F5cd_ft": [f"nn_{m}_n2cpf_{s_ft}" for m in F5CD],
        "F5cd_to": [f"nn_{m}_{s_to}" for m in F5CD],
        "F4_ft": [f"nn_{m}_n2cp_{s_ft}" for m in F4],
        "F4_to": [f"nn_{m}_{s_to}" for m in F4],
        "V13_ft": [f"nn_{m}_n2cpf_{s_ft}" for m in V13],
        "V13_to": [f"nn_{m}_{s_to}" for m in V13],
        "OLD5ep_to": [f"nn_{m}_{s_to}" for m in OLD],
        "OLD5ep_ft": [f"nn_{m}_n2cp_{s_ft}" for m in OLD],
        "F3_ft": [f"nn_final3_{'val_oof' if v else 'test'}"],
        "F3_to": [f"nn_final3_{s_to}"],
        "pre_final": [f"pre_final_{s_x}"],
        "pre2_final": [f"pre2_final_{s_x}"],
        "xgb_x300_luo": [f"xgb_x300_luo_{s_x}"],
        "cb_tr300k_luo": [f"cb_tr300k_luo_{s_x}"],
        "stk2_res4_clean": [f"stk2_res4_clean_{s_x}"],
    }
    return src


def fam_scores(S, split):
    fams, members = {}, {}
    for k, names in fam_sources(split).items():
        zs = [S.z(p(n)) for n in names]
        members[k] = zs
        fams[k] = S.z(np.mean(zs, 0))
    return fams, members


def pf_corr(S, za, zb):
    """mean over feeds of the within-feed Pearson correlation of two per-feed z-scored vectors."""
    return float((np.bincount(S.fid, za * zb, S.n) / S.M).mean())


def n_eff(C):
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum()), float(len(C) / (1 + (len(C) - 1) * (C[np.triu_indices(len(C), 1)].mean())))


def curve(Sx, units, per_feed_fn, max_sub=40, rng=None):
    """Mean ensemble NDCG for subsets of k units (each unit = list of z arrays that enter together)."""
    K = len(units)
    out = {}
    for k in range(1, K + 1):
        combs = list(itertools.combinations(range(K), k))
        if len(combs) > max_sub:
            idx = rng.choice(len(combs), max_sub, replace=False)
            combs = [combs[i] for i in idx]
        vals = []
        for c in combs:
            s = np.mean([z for i in c for z in units[i]], 0)
            vals.append(per_feed_fn(s).mean())
        out[k] = (float(np.mean(vals)), float(np.std(vals)), len(combs))
    return out


def fit_inv(curv):
    """NDCG(k) = a - b/k (variance-of-mean model: error variance (1-rho)/k). Returns a (k->inf), b."""
    ks = np.array(sorted(curv))
    ys = np.array([curv[k][0] for k in ks])
    A = np.stack([np.ones_like(ks, dtype=float), -1.0 / ks], 1)
    (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    try:
        (a2, b2, c2), _ = curve_fit(lambda k, a, b, c: a - b / k ** c, ks.astype(float), ys, p0=[a, b, 1.0], maxfev=20000)
    except Exception:
        a2 = b2 = c2 = float("nan")
    return float(a), float(b), (float(a2), float(b2), float(c2))


def main():
    rng = np.random.default_rng(0)
    rep = {}
    Sv, St = Split("val"), Split("test")
    fv, mv = fam_scores(Sv, "val")
    ft_, mt = fam_scores(St, "test")
    keys = list(fv)
    # family NDCG on val
    rep["family_ndcg_val"] = {k: float(Sv.per_feed(fv[k]).mean()) for k in keys}
    for k in keys:
        print(f"  {k:16s} val {rep['family_ndcg_val'][k]:.5f}  ({len(mv[k])} members)")
    # correlation matrices
    for split, S, f in [("val", Sv, fv), ("test", St, ft_)]:
        C = np.array([[pf_corr(S, f[a], f[b]) for b in keys] for a in keys])
        rep[f"corr_{split}"] = C.tolist()
    rep["keys"] = keys
    Cv, Ct = np.array(rep["corr_val"]), np.array(rep["corr_test"])
    print("\nper-feed Pearson corr of family z-scores (upper: val, lower: test)")
    print(" " * 17 + " ".join(f"{k[:7]:>7s}" for k in keys))
    for i, a in enumerate(keys):
        row = [Cv[i, j] if j > i else (Ct[i, j] if j < i else 1.0) for j in range(len(keys))]
        print(f"{a:16s} " + " ".join(f"{x:7.3f}" for x in row))
    # member-level structure of the v11 pool
    for split, S, m in [("val", Sv, mv), ("test", St, mt)]:
        Z = m["F5_ft"] + m["F5_to"]
        C = np.array([[pf_corr(S, a, b) for b in Z] for a in Z])
        ne = n_eff(C)
        to_to = C[6:, 6:][np.triu_indices(6, 1)].mean()
        ft_ft = C[:6, :6][np.triu_indices(6, 1)].mean()
        ft_to_same = np.mean([C[i, 6 + i] for i in range(6)])
        ft_to_cross = np.mean([C[i, 6 + j] for i in range(6) for j in range(6) if i != j])
        rep[f"v11_pool_{split}"] = {"n_eff_eig": ne[0], "n_eff_rho": ne[1], "to_to": float(to_to), "ft_ft": float(ft_ft),
                                    "ft_to_same": float(ft_to_same), "ft_to_cross": float(ft_to_cross)}
        print(f"\nv11 pool ({split}): N_eff eig {ne[0]:.2f}, N/(1+(N-1)rho) {ne[1]:.2f}; mean corr to-to {to_to:.4f} ft-ft {ft_ft:.4f} "
              f"ft-to same member {ft_to_same:.4f} ft-to cross {ft_to_cross:.4f}")
    # test fold-model diversity of the fine-tuned branch (test ft = mean of 5 fold models)
    fold_c = []
    for mname in F5:
        tf = np.load(f"{ROOT}/preds/nn_{mname}_n2cpf_test_folds.npy").astype(np.float64)
        zz = [St.z(tf[k]) for k in range(5)]
        fold_c.append(np.mean([pf_corr(St, zz[i], zz[j]) for i in range(5) for j in range(i + 1, 5)]))
        zto = St.z(p(f"nn_{mname}_trainonly_test"))
        fold_c.append(-np.mean([pf_corr(St, zz[i], zto) for i in range(5)]))  # negative marker: fold-vs-trainonly
    fc = np.array(fold_c).reshape(-1, 2)
    rep["test_fold_corr"] = {"fold_fold": float(fc[:, 0].mean()), "fold_vs_trainonly": float(-fc[:, 1].mean())}
    print(f"\ntest: corr between the 5 fold models of one fine-tuned member {fc[:, 0].mean():.4f}; fold model vs its train-only "
          f"start {-fc[:, 1].mean():.4f}")
    # diminishing-returns curves on val
    H = Holdout()
    curves = {}
    pv = lambda s: Sv.per_feed(s)  # noqa: E731
    ph = lambda s: H.per_feed(s)  # noqa: E731
    cv = {
        "val_F5_to": [[z] for z in mv["F5_to"]],
        "val_F5_ft": [[z] for z in mv["F5_ft"]],
        "val_F5_mixunit": [[a, b] for a, b in zip(mv["F5_ft"], mv["F5_to"])],
        "val_F4_to": [[z] for z in mv["F4_to"]],
        "val_F4_ft": [[z] for z in mv["F4_ft"]],
        "val_F5+F5cd_to": [[z] for z in mv["F5_to"] + mv["F5cd_to"]],
        "val_F5+F5cd_mixunit": [[a, b] for a, b in zip(mv["F5_ft"] + mv["F5cd_ft"], mv["F5_to"] + mv["F5cd_to"])],
    }
    for k, units in cv.items():
        curves[k] = curve(Sv, units, pv, rng=rng)
    hz = {m: H.z(H.load(m)) for m in F5 + F5CD + F4 + ["v13_lr1_mix1", "v13_lr2_mix1", "v13_lr1_mix0"] +
          ["v9_gru_e5_s52", "v9_gru_e5_s54", "v9_din_e5_s56", "v9_deep_e5_s55", "v9_e5_s51", "v9_e5_s53", "v9_e8_s57", "v9_e8_s58"]}
    ch = {
        "hold_F5_to": [[hz[m]] for m in F5],
        "hold_F5cd_to": [[hz[m]] for m in F5CD],
        "hold_F4_to": [[hz[m]] for m in F4],
        "hold_F5+F5cd_to": [[hz[m]] for m in F5 + F5CD],
        "hold_OLD5ep_to": [[hz[m]] for m in OLD],
    }
    for k, units in ch.items():
        curves[k] = curve(H, units, ph, rng=rng)
    rep["curves"] = {k: {str(kk): vv for kk, vv in c.items()} for k, c in curves.items()}
    rep["fits"] = {}
    print("\ndiminishing returns: mean ensemble NDCG by #units; fit a - b/k")
    for k, c in curves.items():
        a, b, (a2, b2, c2) = fit_inv(c)
        rep["fits"][k] = {"a_inf": a, "b": b, "power_fit": [a2, b2, c2]}
        pts = "  ".join(f"{kk}:{v[0]:.5f}" for kk, v in c.items())
        print(f"  {k:22s} {pts}   | a_inf {a:.5f} b {b:.5f} (gain 1->inf {b:.5f}); power fit a {a2:.5f} c {c2:.2f}")
    json.dump(rep, open(f"{ROOT}/solvers/out/ensemble/exp3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
