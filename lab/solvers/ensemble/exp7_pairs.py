"""(3c) Pairwise 'strength vs diversity' law: for every pair of model families, the equal z-mix gain over the better
member as a function of the per-feed correlation (1 - rho) and the strength gap. Fitted on val (mix families and
weak non-NN models) and on the holdout (train-only families). Used to predict what a new solver type must reach to help.
"""
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Holdout, Split  # noqa: E402
from exp3_diversity import F4, F5, F5CD, OLD, V13  # noqa: E402


def p(n):
    return np.load(f"{ROOT}/preds/{n}.npy").astype(np.float64)


def pf_corr(S, a, b):
    return float((np.bincount(S.fid, a * b, S.n) / S.M).mean())


def fit(rows):
    r = np.array([[x["one_minus_rho"], x["gap"], x["gain_best"]] for x in rows])
    A = np.stack([r[:, 0], -r[:, 1]], 1)
    coef, *_ = np.linalg.lstsq(A, r[:, 2], rcond=None)
    pred = A @ coef
    r2 = 1 - ((r[:, 2] - pred) ** 2).sum() / ((r[:, 2] - r[:, 2].mean()) ** 2).sum()
    return coef.tolist(), float(r2)


def pairs(S, fams):
    nd = {k: S.per_feed(v).mean() for k, v in fams.items()}
    rows = []
    for a, b in itertools.combinations(fams, 2):
        rho = pf_corr(S, fams[a], fams[b])
        e = S.per_feed(fams[a] + fams[b]).mean()
        rows.append({"a": a, "b": b, "rho": rho, "one_minus_rho": 1 - rho, "nd_a": nd[a], "nd_b": nd[b],
                     "gap": abs(nd[a] - nd[b]), "eq": e, "gain_best": e - max(nd[a], nd[b])})
    return rows


def main():
    rep = {}
    S = Split("val")
    def mix(ms, suf):  # noqa: E301
        zf = S.z(np.mean([S.z(p(f"nn_{m}_{suf}_val_oof")) for m in ms], 0))
        zt = S.z(np.mean([S.z(p(f"nn_{m}_trainonly_val")) for m in ms], 0))
        return S.z(zf + zt)
    fams = {"F5": mix(F5, "n2cpf"), "F5cd": mix(F5CD, "n2cpf"), "V13": mix(V13, "n2cpf"), "F4": mix(F4, "n2cp"),
            "OLD5ep": mix(OLD, "n2cp"), "F3": S.z(S.z(p("nn_final3_val_oof")) + S.z(p("nn_final3_trainonly_val"))),
            "pre2_final": S.z(p("pre2_final_val")), "xgb_x300_luo": S.z(p("xgb_x300_luo_val")),
            "stk2_res4_clean": S.z(p("stk2_res4_clean_val"))}
    rows = pairs(S, fams)
    nn_rows = [r for r in rows if r["gap"] < 0.01]
    rep["val_rows"] = rows
    rep["val_fit_all"] = fit(rows)
    rep["val_fit_gap<0.01"] = fit(nn_rows)
    print("val pairs: a b rho nd_a nd_b eq gain_over_best")
    for r in sorted(rows, key=lambda r: -r["gain_best"]):
        print(f"  {r['a']:12s} {r['b']:16s} rho {r['rho']:.4f} {r['nd_a']:.5f} {r['nd_b']:.5f} eq {r['eq']:.5f} gain {r['gain_best']:+.5f}")
    print(f"val fit gain = alpha (1-rho) - beta gap: all pairs {rep['val_fit_all']}, gap<0.01 {rep['val_fit_gap<0.01']}")
    # best weight of a weak model added to v11 (descriptive, in-sample upper bound)
    for k in ["pre2_final", "xgb_x300_luo", "stk2_res4_clean", "F4", "OLD5ep"]:
        best = max(((w, S.per_feed(fams["F5"] + w * fams[k]).mean()) for w in [0.05, 0.1, 0.2, 0.3, 0.5]), key=lambda t: t[1])
        rep[f"best_w_{k}"] = best
        print(f"  in-sample best weight of {k} added to F5: w {best[0]} -> {best[1]:.5f} (F5 {S.per_feed(fams['F5']).mean():.5f})")
    H = Holdout()
    hf = {"F5_to": np.mean([H.z(H.load(m)) for m in F5], 0), "F5cd_to": np.mean([H.z(H.load(m)) for m in F5CD], 0),
          "V13_to": np.mean([H.z(H.load(m)) for m in V13], 0), "F4_to": np.mean([H.z(H.load(m)) for m in F4], 0),
          "OLD5ep_to": np.mean([H.z(H.load(m)) for m in OLD], 0)}
    hf = {k: H.z(v) for k, v in hf.items()}
    hr = []
    nd = {k: H.per_feed(v).mean() for k, v in hf.items()}
    for a, b in itertools.combinations(hf, 2):
        rho = float((np.bincount(H.fid, hf[a] * hf[b], H.n) / H.M).mean())
        e = H.per_feed(hf[a] + hf[b]).mean()
        hr.append({"a": a, "b": b, "rho": rho, "one_minus_rho": 1 - rho, "nd_a": nd[a], "nd_b": nd[b], "gap": abs(nd[a] - nd[b]),
                   "eq": e, "gain_best": e - max(nd[a], nd[b])})
    rep["hold_rows"] = hr
    rep["hold_fit"] = fit(hr)
    print("holdout pairs:")
    for r in sorted(hr, key=lambda r: -r["gain_best"]):
        print(f"  {r['a']:10s} {r['b']:10s} rho {r['rho']:.4f} {r['nd_a']:.5f} {r['nd_b']:.5f} eq {r['eq']:.5f} gain {r['gain_best']:+.5f}")
    print(f"holdout fit {rep['hold_fit']}")
    json.dump(rep, open(f"{ROOT}/solvers/out/ensemble/exp7.json", "w"), indent=1)


if __name__ == "__main__":
    main()
