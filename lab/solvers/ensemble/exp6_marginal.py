"""(3b) Marginal gain of adding j members of a given kind on top of the 6-member nn_final5 pool, on the train holdout
(train-only members) and on val (mix units: a member's fta fine-tune + its train-only prediction enter together),
compared with the within-recipe extrapolation a - b/k fitted on the F5 pool itself.
"""
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Holdout, Split  # noqa: E402
from exp3_diversity import F4, F5, F5CD, OLD, V13  # noqa: E402

KINDS = {"F5cd": (F5CD, "n2cpf"), "V13": (V13, "n2cpf"), "F4": (F4, "n2cp"), "OLD5ep": (OLD, "n2cp")}


def p(n):
    return np.load(f"{ROOT}/preds/{n}.npy").astype(np.float64)


def add_curve(per_feed, base_units, extra_units, max_sub=30, rng=None):
    out = {}
    for j in range(1, len(extra_units) + 1):
        combs = list(itertools.combinations(range(len(extra_units)), j))
        if len(combs) > max_sub:
            combs = [combs[i] for i in rng.choice(len(combs), max_sub, replace=False)]
        vals = []
        for c in combs:
            units = base_units + [extra_units[i] for i in c]
            ft = [u[0] for u in units if u[0] is not None]
            to = [u[1] for u in units]
            s = np.mean(to, 0) if not ft else 0.5 * np.mean(ft, 0) / np.std(np.mean(ft, 0)) + 0.5 * np.mean(to, 0) / np.std(np.mean(to, 0))
            vals.append(per_feed(s).mean())
        out[j] = float(np.mean(vals))
    return out


def within_fit(per_feed, units, rng):
    K = len(units)
    pts = {}
    for k in range(1, K + 1):
        combs = list(itertools.combinations(range(K), k))
        if len(combs) > 30:
            combs = [combs[i] for i in rng.choice(len(combs), 30, replace=False)]
        vals = []
        for c in combs:
            us = [units[i] for i in c]
            ft = [u[0] for u in us if u[0] is not None]
            to = [u[1] for u in us]
            s = np.mean(to, 0) if not ft else 0.5 * np.mean(ft, 0) / np.std(np.mean(ft, 0)) + 0.5 * np.mean(to, 0) / np.std(np.mean(to, 0))
            vals.append(per_feed(s).mean())
        pts[k] = float(np.mean(vals))
    ks = np.array(sorted(pts), float)
    ys = np.array([pts[int(k)] for k in ks])
    (a, b), *_ = np.linalg.lstsq(np.stack([np.ones_like(ks), -1 / ks], 1), ys, rcond=None)
    return pts, float(a), float(b)


def main():
    rng = np.random.default_rng(1)
    rep = {}
    H = Holdout()
    hz = lambda m: H.z(H.load(m))  # noqa: E731
    base_h = [(None, hz(m)) for m in F5]
    pts, a, b = within_fit(H.per_feed, base_h, rng)
    print(f"holdout within-F5 curve {pts}  fit a {a:.5f} b {b:.5f}")
    rep["holdout_within_F5"] = {"pts": pts, "a": a, "b": b}
    for kind, (ms, _) in KINDS.items():
        cur = add_curve(H.per_feed, base_h, [(None, hz(m)) for m in ms], rng=rng)
        ext = {j: a - b / (6 + j) for j in cur}
        rep[f"holdout_add_{kind}"] = {"curve": cur, "within_extrapolation": ext}
        print(f"  holdout +{kind:7s} " + "  ".join(f"+{j}: {v:.5f} (same-recipe {ext[j]:.5f}, diff {v - ext[j]:+.5f})" for j, v in cur.items()))
    S = Split("val")
    unit = lambda m, suf: (S.z(p(f"nn_{m}_{suf}_val_oof")), S.z(p(f"nn_{m}_trainonly_val")))  # noqa: E731
    base_v = [unit(m, "n2cpf") for m in F5]
    pts, a, b = within_fit(S.per_feed, base_v, rng)
    print(f"val within-F5 (mix units) curve {pts}  fit a {a:.5f} b {b:.5f}")
    rep["val_within_F5"] = {"pts": pts, "a": a, "b": b}
    for kind, (ms, suf) in KINDS.items():
        cur = add_curve(S.per_feed, base_v, [unit(m, suf) for m in ms], rng=rng)
        ext = {j: a - b / (6 + j) for j in cur}
        rep[f"val_add_{kind}"] = {"curve": cur, "within_extrapolation": ext}
        print(f"  val +{kind:7s} " + "  ".join(f"+{j}: {v:.5f} (same-recipe {ext[j]:.5f}, diff {v - ext[j]:+.5f})" for j, v in cur.items()))
    json.dump(rep, open(f"{ROOT}/solvers/out/ensemble/exp6.json", "w"), indent=1)


if __name__ == "__main__":
    main()
