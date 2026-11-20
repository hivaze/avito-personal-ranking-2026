"""(2b) Caruana-style greedy ensemble selection (forward, with replacement, init = best single unit) over a library of
27 mix units (member m: 0.5 z(fine-tuned m) + 0.5 z(train-only m)) from F5, F5cd, V13, F4, OLD5ep, with nested user-fold CV:
selection on 4 folds, evaluation on the 5th. 25 steps. Compared with family-equal eq(F5, F5cd, V13) and v11.
"""
import json
import sys
import multiprocessing as mp

import numpy as np

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Split, fmt_slices, slices  # noqa: E402
from exp3_diversity import F4, F5, F5CD, OLD, V13  # noqa: E402

LIB = [(m, "n2cpf") for m in F5 + F5CD + V13] + [(m, "n2cp") for m in F4 + OLD]
STEPS = 25
S = None
U = None


def init():
    global S, U
    S = Split("val")
    U = []
    for m, suf in LIB:
        a = S.z(np.load(f"{ROOT}/preds/nn_{m}_{suf}_val_oof.npy").astype(np.float64))
        b = S.z(np.load(f"{ROOT}/preds/nn_{m}_trainonly_val.npy").astype(np.float64))
        U.append(0.5 * a + 0.5 * b)


def run_fold(k):
    tr = S.fold != k
    single = [S.per_feed(u)[tr].mean() for u in U]
    sel = [int(np.argmax(single))]
    acc = U[sel[0]].copy()
    hist = [(sel[0], single[sel[0]])]
    for _ in range(STEPS - 1):
        best, bv = None, -1
        for j in range(len(U)):
            s = (acc + U[j]) / (len(sel) + 1)
            v = S.per_feed(s)[tr].mean()
            if v > bv:
                best, bv = j, v
        sel.append(best)
        acc += U[best]
        hist.append((best, bv))
    return k, sel, hist


def main():
    with mp.get_context("spawn").Pool(5, initializer=init) as pool:
        res = pool.map(run_fold, range(5))
    init()
    v11 = None
    import importlib
    e8 = importlib.import_module("exp8_candidates")
    D = e8.build(S, "val")
    base = S.per_feed(D["D0_v11"])
    d3 = S.per_feed(D["D3_eq_F5_F5cd_V13"])
    # nested evaluation at each step count
    rep = {"lib": [m for m, _ in LIB], "folds": {}}
    curve = {}
    for t in [1, 3, 5, 10, 15, 20, 25]:
        out = np.zeros(S.n)
        for k, sel, _ in res:
            s = np.mean([U[j] for j in sel[:t]], 0)
            out[S.fold == k] = S.per_feed(s)[S.fold == k]
        curve[t] = out
        sl = slices(S, out - base)
        print(f"caruana nested, {t:2d} steps: {out.mean():.5f}  vs v11 {fmt_slices(sl)}")
        rep[f"steps_{t}"] = {"ndcg": float(out.mean()), "delta": {a: [float(x) for x in b] for a, b in sl.items()}}
    dd = curve[25] - d3
    print(f"caruana(25) - eq(F5,F5cd,V13): {dd.mean():+.5f}+-{dd.std() / np.sqrt(S.n):.5f}  (eq3 {d3.mean():.5f})")
    rep["car25_minus_eq3"] = [float(dd.mean()), float(dd.std() / np.sqrt(S.n))]
    for k, sel, hist in res:
        names = [LIB[j][0] for j in sel]
        cnt = {}
        for n in names:
            cnt[n] = cnt.get(n, 0) + 1
        rep["folds"][k] = {"selected": names, "inner_curve": [h[1] for h in hist]}
        print(f"  fold {k}: inner {hist[0][1]:.5f} -> {hist[-1][1]:.5f}; picks {sorted(cnt.items(), key=lambda t: -t[1])}")
    json.dump(rep, open(f"{ROOT}/solvers/out/ensemble/exp9.json", "w"), indent=1)


if __name__ == "__main__":
    main()
