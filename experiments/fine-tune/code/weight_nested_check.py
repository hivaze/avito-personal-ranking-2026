# Nested check of the weight of the fine-tuned branch in the final score (2026-11-11). For a 19-member pool it
# forms w * z(fine-tuned) + (1 - w) * z(train-only) on val, picks w from {0.33, 0.5, 0.67} on four user folds
# and scores the fifth. The adoption rule was written to a file before the script ran.
# Run on CPU from the project root, as recorded in lab/reports/r5_wft.md:
#   CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 POLARS_MAX_THREADS=16 \
#     .venv/bin/python solvers/r5_wft.py
# Produced solvers/out/r5_wft/r5_wft.log and r5_wft.json (the log is in ../results/weight_nested_check.log).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/solvers/r5_wft.py
#
"""r5_wft: pre-declared nested-CV check of the fine-tuned share w_ft for the Cd ensemble (CPU only, read-only on preds/).

See solvers/out/r5_wft/PREDECLARATION.txt (written before this script ran). Outputs: solvers/out/r5_wft/{r5_wft.json,r5_wft.log}.
"""
import json
import os
import sys

for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = "<repo>"
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/solvers/ensemble")
from stack_val import feed_rank_z  # noqa: E402
from ens_common import Split  # noqa: E402

OUT = f"{ROOT}/solvers/out/r5_wft"
P = f"{ROOT}/preds"
FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1", "v14_s92", "v14_s93", "v14_s94", "v14_s95", "v14_s96"],
    "F7": [f"v17_pre2s_s{s}" for s in range(101, 108)],
}
MEM = [m for f in FAM.values() for m in f]
GRID = [0.33, 0.5, 0.67]
CD_VAL = 0.39901  # CURRENT_STATUS 3o / SOLUTIONS_REVIEW 4.1

S = Split("val")
off32, M = S.off, S.M


def z(x):
    return feed_rank_z(off32, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)


def branch(ms, kind):
    suf = "n2cpv3_val_oof" if kind == "ft" else "trainonly_val"
    return z(np.mean([z(np.load(f"{P}/nn_{m}_{suf}.npy")) for m in ms], 0))


def se(d):
    return d.mean(), d.std() / np.sqrt(len(d))


def se_user(d, user):
    """User-clustered SE of the mean (user 0 feeds treated as singletons)."""
    g = np.where(user == 0, -1 - np.arange(len(user)), user)
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, d - d.mean())
    n = len(d)
    G = len(s)
    return np.sqrt(G / (G - 1) * (s ** 2).sum()) / n


def slices(d):
    w = S.private_w("private")
    masks = {"all": np.ones(S.n, bool), "M<50": S.M < 50, "M>=50": S.M >= 50, "Tue": S.day == 0, "Thu": S.day == 1}
    out = {k: se(d[m]) for k, m in masks.items()}
    x = w / w.mean() * d
    out["priv_w"] = (x.mean(), x.std() / np.sqrt(len(d)))
    return out


def fmt(sl):
    return "  ".join(f"{k} {m:+.5f}+-{s:.5f} (z {m / max(s, 1e-12):+.2f})" for k, (m, s) in sl.items())


def folds(u0_singleton):
    g = S.user.copy()
    if u0_singleton:
        u0 = g == 0
        g = np.where(u0, -1 - np.arange(S.n), g)
    fold = np.full(S.n, -1)
    for k, (_, ho) in enumerate(GroupKFold(n_splits=5).split(np.arange(S.n), groups=g)):
        fold[ho] = k
    return fold


def nested(per, fold):
    res = np.zeros(S.n)
    chosen, tab = [], []
    for k in range(5):
        tr, te = fold != k, fold == k
        sc = {w: per[w][tr].mean() for w in GRID}
        best_val = max(sc.values())
        cands = [w for w in GRID if sc[w] == best_val]
        best = 0.5 if 0.5 in cands else min(cands)
        chosen.append(best)
        res[te] = per[best][te]
        tab.append({"fold": k, "n_feeds": int(te.sum()), "train_scores": {str(w): float(v) for w, v in sc.items()},
                    "train_delta_vs_0.5": {str(w): float(sc[w] - sc[0.5]) for w in GRID},
                    "heldout_delta_vs_0.5": {str(w): float(per[w][te].mean() - per[0.5][te].mean()) for w in GRID},
                    "chosen": best})
    return res, chosen, tab


def main():
    log = []

    def p(s=""):
        print(s, flush=True)
        log.append(s)

    rep = {"members": MEM, "grid": GRID}
    zf, zt = branch(MEM, "ft"), branch(MEM, "to")
    per = {}
    for w in sorted(set(GRID) | {round(0.1 * i, 1) for i in range(11)}):
        per[w] = S.per_feed(w * zf + (1 - w) * zt)
    p(f"Cd s(0.5) val {per[0.5].mean():.5f} (expected {CD_VAL}); ft branch {per[1.0].mean():.5f}; to branch {per[0.0].mean():.5f}")
    p(f"n_feeds {S.n}; user-0 feeds {(S.user == 0).sum()}; Tue {(S.day == 0).sum()} Thu {(S.day == 1).sum()}")
    rep["sanity"] = {"cd_val": float(per[0.5].mean()), "expected": CD_VAL, "ft_only": float(per[1.0].mean()), "to_only": float(per[0.0].mean())}
    assert abs(per[0.5].mean() - CD_VAL) < 6e-6, "Cd reproduction failed"

    for name, u0s in [("PRIMARY (GroupKFold by user_id, user 0 = one group)", False),
                      ("ROBUSTNESS (n2cpv3 fine-tune folds, user-0 singletons)", True)]:
        fold = folds(u0s)
        res, chosen, tab = nested(per, fold)
        d = res - per[0.5]
        m, s = se(d)
        su = se_user(d, S.user)
        p(f"\n== {name}")
        p(f"fold sizes {np.bincount(fold).tolist()}  user-0 feeds per fold {np.bincount(fold[S.user == 0], minlength=5).tolist()}")
        for t in tab:
            p(f"  fold {t['fold']} n={t['n_feeds']:5d}  train d(0.33) {t['train_delta_vs_0.5']['0.33']:+.5f}  d(0.67) {t['train_delta_vs_0.5']['0.67']:+.5f}"
              f"  -> chosen {t['chosen']}   | held-out d(0.33) {t['heldout_delta_vs_0.5']['0.33']:+.5f} d(0.67) {t['heldout_delta_vs_0.5']['0.67']:+.5f}")
        sl = slices(d)
        p(f"nested score {res.mean():.5f} vs w=0.5 {per[0.5].mean():.5f}: delta {m:+.5f} +- {s:.5f} (z {m / max(s, 1e-12):+.2f}); user-clustered SE {su:.5f}")
        p(f"  slices: {fmt(sl)}")
        same = len(set(chosen)) == 1 and chosen[0] != 0.5
        adopt = same and (m / max(s, 1e-12) >= 2.5)
        p(f"  chosen per fold {chosen}; all-5-same-and-not-0.5: {same}; nested z >= 2.5: {m / max(s, 1e-12) >= 2.5} -> ADOPT: {adopt}")
        rep["primary" if not u0s else "robust_u0singleton"] = {
            "chosen": chosen, "folds": tab, "nested_score": float(res.mean()), "delta": float(m), "se": float(s),
            "se_user": float(su), "z": float(m / max(s, 1e-12)), "slices": {k: [float(a), float(b)] for k, (a, b) in sl.items()},
            "all5_same_not_05": bool(same), "adopt": bool(adopt)}

    p("\n== INFORMATION ONLY: in-sample full-val fixed-w deltas vs w=0.5 (never used for the decision)")
    rep["insample"] = {}
    for w in sorted(per):
        d = per[w] - per[0.5]
        sl = slices(d)
        rep["insample"][str(w)] = {"ndcg": float(per[w].mean()), "slices": {k: [float(a), float(b)] for k, (a, b) in sl.items()}}
        if w in (0.0, 0.33, 0.67, 1.0) or abs(w * 10 - round(w * 10)) < 1e-9:
            p(f"  w {w:4.2f}  ndcg {per[w].mean():.5f}  {fmt(sl)}")

    p("\n== INFORMATION ONLY: per-family in-sample fixed-w deltas vs w=0.5")
    rep["family"] = {}
    for f, ms in FAM.items():
        ff, ft_ = branch(ms, "ft"), branch(ms, "to")
        pf = {w: S.per_feed(w * ff + (1 - w) * ft_) for w in [0.0, 0.33, 0.5, 0.67, 1.0]}
        rep["family"][f] = {str(w): float(v.mean()) for w, v in pf.items()}
        p(f"  {f:5s} w0.5 {pf[0.5].mean():.5f} | " + "  ".join(f"d(w={w}) {(pf[w] - pf[0.5]).mean():+.5f}+-{se(pf[w] - pf[0.5])[1]:.5f}"
                                                         for w in [0.0, 0.33, 0.67, 1.0]))

    with open(f"{OUT}/r5_wft.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    with open(f"{OUT}/r5_wft.log", "w") as fh:
        fh.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
