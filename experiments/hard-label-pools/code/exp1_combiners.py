# Compares 8 ways to combine the 12 predictions of one 6-member family (per-feed z-mean, rank mean,
# raw-logit mean, softmax-probability mean, reciprocal rank fusion, Gaussianised ranks, a heavier
# fine-tune weight) and 5 schemes of segment-specific fine-tune weights chosen in nested user folds.
# How it was run: CPU, 22 s, 2026-11-11 17:49 UTC, from lab/solvers/ensemble; the command line is
# not on record.
# What it produced: solvers/out/ensemble/exp1.{log,json}; the log is results/combiners.txt (the
# z-mean ties the logit and probability means; the rank mean loses 0.0020 +- 0.00035 on val).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/solvers/ensemble/exp1_combiners.py
"""(1) Combiners for the v11 member pool (nn_final5: 6 fta fine-tuned + 6 train-only members) on val.

Pre-declared candidates (all equal weight unless stated), each compared paired per feed with v11 (A):
  A  v11: 0.5 z(mean z ft) + 0.5 z(mean z to)            (current)
  B  per-member z-mean (12 members, no family re-z)
  C  rank-mean: 0.5 mean rank01(ft) + 0.5 mean rank01(to)
  D  raw-logit mean of the 12 members
  E  softmax-probability mean of the 12 members (T=1 on raw logits)
  F  v11 with w_ft = 0.67
  G  reciprocal-rank fusion (k=60) of the 12 members
  H  v11 recipe applied with per-family z but member-level rank->z (Gaussianised ranks)
Segment-specific w_ft in {0.33, 0.5, 0.67} per segment, chosen by nested user-fold CV (folds of nn/finetune.py):
  S1 M<50 vs M>=50; S2 cold-share terciles (label-free, LUO popularity == 0); S3 user0 vs known; S4 M x cold (2x2)
  S0 global w_ft from {0.3,0.4,0.5,0.6,0.7} chosen in nested CV (control).
Outputs a table to stdout + solvers/out/ensemble/exp1.json
"""
import json
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Split, fmt_slices, slices  # noqa: E402

MEM = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]


def family_members(split, suffix_ft="n2cpf"):
    sv = "val_oof" if split == "val" else "test"
    st = "trainonly_val" if split == "val" else "trainonly_test"
    ft = [np.load(f"{ROOT}/preds/nn_{m}_{suffix_ft}_{sv}.npy").astype(np.float64) for m in MEM]
    to = [np.load(f"{ROOT}/preds/nn_{m}_{st}.npy").astype(np.float64) for m in MEM]
    return ft, to


def combos(S, ft, to):
    zf = [S.z(x) for x in ft]
    zt = [S.z(x) for x in to]
    Zf, Zt = S.z(np.mean(zf, 0)), S.z(np.mean(zt, 0))
    out = {}
    out["A_v11_z"] = 0.5 * Zf + 0.5 * Zt
    out["B_member_z"] = np.mean(zf + zt, 0)
    rf = [S.rank01(x) for x in ft]
    rt = [S.rank01(x) for x in to]
    out["C_rank_mean"] = 0.5 * np.mean(rf, 0) + 0.5 * np.mean(rt, 0)
    out["D_raw_logit"] = np.mean(ft + to, 0)
    out["E_softmax_prob"] = np.mean([S.softmax(x) for x in ft + to], 0)
    out["F_wft067"] = 0.67 * Zf + 0.33 * Zt
    # RRF: rank from the top (1-based); rank01 is 1 for the best
    rrf = 0
    for r in rf + rt:
        top = (1 - r) * np.maximum(S.M[S.fid] - 1, 1) + 1
        rrf = rrf + 1.0 / (60 + top)
    out["G_rrf60"] = rrf
    gf = [norm.ppf((r * (S.M[S.fid] - 1) + 0.5) / S.M[S.fid]) for r in rf]
    gt = [norm.ppf((r * (S.M[S.fid] - 1) + 0.5) / S.M[S.fid]) for r in rt]
    out["H_gauss_rank"] = 0.5 * S.z(np.mean(gf, 0)) + 0.5 * S.z(np.mean(gt, 0))
    return out, Zf, Zt


def nested_segment(S, Zf, Zt, seg, grid):
    """seg: int array per feed (segment id). Returns nested per-feed NDCG and chosen weights per fold."""
    per = {w: S.per_feed(w * Zf + (1 - w) * Zt) for w in grid}
    res = np.zeros(S.n)
    chosen = []
    for k in range(5):
        tr, te = S.fold != k, S.fold == k
        ck = {}
        for s in np.unique(seg):
            m = seg == s
            best = max(grid, key=lambda w: per[w][tr & m].mean())
            ck[int(s)] = best
            res[te & m] = per[best][te & m]
        chosen.append(ck)
    return res, chosen, per


def main():
    S = Split("val")
    ft, to = family_members("val")
    for m, a, b in zip(MEM, ft, to):
        print(f"  member {m}: ft {S.per_feed(a).mean():.5f} to {S.per_feed(b).mean():.5f}")
    out, Zf, Zt = combos(S, ft, to)
    v11_file = np.load(f"{ROOT}/preds/nn_final5_val_oof.npy"), np.load(f"{ROOT}/preds/nn_final5_trainonly_val.npy")
    v11f = 0.5 * S.z(v11_file[0]) + 0.5 * S.z(v11_file[1])
    base = S.per_feed(out["A_v11_z"])
    print(f"A recomputed {base.mean():.5f}  from nn_final5 files {S.per_feed(v11f).mean():.5f}")
    rep = {"members": MEM, "combiners": {}, "segments": {}}
    for k, s in out.items():
        p = S.per_feed(s)
        sl = slices(S, p - base)
        rep["combiners"][k] = {"ndcg": float(p.mean()), "delta": {a: [float(x) for x in b] for a, b in sl.items()}}
        print(f"{k:16s} {p.mean():.5f}  {fmt_slices(sl)}")
    # segment-specific weights in nested CV
    grid3 = [0.33, 0.5, 0.67]
    cs = S.cold_share
    terc = np.digitize(cs, np.quantile(cs, [1 / 3, 2 / 3]))
    segs = {
        "S0_global_5grid": (np.zeros(S.n, int), [0.3, 0.4, 0.5, 0.6, 0.7]),
        "S0b_global_3grid": (np.zeros(S.n, int), grid3),
        "S1_M50": ((S.M >= 50).astype(int), grid3),
        "S2_cold_terciles": (terc, grid3),
        "S3_user0": ((S.user == 0).astype(int), grid3),
        "S4_Mx_cold": ((S.M >= 50).astype(int) * 2 + (cs > np.median(cs)).astype(int), grid3),
    }
    for name, (seg, grid) in segs.items():
        res, chosen, per = nested_segment(S, Zf, Zt, seg, grid)
        sl = slices(S, res - base)
        # in-sample (optimistic) choice on all val for reference
        ins = np.zeros(S.n)
        for s in np.unique(seg):
            m = seg == s
            best = max(grid, key=lambda w: per[w][m].mean())
            ins[m] = per[best][m]
        rep["segments"][name] = {"nested": float(res.mean()), "insample": float(ins.mean()), "chosen": chosen,
                                 "delta": {a: [float(x) for x in b] for a, b in sl.items()}}
        print(f"{name:18s} nested {res.mean():.5f} (in-sample {ins.mean():.5f})  {fmt_slices(sl)}")
        print(f"   chosen per fold: {chosen}")
    # per-segment curves of w_ft (descriptive)
    for name, m in {"M<50": S.M < 50, "M>=50": S.M >= 50, "Tue": S.day == 0, "Thu": S.day == 1,
                    "cold_lo": terc == 0, "cold_mid": terc == 1, "cold_hi": terc == 2, "user0": S.user == 0}.items():
        curve = {}
        for w in [0.0, 0.33, 0.5, 0.67, 1.0]:
            p = S.per_feed(w * Zf + (1 - w) * Zt)
            curve[w] = float(p[m].mean())
        rep.setdefault("wft_curves", {})[name] = curve
        print(f"  w_ft curve {name:8s} " + "  ".join(f"{w}:{v:.5f}" for w, v in curve.items()))
    json.dump(rep, open(f"{ROOT}/solvers/out/ensemble/exp1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
