"""(1b) Combiners on the train holdout (independent labels, train-only members), family-vs-member weighting on val,
and replication of the M-segment fine-tune-share effect on other member pools.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Holdout, Split, fmt_slices, slices  # noqa: E402
from exp1_combiners import nested_segment  # noqa: E402
from exp3_diversity import F4, F5, F5CD, OLD, V13  # noqa: E402


def p(n):
    return np.load(f"{ROOT}/preds/{n}.npy").astype(np.float64)


def main():
    rep = {}
    # ---------------- (a) holdout combiners ----------------
    H = Holdout()
    hz, hr = {}, {}
    raw = {}
    for m in F5 + F4 + F5CD + V13 + OLD:
        raw[m] = H.load(m)
        hz[m] = H.z(raw[m])
    def hs(d, base):  # noqa: E301
        return f"{d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f} late {d[H.late].mean():+.5f}+-{d[H.late].std() / np.sqrt(H.late.sum()):.5f}"
    zF5 = np.mean([hz[m] for m in F5], 0)
    b = H.per_feed(zF5)
    print(f"holdout F5_to z-mean {b.mean():.5f}")
    rep["holdout"] = {"F5_z": float(b.mean())}
    cands = {
        "rank_mean": np.mean([H.rank01(raw[m]) for m in F5], 0),
        "raw_logit": np.mean([raw[m] for m in F5], 0),
        "softmax_prob": np.mean([H.softmax(raw[m]) for m in F5], 0),
    }
    for k, s in cands.items():
        d = H.per_feed(s) - b
        rep["holdout"][k] = [float(d.mean()), float(d.std() / np.sqrt(H.n))]
        print(f"  holdout F5 {k:14s} {hs(d, b)}")
    # family-equal vs member-equal and cross-kind additions on the holdout
    for name, other in [("F4", F4), ("F5cd", F5CD), ("V13", V13), ("OLD5ep", OLD)]:
        zO = np.mean([hz[m] for m in other], 0)
        fam = H.z(zF5) + H.z(zO)
        mem = np.mean([hz[m] for m in F5 + other], 0)
        df, dm = H.per_feed(fam) - b, H.per_feed(mem) - b
        po = H.per_feed(zO).mean()
        rep["holdout"][f"F5+{name}"] = {"other_alone": float(po), "family_eq": [float(df.mean()), float(df.std() / np.sqrt(H.n))],
                                       "member_eq": [float(dm.mean()), float(dm.std() / np.sqrt(H.n))]}
        print(f"  holdout F5 + {name:7s} ({len(other)} m, alone {po:.5f}): family-equal {hs(df, b)} | member-equal {hs(dm, b)}")
    # ---------------- (b) val: family vs member weighting ----------------
    S = Split("val")
    def mix_members(ms, ft_suf):  # noqa: E301
        zf = [S.z(p(f"nn_{m}_{ft_suf}_val_oof")) for m in ms]
        zt = [S.z(p(f"nn_{m}_trainonly_val")) for m in ms]
        return zf, zt
    pools = {"F5": (F5, "n2cpf"), "F4": (F4, "n2cp"), "F5cd": (F5CD, "n2cpf"), "V13": (V13, "n2cpf"), "OLD5ep": (OLD, "n2cp")}
    Z = {k: mix_members(*v) for k, v in pools.items()}
    def fmix(k):  # noqa: E301
        zf, zt = Z[k]
        return 0.5 * S.z(np.mean(zf, 0)) + 0.5 * S.z(np.mean(zt, 0))
    v11 = fmix("F5")
    base = S.per_feed(v11)
    print(f"\nval v11 {base.mean():.5f}")
    rep["val"] = {}
    for k in pools:
        if k == "F5":
            continue
        fam = S.z(v11) + S.z(fmix(k))
        zf = Z["F5"][0] + Z[k][0]
        zt = Z["F5"][1] + Z[k][1]
        mem = 0.5 * S.z(np.mean(zf, 0)) + 0.5 * S.z(np.mean(zt, 0))
        for lab, s in [("family_eq", fam), ("member_eq", mem)]:
            d = S.per_feed(s) - base
            sl = slices(S, d)
            rep["val"][f"F5+{k}:{lab}"] = {a: [float(x) for x in bb] for a, bb in sl.items()}
            print(f"  F5+{k:6s} {lab:9s} {base.mean() + d.mean():.5f}  {fmt_slices(sl)}")
        print(f"     ({k} mix alone {S.per_feed(fmix(k)).mean():.5f})")
    allfam = sum(S.z(fmix(k)) for k in ["F5", "F5cd", "V13", "F4"])
    d = S.per_feed(allfam) - base
    print(f"  eq(F5,F5cd,V13,F4) {base.mean() + d.mean():.5f}  {fmt_slices(slices(S, d))}")
    # ---------------- (c) M-segment w_ft replication ----------------
    print("\nM-segment fine-tune share replication (w_ft curve on M<50 / M>=50, nested S1 delta vs w=0.5)")
    rep["mseg"] = {}
    pools2 = dict(pools)
    pools2["F5_n2cp"] = (F5, "n2cp")
    pools2["F5_n2cpv2"] = (F5, "n2cpv2")
    for k, (ms, suf) in pools2.items():
        zf, zt = mix_members(ms, suf)
        Zf, Zt = S.z(np.mean(zf, 0)), S.z(np.mean(zt, 0))
        cur = {}
        for w in [0.0, 0.33, 0.5, 0.67, 1.0]:
            pp = S.per_feed(w * Zf + (1 - w) * Zt)
            cur[w] = (float(pp[S.M < 50].mean()), float(pp[S.M >= 50].mean()))
        b5 = S.per_feed(0.5 * Zf + 0.5 * Zt)
        res, chosen, _ = nested_segment(S, Zf, Zt, (S.M >= 50).astype(int), [0.33, 0.5, 0.67])
        dd = res - b5
        dft = S.per_feed(Zf) - S.per_feed(Zt)
        rep["mseg"][k] = {"curve": {str(w): v for w, v in cur.items()}, "nested_S1": [float(dd.mean()), float(dd.std() / np.sqrt(S.n))],
                          "ft_minus_to_M<50": [float(dft[S.M < 50].mean()), float(dft[S.M < 50].std() / np.sqrt((S.M < 50).sum()))],
                          "ft_minus_to_M>=50": [float(dft[S.M >= 50].mean()), float(dft[S.M >= 50].std() / np.sqrt((S.M >= 50).sum()))],
                          "chosen": chosen}
        print(f"  {k:10s} ft-to M<50 {dft[S.M < 50].mean():+.5f}+-{dft[S.M < 50].std() / np.sqrt((S.M < 50).sum()):.5f}  "
              f"M>=50 {dft[S.M >= 50].mean():+.5f}+-{dft[S.M >= 50].std() / np.sqrt((S.M >= 50).sum()):.5f}  "
              f"nested S1 {dd.mean():+.5f}+-{dd.std() / np.sqrt(S.n):.5f} chosen {[list(c.values()) for c in chosen]}")
    json.dump(rep, open(f"{ROOT}/solvers/out/ensemble/exp4.json", "w"), indent=1)


if __name__ == "__main__":
    main()
