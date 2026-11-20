"""(4) Pre-declared final-design candidates built from the families that exist now, evaluated on val (paired vs v11,
standard slices + calib private weights + adversarial private-like weights), on the holdout (train-only branch) and by
test-side similarity to v11 (mean per-feed Pearson of z and Spearman). Writes candidate test/val score files to
solvers/out/ensemble/ (never to preds/).

  D0 v11 = mix(F5)                       mix(X) = 0.5 z(mean z ft) + 0.5 z(mean z to)
  D1 eq(F5, F5cd)
  D2 eq(F5, V13)
  D3 eq(F5, F5cd, V13)                   family-equal
  D4 member-equal over F5+F5cd+V13 (15 members per branch), 0.5/0.5 branches
  D5 eq(F5, F5cd, V13, F4)
"""
import json
import sys

import numpy as np

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Holdout, Split, fmt_slices, slices  # noqa: E402
from exp3_diversity import F4, F5, F5CD, V13  # noqa: E402

OUT = f"{ROOT}/solvers/out/ensemble"
POOLS = {"F5": (F5, "n2cpf"), "F5cd": (F5CD, "n2cpf"), "V13": (V13, "n2cpf"), "F4": (F4, "n2cp")}


def p(n):
    return np.load(f"{ROOT}/preds/{n}.npy").astype(np.float64)


def branches(S, split, k):
    ms, suf = POOLS[k]
    sf, st = ("val_oof", "trainonly_val") if split == "val" else ("test", "trainonly_test")
    return [S.z(p(f"nn_{m}_{suf}_{sf}")) for m in ms], [S.z(p(f"nn_{m}_{st}")) for m in ms]


def build(S, split):
    B = {k: branches(S, split, k) for k in POOLS}
    mix = {k: S.z(0.5 * S.z(np.mean(B[k][0], 0)) + 0.5 * S.z(np.mean(B[k][1], 0))) for k in POOLS}
    D = {"D0_v11": mix["F5"],
         "D1_eq_F5_F5cd": mix["F5"] + mix["F5cd"],
         "D2_eq_F5_V13": mix["F5"] + mix["V13"],
         "D3_eq_F5_F5cd_V13": mix["F5"] + mix["F5cd"] + mix["V13"]}
    ft = B["F5"][0] + B["F5cd"][0] + B["V13"][0]
    to = B["F5"][1] + B["F5cd"][1] + B["V13"][1]
    D["D4_member_eq15"] = 0.5 * S.z(np.mean(ft, 0)) + 0.5 * S.z(np.mean(to, 0))
    D["D5_eq_F5_F5cd_V13_F4"] = D["D3_eq_F5_F5cd_V13"] + mix["F4"]
    return {k: S.z(v) for k, v in D.items()}


def spearman_pf(S, a, b):
    ra, rb = S.z(S.rank01(a)), S.z(S.rank01(b))
    return float((np.bincount(S.fid, ra * rb, S.n) / S.M).mean())


def main():
    rep = {}
    Sv, St = Split("val"), Split("test")
    Dv, Dt = build(Sv, "val"), build(St, "test")
    adv = np.load(f"{OUT}/adv_w_private_val.npy").astype(np.float64)
    base = Sv.per_feed(Dv["D0_v11"])
    H = Holdout()
    hz = lambda ms: H.z(np.mean([H.z(H.load(m)) for m in ms], 0))  # noqa: E731
    hm = {"F5": hz(F5), "F5cd": hz(F5CD), "V13": hz(V13), "F4": hz(F4)}
    hD = {"D0_v11": hm["F5"], "D1_eq_F5_F5cd": hm["F5"] + hm["F5cd"], "D2_eq_F5_V13": hm["F5"] + hm["V13"],
          "D3_eq_F5_F5cd_V13": hm["F5"] + hm["F5cd"] + hm["V13"],
          "D4_member_eq15": np.mean([H.z(H.load(m)) for m in F5 + F5CD + V13], 0),
          "D5_eq_F5_F5cd_V13_F4": hm["F5"] + hm["F5cd"] + hm["V13"] + hm["F4"]}
    hb = H.per_feed(hD["D0_v11"])
    for k in Dv:
        pv = Sv.per_feed(Dv[k])
        d = pv - base
        sl = slices(Sv, d)
        x = adv * d
        hd = H.per_feed(hD[k]) - hb
        pr = Sv.private_w("private")
        rep[k] = {"val": float(pv.mean()), "delta": {a: [float(u) for u in b] for a, b in sl.items()},
                  "adv_w": [float(x.mean()), float(x.std() / np.sqrt(Sv.n))],
                  "holdout_to": float(H.per_feed(hD[k]).mean()), "holdout_delta": [float(hd.mean()), float(hd.std() / np.sqrt(H.n))],
                  "holdout_late": [float(hd[H.late].mean()), float(hd[H.late].std() / np.sqrt(H.late.sum()))],
                  "pred_private": float((pr * pv).mean()), "pred_public": float((Sv.private_w("public") * pv).mean()),
                  "test_pearson_vs_v11": float((np.bincount(St.fid, Dt[k] * Dt["D0_v11"], St.n) / St.M).mean()),
                  "test_spearman_vs_v11": spearman_pf(St, Dt[k], Dt["D0_v11"]),
                  "delta_sd_per_feed": float(d.std())}
        r = rep[k]
        print(f"{k:22s} val {r['val']:.5f}  {fmt_slices(sl)}  adv {r['adv_w'][0]:+.5f}+-{r['adv_w'][1]:.5f}")
        print(f"{'':22s} holdout(to) {r['holdout_to']:.5f} d {r['holdout_delta'][0]:+.5f}+-{r['holdout_delta'][1]:.5f} "
              f"late {r['holdout_late'][0]:+.5f}+-{r['holdout_late'][1]:.5f}  calib pub/priv {r['pred_public']:.5f}/{r['pred_private']:.5f}  "
              f"test pearson {r['test_pearson_vs_v11']:.4f} spearman {r['test_spearman_vs_v11']:.4f}")
        assert np.isfinite(Dt[k]).all() and len(Dt[k]) == St.N
        np.save(f"{OUT}/cand_{k}_test.npy", Dt[k].astype(np.float32))
        np.save(f"{OUT}/cand_{k}_val.npy", Dv[k].astype(np.float32))
    json.dump(rep, open(f"{OUT}/exp8.json", "w"), indent=1)


if __name__ == "__main__":
    main()
