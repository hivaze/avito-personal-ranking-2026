"""Information only (no selection): why the phase-A pools gain less over v11 than sol_ensemble's D4, and what the n2cpf
fine-tune halves of F5cd/H contribute. Reads preds/ and the phase-A cand files; writes info.json/info.log in the out dir.

Rows (member-equal, 0.5 z(FT mean) + 0.5 z(TO mean)):
  D4f    = F5 + F5cd + V13 with every FT = n2cpf (reproduces sol_ensemble D4, 0.39908)
  D4v3   = same members, FT = n2cpv3 where it exists (F5, v13_lr1_mix1), else n2cpf
  D4v3+  = D4v3 + v14_mix1_s91 (= Q3 without v14_s92..s96)
  Q3     = phase-A choice (F5+F6+F5cd+H), Q3h = Q3 with FT branch from n2cpv3 members only
  famQ1/2/3 = family-equal z-mean of the family mixes (the lead's family-equal form); holdout = family-equal TO halves
"""
import json
import os
import sys

for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import HOLD, PREDS, Holdout, Split  # noqa: E402

OUT = "<repo>/solvers/out/CPU1_xrecipe_member_equal_pool_v3"
F5 = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]
F6 = ["v13_lr1_mix1", "v14_s92", "v14_s93", "v14_s94", "v14_s95", "v14_s96"]
F5CD = ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"]
H = ["v13_lr1_mix0", "v13_lr2_mix1", "v14_mix1_s91"]
V13 = ["v13_lr1_mix1", "v13_lr2_mix1", "v13_lr1_mix0"]

Sv, Ho = Split("val"), Holdout()
c = {}


def z(stem):
    if stem not in c:
        c[stem] = Sv.z(np.load(f"{PREDS}/{stem}.npy").astype(np.float64))
    return c[stem]


def hz(m):
    k = ("h", m)
    if k not in c:
        c[k] = Ho.z(np.load(f"{HOLD}/nn_{m}_holdout.npy").astype(np.float64))
    return c[k]


def ftstem(m, pref):
    for suf in pref:
        if os.path.exists(f"{PREDS}/nn_{m}_{suf}_val_oof.npy"):
            return f"nn_{m}_{suf}_val_oof"
    raise FileNotFoundError(m)


def mix(ft_members, to_members, pref=("n2cpv3", "n2cpf")):
    return Sv.z(0.5 * Sv.z(np.mean([z(ftstem(m, pref)) for m in ft_members], 0)) +
                0.5 * Sv.z(np.mean([z(f"nn_{m}_trainonly_val") for m in to_members], 0)))


def hmix(members):
    return Ho.per_feed(Ho.z(np.mean([hz(m) for m in members], 0)))


rows = {
    "R": (Sv.z(0.5 * z("nn_final5v3_val_oof") + 0.5 * z("nn_final5_trainonly_val")), F5),
    "v11": (Sv.z(0.5 * z("nn_final5_val_oof") + 0.5 * z("nn_final5_trainonly_val")), F5),
    "D4f": (mix(F5 + F5CD + V13, F5 + F5CD + V13, ("n2cpf",)), F5 + F5CD + V13),
    "D4v3": (mix(F5 + F5CD + V13, F5 + F5CD + V13), F5 + F5CD + V13),
    "D4v3+s91": (mix(F5 + F5CD + V13 + ["v14_mix1_s91"], F5 + F5CD + V13 + ["v14_mix1_s91"]), F5 + F5CD + V13 + ["v14_mix1_s91"]),
    "Q1": (np.load(f"{OUT}/cand_A_Q1_val.npy").astype(np.float64), F5 + F6),
    "Q3": (np.load(f"{OUT}/cand_A_Q3_val.npy").astype(np.float64), F5 + F6 + F5CD + H),
    "Q3h": (np.load(f"{OUT}/cand_A_Q3h_val.npy").astype(np.float64), F5 + F6 + F5CD + H),
}
fam = {k: mix(v, v) for k, v in {"F5": F5, "F6": F6, "F5cd": F5CD, "H": H}.items()}
rows["famQ1"] = (Sv.z(fam["F5"] + fam["F6"]), F5 + F6)
rows["famQ2"] = (Sv.z(fam["F5"] + fam["F6"] + fam["F5cd"]), F5 + F6 + F5CD)
rows["famQ3"] = (Sv.z(sum(fam.values())), F5 + F6 + F5CD + H)
rows["Q2"] = (np.load(f"{OUT}/cand_A_Q2_val.npy").astype(np.float64), F5 + F6 + F5CD)
FAMS = {"famQ1": [F5, F6], "famQ2": [F5, F6, F5CD], "famQ3": [F5, F6, F5CD, H]}  # family-equal holdout TO halves
rows["F6mix"] = (fam["F6"], F6)
rows["F5cd+H (no F5/F6)"] = (mix(F5CD + H, F5CD + H), F5CD + H)
SL = {"all": np.ones(Sv.n, bool), "M<50": Sv.M < 50, "M>=50": Sv.M >= 50, "Tue": Sv.day == 0, "Thu": Sv.day == 1}
pf = {k: Sv.per_feed(v[0]) for k, v in rows.items()}
ho = {k: hmix(v[1]) for k, v in rows.items()}
for k, fl in FAMS.items():
    ho[k] = Ho.per_feed(Ho.z(sum(Ho.z(np.mean([hz(m) for m in f], 0)) for f in fl)))
out = {}
pairs = [("D4f", "v11"), ("D4v3", "v11"), ("D4v3", "R"), ("D4v3+s91", "R"), ("Q3", "R"), ("Q3", "D4v3+s91"), ("Q3", "D4v3"),
         ("Q3", "Q3h"), ("Q3", "Q1"), ("famQ3", "R"), ("famQ3", "Q3"), ("famQ1", "Q1"), ("famQ2", "Q2"), ("F6mix", "R"), ("F5cd+H (no F5/F6)", "R"),
         ("F5cd+H (no F5/F6)", "Q3")]
lines = []
for k in rows:
    lines.append(f"{k:18s} val {pf[k].mean():.5f}  holdout TO-half {ho[k].mean():.5f}  n_members {len(rows[k][1])}")
for a, b in pairs:
    d = pf[a] - pf[b]
    hd = ho[a] - ho[b]
    o = {s: [float(d[m].mean()), float(d[m].std() / np.sqrt(m.sum()))] for s, m in SL.items()}
    o["holdout"] = [float(hd.mean()), float(hd.std() / np.sqrt(len(hd)))]
    o["holdout_late20"] = [float(hd[Ho.late].mean()), float(hd[Ho.late].std() / np.sqrt(Ho.late.sum()))]
    out[f"{a} - {b}"] = o
    lines.append(f"{a:18s} - {b:9s} " + "  ".join(f"{s} {o[s][0]:+.5f}+-{o[s][1]:.5f}" for s in SL) +
                 f"  | holdout {o['holdout'][0]:+.5f}+-{o['holdout'][1]:.5f} late {o['holdout_late20'][0]:+.5f}")
# family-mix correlations on val (per-feed Pearson of z)
fc = {}
ks = list(fam)
for i, x in enumerate(ks):
    for y in ks[i + 1:]:
        fc[f"{x}|{y}"] = float((np.bincount(Sv.fid, fam[x] * fam[y], Sv.n) / Sv.M).mean())
lines.append("family-mix per-feed corr (val): " + "  ".join(f"{k} {v:.4f}" for k, v in fc.items()))
lines.append("family-mix val: " + "  ".join(f"{k} {Sv.per_feed(v).mean():.5f}" for k, v in fam.items()))
print("\n".join(lines))
open(f"{OUT}/info.log", "w").write("\n".join(lines) + "\n")
json.dump({"levels": {k: [float(pf[k].mean()), float(ho[k].mean())] for k in rows}, "pairs": out, "family_corr": fc},
          open(f"{OUT}/info.json", "w"), indent=1)
