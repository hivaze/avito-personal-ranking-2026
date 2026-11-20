"""why/fit: the model list of fwd_train.py with the stored val / holdout prediction paths of the same (selected) checkpoints (CPU import,
no torch / CUDA)."""
from pathlib import Path

ROOT = Path("<repo>")
MOE = ROOT / "lab/nn_x/w12/moe/out"
W5 = ROOT / "nn_x/w5"
FAMS = [("kd05w192", range(951, 959)), ("kd05w192moeffn", range(951, 955)), ("kd05w192moevo", range(951, 955)),
        ("kd05w192b3ff1", range(951, 955)), ("kd05", range(951, 955)), ("kd05w224", range(951, 955)), ("kd08w192", range(951, 955)),
        ("kwd2", range(951, 955)), ("kw256d2", range(951, 955)), ("kwh6", range(951, 955)), ("kw_ff4", (951, 952)),
        ("kw_hl2", (951, 952)), ("kw_drop0", (951, 952)), ("kw_e20", (951, 952)), ("base", range(951, 955))]


def pred_list(suf=""):
    out = []
    for fam, seeds in FAMS:
        for s in seeds:
            v = f"w5_e10_{fam}_s{s}"
            B = MOE if ("moe" in fam or "b3ff1" in fam) else W5
            out.append((v + suf, B / "preds" / f"nn_{v}{suf}_trainonly_val.npy", B / "preds/holdout" / f"nn_{v}{suf}_holdout.npy"))
    return out
