"""Family / arm definitions and curve helpers for the dynamics analysis (CPU, logs only)."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("<repo>/nn_x/insights/dynamics")
T_COOL = 78680          # cooldown start of every 10-epoch run (8.0 epochs x 9835 steps)
K_COOL = 19670          # cooldown length (2 epochs)
SNAP = {77500: 1180, 70000: 8680, 60000: 18680}  # snapshot step -> stable steps the original still ran before t_cool


def load():
    C = pd.read_csv(OUT / "curves.csv")
    R = pd.read_csv(OUT / "runs.csv").set_index("run")
    return C, R


def seeds(prefix, ids):
    return [f"{prefix}{i}" for i in ids]


FAM10 = {  # 10-epoch from-scratch runs, eval every 2500, cooldown = last 2 epochs (steps > 78680)
    "F5 cvnd lr2 mix.3": ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84",
                          "v12_allprew_s85"],
    "F6 cvnd lr1 mix.1": ["v13_lr1_mix1"] + seeds("v14_s", range(92, 97)),
    "cvnd lr2 mix.1": ["v13_lr2_mix1", "v14_mix1_s91"],
    "cvnd lr1 mix0 (s23)": ["v13_lr1_mix0"],
    "cvnd lr1 mix.3 (s23)": ["v13_lr1_mix3"],
    "F7 pre2s lr2 mix.1": seeds("v17_pre2s_s", range(101, 108)),
    "F9 pre2s lr1 mix.1": seeds("v22_pre2s_lr1_s", range(401, 407)),
    "F10 GRU lr2 mix.1": seeds("v23_gru_pre2s_s", range(501, 507)),
    "F11 DIN lr2 mix.1": seeds("v24_din_pre2s_s", range(601, 605)),
    "F12 d192 lr2 mix.1": seeds("v26_d192_pre2s_s", range(701, 708)),
    "a4u e10 lr2 mix.1": seeds("r6_a4u_e10_s", range(101, 107)),
    "cset e10 lr2 mix.1": seeds("r6_cset_e10_s", range(101, 107)),
    "F13g GRU lr2 m02": seeds("v27_gru_m02_s", range(511, 519)),
    "F13l pre2s lr1 m02": seeds("v28_lr1_m02_s", range(411, 418)),
    "d192 lr2 m02 (v29)": seeds("v29_d192_m02_s", range(711, 716)),
    "F7 lr2 m02 scratch (v30)": seeds("v30_f7_m02_s", range(901, 907)),
}
# matched pairs mix.1 -> m02 (from scratch), and lr pairs
PAIRS_MIX = [("F7 pre2s lr2 mix.1", "F7 lr2 m02 scratch (v30)"), ("F9 pre2s lr1 mix.1", "F13l pre2s lr1 m02"),
             ("F10 GRU lr2 mix.1", "F13g GRU lr2 m02"), ("F12 d192 lr2 mix.1", "d192 lr2 m02 (v29)")]
PAIRS_LR = [("F7 pre2s lr2 mix.1", "F9 pre2s lr1 mix.1"), ("cvnd lr2 mix.1", "F6 cvnd lr1 mix.1"),
            ("F7 lr2 m02 scratch (v30)", "F13l pre2s lr1 m02")]

ROOTS = range(101, 108)
RECOOL = {  # r6 cooldown continuations from F7 snapshots (eval every 1000; step 0 = snapshot)
    "c0 (mix.1, cs811)": ([f"r6_cd_c0_r{r}" for r in ROOTS], 77500),
    "c0 (mix.1, cs813)": ([f"r6_cd_c0_r{r}_cs813" for r in range(101, 105)], 77500),
    "m02 (cs811) =F7x": ([f"r6_cd_m02_r{r}" for r in ROOTS], 77500),
    "m02 (cs813)": ([f"r6_cd_m02_r{r}_cs813" for r in range(101, 105)], 77500),
    "m02 p70k (cs812) =F7x70": ([f"r6_cd_m02_p70k_r{r}" for r in ROOTS], 70000),
    "m0 (no pool)": ([f"r6_cd_m0_r{r}" for r in ROOTS], 77500),
    "aa5 (mix.1)": ([f"r6_cd_aa5_r{r}" for r in range(101, 105)], 77500),
    "w10 (mix.1)": ([f"r6_cd_w10_r{r}" for r in range(101, 105)], 77500),
    "d0 (mix.1)": ([f"r6_cd_d0_r{r}" for r in range(101, 105)], 77500),
    "d20 (mix.1)": ([f"r6_cd_d20_r{r}" for r in range(101, 105)], 77500),
    "pp02 (mix.1)": ([f"r6_cd_pp02_r{r}" for r in range(101, 105)], 77500),
    "aug (mix.1)": ([f"r6_cd_aug_r{r}" for r in range(101, 105)], 77500),
    "e3 (m02, 3ep)": ([f"r6_cd_e3_r{r}" for r in range(101, 105)], 77500),
    "cf4 (m02, p60k 4ep)": ([f"r6_cd_cf4_p60k_r{r}" for r in range(101, 105)], 60000),
    "s2c2 (m02, 2 stable+2 cd)": ([f"r6_cd_s2c2_r{r}" for r in range(101, 105)], 77500),
}


def root_of(run):
    import re
    m = re.search(r"_r(\d{3})", run)
    return int(m[1]) if m else None


def orig_of(run):
    """F7 original (v17) of a re-cool run."""
    r = root_of(run)
    return f"v17_pre2s_s{r}" if r else None


def curve(C, run):
    return C[C.run == run].sort_values("step").reset_index(drop=True)


def plateau(c, lo=67500, hi=77500):
    w = (c.step >= lo) & (c.step <= hi)
    return c.loc[w, "val"].mean(), c.loc[w, "hold"].mean(), int(w.sum())


def ma3(y):
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    out = np.convolve(y, np.ones(3) / 3, mode="same")
    out[0], out[-1] = y[:2].mean(), y[-2:].mean()
    return out


def mse(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan), len(x)


def fmt(m, s, n=None, d=5):
    if np.isnan(m):
        return "   n/a"
    t = f"{m:+.{d}f}"
    if s is not None and not np.isnan(s):
        t += f"+-{s:.{d}f}"
    if n is not None:
        t += f" (n{n})"
    return t
