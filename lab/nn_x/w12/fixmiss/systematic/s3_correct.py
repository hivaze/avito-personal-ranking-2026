"""Step 3: cross-fitted corrections for the slices the scan flagged (fit on user half A, apply to B, and reverse).

new logit = base + X b, b fitted by listwise CE (each positive's softmax) on one half with a small L2;
base = s / T (global T) or s * exp(G w) (per-feed temperature, w fitted on the same half; changes no order).
Also a direct NDCG grid for one-parameter offsets (best b on A applied to B and reverse).

usage: s3_correct.py [final|ft|to]   (column score / s_ft / s_to)
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *  # noqa: F401,F403
from core import _feed_lse

which = sys.argv[1] if len(sys.argv) > 1 else "final"
colname = {"final": "score", "ft": "s_ft", "to": "s_to"}[which]
F, M, off = layout()
half = np.load(HERE / "data/half.npy")
fid = np.repeat(np.arange(len(M)), M)
D = load_table()
y = D["label"].to_numpy().astype(np.float64)
s = D[colname].to_numpy().astype(np.float64)
nd0 = ndcg_per_feed(s, y, off, M)
print(f"[{which}] base NDCG {nd0.mean():.5f}")


def col(c):
    return np.nan_to_num(D[c].to_numpy().astype(np.float64), nan=np.nan)


def zc(x):
    x = np.asarray(x, np.float64).copy()
    ok = ~np.isnan(x)
    x[ok] = (x[ok] - x[ok].mean()) / x[ok].std()
    x[~ok] = 0.0
    return x


cni = col("cat_newest_idx")
nov8 = ((cni < 0) | (cni >= 8)).astype(float)
Mr = M[fid].astype(float)
first = off
# per-feed temperature descriptors (same as s1b)
G = np.column_stack([np.ones(len(M)), np.log(M)] + [np.nan_to_num(col(c)[first], nan=0.0) for c in
                     ["h_brk10", "h_ent", "n_hist_cats32", "h_nov32r"]] +
                    [col("H")[first] / 256.0, col("guest")[first], np.nan_to_num(col("h_topsh")[first], nan=0.0),
                     col("uvec_missing")[first]])
G[:, 1:] = (G[:, 1:] - G[:, 1:].mean(0)) / G[:, 1:].std(0)
wAB = np.load(HERE / "data/temper_w.npy") if which == "final" else None

feat = {
    "hmean5_q9": (col("hmean5") >= np.nanquantile(col("hmean5"), 0.9)).astype(float),
    "hmean5_z": zc(col("hmean5")), "h1_z": zc(col("h1")), "f1_z": zc(col("f1")), "f3_z": zc(col("f3")),
    "cat_hist_share_z": zc(col("cat_hist_share")), "cat_share_feed_z": zc(col("cat_n_in_feed") / Mr),
    "seqcv_z": zc(np.log1p(1e4 * col("seq_cv_invd20"))), "ucos_z": zc(col("ucos")), "hmax_z": zc(col("hmax")),
    "nov8": nov8, "cat50": (col("cat") == 50).astype(float), "nodup": (col("n_cand_dup95") == 0).astype(float),
    "catrank1": (col("cat_user_rank") == 1).astype(float),
    "nov8_focus": nov8 * (col("h_brk10") <= 1), "nov8_Mbig": nov8 * (Mr > 150),
    "nov8_entlow": nov8 * (col("h_ent") < np.nanquantile(col("h_ent")[first], 0.2)),
    "nov8_newestview": nov8 * (col("newest_type") == 1),
    "f4_z": zc(col("f4")), "young": (col("f4") <= np.nanquantile(col("f4"), 0.1)).astype(float),
    "cold": (col("xp_imp_c50") == 0).astype(float),
}
specs = {
    "hmean5_q9 (only slice surviving per-feed T)": ["hmean5_q9"],
    "hmean5 linear": ["hmean5_z"],
    "nov8 (new-to-user category)": ["nov8"],
    "nov8 x focused user (h_brk10<=1)": ["nov8_focus"],
    "nov8 x M>150": ["nov8_Mbig"],
    "nov8 x low entropy": ["nov8_entlow"],
    "nov8 + nov8 x {focus, M>150, lowent, newest=view}": ["nov8", "nov8_focus", "nov8_Mbig", "nov8_entlow", "nov8_newestview"],
    "familiarity set (12 linear/indicator)": ["hmean5_z", "h1_z", "f1_z", "f3_z", "cat_hist_share_z", "cat_share_feed_z",
                                              "seqcv_z", "ucos_z", "hmax_z", "nov8", "cat50", "nodup"],
    "all flagged slices (19 indicators + 4)": ["hmean5_q9", "nov8", "nov8_focus", "nov8_Mbig", "nov8_entlow", "catrank1",
                                              "cat50", "nodup", "nov8_newestview"],
    "age / cold (staleness)": ["f4_z", "young", "cold"],
}


def nd(score):
    return ndcg_per_feed(score, y, off, M)


rows_B = half[fid] == 1
results = {}
for tmode in (["global", "perfeed"] if which == "final" else ["global"]):
    print(f"\n--- base temperature: {tmode} ---")
    if tmode == "global":
        TA = fit_T(s, y, off, M, half == 0); TB = fit_T(s, y, off, M, half == 1)
        baseA, baseB = s / TA, s / TB              # base used when fitting on A / on B
    else:
        baseA, baseB = s * np.exp(G @ wAB[0])[fid], s * np.exp(G @ wAB[1])[fid]
    for name, fs in specs.items():
        X = np.column_stack([feat[f] for f in fs])
        bA = clogit_fit(baseA, X, y, off, M, half == 0, l2=1e-4)
        bB = clogit_fit(baseB, X, y, off, M, half == 1, l2=1e-4)
        new = np.where(rows_B, baseA + X @ bA, baseB + X @ bB)
        r = report_delta(nd(new), nd0, M, half, f"{name[:40]}")
        print(f"      b(A) {np.round(bA, 3).tolist()}  b(B) {np.round(bB, 3).tolist()}")
        results[(tmode, name)] = r

# direct NDCG grid for single-parameter offsets (in units of the global-T logit)
print("\n--- direct NDCG grid, one-parameter offsets: best b on one half applied to the other ---")
grid = np.array([-0.4, -0.3, -0.2, -0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
for f in ["hmean5_q9", "hmean5_z", "nov8", "nov8_focus", "nov8_Mbig", "f3_z", "cat_share_feed_z", "ucos_z", "young", "f4_z"]:
    x = feat[f]
    curves = np.array([nd(s + b * x) - nd0 for b in grid])          # (grid, feeds)
    mA = curves[:, half == 0].mean(1); mB = curves[:, half == 1].mean(1)
    bA, bB = grid[mA.argmax()], grid[mB.argmax()]
    d = np.where(half == 1, curves[mA.argmax()], curves[mB.argmax()])
    r = report_delta(d + nd0, nd0, M, half, f"grid {f} (bA {bA:+.3f}, bB {bB:+.3f})")
    print(f"      in-sample best A {mA.max():+.5f} B {mB.max():+.5f}")
