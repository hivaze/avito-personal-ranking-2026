"""Step 6: value of a PERFECT correction. If the detected calibration bias were exactly true (truth q = softmax(s/T + d),
d = X b fitted by listwise CE on ALL of val, in-sample), how much NDCG would re-ranking by s/T + d gain over s?
Labels are drawn from q (Plackett-Luce, the feed's own number of positives), both orders scored on the same draws.
This bounds any loss / feature / post-hoc fix that only removes these biases.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *  # noqa: F401,F403

R = 40
F, M, off = layout()
half = np.load(HERE / "data/half.npy")
fid = np.repeat(np.arange(len(M)), M)
D = load_table()
y = D["label"].to_numpy().astype(np.float64)
s = D["score"].to_numpy().astype(np.float64)
npos = np.bincount(fid, y, len(M)).astype(np.int64)
T = fit_T(s, y, off, M, np.ones(len(M), bool))
rk = model_rank(s, off, M)


def col(c):
    return D[c].to_numpy().astype(np.float64)


def zc(x):
    x = x.copy(); ok = ~np.isnan(x); x[ok] = (x[ok] - x[ok].mean()) / x[ok].std(); x[~ok] = 0; return x


cni = col("cat_newest_idx"); nov8 = ((cni < 0) | (cni >= 8)).astype(float); Mr = M[fid].astype(float)
feat = {"hmean5_q9": (col("hmean5") >= np.nanquantile(col("hmean5"), 0.9)).astype(float), "nov8": nov8,
        "hmean5_z": zc(col("hmean5")), "h1_z": zc(col("h1")), "f1_z": zc(col("f1")), "f3_z": zc(col("f3")),
        "cat_hist_share_z": zc(col("cat_hist_share")), "cat_share_feed_z": zc(col("cat_n_in_feed") / Mr),
        "seqcv_z": zc(np.log1p(1e4 * col("seq_cv_invd20"))), "ucos_z": zc(col("ucos")), "hmax_z": zc(col("hmax")),
        "cat50": (col("cat") == 50).astype(float), "nodup": (col("n_cand_dup95") == 0).astype(float),
        "nov8_focus": nov8 * (col("h_brk10") <= 1), "nov8_Mbig": nov8 * (Mr > 150)}
sets = {"hmean5_q9": ["hmean5_q9"], "nov8": ["nov8"], "nov8 x focus": ["nov8_focus"],
        "familiarity set (12)": ["hmean5_z", "h1_z", "f1_z", "f3_z", "cat_hist_share_z", "cat_share_feed_z", "seqcv_z",
                                 "ucos_z", "hmax_z", "nov8", "cat50", "nodup"],
        "x3 familiarity set (bias tripled)": None}
dummy_disc = DISC
for name, fs in sets.items():
    if fs is None:
        fs = sets["familiarity set (12)"]; mult = 3.0
    else:
        mult = 1.0
    X = np.column_stack([feat[f] for f in fs])
    b = clogit_fit(s / T, X, y, off, M, np.ones(len(M), bool), l2=1e-4) * mult
    x_true = s / T + X @ b
    rows, _ = redraw(x_true, rk, off, M, npos, R, 99, DISC)
    d = []
    for r in range(R):
        yr = np.zeros(len(s)); yr[rows[r]] = 1
        d.append(ndcg_per_feed(x_true, yr, off, M) - ndcg_per_feed(s, yr, off, M))
    d = np.array(d)                       # R x feeds
    dm = d.mean(0)                        # per-feed expected gain
    print(f"{name:36s} b {np.round(b, 3).tolist()[:12]}\n    value of perfect correction: val {dm.mean():+.5f} "
          f"(MC sd {d.mean(1).std() / np.sqrt(R):.5f}) | M<50 {dm[M < 50].mean():+.5f} | M>=50 {dm[M >= 50].mean():+.5f} | "
          f"pmix-pub {np.average(dm, weights=pmix.weights()['public']):+.5f} | rows whose rank moves "
          f"{(model_rank(x_true, off, M) != rk).mean():.3f}", flush=True)
