"""INVENTORY B, step 9: where does the conditional information of the short-list columns sit?
For each column: conditional score z (given pool + the ranker's own inputs) on feed slices and under the public / private feed-mix weights
(src/pmix.py), plus coverage (share of rows with a non-default value) on val and holdout.  Slices are label-free.
  val: day 20718 (2 d after the train window) vs 20720+ (4 d); M < 50 vs M >= 50; cold-share terciles of the feed (share of candidates with it_hist_all_luo = 0)
  holdout: Sep 6-8 / Sep 9-16 / Sep 17-19.
Usage: s09_slices.py     Output: out/slices.txt.  CPU only, 12 threads, read-only."""
import sys, json, time
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
sys.argv = [sys.argv[0], "val", "TO16"]
from lens import *
import cand_specs as cs, condlib, pmix
t0 = time.time()
SP = {f"{g}:{n}": s for g, n, s in cs.specs()}
COLS = ["ccb:c09_rp3b05", "ccb:c09_p3a", "cbB_c:kn_cos1", "cbB_c:kn_cos20", "store:tte_it_coec", "store:tte_it_ctr", "store:xp_ctr_c", "ccb:c06_theta_l3", "timegap:tg_n_days_active",
        "timegap:tg_gap_first", "ccb:c06_seen_l3", "codes:ex_X3_s1s2_lf", "codes:ex_n_s2_s1s2_lf", "store:tte_ui_imp", "cbB_a:ue_k05_coec", "cbB_b:tr_cat_base", "store:tte_it_imp", "store:xp_imp_c50",
        "store:ixh_mf1", "store:ixh_good", "store:it_hist_hz_all", "store:trd_kde_sday", "store:clu_pmi1024_mean"]
VALONLY = ["lit_gnn:walk_train_c0", "lit_gnn:walk_all_c0", "itemprior:ip_x_val_T", "emb:emb_proj_mu", "emb:emb_chg_d"]


def score_f(N, x):
    """per-feed efficient score of the candidate column (as condlib.Null.score, returning the per-feed terms)"""
    D = N.D; x = np.asarray(x, np.float32); p = N.r["p"]
    mx = np.bincount(N.fid, p * x, D["n"]); g = np.bincount(N.fid, D["y"] * x, D["n"]) - N.npos * mx
    Ixc = (N.rw * x) @ N.X - mx @ N.nMU
    return g - N.GA @ Ixc


def cz(D, Uf, w=None):
    w = np.ones(len(Uf)) if w is None else w
    Uc = np.bincount(D["clus"], Uf * w); s = np.sqrt((Uc ** 2).sum())
    return float((Uf * w).sum() / s) if s > 0 else 0.0


out = open(OUT / "slices.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n"); out.flush()

for sp in ("val", "hold"):
    D = load_split(sp, "TO16"); condlib.sp = sp
    nm, cl = condlib.control_block(D); N = condlib.Null(D, nm, cl)
    day = (D["ts"].astype(np.int64) + 3 * 3600) // 86400
    luo = cs.load(SP["store:it_hist_all_luo"][sp], D); fid = np.repeat(np.arange(D["n"]), D["M"])
    cold = np.bincount(fid, (np.nan_to_num(luo) == 0).astype(np.float64), D["n"]) / D["M"]
    if sp == "val":
        W = pmix.weights(); q = np.quantile(cold, [1 / 3, 2 / 3])
        SL = [("all", None), ("day 20718", (day == 20718) * 1.0), ("day 20720+", (day >= 20720) * 1.0), ("M<50", (D["M"] < 50) * 1.0), ("M>=50", (D["M"] >= 50) * 1.0),
              ("cold share low", (cold <= q[0]) * 1.0), ("cold mid", ((cold > q[0]) & (cold <= q[1])) * 1.0), ("cold high", (cold > q[1]) * 1.0), ("pmix-public w", W["public"]), ("pmix-private w", W["private"])]
        P(f"[val] {D['n']} feeds; days {dict(zip(*np.unique(day, return_counts=True)))}; feed cold share terciles at {q.round(3).tolist()}; mean cold share {cold.mean():.3f}")
    else:
        SL = [("all", None), ("Sep 6-8", (day <= 20704) * 1.0), ("Sep 9-16", ((day > 20704) & (day < 20713)) * 1.0), ("Sep 17-19", (day >= 20713) * 1.0),
              ("cold share low", (cold <= np.quantile(cold, 1 / 3)) * 1.0), ("cold high", (cold > np.quantile(cold, 2 / 3)) * 1.0)]
        P(f"\n[holdout] {D['n']} feeds; days {dict(zip(*np.unique(day, return_counts=True)))}; mean cold share {cold.mean():.3f}")
    P(f"{'column (z form)':28s} coverage  " + "  ".join(f"{n:>14s}" for n, _ in SL))
    for k in COLS + (VALONLY if sp == "val" else []):
        if sp not in SP[k]: continue
        x = cs.load(SP[k][sp], D); F = forms(x, D); Uf = score_f(N, F["z"])
        xx = np.nan_to_num(x, nan=0.0); mode = np.bincount((xx == xx[0]).astype(int)).argmax()
        vals, cnt = np.unique(np.round(xx[:: max(1, len(xx) // 200000)], 6), return_counts=True); dflt = vals[cnt.argmax()]
        cov = float(((xx != dflt) & ~np.isnan(x)).mean())
        P(f"{k:28s} {cov:7.3f}   " + "  ".join(f"{cz(D, Uf, w):+14.2f}" for _, w in SL))
P(f"[{time.time() - t0:.0f}s] done")
