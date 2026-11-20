"""INVENTORY B, step 7b (the recommended bundles; same reads as s07_shortlist.py, output out/shortlist_R.json): the short list, read as GROUPS with three reads each, next to real positive controls read the same way.
  (a) LR(group | pool + the ranker's own inputs)        in-sample, k df, on val (train-only pool and the FD16P20 file) and on the holdout
  (b) nested NDCG of pool + group (5 user-grouped folds) on each witness                       [the old residual form, for the record]
  (c) CROSS-WITNESS transfer: coefficients of [pool, group] fitted on the holdout, applied unchanged to val, and the reverse.
      Out of sample in feeds, users and period; the only read here that is not an in-sample selection.
Positive controls: the 8 popularity inputs on the popularity-light wide students; the encoder scalars on the members trained without them.
Usage: s07_shortlist.py     Output: out/shortlist.json + stdout.  CPU only, 12 threads, read-only."""
import sys, json, time
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
sys.argv = [sys.argv[0], "val", "TO16"]
from lens import *
import cand_specs as cs
import importlib.util
t0 = time.time()
SP = {f"{g}:{n}": s for g, n, s in cs.specs()}
POP8 = "seq_cv_invd20 seq_cv_pop seq_cv_norm it_hist_all_luo it_hist_et1_luo it_hist_et2_luo it_hist_et3_luo it_hist_et4_luo".split()
PRE2 = "pre2_fe_z pre2_new_z pre2_hmax32".split()
GROUPS = {
    "R1 recommended bundle, ranks 1-5 (p3a, rp3b05, kn_cos1, kn_cos20, X3, n_s2, bt_theta, gap_first, never_seen)": ["ccb:c09_p3a", "ccb:c09_rp3b05", "cbB_c:kn_cos1", "cbB_c:kn_cos20", "codes:ex_X3_s1s2_lf", "codes:ex_n_s2_s1s2_lf", "ccb:c06_theta_l3", "timegap:tg_gap_first", "timegap:tg_never_seen"],
    "R2 ranks 1-3 only (p3a, rp3b05, kn_cos1, kn_cos20, X3, n_s2)": ["ccb:c09_p3a", "ccb:c09_rp3b05", "cbB_c:kn_cos1", "cbB_c:kn_cos20", "codes:ex_X3_s1s2_lf", "codes:ex_n_s2_s1s2_lf"],
}


def cols_for(keys, D, both_forms=True):
    out = []
    for k in keys:
        F = forms(cs.load(SP[k][D["split"]], D), D); out.append(F["z"])
        if both_forms: out.append(F["rk"])
    return out


def transfer(Dfit, Dapp, cf, ca):
    """coefficients of [pool, group] fitted on Dfit, applied to Dapp; NDCG delta vs the pool of Dapp"""
    f0 = fit(Dfit, []); r = fit(Dfit, cf, theta0=[f0["theta"][0]] + [0.0] * len(cf)); th = r["theta"]
    s = (np.stack([Dapp["base"]] + ca, 1).astype(np.float32) @ th.astype(np.float32)).astype(np.float32)
    return ndcg(s, Dapp) - Dapp["ndcg0"], th


import condlib  # conditional machinery of s06 (control block, null model with K columns)


def cond_LR(D, keys, exclude=()):
    condlib.sp = D["split"]
    nm, cl = condlib.control_block(D, exclude=set(exclude)); N = condlib.Null(D, nm, cl)
    g = cols_for(keys, D); X = np.ascontiguousarray(np.concatenate([N.X, np.stack(g, 1).astype(np.float32)], 1))
    r = condlib.fit_big(D, X, theta0=list(N.r["theta"]) + [0.0] * len(g))
    return float(2 * (r["L"] - N.r["L"])), len(g), float((r["L"] - N.r["L"]) / D["n"])


res = {}
DV = load_split("val", "TO16"); DF = load_split("val", "FILE"); DH = load_split("hold", "TO16")
import pmix
wp = pmix.weights()["private"]
print(f"[{time.time() - t0:.0f}s] pools: val train-only {DV['ndcg0'].mean():.5f}, val file {DF['ndcg0'].mean():.5f}, holdout {DH['ndcg0'].mean():.5f}", flush=True)
from scipy.stats import chi2


def run(tag, keys, DV, DF, DH, exclude=()):
    o = {}
    cv, ch = cols_for(keys, DV), cols_for(keys, DH)
    for nm, D, c in (("val", DV, cv), ("hold", DH, ch)):
        f0 = fit(D, []); r = fit(D, c, theta0=[f0["theta"][0]] + [0.0] * len(c)); d = nested(D, c, r["theta"])
        o[f"LR_pool_{nm}"] = float(2 * (r["L"] - f0["L"])); o[f"nest_{nm}"] = float(d.mean()); o[f"nest_{nm}_se"] = float(d.std() / np.sqrt(len(d)))
        if nm == "val": o["nest_val_pmix"] = float(np.average(d, weights=wp))
        lr, k, dn = cond_LR(D, keys, exclude); o[f"condLR_{nm}"] = lr; o["k"] = k; o[f"cond_dnats_{nm}"] = dn; o[f"cond_p_{nm}"] = float(chi2.sf(lr, k))
    if DF is not None:
        lr, k, dn = cond_LR(DF, keys, exclude); o["condLR_valFILE"] = lr; o["cond_p_valFILE"] = float(chi2.sf(lr, k))
        cf = cols_for(keys, DF); d, _ = transfer(DH, DF, ch, cf); o["xfer_h2vFILE"] = float(d.mean()); o["xfer_h2vFILE_se"] = float(d.std() / np.sqrt(len(d)))
    d, th = transfer(DH, DV, ch, cv); o["xfer_h2v"] = float(d.mean()); o["xfer_h2v_se"] = float(d.std() / np.sqrt(len(d))); o["xfer_h2v_pmix"] = float(np.average(d, weights=wp)); o["theta_hold"] = th.tolist()
    d2, th2 = transfer(DV, DH, cv, ch); o["xfer_v2h"] = float(d2.mean()); o["xfer_v2h_se"] = float(d2.std() / np.sqrt(len(d2))); o["theta_val"] = th2.tolist()
    print(f"\n== {tag}  ({o['k']} columns: z and rank form of {len(keys)})\n"
          f"   (a) LR | pool + own inputs: val {o['condLR_val']:.1f} (p {o['cond_p_val']:.1e})" + (f", val FILE {o['condLR_valFILE']:.1f} (p {o['cond_p_valFILE']:.1e})" if DF is not None else "")
          + f", holdout {o['condLR_hold']:.1f} (p {o['cond_p_hold']:.1e}); nats/feed {o['cond_dnats_val']:+.5f} / {o['cond_dnats_hold']:+.5f}\n"
          f"   (b) nested NDCG, pool + group: val {o['nest_val']:+.5f} +- {o['nest_val_se']:.5f} (pmix-private {o['nest_val_pmix']:+.5f}), holdout {o['nest_hold']:+.5f} +- {o['nest_hold_se']:.5f}\n"
          f"   (c) cross-witness: holdout coefficients on val {o['xfer_h2v']:+.5f} +- {o['xfer_h2v_se']:.5f} (z {o['xfer_h2v'] / o['xfer_h2v_se']:+.2f}; pmix-private {o['xfer_h2v_pmix']:+.5f})"
          + (f", on the val FILE {o['xfer_h2vFILE']:+.5f} +- {o['xfer_h2vFILE_se']:.5f}" if DF is not None else "")
          + f"; val coefficients on the holdout {o['xfer_v2h']:+.5f} +- {o['xfer_v2h_se']:.5f} (z {o['xfer_v2h'] / o['xfer_v2h_se']:+.2f})", flush=True)
    return o


for tag, keys in GROUPS.items():
    res[tag] = run(tag, keys, DV, DF, DH); json.dump(res, open(OUT / "shortlist_R.json", "w"), indent=1)

print(f"[{time.time() - t0:.0f}s] done", flush=True)
