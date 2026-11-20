"""R5 refit evaluation (CPU). Pre-registration: scratchpad round5/refit/PREREG.txt.

  CUDA_VISIBLE_DEVICES="" python solvers/r5_refit_eval.py nested   # decision on the nested simulation
  CUDA_VISIBLE_DEVICES="" python solvers/r5_refit_eval.py test     # diagnostics of the full-val refit test files
Inputs: nn_x/solvers/preds/refit/nested_<v>.npz, nn_x/solvers/preds/refit/nn_<v>_n2cpv3full_test.npy,
        preds/nn_<v>_n2cpv3_{val_oof,test}.npy, preds/nn_<v>_trainonly_test.npy (read-only).
Output: solvers/out/r5_refit/{nested,test}.json
"""
import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(k, "16")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>")
RF = ROOT / "nn_x/solvers/preds/refit"
OUT = ROOT / "solvers/out/r5_refit"
OUT.mkdir(parents=True, exist_ok=True)
NESTED = ["v15_cd_all_e10", "v14_s93", "v17_pre2s_s102"]
CD = (["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84",
       "v15_cd_allprew_s85", "v13_lr1_mix1"] + [f"v14_s{s}" for s in range(92, 97)]
      + [f"v17_pre2s_s{s}" for s in range(101, 108)])


def meta(split):
    f = pl.read_parquet(ROOT / f"data/flat/{split}/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    return f, M, off


def fz(s, M):
    s = np.asarray(s, np.float64)
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


def ms(d, m=None):
    d = d if m is None else d[m]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def nested():
    f, M, off = meta("val")
    y = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
    day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
    tue = day <= day.min() + 1
    luo = np.load(ROOT / "nn/feat_cache/val/it_hist_all_luo.npy").astype(np.float64)
    fid = np.repeat(np.arange(len(M)), M)
    cold = np.bincount(fid, (np.nan_to_num(luo) == 0).astype(np.float64), len(M)) / M
    hi_cold = cold >= np.quantile(cold, 2 / 3)
    slices = {"all": np.ones(len(M), bool), "M<50": M < 50, "M>=50": M >= 50, "Tue": tue, "Thu": ~tue,
              "cold_top_tercile": hi_cold}
    ev = lambda s: ndcg_per_feed(s, y, off, M)  # noqa: E731
    res = {"members": {}, "slices_n": {k: int(v.sum()) for k, v in slices.items()}}
    # POSTHOC (not pre-declared, information only): the production OOF fold-k model is a second-seed 80% refit on
    # exactly R_k (same GroupKFold call), so 0.5*(ii) + 0.5*prod = a 2-seed refit
    D = {"ii": [], "iii": [], "ii_iii": [], "iiiz": [], "aa": [], "prod_vs_i": [], "POSTHOC_2seed_refit_vs_i": []}
    outer = None
    nested_list = sys.argv[2].split(",") if len(sys.argv) > 2 else NESTED
    for v in nested_list:
        z = np.load(RF / f"nested_{v}.npz")
        Pi, Pii = z["P_i"].astype(np.float64), z["P_ii"].astype(np.float64)
        outer = z["outer"] if outer is None else outer
        assert (outer == z["outer"]).all()
        prod = np.load(ROOT / f"preds/nn_{v}_n2cpv3_val_oof.npy").astype(np.float64)
        n_i, n_ii, n_iii = ev(Pi), ev(Pii), ev(0.5 * Pi + 0.5 * Pii)
        n_iiiz = ev(0.5 * fz(Pi, M) + 0.5 * fz(Pii, M))
        n_prod = ev(prod)
        n_2s = ev(0.5 * Pii + 0.5 * prod)
        r = {"ndcg_i": float(n_i.mean()), "ndcg_ii": float(n_ii.mean()), "ndcg_iii": float(n_iii.mean()),
             "ndcg_iii_z": float(n_iiiz.mean()), "ndcg_prod_oof_5fold_80pct": float(n_prod.mean()),
             "d_ii_i": ms(n_ii - n_i), "d_iii_i": ms(n_iii - n_i), "d_ii_iii": ms(n_ii - n_iii),
             "d_iiiz_i": ms(n_iiiz - n_i),
             "AA_ii_minus_prod (two seeds of an 80% model)": ms(n_ii - n_prod),
             "d_prod_i (80% model, other seed, vs inner mean)": ms(n_prod - n_i),
             "per_outer_fold_d_ii_i": [ms(n_ii - n_i, outer == k)[0] for k in range(5)],
             "per_outer_fold_d_iii_i": [ms(n_iii - n_i, outer == k)[0] for k in range(5)],
             "corr_z_i_ii": float(np.corrcoef(fz(Pi, M), fz(Pii, M))[0, 1]),
             "POSTHOC_d_2seed_refit_i": ms(n_2s - n_i)}
        res["members"][v] = r
        D["ii"].append(n_ii - n_i)
        D["iii"].append(n_iii - n_i)
        D["ii_iii"].append(n_ii - n_iii)
        D["iiiz"].append(n_iiiz - n_i)
        D["aa"].append(n_ii - n_prod)
        D["prod_vs_i"].append(n_prod - n_i)
        D["POSTHOC_2seed_refit_vs_i"].append(n_2s - n_i)
    pooled = {k: np.mean(v, 0) for k, v in D.items()}
    res["pooled"] = {k: {s: ms(d, m) for s, m in slices.items()} for k, d in pooled.items()}
    dec = {}
    for x in ("ii", "iii"):
        mu, se = res["pooled"][x]["all"]
        mem_ok = all(res["members"][v][f"d_{x}_i"][0] >= 0 for v in nested_list)
        dec[x] = {"pooled": mu, "se": se, "z": mu / se, "gt_2se": bool(mu > 2 * se), "no_member_negative": mem_ok,
                  "PASS": bool(mu > 2 * se and mem_ok)}
    mu, se = res["pooled"]["ii_iii"]["all"]
    if dec["ii"]["PASS"] and dec["iii"]["PASS"]:
        choice = "ii" if mu > 2 * se else "iii"
    elif dec["ii"]["PASS"] or dec["iii"]["PASS"]:
        choice = "ii" if dec["ii"]["PASS"] else "iii"
    else:
        choice = "none (keep 5-fold mean)"
    res["decision"] = dict(dec, choice=choice)
    (OUT / "nested.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decision"], indent=1))
    for v, r in res["members"].items():
        print(v, {k: (np.round(x, 5).tolist() if isinstance(x, (list, tuple, float)) else x) for k, x in r.items()})
    for k, d in res["pooled"].items():
        print(f"pooled {k}: " + "  ".join(f"{s} {a:+.5f}+-{b:.5f}" for s, (a, b) in d.items()))


def test():
    f, M, off = meta("test")
    fv, Mv, offv = meta("val")
    res = {"members": {}}
    Zfm, Zfull, Zmix, Zto = [], [], [], []
    for v in CD:
        fm = np.load(ROOT / f"preds/nn_{v}_n2cpv3_test.npy").astype(np.float64)
        full = np.load(RF / f"nn_{v}_n2cpv3full_test.npy").astype(np.float64)
        to = np.load(ROOT / f"preds/nn_{v}_trainonly_test.npy").astype(np.float64)
        assert len(fm) == len(full) == len(to) == M.sum() and np.isfinite(full).all()
        mix = 0.5 * fm + 0.5 * full
        z1, z2, z3, z0 = fz(fm, M), fz(full, M), fz(mix, M), fz(to, M)
        res["members"][v] = {"corr_z_foldmean_full": float(np.corrcoef(z1, z2)[0, 1]),
                             "corr_z_foldmean_trainonly": float(np.corrcoef(z1, z0)[0, 1]),
                             "corr_z_full_trainonly": float(np.corrcoef(z2, z0)[0, 1]),
                             "raw_sd_foldmean": float(fm.std()), "raw_sd_full": float(full.std())}
        Zfm.append(z1); Zfull.append(z2); Zmix.append(z3); Zto.append(z0)
    ft_fm, ft_full, ft_mix, to = (fz(np.mean(Z, 0), M) for Z in (Zfm, Zfull, Zmix, Zto))
    cd = 0.5 * ft_fm + 0.5 * to
    cd_ii = 0.5 * ft_full + 0.5 * to
    cd_iii = 0.5 * ft_mix + 0.5 * to
    res["ensemble_corr"] = {"ftbranch_foldmean_vs_full": float(np.corrcoef(ft_fm, ft_full)[0, 1]),
                            "ftbranch_foldmean_vs_mix": float(np.corrcoef(ft_fm, ft_mix)[0, 1]),
                            "Cd_vs_Cd_ii": float(np.corrcoef(cd, cd_ii)[0, 1]),
                            "Cd_vs_Cd_iii": float(np.corrcoef(cd, cd_iii)[0, 1])}
    np.save(OUT / "Cd_ii_test_score.npy", cd_ii.astype(np.float32))
    np.save(OUT / "Cd_iii_test_score.npy", cd_iii.astype(np.float32))
    np.save(OUT / "Cd_ref_test_score.npy", cd.astype(np.float32))
    (OUT / "test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["ensemble_corr"], indent=1))
    for v, r in res["members"].items():
        print(v, {k: round(x, 4) for k, x in r.items()})


if __name__ == "__main__":
    {"nested": nested, "test": test}[sys.argv[1]]()
