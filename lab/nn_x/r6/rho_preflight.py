"""P3 rho70 PRE-FLIGHT (reports/RESEARCH_SWEEP_1112.md P3; rules fixed in the wave-4 task before any excess number
existed). CPU only, read-only on nn_x/r6/rho/excess_r<root>.npz (rho_score.py) and the reference run's log json.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/rho_preflight.py \
      [--ref r6_rhoref_pool_s1] [--roots 101,102,103,104]
  -> nn_x/r6/rho/preflight.{txt,json}; writes the marker rho/PREFLIGHT_PASS iff ALL checks pass (cd_lane.sh skips
     rho70 jobs without it), otherwise rho/PREFLIGHT_FAIL.

Checks (rho70 is NOT launched if any fails):
  (1) reference val NDCG >= 0.36 (val read only for this check: the reference run's selected-checkpoint val NDCG);
  (2) per root: Spearman(excess, L_snap) < 0.9 over the production feeds (otherwise rho70 = hard-example mining,
      and focal loss is closed);
  (3) per root: the kept feeds (top 70% by excess) have a LOWER share of positives at production positions 0-2 than
      the dropped feeds (pooled: sum ntop3 / sum npos per set): the selection filters position-driven positives.
Diagnostics (not gates): Pearson(excess, L_snap), Spearman(excess, -L_ref), per-feed mean top-3 share, M<50 share,
mean L_snap / L_ref of kept vs dropped, kept-set overlap across roots, the reference's selection-holdout NDCG.
"""
import argparse
import json
import os

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np  # noqa: E402

R6 = "<repo>/nn_x/r6"


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="r6_rhoref_pool_s1")
    ap.add_argument("--roots", default="101,102,103,104")
    ap.add_argument("--keep", type=float, default=0.7)
    a = ap.parse_args()
    roots = [int(r) for r in a.roots.split(",")]
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))  # noqa: E731
    js = json.load(open(f"{R6}/logs/{a.ref}.json"))
    hb = [h for h in js["history"] if h["step"] == js["best_step"]][0]
    ref_val, ref_hold = js["val_ndcg"], hb["holdout_ndcg"]
    out = dict(ref=a.ref, ref_val=ref_val, ref_holdout_selected=ref_hold, best_step=js["best_step"], roots={})
    c1 = ref_val >= 0.36
    P(f"P3 rho70 pre-flight ({a.ref}; keep {a.keep})")
    P(f"(1) reference val {ref_val:.5f} (>= 0.36: {'PASS' if c1 else 'FAIL'}); selection-holdout {ref_hold:.5f} at step "
      f"{js['best_step']} (pool-only, --only_random 0.5, holdout excluded)")
    kept_sets, ok = {}, c1
    for r in roots:
        z = np.load(f"{R6}/rho/excess_r{r}.npz")
        exc, ls, lr_, M = z["excess"], z["L_snap"], z["L_ref"], z["M"]
        npos, ntop = z["npos"], z["ntop3"]
        n_keep = int(round(a.keep * len(exc)))
        order = np.lexsort((z["fidx"], -exc))
        k = np.zeros(len(exc), bool)
        k[order[:n_keep]] = True
        kept_sets[r] = z["fidx"][k]
        sp = spearman(exc, ls)
        sh_k, sh_d = ntop[k].sum() / npos[k].sum(), ntop[~k].sum() / npos[~k].sum()
        c2, c3 = sp < 0.9, sh_k < sh_d
        ok = ok and c2 and c3
        d = dict(spearman_excess_Lsnap=sp, pearson_excess_Lsnap=float(np.corrcoef(exc, ls)[0, 1]),
                 spearman_excess_negLref=spearman(exc, -lr_), top3_share_kept=float(sh_k), top3_share_dropped=float(sh_d),
                 top3_feedmean_kept=float((ntop[k] / npos[k]).mean()), top3_feedmean_dropped=float((ntop[~k] / npos[~k]).mean()),
                 m_lt50_kept=float((M[k] < 50).mean()), m_lt50_dropped=float((M[~k] < 50).mean()),
                 Lsnap_kept=float(ls[k].mean()), Lsnap_dropped=float(ls[~k].mean()), Lref_kept=float(lr_[k].mean()),
                 Lref_dropped=float(lr_[~k].mean()), excess_cut=float(exc[order[n_keep - 1]]),
                 check2=bool(c2), check3=bool(c3), snap=str(z["snap"]), ref_file=str(z["ref"]))
        out["roots"][r] = d
        P(f"r{r}: (2) Spearman(excess, L_snap) {sp:.4f} (< 0.9: {'PASS' if c2 else 'FAIL'}; Pearson "
          f"{d['pearson_excess_Lsnap']:.4f}; Spearman(excess, -L_ref) {d['spearman_excess_negLref']:.4f})")
        P(f"     (3) positives at production positions 0-2: kept {sh_k:.4f} vs dropped {sh_d:.4f} (kept lower: "
          f"{'PASS' if c3 else 'FAIL'}; per-feed means {d['top3_feedmean_kept']:.4f} vs {d['top3_feedmean_dropped']:.4f})")
        P(f"     diag: M<50 share kept {d['m_lt50_kept']:.3f} dropped {d['m_lt50_dropped']:.3f}; L_snap kept "
          f"{d['Lsnap_kept']:.4f} dropped {d['Lsnap_dropped']:.4f}; L_ref kept {d['Lref_kept']:.4f} dropped "
          f"{d['Lref_dropped']:.4f}; excess cut {d['excess_cut']:+.4f}")
    if len(roots) > 1:
        jac = {f"{r1}-{r2}": float(len(np.intersect1d(kept_sets[r1], kept_sets[r2]))
                                   / len(np.union1d(kept_sets[r1], kept_sets[r2])))
               for i, r1 in enumerate(roots) for r2 in roots[i + 1:]}
        out["kept_jaccard"] = jac
        P("diag: kept-set Jaccard across roots: " + " ".join(f"{k} {v:.3f}" for k, v in jac.items()))
    out["PASS"] = bool(ok)
    P(f"==> PRE-FLIGHT {'PASS: rho70 may be launched' if ok else 'FAIL: do NOT launch rho70'}")
    open(f"{R6}/rho/preflight.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{R6}/rho/preflight.json", "w"), indent=1)
    for f in ("PREFLIGHT_PASS", "PREFLIGHT_FAIL"):
        if os.path.exists(f"{R6}/rho/{f}"):
            os.remove(f"{R6}/rho/{f}")
    open(f"{R6}/rho/{'PREFLIGHT_PASS' if ok else 'PREFLIGHT_FAIL'}", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
