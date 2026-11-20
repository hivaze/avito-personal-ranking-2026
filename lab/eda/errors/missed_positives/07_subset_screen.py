"""Pairwise residual test (as in 03_screen) on pair subsets, for a curated feature list incl. the new derived features:
  ALL      all within-feed (pos, neg) pairs
  POSNOV   pairs whose positive is in a category absent from the 32 newest events (the missed-positive population)
  BOTHNOV  both candidates in such categories (can anything order novel items?)
  SAMECAT  both in the same category (within-category ordering)
  CROSS    different categories
Base = v5 blend z (best) and nn_final2_trainonly z (clean). Leak-free feeds only. t = cluster-robust (feed) t of the
feature coefficient in the conditional logit; dll = log-likelihood gain (feed-weighted pairs).
"""
import importlib.util, sys
sys.path.insert(0, "<repo>/src")
from common import *

D = Path("<repo>/eda/errors/missed_positives")
spec = importlib.util.spec_from_file_location("scr", D / "03_screen.py"); scr = importlib.util.module_from_spec(spec); spec.loader.exec_module(scr)

FEATS_LIST = ["f1", "f2", "f3", "f4", "f5", "f3_rk", "cat_cnt_all", "cat_cnt_20", "cat_min_hpos", "cat_share_all", "feed_cat_share",
              "maxsim32", "cos_u", "cos_h10", "attn32", "seq_max_all", "ctx_hs_k256", "pre_v3_fe", "pre_v3_new", "pre_v3_fe_noid",
              "pre_v3_id_contrib", "pre_v3_idmax32", "pre_v3_hmax32", "it_hist_all_luo", "it_hist_et1_luo", "it_cand_cnt_train", "it_tr_ctr",
              "tte_it_ctr", "seq_cv_invd20", "seq_cv_norm", "seq_cv_pop", "ctx_nd_c90", "ctx_nd_max", "clu_ccos1024", "clu_hmin1024",
              "clu_u1024i1024_lift", "clu_pmi11024_mean", "trd_skde_s6h", "trd_kde_sday", "trd_tp_mx20", "trd_tp_m20", "trd_tp_l1",
              "trd_fp_ud1", "trd_fp_ud3", "emb_missing", "item_norm",
              "d_dup95_h", "d_dup90_h", "d_dup95_hmin", "d_cat_last1", "d_cat_last3cnt", "d_ncat_feed",
              "d_cat_run", "d_cat_run_e1", "d_cat_pur20", "d_cat_pur5", "d_top_cat20", "d_f3_resid"]


def load(n):
    p = FEATS / "val" / f"{n}.npy"
    if not p.exists():
        p = D / "derived/val" / f"{n}.npy"
    return np.load(p).astype(np.float64)


def exists(n):
    return (FEATS / "val" / f"{n}.npy").exists() or (D / "derived/val" / f"{n}.npy").exists()


def main():
    global FEATS_LIST
    miss = [n for n in FEATS_LIST if not exists(n)]
    print("missing (deleted by other agents):", miss)
    FEATS_LIST = [n for n in FEATS_LIST if exists(n)] + ["als_exp5", "als_last10"]
    fr = np.load(D / "frame.npz")
    y, best, zc, M, off, leakf = (fr[k] for k in ["y", "best", "z_clean", "M", "off", "leakf"])
    nf = len(M)
    ip, ineg, pf, w0 = scr.build_pairs(off.astype(np.int64), M.astype(np.int64), y, ~leakf)
    cat = np.load(FLAT / "val/cand_category_id.npy")
    cmin = load("cat_min_hpos"); ccnt = load("cat_cnt_all")
    nov = (ccnt == 0) | (cmin >= 32)
    subsets = {"ALL": np.ones(len(ip), bool), "POSNOV": nov[ip], "BOTHNOV": nov[ip] & nov[ineg],
               "SAMECAT": cat[ip] == cat[ineg], "CROSS": cat[ip] != cat[ineg]}
    rows = []
    for sname, sm in subsets.items():
        P, N, F = ip[sm], ineg[sm], pf[sm]
        # re-weight: every feed total weight 1 inside the subset
        cnt = np.bincount(F, minlength=nf).astype(float)
        w = 1.0 / cnt[F]
        dzb = best[P] - best[N]; dzc = zc[P] - zc[N]
        ab, _, _, llb = scr.fit(dzb, np.zeros_like(dzb), w, F, nf)
        ac, _, _, llc = scr.fit(dzc, np.zeros_like(dzc), w, F, nf)
        acc_b = float(np.sum(w * (dzb > 0)) / w.sum())
        print(f"{sname}: pairs {sm.sum()}, feeds {(cnt > 0).sum()}, model pair accuracy (best) {acc_b:.3f}, a_best {ab:.3f}", flush=True)
        for n in FEATS_LIST:
            x = load(n)
            u = scr.rank_gauss(x)
            dx = u[P] - u[N]
            if np.abs(dx).max() == 0:
                continue
            _, bb, tb, lb = scr.fit(dzb, dx, w, F, nf, ab)
            _, bc, tc, lc = scr.fit(dzc, dx, w, F, nf, ac)
            facc = float(np.sum(w * ((dx > 0) + 0.5 * (dx == 0))) / w.sum())
            rows.append(dict(subset=sname, feat=n, feat_pair_acc=facc, b_best=bb, t_best=tb, dll_best=lb - llb, t_clean=tc, dll_clean=lc - llc))
    df = pl.DataFrame(rows)
    df.write_parquet(D / "subset_screen.parquet")
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(3)
    for sname in subsets:
        print(f"\n=== {sname} (sorted by t_best) ===")
        print(df.filter(pl.col("subset") == sname).sort(pl.col("t_best").abs(), descending=True).head(25).drop("subset"))


if __name__ == "__main__":
    main()
