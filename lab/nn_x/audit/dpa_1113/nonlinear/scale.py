"""How the trainer's set_norm standardisation (checkpoint f_mean/f_std of r6_cd_m02_r101, train-batch statistics,
clamp +-6) maps the val / holdout inputs: clip fractions, standardised quantiles, mass points. CPU-replica columns only."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, D
mu, sd = np.load(f"{D}/fstats_r101.npy")
sd = np.maximum(sd, 1e-3)
XT = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo",
      "it_hist_et3_luo", "it_hist_et4_luo", "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev",
      "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w",
      "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128",
      "ctx_hs_e1", "ctx_hs_t10", "pre2_fe_z", "pre2_new_z", "has_pre2_fe"]
COLS = {0: "f1", 3: "f2", 6: "f3", 9: "f4", 12: "f5", 17: "d_cos_u", 19: "d_smax", 20: "d_smax8", 21: "d_smax32",
        22: "d_smean", 23: "d_lse", 24: "d_top3", 25: "d_mx_e1", 26: "d_mx_e2", 27: "d_mx_e3", 28: "d_mx_e4",
        35: "d_ccnt", 36: "d_cshare", 37: "d_cc_e1", 38: "d_cc_e2", 39: "d_cc_e3", 40: "d_cc_e4", 47: "d_firstrec", 48: "d_fcs"}
for j, n in enumerate(XT):
    COLS[63 + j] = "xt_" + n
for split in ["val", "hold"]:
    d = load(split)
    print(f"=== {split}: standardised by the checkpoint's train statistics (clamp +-6) ===")
    print(f"  {'col':4s} {'input':24s} {'mean_z':>7s} {'sd_z':>6s} {'p1':>6s} {'p50':>6s} {'p99':>6s} {'clip%':>6s} {'top_mass%':>9s} {'n_unique':>8s}")
    for j, n in COLS.items():
        x = d[n].astype(np.float64)
        z = (x - mu[j]) / sd[j]
        clip = np.mean(np.abs(z) > 6) * 100
        vals, cnt = np.unique(np.round(x, 6), return_counts=True)
        top = cnt.max() / len(x) * 100
        print(f"  {j:4d} {n:24s} {z.mean():7.2f} {z.std():6.2f} {np.percentile(z,1):6.2f} {np.percentile(z,50):6.2f} {np.percentile(z,99):6.2f} {clip:6.2f} {top:9.1f} {len(vals):8d}")
