"""Summaries of out/feature_table.csv: per usage class, ranked shift / label-relation / support tables."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
t = pd.read_csv(HERE / "out/feature_table.csv")
t = t.set_index("feature")

XT = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo',
      'it_hist_et4_luo', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u', 'ctx_nd_knn_u_dev',
      'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'ctx_hn_top1w', 'ctx_hn_rrw', 'ctx_hn_bestrk', 'ctx_hn_margin',
      'ctx_hn_soft05', 'ctx_hn_e1_rrw', 'ctx_hs_mutual', 'ctx_hs_k128', 'ctx_hs_e1', 'ctx_hs_t10', 'pre2_fe_z', 'pre2_new_z',
      'pre2_fe_s', 'pre2_new_s', 'pre2_hmax32', 'has:pre2_fe']
INMODEL = ['f1', 'f2', 'f3', 'f4', 'f5', 'emb_missing', 'item_norm', 'cos_u', 'dot_u', 'maxsim32', 'meansim32', 'cos_h10', 'cos_hall',
           'attn32', 'cos_het1', 'cos_het2', 'cos_het3', 'cos_het4', 'cat_cnt_all', 'cat_share_all', 'cat_cnt_et1', 'cat_cnt_et2',
           'cat_min_hpos', 'feed_cat_share', 'M', 'H']
FT = ['centrality', 'it_hist_all_luo', 'it_hist_all_luo_rk', 'it_hist_all_luo_zsl', 'it_hist_all_luo_gap', 'it_hist_et1_luo',
      'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'it_hist_et5_luo', 'it_hist_et6_luo', 'it_hist_et7_luo',
      'it_hist_et8_luo', 'it_hist_et9_luo', 'it_hist_et10_luo', 'seq_cv_invd20', 'seq_cv_invd20_rk', 'seq_cv_pop', 'seq_max_all',
      'seq_top3', 'seq_argmax_hpos', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u',
      'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'pre_fe', 'pre_fe_z', 'pre_new', 'pre_new_z',
      'pre_hmax32_c', 'pre_hmax8_c', 'pre_fe_noid', 'pre_id_contrib_v2', 'pre_idmax32', 'pre_idmax32_has', 'pre_hmax32_sent']
cls = {}
for n in t.index:
    c = []
    if n in XT: c.append("XT")
    if n in INMODEL: c.append("IM")
    if n in FT: c.append("FT")
    cls[n] = "+".join(c) if c else "unused"
t["class"] = pd.Series(cls)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
f = lambda c: t[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")

def dev(c):  # |AUC - 0.5|
    return (t[c] - 0.5).abs()

print("=== A. USED features: shift and label relation (cAUC = candidate-level adversarial AUC; wAUC = within-feed label AUC) ===")
cols = ["class", "cAUC_trRL~val50", "cAUC_trProd~val50", "cAUC_trRL~priv50", "cAUC_val~pub", "cAUC_val~priv", "cAUC_pub~priv",
        "fAUC_mean_trRL~val50", "fAUC_zeroshare_val~priv", "fAUC_nanshare_val~priv",
        "wAUC_trProd", "wAUC_trRL", "wAUC_valAll", "wAUC_val50", "wAUC_valTue", "wAUC_valThu",
        "slope_trProd", "slope_trRL", "slope_valAll", "out_q_priv", "out_q_cold_priv", "zcold_trRL", "zcold_val", "zcold_priv",
        "nan_val", "nan_priv", "zero_val", "zero_priv", "smd_trRL_val", "smd_val_priv"]
used = t[t["class"] != "unused"].copy()
used["shift_tr"] = dev("cAUC_trRL~val50")[used.index]; used["shift_priv"] = dev("cAUC_val~priv")[used.index]
used["flip"] = np.sign(used["slope_trProd"]) != np.sign(used["slope_valAll"])
used["dslope"] = used["slope_trProd"] - used["slope_valAll"]
print(used[cols].round(3).to_string())
print("\n=== A2. USED features ranked by train(random-like)->val shift ===")
print(used.sort_values("shift_tr", ascending=False)[["class", "cAUC_trRL~val50", "cAUC_trProd~val50", "cAUC_val~priv", "zero_trRL", "zero_val", "zero_priv", "nan_trRL", "nan_val", "nan_priv"]].head(25).round(3).to_string())
print("\n=== A3. USED features ranked by val->private shift ===")
print(used.sort_values("shift_priv", ascending=False)[["class", "cAUC_val~pub", "cAUC_val~priv", "cAUC_pub~priv", "fAUC_mean_val~priv", "fAUC_zeroshare_val~priv", "out_q_priv", "out_q_cold_priv", "smd_val_priv"]].head(25).round(3).to_string())
print("\n=== A4. USED features: label relation production vs exploration (slope = 1-D conditional-logit on within-feed z) ===")
print(used.sort_values("dslope")[["class", "wAUC_trProd", "wAUC_trRL", "wAUC_valAll", "slope_trProd", "slope_trRL", "slope_valAll", "slope_valTue", "slope_valThu", "flip"]].round(3).to_string())
print("\n=== B. ALL features: sign flips production vs val (|slope| > 0.03 on both) ===")
fl = t[(np.sign(t["slope_trProd"]) != np.sign(t["slope_valAll"])) & (t["slope_trProd"].abs() > 0.03) & (t["slope_valAll"].abs() > 0.03)]
print(fl[["class", "wAUC_trProd", "wAUC_trRL", "wAUC_valAll", "slope_trProd", "slope_trRL", "slope_valAll"]].round(3).to_string())
print("\n=== C. ALL features ranked by trRL->val50 candidate shift (top 40) ===")
t["shift_tr"] = dev("cAUC_trRL~val50"); t["shift_priv"] = dev("cAUC_val~priv")
print(t.sort_values("shift_tr", ascending=False)[["class", "cAUC_trRL~val50", "cAUC_trProd~trRL", "cAUC_val~priv", "wAUC_trRL", "wAUC_valAll"]].head(40).round(3).to_string())
print("\n=== D. ALL features ranked by val->private candidate shift (top 40) ===")
print(t.sort_values("shift_priv", ascending=False)[["class", "cAUC_val~pub", "cAUC_val~priv", "cAUC_pub~priv", "out_q_priv", "out_q_cold_priv", "nan_val", "nan_priv", "zero_val", "zero_priv"]].head(40).round(3).to_string())
print("\n=== E. Private rows outside val support (q0.1%-99.9%), top 25 ===")
print(t.sort_values("out_q_priv", ascending=False)[["class", "out_minmax_priv", "out_q_pub", "out_q_priv", "out_q_cold_priv", "smd_val_priv"]].head(25).round(4).to_string())
print("\n=== F. UNUSED features with the strongest val label relation (wAUC_valAll), top 40 ===")
un = t[t["class"] == "unused"].copy(); un["w"] = (un["wAUC_valAll"] - 0.5).abs()
print(un.sort_values("w", ascending=False)[["wAUC_valAll", "wAUC_trRL", "wAUC_trProd", "slope_valAll", "slope_trProd", "cAUC_trRL~val50", "cAUC_val~priv", "nan_val"]].head(40).round(3).to_string())
t.to_csv(HERE / "out/feature_table_classed.csv")
