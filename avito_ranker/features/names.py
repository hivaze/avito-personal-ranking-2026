"""File names of the pre-computed candidate inputs.

The names are the ones the arrays have on disk; trained checkpoints refer to
them, so they are kept. What each one means is described in the module that
builds it.
"""
from __future__ import annotations

# Popularity inputs: co-visitation and leave-user-out occurrence counts. The popularity-light
# view of the ranker drops these eight and reads EXPOSURE_INPUT instead.
POPULARITY_INPUTS = (
    "seq_cv_invd20", "seq_cv_pop", "seq_cv_norm",
    "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo",
)
FEED_CONTEXT_INPUTS = (
    "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
    "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent",
    "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw",
    "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10",
)
# The last one, pre2_fe, is read only for the presence flag "this row has encoder inputs".
ENCODER_INPUTS = ("pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32", "pre2_fe")

# The 34 arrays the full-view ranker reads, for train, val and test.
RANKER_INPUTS = POPULARITY_INPUTS + FEED_CONTEXT_INPUTS + ENCODER_INPUTS

# The one array the popularity-light view reads in place of POPULARITY_INPUTS.
EXPOSURE_INPUT = "xp_imp_c50"

# The 42 arrays of the fine-tune stage, val and test only, in the column order of the fine-tune MLP.
FINETUNE_EXTRAS = (
    "centrality",
    "it_hist_all_luo", "it_hist_all_luo_rk", "it_hist_all_luo_zsl", "it_hist_all_luo_gap",
    *(f"it_hist_et{e}_luo" for e in range(1, 11)),
    "seq_cv_invd20", "seq_cv_invd20_rk", "seq_cv_pop",
    "seq_max_all", "seq_top3", "seq_argmax_hpos",
    "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
    "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent",
    "pre_fe", "pre_fe_z", "pre_new", "pre_new_z", "pre_hmax32_c", "pre_hmax8_c", "pre_fe_noid",
    "pre_id_contrib_v2", "pre_idmax32", "pre_idmax32_has", "pre_hmax32_sent",
)
