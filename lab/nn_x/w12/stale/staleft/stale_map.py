"""staleft: which val input files the stale view reads (name -> counterfactual file). Pure paths, no torch import.
STALE_XT = the project's stale-val scenario 'both_pre2both' (nn_x/w5/stale_pred.py OVR, unchanged).
STALE_XE = the same scenario for the fine-tune extras (n2c_pre_v3): existing both_* files + the columns built by build_cf.py.
Columns listed in logs/gate_exclude.txt (one name per line; written only if the label-free gate fails a column) are removed
from STALE_XE, i.e. they stay fresh."""
from pathlib import Path

ROOT = Path("<repo>")
HERE = ROOT / "nn_x/w12/stale/staleft"
CF_BOTH = ROOT / "nn_x/r6/stale_cf"
CF_PRE2 = ROOT / "solvers/out/stale/cf"
CF_MINE = HERE / "cf"

_B = ("it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo", "seq_cv_invd20", "seq_cv_norm", "seq_cv_pop")
STALE_XT = {**{n: CF_BOTH / f"both_{n}.npy" for n in _B},
            **{n: CF_PRE2 / f"pre2both_{n}.npy" for n in ("pre2_fe", "pre2_fe_z", "pre2_fe_s")}}
_MINE = ["it_hist_all_luo_rk", "it_hist_all_luo_zsl", "it_hist_all_luo_gap", *[f"it_hist_et{i}_luo" for i in range(5, 11)],
         "seq_cv_invd20_rk", "pre_fe", "pre_fe_z", "pre_id_contrib_v2", "pre_idmax32", "pre_idmax32_has"]
STALE_XE = {**{n: CF_BOTH / f"both_{n}.npy" for n in _B if n != "seq_cv_norm"}, **{n: CF_MINE / f"both_{n}.npy" for n in _MINE}}
_ex = HERE / "logs/gate_exclude.txt"
if _ex.exists():
    for _n in _ex.read_text().split():
        STALE_XE.pop(_n, None)

N2C_PRE_V3 = ['centrality', 'it_hist_all_luo', 'it_hist_all_luo_rk', 'it_hist_all_luo_zsl', 'it_hist_all_luo_gap',
              *[f'it_hist_et{i}_luo' for i in range(1, 11)], 'seq_cv_invd20', 'seq_cv_invd20_rk', 'seq_cv_pop', 'seq_max_all',
              'seq_top3', 'seq_argmax_hpos', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u',
              'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'pre_fe', 'pre_fe_z', 'pre_new',
              'pre_new_z', 'pre_hmax32_c', 'pre_hmax8_c', 'pre_fe_noid', 'pre_id_contrib_v2', 'pre_idmax32', 'pre_idmax32_has',
              'pre_hmax32_sent']
XT_NAMES = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo',
            'it_hist_et4_luo', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u', 'ctx_nd_knn_u_dev',
            'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'ctx_hn_top1w', 'ctx_hn_rrw', 'ctx_hn_bestrk', 'ctx_hn_margin',
            'ctx_hn_soft05', 'ctx_hn_e1_rrw', 'ctx_hs_mutual', 'ctx_hs_k128', 'ctx_hs_e1', 'ctx_hs_t10', 'pre2_fe_z', 'pre2_new_z',
            'pre2_fe_s', 'pre2_new_s', 'pre2_hmax32', 'has:pre2_fe']
