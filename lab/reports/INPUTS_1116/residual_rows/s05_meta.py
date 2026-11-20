"""INVENTORY B, step 5: per-column metadata of the feature store.
Parses reports/featrev_inventory.md Appendix A (256 features: builder, splits, users, in-model analogue, residual reads of 11-11, leak status)
and adds: is the column one of the 34 train-stage inputs / the 42 fine-tune extras / an in-model column, and the array row counts on disk.
Output: out/store_meta.tsv. Read-only."""
import re, os, json
import numpy as np
from numpy.lib import format as npf
ROOT = "<repo>"
XT34 = ("seq_cv_invd20 seq_cv_pop seq_cv_norm it_hist_all_luo it_hist_et1_luo it_hist_et2_luo it_hist_et3_luo it_hist_et4_luo ctx_nd_hcent ctx_nd_knn_f1 ctx_nd_knn_h "
        "ctx_nd_knn_h_dev ctx_nd_knn_u ctx_nd_knn_u_dev ctx_nd_max ctx_nd_top5h ctx_nd_top5u ctx_nd_ucent ctx_hn_top1w ctx_hn_rrw ctx_hn_bestrk ctx_hn_margin ctx_hn_soft05 "
        "ctx_hn_e1_rrw ctx_hs_mutual ctx_hs_k128 ctx_hs_e1 ctx_hs_t10 pre2_fe_z pre2_new_z pre2_fe_s pre2_new_s pre2_hmax32").split()   # + the flag has:pre2_fe
FT42 = ['centrality', 'it_hist_all_luo', 'it_hist_all_luo_rk', 'it_hist_all_luo_zsl', 'it_hist_all_luo_gap', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'it_hist_et5_luo', 'it_hist_et6_luo', 'it_hist_et7_luo', 'it_hist_et8_luo', 'it_hist_et9_luo', 'it_hist_et10_luo', 'seq_cv_invd20', 'seq_cv_invd20_rk', 'seq_cv_pop', 'seq_max_all', 'seq_top3', 'seq_argmax_hpos', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u', 'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'pre_fe', 'pre_fe_z', 'pre_new', 'pre_new_z', 'pre_hmax32_c', 'pre_hmax8_c', 'pre_fe_noid', 'pre_id_contrib_v2', 'pre_idmax32', 'pre_idmax32_has', 'pre_hmax32_sent']
A = {}
on = False
for l in open(f"{ROOT}/reports/featrev_inventory.md"):
    if l.startswith("## Appendix A"): on = True; continue
    if l.startswith("## Appendix B"): break
    if on and l.startswith("| ") and not l.startswith("| feature") and not l.startswith("|---"):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) >= 16:
            A[c[0]] = dict(family=c[1], builder=c[2], splits=c[3], usedby=c[5], inmodel=c[6], sauc=c[7], resid_mix_lin=c[11], resid_mix_gbm=c[12], resid_to=c[13], resid_trH=c[14], leak=c[15])
def rows(p):
    if not os.path.exists(p): return ""
    with open(p, "rb") as fp:
        v = npf.read_magic(fp); sh = (npf.read_array_header_1_0(fp) if v == (1, 0) else npf.read_array_header_2_0(fp))[0]
    return "x".join(str(s) for s in sh)
out = open(f"{ROOT}/reports/INPUTS_1116/residual_rows/out/store_meta.tsv", "w")
out.write("name\tdir\trows_train\trows_val\trows_test\tin_xt34\tin_ft42\tfamily\tbuilder\tusedby_1111\tinmodel_analogue\tsAUC_val_trH\tresid_mix_LIN_1111\tresid_mix_GBM_1111\tresid_trH_LIN_1111\tleak_status_1111\n")
n = 0
for d in ("data/feats", "data/feats_new"):
    names = set()
    for s in ("train", "val", "test"):
        names |= {f[:-4] for f in os.listdir(f"{ROOT}/{d}/{s}") if f.endswith(".npy")}
    for nme in sorted(names):
        a = A.get(nme, {})
        out.write("\t".join([nme, d] + [rows(f"{ROOT}/{d}/{s}/{nme}.npy") for s in ("train", "val", "test")] + [str(int(nme in XT34)), str(int(nme in FT42))]
                            + [a.get(k, "") for k in ("family", "builder", "usedby", "inmodel", "sauc", "resid_mix_lin", "resid_mix_gbm", "resid_trH", "leak")]) + "\n"); n += 1
print("appendix A rows parsed:", len(A), "| store columns written:", n)
