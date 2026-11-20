"""List the fine-tune's inputs (xt of a W16 member + the n2c_pre_v3 extras) and which have counterfactual files. CPU, read-only."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
R = Path("<repo>")
ITEM_PRESET = ["it_cand_cnt_split", "it_cand_cnt_split_rk", "it_cand_cnt_split_z", "it_cand_cnt_split_gap",
               "it_cand_cnt_train", "it_hist_all", "it_hist_all_rk", "it_hist_all_z", "it_hist_all_gap",
               *[f"it_hist_et{i}" for i in range(1, 11)], "it_tr_coec", "it_tr_coec_rk", "it_tr_coec_z",
               "it_tr_coec_gap", "it_tr_ctr", "it_tr_pos", "u_tr_ctr", "u_tr_feeds", "hour", "centrality"]
ITEM_SAFE = [n for n in ITEM_PRESET if n not in ("it_cand_cnt_split", "it_cand_cnt_split_gap")]
ITEM_LUO = [n for n in ITEM_SAFE if not n.startswith("it_hist_")] + [
    "it_hist_all_luo", "it_hist_all_luo_rk", "it_hist_all_luo_z", "it_hist_all_luo_gap",
    *[f"it_hist_et{i}_luo" for i in range(1, 11)]]
COVIS_PRESET = ["seq_cv_invd20", "seq_cv_invd20_rk", "seq_cv_invd20_z", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1",
                "seq_cv_seqs", "seq_cv_pop", "seq_cv_norm"]
SEQ_PRESET = ["seq_max_all", "seq_top3", "seq_dcos_hl5", "seq_nd90_cnt", "seq_argmax_hpos"]
CTX_PRESET = ["ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_dupb_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_u",
              "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
              "ctx_nd_max", "ctx_nd_mean", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent"]
ITEM_CLEAN = [n for n in ITEM_LUO if not n.startswith("it_cand_cnt_split")]
PRE = ["pre_fe", "pre_fe_z", "pre_new", "pre_new_z", "pre_hmax32", "pre_hmax8", "pre_fe_noid", "pre_id_contrib", "pre_idmax32"]
n2clean = ITEM_CLEAN + COVIS_PRESET + SEQ_PRESET + CTX_PRESET
n2c_pre = n2clean + PRE
v2 = [n for n in n2c_pre if n != "hour" and not n.startswith(("it_tr_", "u_tr_", "it_cand_cnt_train"))]
v2 = [("pre_id_contrib_v2" if n == "pre_id_contrib" else n) for n in v2]
PHANTOM = {"seq_cv_invd20_z", "seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "seq_dcos_hl5", "seq_nd90_cnt",
           "ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_dupb_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_u", "ctx_nd_mean"}
REN = {"pre_hmax32": "pre_hmax32_c", "pre_hmax8": "pre_hmax8_c", "it_hist_all_luo_z": "it_hist_all_luo_zsl"}
v3 = [REN.get(n, n) for n in v2 if n not in PHANTOM and n != "seq_cv_norm"] + ["pre_idmax32_has", "pre_hmax32_sent"]
FC = R / "nn_x/w5/feat_cache"
print("n2c_pre_v3:", len(v3))
for n in v3:
    ex = [(FC / s / f"{n}.npy").exists() for s in ("val", "test")]
    print(f"  {n:28s} val/test exists {ex}")
ck = torch.load(R / "nn_x/w5/ckpt/w5_e10_kd05w192_s951.pt", map_location="cpu", weights_only=False)
xt = ck["cfg"].get("xt_names")
print("xt names", len(xt))
for i, n in enumerate(xt):
    print(f"  {i:2d} {n}")
print("cfg keys", {k: v for k, v in ck["cfg"].items() if k != "xt_names"})
print("args", ck["args"])
