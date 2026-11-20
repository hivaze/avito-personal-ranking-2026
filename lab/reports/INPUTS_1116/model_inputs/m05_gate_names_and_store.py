"""INVENTORY A, measurement M5 (CPU, read-only): (a) how the name-based input audit of src/presub_gate.py (check C; patterns
copied verbatim from that file, l.33-40) classifies names a trained-input arm could carry; (b) which per-candidate arrays
already exist for train AND val AND test in data/feats and data/feats_new and are not read by a current member.
Writes out/m05_gate_names_and_store.txt."""
import os, re, collections
R = "<repo>"; out = []
BANNED_ANY = [r"^it_hist_(all|et\d+)(_rk|_z|_gap)?$", r"^lh_", r"later"]
BANNED_TRAIN = [r"^pre_(fe|new|hmax|idmax|fe_noid|id_contrib)", r"_id_contrib"]
WARN_TRAIN = [r"^prew_fe$", r"^prew_new$", r"^pre2_fe$", r"^pre2_new$"]
WARN_ANY = [r"^als_", r"^hour$", r"^it_tr_", r"^u_tr_", r"^it_cand_cnt_train$"]
src = open(f"{R}/src/presub_gate.py").read()
for pat in BANNED_ANY + BANNED_TRAIN + WARN_TRAIN + WARN_ANY:
    assert pat in src, pat  # the copied patterns are the ones in the gate
def cls(f):
    g = f[4:] if f.startswith("has:") else f
    if any(re.search(p, g) for p in BANNED_ANY + BANNED_TRAIN): return "FAIL"
    if any(re.search(p, g) for p in WARN_ANY) or (not f.startswith("has:") and any(re.search(p, g) for p in WARN_TRAIN)): return "WARN"
    return "pass"
def xt_add_ok(n):  # nn_x/abl/abl_arms.py xt_add_names: seq_cv* / it_hist* names are refused
    return "refused by --xt_add" if n.startswith(("seq_cv", "it_hist")) else "accepted"
names = ("xp_imp_c50 xp_ctr_c tte_it_coec tte_it_ctr tte_it_imp tte_u_ctr tte_ui_pos tte_uc_ctr tte_c_ctr it_tr_coec u_tr_ctr it_cand_cnt_train "
         "it_cand_cnt_split hour tm_weekend ixh_mf1 seq_cvlag_invd20 hzcv_expo_slog src_expo_h it_hist_hz_all it_hist_all it_hist_et5_luo "
         "it_hist_all_luo_rk pre_fe pre2_fe pre2_fe_s pre2_id_contrib pre2_fe_noid clu_pmi1024_mean trd_kde_s6h als_exp5 centrality seq_max_all lh_any later_overlay").split()
out.append("(a) gate check C (name-based) and --xt_add name rule")
for n in names:
    out.append(f"   {n:22s} gate: {cls(n):5s} xt_add: {xt_add_ok(n)}")
S = {}
for d in ("data/feats", "data/feats_new", "nn/feat_cache"):
    for s in ("train", "val", "test"):
        p = f"{R}/{d}/{s}"
        S[(d, s)] = set(f[:-4] for f in os.listdir(p) if f.endswith(".npy")) if os.path.isdir(p) else set()
cur = set("seq_cv_invd20 seq_cv_pop seq_cv_norm it_hist_all_luo it_hist_et1_luo it_hist_et2_luo it_hist_et3_luo it_hist_et4_luo ctx_nd_hcent ctx_nd_knn_f1 ctx_nd_knn_h ctx_nd_knn_h_dev ctx_nd_knn_u ctx_nd_knn_u_dev ctx_nd_max ctx_nd_top5h ctx_nd_top5u ctx_nd_ucent ctx_hn_top1w ctx_hn_rrw ctx_hn_bestrk ctx_hn_margin ctx_hn_soft05 ctx_hn_e1_rrw ctx_hs_mutual ctx_hs_k128 ctx_hs_e1 ctx_hs_t10 pre2_fe_z pre2_new_z pre2_fe_s pre2_new_s pre2_hmax32 pre2_fe xp_imp_c50".split())
out.append("\n(b) arrays present for train AND val AND test")
for d in ("data/feats", "data/feats_new", "nn/feat_cache"):
    a = S[(d, "train")] & S[(d, "val")] & S[(d, "test")]
    out.append(f"   {d}: files train {len(S[(d, 'train')])} / val {len(S[(d, 'val')])} / test {len(S[(d, 'test')])}; in all three {len(a)}; of these not read by the wide or the pop1 student {len(a - cur)}")
    g = collections.defaultdict(list)
    for n in sorted(a - cur):
        g[n.split("_")[0]].append(n)
    for k in sorted(g):
        out.append(f"      {k:10s} {len(g[k]):3d}: {' '.join(g[k])}")
open(f"{R}/reports/INPUTS_1116/model_inputs/out/m05_gate_names_and_store.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out[:44]))
