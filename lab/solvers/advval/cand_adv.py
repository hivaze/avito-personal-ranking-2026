"""Candidate-level adversarial validation on the live model inputs.
  NN train-stage inputs: f1..f5, category, M, H, XT_PRESETS['cvnd'] (nn/train.py:32), cvnd_prew extras (3 of the 6
  nn_final5 members), cvnd_pre2s extras (nn_final7), embedding proxies (maxsim32, cos_u, cos_h10, cat_cnt_20, emb_missing).
  Fine-tune extras: preset n2c_pre (nn/finetune.py:153; the 51 names that exist on disk), val/test only.
Rows: up to 30 random candidates per feed (equal feed weight). Groups: user (user 0 -> feed).
Writes OUT/cand_adv.json and OUT/cand_oof_<comp>_<set>.npz."""
import json, time
from common_av import *
from adv_lib import *

t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

CVND = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo",
        "it_hist_et3_luo", "it_hist_et4_luo", "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev",
        "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w",
        "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128",
        "ctx_hs_e1", "ctx_hs_t10"]
PREW = ["prew_fe_z", "prew_new_z", "prew_fe", "prew_new", "prew_hmax32"]
PRE2S = ["pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32"]
EMBP = ["maxsim32", "cos_u", "cos_h10", "cat_cnt_20", "emb_missing"]
N2C_PRE = ['it_cand_cnt_train', 'it_tr_coec', 'it_tr_coec_rk', 'it_tr_coec_z', 'it_tr_coec_gap', 'it_tr_ctr', 'it_tr_pos',
           'u_tr_ctr', 'u_tr_feeds', 'hour', 'centrality', 'it_hist_all_luo', 'it_hist_all_luo_rk', 'it_hist_all_luo_z',
           'it_hist_all_luo_gap', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo',
           'it_hist_et5_luo', 'it_hist_et6_luo', 'it_hist_et7_luo', 'it_hist_et8_luo', 'it_hist_et9_luo',
           'it_hist_et10_luo', 'seq_cv_invd20', 'seq_cv_invd20_rk', 'seq_cv_pop', 'seq_cv_norm', 'seq_max_all',
           'seq_top3', 'seq_argmax_hpos', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev',
           'ctx_nd_knn_u', 'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'pre_fe',
           'pre_fe_z', 'pre_new', 'pre_new_z', 'pre_hmax32', 'pre_hmax8', 'pre_fe_noid', 'pre_id_contrib', 'pre_idmax32']
BASE = ["f1", "f2", "f3", "f4", "f5", "category", "M", "H"] + CVND + PREW + PRE2S + EMBP
EXTRA = [n for n in N2C_PRE if n not in BASE and n != "hour"]
TIME = ["ts", "hour", "weekday"]
COV_PREFIX = ("it_hist", "seq_cv", "prew_", "it_tr_", "it_cand_cnt_train", "pre_idmax32", "pre_id_contrib", "u_tr_")
PER = 30


def sample_rows(split, fsel, seed):
    f, M, off = meta(split)
    rng = np.random.default_rng(seed)
    rows, fk = [], []
    for fi in fsel:
        m = M[fi]
        k = rng.choice(m, min(m, PER), replace=False) if m > PER else np.arange(m)
        rows.append(off[fi] + np.sort(k)); fk.append(np.full(len(k), fi))
    return np.concatenate(rows), np.concatenate(fk)


def load_block(split, fsel, with_extra, seed):
    f, M, off = meta(split)
    rows, fk = sample_rows(split, fsel, seed)
    ts = f["timestamp"].to_numpy().astype(np.int64)[fk]
    cols = {"ts": (ts - BOUND) / 86400.0, "hour": ((ts + MSK) % 86400) // 3600, "weekday": (((ts + MSK) // 86400) + 3) % 7,
            "M": M[fk], "H": f["H"].to_numpy()[fk]}
    for i in range(1, 6):
        cols[f"f{i}"] = np.load(f"{FLAT}/{split}/cand_f{i}.npy", mmap_mode="r")[rows]
    cols["category"] = np.load(f"{FLAT}/{split}/cand_category_id.npy", mmap_mode="r")[rows].astype(np.float32)
    for n in CVND + PREW + PRE2S + EMBP + (EXTRA if with_extra else []):
        cols[n] = load_feat(split, n, rows)
    uid = f["user_id"].to_numpy()[fk]
    return cols, uid, fk


def main():
    fv, Mv, _ = meta("val"); ft, Mt, _ = meta("test")
    tt = ft["timestamp"].to_numpy()
    pub = np.where(tt < BOUND)[0]; pri = np.where(tt >= BOUND)[0]
    val_all = np.arange(len(fv))
    tr = train_random_like()
    ftr = meta("train")[0]
    dtr = (ftr["timestamp"].to_numpy().astype(np.int64) + MSK) // 86400
    d9 = 20705  # MSK day number of 2026-09-09 (20702 = Sep 6, build_feed.py log)
    tr9 = tr[dtr[tr] >= d9]
    tr9 = np.sort(np.random.default_rng(1).choice(tr9, min(30000, len(tr9)), replace=False))
    log("train rl Sep9+ sampled feeds", len(tr9), "min day", dtr[tr9].min())
    blocks = {
        "val": load_block("val", val_all, True, 11),
        "public": load_block("test", pub, True, 12),
        "private": load_block("test", pri, True, 13),
        "val50": load_block("val", val_all[Mv >= 50], False, 14),
        "private50": load_block("test", pri[Mt[pri] >= 50], False, 15),
        "train9": load_block("train", tr9, False, 16),
    }
    log("loaded", {k: len(b[1]) for k, b in blocks.items()})
    comps = {"a_val_vs_private": ("val", "private", True), "b_val_vs_public": ("val", "public", True),
             "c_public_vs_private": ("public", "private", True),
             "d_train9_vs_val50": ("train9", "val50", False), "e_train9_vs_private50": ("train9", "private50", False)}
    out = {}
    for comp, (A, B, ext) in comps.items():
        allf = TIME + BASE + (EXTRA if ext else [])
        sets = {"all": allf, "notime": [c for c in allf if c not in TIME],
                "nocov": [c for c in allf if c not in TIME and not c.startswith(COV_PREFIX)]}
        ca, ua, fa = blocks[A]; cb, ub, fb = blocks[B]
        y = np.r_[np.zeros(len(ua)), np.ones(len(ub))].astype(np.int8)
        # group key: user; user-0 rows grouped by (block, feed)
        fkey = np.r_[fa, fb + 10 ** 8]
        g = groups_of(np.r_[ua, ub], fkey)
        out[comp] = {"nA": int(len(ua)), "nB": int(len(ub))}
        for sname, fl in sets.items():
            X = np.stack([np.r_[np.asarray(ca[n], np.float32), np.asarray(cb[n], np.float32)] for n in fl], 1)
            cat = ["category"] if "category" in fl else None
            res = run_adv(X, y, g, fl, P_CAND, 300, cat=cat, shap_rows=30000)
            uni = univariate(X, y, fl)
            out[comp][sname] = dict(auc=res["auc"], fold_auc=res["fold_auc"].tolist(),
                                    imp={n: dict(shap=float(res["shap"][j]), gain=float(res["gain"][j]), **uni[n])
                                         for j, n in enumerate(fl)}, table=top_table(res, uni, 15))
            np.savez(f"{OUT}/cand_oof_{comp}_{sname}.npz", oof=res["oof"].astype(np.float32), y=y)
            log(comp, sname, f"AUC {res['auc']:.4f} folds {np.round(res['fold_auc'], 4)}")
            del X
        json.dump(out, open(f"{OUT}/cand_adv.json", "w"), indent=1, default=float)
    for comp in comps:
        print(f"\n## {comp} (A rows {out[comp]['nA']}, B rows {out[comp]['nB']})")
        for s in ["all", "notime", "nocov"]:
            print(f"{s}: AUC {out[comp][s]['auc']:.4f}")
        print("notime"); print(out[comp]["notime"]["table"])
        print("nocov"); print(out[comp]["nocov"]["table"])


if __name__ == "__main__":
    main()
