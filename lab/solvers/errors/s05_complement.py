"""(3) Complementarity of existing solver outputs vs v11 on val.
Per solver: standalone NDCG, per-feed win/loss vs v11, oracle per-feed switch, z-corr, pre-declared blends
fz(v11) + w*fz(S) for w in {0.1, 0.2, 1.0} (paired delta +- SE, slices) and a nested choice of w in {0, .1, .2, 1}
over the nn/finetune.py user folds."""
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
s = np.load(f"{SCR}/out/v11_val.npy")
zs = fz(s, fid, M)
pf = pl.read_parquet(f"{SCR}/out/perfeed.parquet")
base = ndcg_per_feed(s, y, off, M)
uid = f["user_id"].to_numpy().astype(np.int64)
folds = list(GroupKFold(n_splits=5).split(np.arange(n), groups=uid))
fold_of = np.zeros(n, int)
for k, (_, ho) in enumerate(folds):
    fold_of[ho] = k

S = {  # name: list of pred files (equal z-mean if several)
    "FT5 (nn_final5 fta OOF)": ["nn_final5_val_oof"],
    "TO5 (nn_final5 train-only)": ["nn_final5_trainonly_val"],
    "FT5v2 (W3 hygiene re-FT)": ["nn_final5v2_val_oof"],
    "FT5cd": ["nn_final5cd_val_oof"], "TO5cd": ["nn_final5cd_trainonly_val"],
    "FT5cd12": ["nn_final5cd12_val_oof"], "TO5cd12": ["nn_final5cd12_trainonly_val"],
    "FT4": ["nn_final4_val_oof"], "TO4": ["nn_final4_trainonly_val"],
    "mix4 (v10 w/o pre)": ["nn_final4_val_oof", "nn_final4_trainonly_val"],
    "FT3": ["nn_final3_val_oof"], "TO3": ["nn_final3_trainonly_val"],
    "FT2": ["nn_final2_val_oof"], "TO2": ["nn_final2_trainonly_val"],
    "pre_final (seq encoder)": ["pre_final_val"],
    "xgb_x300_luo (train GBDT)": ["xgb_x300_luo_val"],
    "cb_tr300k_luo (YetiRank)": ["cb_tr300k_luo_val"],
    "stk2_res4_clean (val stacker)": ["stk2_res4_clean_val"],
    "v16_pre2 e3 (2 seeds)": ["nn_v16_pre2_e3_s23_trainonly_val", "nn_v16_pre2_e3_s42_trainonly_val"],
    "v13 lr1 mix0 (no-pool cooldown)": ["nn_v13_lr1_mix0_trainonly_val"],
    "v13 lr2 mix1": ["nn_v13_lr2_mix1_trainonly_val"],
    "v9 din+gru+deep e5": ["nn_v9_din_e5_s56_trainonly_val", "nn_v9_gru_e5_s52_trainonly_val", "nn_v9_deep_e5_s55_trainonly_val"],
    "bce (pointwise, r1)": ["nn_bce_trainonly_val"],
    "nohist (r1)": ["nn_nohist_trainonly_val"],
    "noset (r1)": ["nn_noset_trainonly_val"],
}
day = pf["day"].to_numpy()
pc = pf["poscold"].to_numpy()
ut = np.where(pf["u0"].to_numpy(), "u0", np.where(pf["cold_user"].to_numpy(), "new", "trainU"))
SL = {"M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1, "allcold": pc == "all_cold", "allwarm": pc == "all_warm",
      "mixed": pc == "mixed", "newU": ut == "new", "trainU": ut == "trainU", "u0": ut == "u0"}
rows, seg = [], []
cache = {}
for nm, files in S.items():
    z = np.mean([fz(np.load(f"{ROOT}/preds/{x}.npy"), fid, M) for x in files], 0)
    cache[nm] = z
    nd = ndcg_per_feed(z, y, off, M)
    d = nd - base
    win, loss_ = (d > 1e-9).mean(), (d < -1e-9).mean()
    orc = np.maximum(nd, base).mean() - base.mean()
    zc = np.corrcoef(z, zs)[0, 1]
    r = dict(solver=nm, ndcg=nd.mean(), d_alone=d.mean(), se_alone=d.std() / np.sqrt(n), win=win, loss=loss_, oracle=orc, zcorr=zc)
    perw = {}
    for w in (0.1, 0.2, 1.0):
        ndw = ndcg_per_feed(zs + w * z, y, off, M)
        dw = ndw - base
        perw[w] = dw
        r[f"w{w}"] = dw.mean()
        r[f"se{w}"] = dw.std() / np.sqrt(n)
    # nested choice of w in {0, .1, .2, 1}
    cand_w = [0.0, 0.1, 0.2, 1.0]
    D = np.stack([np.zeros(n)] + [perw[w] for w in cand_w[1:]], 0)
    nest = np.zeros(n)
    ch = []
    for k in range(5):
        trm = fold_of != k
        j = int(np.argmax(D[:, trm].mean(1)))
        ch.append(cand_w[j])
        nest[fold_of == k] = D[j, fold_of == k]
    r["nested"] = nest.mean()
    r["se_n"] = nest.std() / np.sqrt(n)
    r["w_chosen"] = "/".join(str(c) for c in ch)
    rows.append(r)
    sr = dict(solver=nm)
    for k, m in SL.items():
        sr[k + "_alone"] = d[m].mean()
        sr[k + "_w0.2"] = perw[0.2][m].mean()
        sr[k + "_w0.2_z"] = perw[0.2][m].mean() / (perw[0.2][m].std() / np.sqrt(m.sum()))
        sr[k + "_orc"] = np.maximum(nd[m], base[m]).mean() - base[m].mean()
    seg.append(sr)
    np.save(f"{SCR}/out/nd_{files[0]}.npy", nd)
T = pl.DataFrame(rows)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(300)
pl.Config.set_float_precision(5)
pl.Config.set_fmt_str_lengths(40)
print(T.select(["solver", "ndcg", "d_alone", "se_alone", "win", "loss", "oracle", "zcorr"]))
print(T.select(["solver", "w0.1", "se0.1", "w0.2", "se0.2", "w1.0", "se1.0", "nested", "se_n", "w_chosen"]))
T.write_csv("<repo>/solvers/out/errors/s05_complement.csv")
G = pl.DataFrame(seg)
G.write_csv("<repo>/solvers/out/errors/s05_segments.csv")
print("\nsegment deltas of the w=0.2 blend (z in brackets):")
for sr in seg:
    print(f"  {sr['solver'][:32]:32s} " + " ".join(f"{k} {sr[k + '_w0.2']:+.4f}({sr[k + '_w0.2_z']:+.1f})" for k in SL))
print("\nsegment deltas of the solver ALONE vs v11:")
for sr in seg:
    print(f"  {sr['solver'][:32]:32s} " + " ".join(f"{k} {sr[k + '_alone']:+.4f}" for k in SL))
print("\nsegment per-feed oracle (max(v11, S)) gain:")
for sr in seg:
    print(f"  {sr['solver'][:32]:32s} " + " ".join(f"{k} {sr[k + '_orc']:+.4f}" for k in SL))
print("\nsegment base v11 NDCG and share:", " ".join(f"{k} {base[m].mean():.4f}/{m.mean():.3f}" for k, m in SL.items()))
