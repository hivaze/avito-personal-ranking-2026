"""Battery of simple additive rules score + w*g(x), cross-fitted over kit.folds (choose w on 4 folds)."""
import json
from util import *

df = load()
c = Ctx(df)
s = c.base
W = [-0.3, -0.2, -0.15, -0.1, -0.07, -0.05, -0.03, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]


def col(n):
    return df[n].to_numpy().astype(np.float64)


def nz(a, v=0.0):
    a = a.copy(); a[~np.isfinite(a)] = v; return a


G = {}
G["f1"] = col("f1")
G["f1_gt74"] = (col("f1") > 0.74).astype(float)
G["cur0"] = (col("cat_user_rank") == 0).astype(float)
G["hmax_lt70"] = (nz(col("hmax"), 0) < 0.70).astype(float)
G["hmax"] = nz(col("hmax"), 0.5)
G["hty4"] = (col("hmax_type") == 4).astype(float)
G["hidx0"] = (col("hmax_idx") == 0).astype(float)
G["hidx_le2"] = ((col("hmax_idx") >= 0) & (col("hmax_idx") <= 2)).astype(float)
G["hd95_log"] = np.log1p(col("n_hdup95"))
G["hd95_ge1"] = (col("n_hdup95") >= 1).astype(float)
G["cd95_1_3"] = ((col("n_cand_dup95") >= 1) & (col("n_cand_dup95") <= 3)).astype(float)
G["cd95_log"] = np.log1p(col("n_cand_dup95"))
G["f2_gt8"] = (col("f2") > 0.8).astype(float)
G["f2"] = col("f2")
G["f2_mid"] = ((col("f2") > 0.2) & (col("f2") <= 0.8)).astype(float)
G["ccon_ge95"] = (nz(col("c_contact"), 0) >= 0.95).astype(float)
G["ucos"] = nz(col("ucos"), 0.45)
G["ucos_lowq"] = (nz(col("ucos"), 0.45) < 0.4028).astype(float)
G["sft_minus_sto"] = col("s_ft") - col("s_to")
G["scn"] = col("same_cat_newest").astype(float)
G["cv_log"] = np.log1p(col("seq_cv_invd20"))
G["cold"] = (col("xp_imp_c50") == 0).astype(float)
G["f4_fresh10"] = (col("f4") <= 0.1224).astype(float)
G["cmax_95_98"] = ((col("cand_maxcos") > 0.95) & (col("cand_maxcos") <= 0.98)).astype(float)
G["hsrc6"] = (col("hmax_src") == 6).astype(float)
G["f5"] = col("f5")
G["cat_hist_share"] = col("cat_hist_share")
G["hidx_32_100"] = ((col("hmax_idx") > 32) & (col("hmax_idx") <= 100)).astype(float)

res = {}
for name, g in G.items():
    r = c.crossfit(lambda w: s + w * g + 0 * s, W)
    res[name] = dict(cf=r["cf"], se=r["se"], z=r["z"], chosen=r["chosen"], ins_best=r["ins_best"], ins=r["ins_delta"])
    print(f"{name:16s} cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins_best={r['ins_best']} ins={r['ins_delta']:+.5f}", flush=True)
json.dump(res, open("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/battery1.json", "w"), indent=1)
print("variants:", len(G) * len(W))
