"""Category-level ranking quality on val: NN-implied category ranking vs simple category signals.

Category NDCG: per feed, rank the feed's categories by a score; relevance = category holds >= 1 positive;
pessimistic ties; evaluated on non-leak val feeds with >= 1 positive and >= 2 categories.
"""
import json, time
import numpy as np
from catlib import *

t0 = time.time()
v = load_view("val", cand_feats=["tte_uc_ctr", "tte_uc_imp", "cos_u", "maxsim32"])
v = build_groups(v)
leak = leak_mask_val()
npos_f = np.bincount(np.repeat(np.arange(len(v["M"])), v["M"]), weights=v["c_target"], minlength=len(v["M"]))
base = (~leak) & (npos_f > 0)
mask = base & (v["G"] >= 2)
print(f"feeds {len(v['M'])} leak {leak.sum()} haspos {int((npos_f>0).sum())} base {base.sum()} mask(G>=2) {mask.sum()}"
      f" | groups {len(v['g_cat'])} mean G {v['G'].mean():.2f}", flush=True)
# hist group stats
src = v["h_event_source"].astype(np.int64)
top_src = np.argsort(-np.bincount(src, minlength=N_SRC))[:6].astype(np.int64)
hg = hist_group_feats(v, top_src)
H = {n: hg[:, i] for i, n in enumerate(HIST_NAMES)}
gn = v["g_n"].astype(np.float64)
tb = 1e-6 * gn  # tie-break by category size (a free prior)
rng = np.random.default_rng(0)

preds = {"nn_trainonly": np.load(PREDS / "nn_final2_trainonly_val.npy").astype(np.float64),
         "nn_oof": np.load(PREDS / "nn_final2_val_oof.npy").astype(np.float64),
         "stk2": np.load(PREDS / "stk2_res4_clean_val.npy").astype(np.float64)}
preds["stk2"] = feed_z(v["off"], v["M"], preds["stk2"])
preds["v5blend"] = 0.5 * feed_z(v["off"], v["M"], preds["nn_oof"]) + 0.5 * preds["stk2"]

sig = {}
sig["random"] = None
sig["n_cands (feed cat share)"] = gn
sig["f3 (raw, pessimistic ties)"] = gmax(v, v["c_f3"])
sig["f3 + tb(n)"] = gmax(v, v["c_f3"]) + tb
sig["f1 max + tb"] = gmax(v, v["c_f1"]) + tb
sig["f1 mean + tb"] = gmean(v, v["c_f1"]) + tb
sig["hist cnt all + tb"] = H["h_cnt"] + tb
sig["hist cnt last20 + tb"] = H["h_cnt20"] + tb
sig["hist cnt last5 + tb"] = H["h_cnt5"] + tb
sig["hist exp-decay hl10 + tb"] = H["h_exp10"] + tb
sig["hist recency (-min hpos) + tb"] = -H["h_minh"] + tb
for e in (1, 2, 3, 4, 5, 10):
    sig[f"hist cnt et{e} + tb"] = H[f"h_et{e}"] + tb
sig["tte_uc_ctr (train user x cat CTR) + tb"] = gmax(v, np.nan_to_num(v["x_tte_uc_ctr"], nan=0.0159)) + tb
sig["cos_u mean + tb"] = gmean(v, v["x_cos_u"]) + tb
sig["maxsim32 max + tb"] = gmax(v, np.nan_to_num(v["x_maxsim32"], nan=-1)) + tb
for name, s in preds.items():
    sig[f"{name}: max"] = gmax(v, s)
    sig[f"{name}: mean"] = gmean(v, s)
    for T in (0.5, 1.0, 2.0):
        sig[f"{name}: LSE T={T}"] = T * glse(v, s / T)

res = {}
ref = {}
for name, g in sig.items():
    if g is None:
        nd = np.mean([cat_ndcg(v, rng.random(len(gn))) for _ in range(5)], axis=0)
        top1 = None
    else:
        nd = cat_ndcg(v, g)
    # top-1 category hit rate (exact, with pessimistic ties ignored: argmax with tb)
    res[name] = nd
    m, se = mse(nd, mask)
    print(f"{name:45s} catNDCG {m:.4f} ± {se:.4f}", flush=True)

out = {}
for name, nd in res.items():
    m, se = mse(nd, mask)
    d_f3, se_f3 = paired(nd, res["f3 + tb(n)"], mask)
    d_nn, se_nn = paired(nd, res["nn_trainonly: LSE T=1.0"], mask)
    out[name] = dict(catndcg=m, se=se, d_vs_f3=d_f3, se_vs_f3=se_f3, d_vs_nnLSE=d_nn, se_vs_nnLSE=se_nn)

# top-1 hit rate and "category of first positive" rank for selected signals
def top1_hit(g):
    gs = g + 1e-9 * rng.random(len(g))
    best = np.full(len(v["M"]), -np.inf); np.maximum.at(best, v["g_feed"], gs)
    ist = gs == best[v["g_feed"]]
    hit = np.bincount(v["g_feed"], weights=ist * v["g_y"], minlength=len(v["M"]))
    return hit
for name in ["n_cands (feed cat share)", "f3 + tb(n)", "f1 max + tb", "hist exp-decay hl10 + tb",
             "nn_trainonly: LSE T=1.0", "nn_trainonly: max", "nn_oof: LSE T=1.0", "v5blend: LSE T=1.0"]:
    h = top1_hit(sig[name])
    out[name]["top1_hit"] = float(h[mask].mean())
    print(f"top1 hit {name:40s} {h[mask].mean():.4f}")

# item-level reference numbers on the same masks
for name, s in preds.items():
    nd = item_ndcg(v, s)
    out[f"ITEM {name}"] = dict(all_nonleak=mse(nd, ~leak), base=mse(nd, base), mask=mse(nd, mask))
    # oracle: positives' categories first, model order inside
    orc = s + 100.0 * v["g_y"][v["cand_g"]]
    ndo = item_ndcg(v, orc)
    out[f"ITEM {name}"]["oracle_cat"] = mse(ndo, ~leak)
    print(f"ITEM {name}: nonleak {nd[~leak].mean():.4f} | mask {nd[mask].mean():.4f} | cat-oracle {ndo[~leak].mean():.4f}")

# stats about category structure
st = dict(mean_G=float(v["G"][base].mean()), share_G1=float((v["G"][base] == 1).mean()),
          mean_pos_cats=float(np.bincount(v["g_feed"], weights=v["g_y"], minlength=len(v["M"]))[base].mean()),
          share_cands_in_pos_cat=float((v["g_y"][v["cand_g"]].astype(float) * 1).sum() / len(v["cand_g"])))
out["_stats"] = st
print(st)
json.dump(out, open(OUT / "01_catlevel_val.json", "w"), indent=1)
np.savez(OUT / "01_val_catndcg_perfeed.npz", mask=mask, **{k.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace("=", ""): x for k, x in res.items()})
print("done", time.time() - t0)
