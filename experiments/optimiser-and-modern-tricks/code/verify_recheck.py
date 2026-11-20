# Independent re-computation of every paired difference claimed by the three groups, from the saved
# per-candidate predictions, with two extra standard errors: seeds treated as independent, and the spread
# between the per-seed differences added. It also prints same-recipe seed pairs as a null calibration.
# Run on CPU from lab/nn_x/verify; the command line is not on record.
# Produced recheck.out (copied to ../results/verify_recheck.txt).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/verify/recheck.py
#
"""Independent recomputation of every claimed paired delta from the saved per-candidate predictions."""
import os, sys, itertools
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
FL = "<repo>/data/flat"
f = pl.read_parquet(f"{FL}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{FL}/val/cand_target.npy").astype(np.float64)
O = "<repo>/preds_x/optim"; A = "<repo>/preds_x/attn"; F = "<repo>/preds_x/feat"
cache = {}
def val(p):
    if p not in cache:
        s = np.load(p); assert s.shape == y.shape, (p, s.shape)
        cache[p] = ndcg_per_feed(s, y, off, M)
    return cache[p]
hcache = {}
def hold(p):
    if p not in hcache:
        h = np.load(p); Mh = h["M"].astype(np.int64); oh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
        hcache[p] = (ndcg_per_feed(h["scores"], h["y"], oh, Mh), Mh, h["y"])
    return hcache[p]
def cmp(name, base, treat, seeds=(23, 24), vs="_trainonly_val.npy", ts=None):
    ts = ts or vs
    dv, dh, per = [], [], []
    for s in seeds:
        b, t = base.format(seed=s), treat.format(seed=s)
        vb, vt = val(b + vs), val(t + ts); dv.append(vt - vb)
        ps = f"s{s}: v {vb.mean():.5f}->{vt.mean():.5f} ({(vt-vb).mean():+.5f})"
        if os.path.exists(b + "_holdout.npz") and os.path.exists(t + "_holdout.npz"):
            hb, Mb, yb = hold(b + "_holdout.npz"); ht, Mt, yt = hold(t + "_holdout.npz")
            assert (Mb == Mt).all() and (yb == yt).all()
            dh.append(ht - hb); ps += f" h {hb.mean():.5f}->{ht.mean():.5f} ({(ht-hb).mean():+.5f})"
        per.append(ps)
    out = [name]
    for k, D in (("val", dv), ("hold", dh)):
        if not D: continue
        d = np.mean(D, 0); se = d.std(ddof=1) / np.sqrt(len(d))
        seed_means = [x.mean() for x in D]
        # naive independent-seed SE and a conservative SE adding the between-seed spread of the per-seed deltas
        se_ind = np.sqrt(sum((x.std(ddof=1) / np.sqrt(len(x))) ** 2 for x in D)) / len(D)
        se_btw = np.std(seed_means, ddof=1) / np.sqrt(len(D)) if len(D) > 1 else np.nan
        se_cons = np.sqrt(se ** 2 + se_btw ** 2)
        out.append(f"{k}: d={d.mean():+.5f} se={se:.5f} z={d.mean()/se:+.2f} | se_ind={se_ind:.5f} se_btw={se_btw:.5f} z_cons={d.mean()/se_cons:+.2f}")
    print(" ; ".join(out)); print("     " + " | ".join(per))
B = O + "/nn_base_s{seed}"; S2B = O + "/nn_s2_base_s{seed}"
print("=== ATTN (train-only vs base) ===")
for n in ["attn_gate", "attn_fox", "attn_qkss", "attn_retr", "attn_gate_retr", "attn_moe", "attn_gate_retr_moe", "attn_retrmeta"]:
    cmp(n, B, A + f"/nn_{n}_s{{seed}}")
cmp("A1 incr over A4", A + "/nn_attn_retr_s{seed}", A + "/nn_attn_gate_retr_s{seed}")
cmp("A5 incr over A1+A4", A + "/nn_attn_gate_retr_s{seed}", A + "/nn_attn_gate_retr_moe_s{seed}")
cmp("A4meta vs A4", A + "/nn_attn_retr_s{seed}", A + "/nn_attn_retrmeta_s{seed}")
print("=== ATTN stage 2 (vs s2_base) ===")
for n in ["attn_retr", "attn_gate_retr", "attn_gate_retr_moe"]:
    cmp("s2_" + n, S2B, A + f"/nn_s2_{n}_s{{seed}}")
print("=== ATTN fine-tune OOF (vs ftb) ===")
for n in ["attn_retr", "attn_gate_retr", "attn_gate_retr_moe"]:
    cmp("ftb_" + n, O + "/nn_ftb_s{seed}", A + f"/nn_ftb_{n}_s{{seed}}", vs="_val_oof.npy")
cmp("ftb A1 incr", A + "/nn_ftb_attn_retr_s{seed}", A + "/nn_ftb_attn_gate_retr_s{seed}", vs="_val_oof.npy")
print("=== OPTIM ===")
for n in ["base_s{seed}_ema0.999", "optim_muon_s{seed}", "optim_muon_lr2_s{seed}", "optim_kd_s{seed}", "optim_wsd_s{seed}", "optim_muwsd_s{seed}", "optim_mu2wsd_s{seed}", "optim_mu2wsdkd_s{seed}"]:
    cmp(n + " vs base", B, O + "/nn_" + n)
for n in ["optim_wsd_s{seed}", "optim_muwsd_s{seed}", "s2_optim_muon_s{seed}"]:
    cmp(n + " vs s2_base", S2B, O + "/nn_" + n)
cmp("muon lr2 vs lr1", O + "/nn_optim_muon_s{seed}", O + "/nn_optim_muon_lr2_s{seed}")
cmp("mu2wsd vs muwsd", O + "/nn_optim_muwsd_s{seed}", O + "/nn_optim_mu2wsd_s{seed}")
cmp("s2 on muwsd", O + "/nn_optim_muwsd_s{seed}", O + "/nn_s2_optim_muwsd_s{seed}")
cmp("s2 on wsd", O + "/nn_optim_wsd_s{seed}", O + "/nn_s2_optim_wsd_s{seed}")
print("=== OPTIM fine-tune OOF (vs ftb) ===")
for n in ["fta_s{seed}", "ftlp_s{seed}", "fta_optim_muon_s{seed}", "ftb_optim_muon_s{seed}", "ftmu_optim_muon_s{seed}", "fta_optim_muwsd_s{seed}", "fta_optim_mu2wsd_s{seed}"]:
    cmp(n, O + "/nn_ftb_s{seed}", O + "/nn_" + n, vs="_val_oof.npy")
cmp("fta muwsd vs fta base", O + "/nn_fta_s{seed}", O + "/nn_fta_optim_muwsd_s{seed}", vs="_val_oof.npy")
print("=== FEAT ===")
for n in ["feat_ple", "feat_prop", "feat_sid", "feat_ts", "feat_tabm"]:
    cmp(n, B, F + f"/nn_{n}_s{{seed}}")
print("=== NULL CALIBRATION: same config, different seeds / reruns ===")
nulls = [("base s24 vs s23", O + "/nn_base_s23", O + "/nn_base_s24"),
         ("base s25 vs s23", O + "/nn_base_s23", F + "/nn_feat_base_s25"),
         ("base s25 vs s24", O + "/nn_base_s24", F + "/nn_feat_base_s25"),
         ("rerun v2_feed_k9 vs base_s23", "<repo>/preds/nn_v2_feed_k9", O + "/nn_base_s23")]
for n, b, t in nulls:
    vb, vt = val(b + "_trainonly_val.npy"), val(t + "_trainonly_val.npy"); d = vt - vb
    msg = f"{n}: val d={d.mean():+.5f} se={d.std(ddof=1)/np.sqrt(len(d)):.5f} z={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f}"
    if os.path.exists(b + "_holdout.npz") and os.path.exists(t + "_holdout.npz"):
        hb = hold(b + "_holdout.npz")[0]; ht = hold(t + "_holdout.npz")[0]; dh = ht - hb
        msg += f" | hold d={dh.mean():+.5f} se={dh.std(ddof=1)/np.sqrt(len(dh)):.5f} z={dh.mean()/(dh.std(ddof=1)/np.sqrt(len(dh))):+.2f}"
    print(msg)
P = "<repo>/preds/nn_"
for grp in [["v4_cv", "v4_cv_s33", "v4_cv_s38"], ["aux_sim_s4", "aux_sim_s5", "aux_sim_s9", "aux_sim_s11", "aux_sim_s12"]]:
    zs = []
    for a_, b_ in itertools.combinations(grp, 2):
        d = val(P + b_ + "_trainonly_val.npy") - val(P + a_ + "_trainonly_val.npy")
        zs.append(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        print(f"  {b_} - {a_}: d={d.mean():+.5f} z={zs[-1]:+.2f}")
    print(f"  group {grp[0]}: rms z = {np.sqrt(np.mean(np.square(zs))):.2f} over {len(zs)} pairs")
