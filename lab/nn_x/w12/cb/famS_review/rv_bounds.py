"""famS_review step 3: can the negative verdict be broken by ANY combination of the family's saved predictions? (undeclared looks, for bounding only; never a choice)
 (a) level of wider z-mean bundles (all famS models; the top-level ones; with the LightGBM students) against the stage-2 gate val >= 0.3900
 (b) in-sample UPPER BOUND of the nats of the family: conditional logit [zW, all famS models as separate covariates] (free weights fitted on the witness itself)
 (c) other encodings of the best student: raw centred score next to z (the per-feed scale that z removes), rank-normal score
 (d) joint read by M bin on val; cross-witness coefficients (fit on one witness, read on the other) for the bundle and the hard-label tree"""
import sys, json, time
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/famS_review")
import rv_own as o
from rv_own import ROOT, FAM, INFRA, pse
t0 = time.time()
W = o.witnesses(); ZW = {sp: o.w16_pool(sp, W[sp]) for sp in ("val", "hold")}
E0 = {sp: o.ndcg(ZW[sp], W[sp]) for sp in ("val", "hold")}
CL = {sp: o.CL(W[sp]) for sp in ("val", "hold")}
L0 = {sp: CL[sp].fit([ZW[sp]])[1] for sp in ("val", "hold")}
A = "s1f_rmse_d8_lr030"; H = "ts_lr020_s0"
FAMS = ["s1_rmse_d8_lr010_s0", f"{A}_s0", f"{A}_s1", f"{A}_s2", f"{A}_s1long", "s3f_pca_d8_lr030_s0", "s2_rmse_d10_lr030_s0", "s4_qsm_kdmix_lr030_s0", "s4b_qsm_soft_lr030_s0",
        "s4y_yeti_teacher_lr030_s0"]
EXT = {"lgbK": ("nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_val.npy", "nn_x/w12/recheck/long_bets/out/gbdt_K_xm_134118_trH.npy"),
       "lgbKL0": ("nn_x/w12/cb/audit/out/KL0_60k_r2400_val.npy", "nn_x/w12/cb/audit/out/KL0_60k_r2400_trH.npy")}


def raw(t, sp):
    if t in EXT:
        return np.load(ROOT / EXT[t][0 if sp == "val" else 1])
    for d in (FAM / "preds", INFRA / "out"):
        p = d / f"{t}_{sp}.npy"
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(t)


Z = {(t, sp): o.zfeed(raw(t, sp), W[sp]["off"], W[sp]["M"]) for t in FAMS + [H] + list(EXT) for sp in ("val", "hold")}
out = {}
print("== (a) wider bundles (member-equal z-mean): level, gap to the gate 0.3900, nats, member rule")
SETS = {"all 10 famS models": FAMS, "top 5 by level (S1, S1f s0, S1f s2, S1f long, S4b)": [FAMS[0], FAMS[1], FAMS[3], FAMS[4], FAMS[8]],
        "S4b + S1f long + S1": [FAMS[8], FAMS[4], FAMS[0]], "all 10 famS + 2 LightGBM students": FAMS + list(EXT), "S4b + S1f long + S1 + LightGBM KL0 plateau": [FAMS[8], FAMS[4], FAMS[0], "lgbKL0"],
        "all 10 famS + 2 LightGBM + hard-label tree": FAMS + list(EXT) + [H]}
for nm, ts in SETS.items():
    r = {}
    for sp in ("val", "hold"):
        w = W[sp]; zx = np.mean([Z[(t, sp)] for t in ts], 0); lv = o.ndcg(zx, w)
        _, l2 = CL[sp].fit([ZW[sp], zx]); d = L0[sp] - l2
        dm = o.ndcg(0.9 * ZW[sp] + 0.1 * o.zfeed(zx, w["off"], w["M"]), w) - E0[sp]
        r[sp] = dict(level=float(lv.mean()), nats=pse(d), mix=pse(dm))
    pz = (r["val"]["mix"][0] / r["val"]["mix"][1] + r["hold"]["mix"][0] / r["hold"]["mix"][1]) / np.sqrt(2)
    out[nm] = r
    print(f"[{nm}] level {r['val']['level']:.5f} / {r['hold']['level']:.5f} (gate short by {0.39 - r['val']['level']:+.5f}) | NATS {r['val']['nats'][0]:+.5f}+-{r['val']['nats'][1]:.5f} / "
          f"{r['hold']['nats'][0]:+.5f}+-{r['hold']['nats'][1]:.5f} | member {r['val']['mix'][0]:+.5f}+-{r['val']['mix'][1]:.5f} / {r['hold']['mix'][0]:+.5f}+-{r['hold']['mix'][1]:.5f} z {pz:+.2f}", flush=True)

print("== (b) free-weight upper bound (in-sample; optimism about k / 2n = 0.00035 val, 0.00017 holdout for k = 10)")
for nm, ts in (("W16 + 10 famS models, free weights", FAMS), ("W16 + 10 famS + hard-label tree, free weights", FAMS + [H]), ("W16 + 10 famS + hard + 2 LightGBM, free weights", FAMS + [H] + list(EXT))):
    r = {}
    for sp in ("val", "hold"):
        _, l2 = CL[sp].fit([ZW[sp]] + [Z[(t, sp)] for t in ts]); r[sp] = pse(L0[sp] - l2)
    out[nm] = r
    print(f"[{nm}] NATS {r['val'][0]:+.5f}+-{r['val'][1]:.5f} / {r['hold'][0]:+.5f}+-{r['hold'][1]:.5f}  (bar 0.0016 on both)", flush=True)

print("== (c) other encodings of the best student (3-seed bundle members' mean)")
for sp in ("val", "hold"):
    w = W[sp]; fid = o.fid_of(w["M"])
    zs = np.mean([Z[(f"{A}_s{k}", sp)] for k in range(3)], 0)
    rawm = np.mean([raw(f"{A}_s{k}", sp).astype(np.float64) for k in range(3)], 0); rawc = rawm - (np.add.reduceat(rawm, w["off"]) / w["M"])[fid]
    sdf = np.sqrt(np.add.reduceat(rawc ** 2, w["off"]) / w["M"])
    _, l_raw = CL[sp].fit([ZW[sp], rawc]); _, l_both = CL[sp].fit([ZW[sp], zs, rawc]); _, l_sc = CL[sp].fit([ZW[sp], ZW[sp] * sdf[fid]]); _, l_sc2 = CL[sp].fit([ZW[sp], ZW[sp] * sdf[fid], zs])
    out[f"enc {sp}"] = dict(raw=pse(L0[sp] - l_raw), both=pse(L0[sp] - l_both), w16_scaled_by_student_sd=pse(L0[sp] - l_sc), z_given_scaled=pse(l_sc - l_sc2))
    print(f"[{sp}] [zW, raw centred student] {pse(L0[sp] - l_raw)[0]:+.5f}+-{pse(L0[sp] - l_raw)[1]:.5f} | [zW, z, raw] {pse(L0[sp] - l_both)[0]:+.5f}+-{pse(L0[sp] - l_both)[1]:.5f} | "
          f"[zW, zW x student's per-feed sd] {pse(L0[sp] - l_sc)[0]:+.5f}+-{pse(L0[sp] - l_sc)[1]:.5f} (a scale effect, not ranking information) | student z on top of that {pse(l_sc - l_sc2)[0]:+.5f}+-{pse(l_sc - l_sc2)[1]:.5f}", flush=True)

print("== (d) joint read by M bin (val) and cross-witness coefficients")
w = W["val"]; zs = np.mean([Z[(f"{A}_s{k}", "val")] for k in range(3)], 0); zh = Z[(H, "val")]
for nm, k in (("M>=50", w["M"] >= 50), ("M<50", w["M"] < 50)):
    c = o.CL(w, feeds=k); _, l0 = c.fit([ZW["val"]]); _, ls = c.fit([ZW["val"], zs]); _, lh = c.fit([ZW["val"], zh]); _, lsh = c.fit([ZW["val"], zs, zh])
    out[f"joint val {nm}"] = dict(S=pse(l0 - ls), H=pse(l0 - lh), S_given_H=pse(lh - lsh), H_given_S=pse(ls - lsh))
    print(f"[val {nm}] S {pse(l0 - ls)[0]:+.5f}+-{pse(l0 - ls)[1]:.5f} | H {pse(l0 - lh)[0]:+.5f}+-{pse(l0 - lh)[1]:.5f} | S | W16,H {pse(lh - lsh)[0]:+.5f}+-{pse(lh - lsh)[1]:.5f} | H | W16,S {pse(ls - lsh)[0]:+.5f}+-{pse(ls - lsh)[1]:.5f}", flush=True)
# cross-witness: coefficients fitted on one witness, loss read on the other (no in-sample optimism)
for nm, cols in (("student bundle", lambda sp: [ZW[sp], np.mean([Z[(f"{A}_s{k}", sp)] for k in range(3)], 0)]), ("hard-label tree", lambda sp: [ZW[sp], Z[(H, sp)]])):
    for a, b in (("val", "hold"), ("hold", "val")):
        ba, _ = CL[a].fit(cols(a)); b0, _ = CL[a].fit([ZW[a]])
        Xb = np.stack([np.asarray(c)[CL[b].rows] for c in cols(b)], 1)
        lfull, _ = CL[b]._lossgrad(ba, Xb); lbase, _ = CL[b]._lossgrad(b0, Xb[:, :1])
        out[f"cross {nm} fit {a} read {b}"] = pse(lbase - lfull)
        print(f"[{nm}] coefficients fitted on {a}, read on {b}: NATS {pse(lbase - lfull)[0]:+.5f}+-{pse(lbase - lfull)[1]:.5f} (beta {np.round(ba, 3).tolist()})", flush=True)
(ROOT / "nn_x/w12/cb/famS_review/out/rv_bounds.json").write_text(json.dumps(out, indent=1))
print(f"DONE {time.time() - t0:.0f}s")
