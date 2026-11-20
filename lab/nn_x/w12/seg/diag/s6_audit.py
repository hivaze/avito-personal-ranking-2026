"""C: stored architecture / history arms read by H bucket (pool of the arm's stored members vs the same-seed control pool).
Reading rule: DECLARATION block 0 (flag only if segment z >= 3.3 AND interaction z >= 2 AND complement delta >= -1 SE)."""
from lib import *
A, W5, R6, R7, W9, PR = ROOT / "nn_x/abl/preds", ROOT / "nn_x/w5/preds", ROOT / "nn_x/r6/preds", ROOT / "nn_x/r7/preds", ROOT / "nn_x/w9/hs1/preds", ROOT / "preds"
def files(d, pat, seeds, pre="s"): return [d / pat.format(s=f"{pre}{x}") for x in seeds]
S2 = (23, 24); E2 = (951, 952); E4 = (951, 952, 953, 954)
ARMS = []
def add(label, arm, ctl, note=""): ARMS.append((label, arm, ctl, note))
abl_base = files(A, "nn_abl_base_{s}_trainonly_val.npy", S2)
for a, lab in (("retr32", "retrieval depth 32 (2 ep)"), ("retr32u", "retrieval 32 + unique slots + type counts (2 ep)"), ("retr_cat", "second same-category retrieval (2 ep)"), ("no_retr", "INFO: retrieval REMOVED (2 ep)"),
               ("no_sim", "INFO: cosine sim_bias REMOVED (2 ep)"), ("blocks3", "3 blocks (2 ep)"), ("kv64", "frozen encoder hidden-state K/V branch (2 ep)"), ("xdrop_pre2", "INFO: pre2 inputs REMOVED (2 ep)"),
               ("xdrop_pop", "INFO: popularity inputs REMOVED (2 ep)"), ("pop1", "INFO: pop1 view (2 ep)")):
    add(lab, files(A, "nn_abl_" + a + "_{s}_trainonly_val.npy", S2), abl_base)
e10_base = files(W5, "nn_w5_e10_base_{s}_trainonly_val.npy", E2)
for a, lab in (("retr32", "retrieval depth 32 (10 ep, hard labels)"), ("retr32u", "retrieval 32 + unique + type counts (10 ep)"), ("blocks3", "3 blocks (10 ep)")):
    add(lab, files(A, "nn_abl_e10_" + a + "_{s}_trainonly_val.npy", E2), e10_base)
kd = lambda seeds: files(W5, "nn_w5_e10_kd05w192_{s}_trainonly_val.npy", seeds); kdt = lambda seeds: files(W5, "nn_w5_e10_kd05t25w192_{s}_trainonly_val.npy", seeds)
for a, seeds, lab in (("kw_hl2", E2, "2 history layers under KD (d192)"), ("kw_id", E2, "hashed item-ID table under KD"), ("kw_ff4", E2, "FFN x4 under KD"), ("kwh30", E4, "history-drop 0.3 function matching under KD"),
                      ("kd05w192gru", E4, "GRU history encoder under KD"), ("kd05w224", E4, "width 224 under KD")):
    add(lab, files(W5, "nn_w5_e10_" + a + "_{s}_trainonly_val.npy", seeds), kd(seeds))
for a, lab in (("kd05t25w192b3", "3 blocks under KD (teacher 25)"), ("kd05t25w256", "width 256 under KD (teacher 25)")):
    add(lab, files(W5, "nn_w5_e10_" + a + "_{s}_trainonly_val.npy", E2), kdt(E2))
r7b = files(R7, "nn_r7_base_{s}_trainonly_val.npy", S2)
add("r7 hl2: second history layer (2 ep)", files(R7, "nn_r7_hl2_{s}_trainonly_val.npy", S2), r7b); add("r7 hc4: 4 hyper-connection streams (2 ep)", files(R7, "nn_r7_hc4_{s}_trainonly_val.npy", S2), r7b)
r6b = files(R6, "nn_r6_base_{s}_trainonly_val.npy", (23,))
for a, lab in (("typed", "r6 typed cross-attention (target actions only in the last block)"), ("a4u", "r6 a4u: unique retrieval slots + repeat count (2 ep)"), ("a4w", "r6 a4w: DIN product term in retrieval values (2 ep)"), ("cset", "r6 cset: candidate-set retrieval (2 ep)")):
    add(lab, files(R6, "nn_r6_" + a + "_{s}_trainonly_val.npy", (23,)), r6b, "seed 23 only (no stored s24 control)")
v17 = files(PR, "nn_v17_pre2s_{s}_trainonly_val.npy", range(101, 107))
add("r6 a4u at 10 ep (6 seeds vs F7 same seeds)", files(R6, "nn_r6_a4u_e10_{s}_trainonly_val.npy", range(101, 107)), v17); add("r6 cset at 10 ep (6 seeds vs F7 same seeds)", files(R6, "nn_r6_cset_e10_{s}_trainonly_val.npy", range(101, 107)), v17)
cdm = files(R6, "nn_r6_cd_m02_{s}_trainonly_val.npy", range(101, 105), "r")
for a, lab in (("hcap", "hcap: per-type history caps in the cooldown (history parity)"), ("rec33", "rec33: cooldown on the last 3 train days"), ("aug", "aug: candidate subset + history drop 0.15 on cooldown pool batches")):
    add(lab, files(R6, "nn_r6_cd_" + a + "_{s}_trainonly_val.npy", range(101, 105), "r"), cdm)
for a, lab in (("hs1", "HS1: history restricted to feed-surface events (2 ep)"), ("hs1b", "HS1b: history restricted to target actions (2 ep)")):
    add(lab, files(W9, "nn_w9_" + a + "_{s}_trainonly_val.npy", S2), abl_base, "control = abl_base same seeds")
add("GRU history encoder family v23 (6 seeds) vs F7 v17 (6 seeds; different seeds)", files(PR, "nn_v23_gru_pre2s_{s}_trainonly_val.npy", range(501, 507)), v17, "families, not seed-paired")
SG = {"ALL": np.ones(NF, bool), "H=256": CAP, "H<=9": Hv <= 9, "uid0": uid == 0}
pool = lambda fs: np.mean([Z(np.load(f)) for f in fs], 0)
rows = []; zs = []; cache = {}
for lab, arm, ctl, note in ARMS:
    miss = [f.name for f in arm + ctl if not f.exists()]
    if miss: rows.append([lab, len(arm), "MISSING " + miss[0]] + [""] * 9); continue
    key = tuple(map(str, ctl))
    if key not in cache: cache[key] = ev(pool(ctl))
    d = ev(pool(arm)) - cache[key]; r = [lab + (f" [{note}]" if note else ""), len(arm)]
    mu, se, _ = cse(d); r.append(f"{mu:+.5f}+-{se:.5f} (z {mu / se:+.1f})")
    for sg in ("H=256", "H<=9", "uid0"):
        m = SG[sg]; mu, se, _ = cse(d, m); mc, sc, _ = cse(d, ~m); di, si = cdiff(d, m, ~m); info = lab.startswith("INFO")
        flag = (mu / se >= 3.3) and (di / si >= 2) and (mc >= -sc)
        r += [f"{mu:+.5f}+-{se:.5f} (z {mu / se:+.1f})", f"{di:+.5f} (z {di / si:+.1f})", "FLAG" if flag and not info else "-"]
        if not info: zs.append((mu / se, di / si, lab, sg))
    rows.append(r)
print(tab(rows, ["arm (pool of stored train-only val predictions vs same-seed control pool)", "members", "ALL", "H=256", "H=256 minus rest", "flag", "H<=9", "H<=9 minus rest", "flag", "uid0", "uid0 minus rest", "flag"]))
z = np.array([t[0] for t in zs]); zi = np.array([t[1] for t in zs])
print(f"\nnon-INFO looks: {len(z)} segment cells. |segment z| >= 2: {(np.abs(z) >= 2).sum()} (expected by chance {0.0455 * len(z):.1f}); segment z >= +2: {(z >= 2).sum()}; z >= 3.3: {(z >= 3.3).sum()}; |interaction z| >= 2: {(np.abs(zi) >= 2).sum()} (expected {0.0455 * len(zi):.1f})")
for t in sorted(zs, key=lambda t: -t[0])[:6]: print(f"   largest positive segment z: {t[0]:+.2f} (interaction z {t[1]:+.2f})  {t[2]} @ {t[3]}")
for t in sorted(zs, key=lambda t: t[0])[:4]: print(f"   largest negative segment z: {t[0]:+.2f} (interaction z {t[1]:+.2f})  {t[2]} @ {t[3]}")
# finer H profile for the three arms the brief names (retrieval depth, second history layer, GRU) and the value of retrieval itself
bn = [n for n, _, _ in BUCK]; rows = []
for lab, arm, ctl, note in ARMS:
    if not any(k in lab for k in ("retrieval depth 32", "2 history layers", "second history layer", "GRU history encoder under KD", "retrieval REMOVED", "history-drop", "sim_bias REMOVED", "pre2 inputs REMOVED", "popularity inputs REMOVED", "HS1:")): continue
    d = ev(pool(arm)) - ev(pool(ctl)); rows.append([lab] + ["%+.4f+-%.4f" % cse(d, bmask[n])[:2] for n in bn])
print("\n### delta by H bucket for the arms named in the brief (and the worth of components, INFO rows = component REMOVED, so a negative number is the component's worth)\n" + tab(rows, ["arm"] + bn))
