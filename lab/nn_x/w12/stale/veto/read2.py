"""veto read 2: REVERSE CHECK (settings adopted with the in-period holdout) + information reads of other holdout-closed arms.
Rules: PREDECLARED.txt items 4-5 (written 02:23Z). Stored predictions only (CPU). Output: logs/read2.txt + out/read2.json
"""
import json
import sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *

c = Ctx()
R = []
def go(label, A, B, **kw):
    try:
        r = read(c, label, A, B, **kw)
    except FileNotFoundError as e:
        print(f"== {label}: MISSING {e}"); return None
    show(r); R.append(r); return r

rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]
W192 = rng("w5_e10_kd05w192_s", 951, 958); T192 = rng("w5_e10_kd05t25w192_s", 951, 958); W16 = W192 + T192
W8 = W192[:4] + T192[:4]


def member_mean(names, kind, sufA, sufB):
    """Mean over runs of the per-feed NDCG delta (member level): A = name+sufA, B = name+sufB."""
    ev = c.nh if kind == "hold" else c.nv
    d = np.mean([ev(c.zfile(v + sufA, kind)) - ev(c.zfile(v + sufB, kind)) for v in names], 0)
    return d


print("##### R1. CHECKPOINT SELECTION: holdout-SELECTED (S) minus @last (L); pool level; + = the holdout-selected checkpoint is better")
go("W16: S - L (16 wide students)", W16, W16, sufA="", sufB="@last")
go("W8:  S - L", W8, W8, sufA="", sufB="@last")
diff9 = [W192[i] for i in (0, 2, 4, 5, 7)] + [T192[i] for i in (0, 2, 3, 4)]
go("W16 members whose selected step != last step (9): S - L", diff9, diff9, sufA="", sufB="@last")
for fam, names in (("kd05 d128 x4", rng("w5_e10_kd05_s", 951, 954)), ("kd05t25 d128 x4", rng("w5_e10_kd05t25_s", 951, 954)),
                   ("kd08 d128 x4", rng("w5_e10_kd08_s", 951, 954)), ("hard base x6", rng("w5_e10_base_s", 951, 956)),
                   ("hard pop1 x8", rng("abl_e10_pop1_s", 951, 958)), ("pop1kd d128 x4", rng("abl_e10_pop1kd05t25_s", 951, 954)),
                   ("pop1kd w192 x4", rng("abl_e10_pop1kd05w192_s", 951, 954)), ("hard pop-free x4", rng("abl_e10_xdrop_pop_s", 951, 954)),
                   ("hard e20 x4 (long cooldown)", rng("w5_e10_e20_s", 951, 954))):
    go(f"{fam}: S - L", names, names, sufA="", sufB="@last")
print("--- hard-label re-cool arms (stored GPU stale-val for S and L), 4 roots each")
ARMS = ["m02ref", "hzcd", "auxf", "truncp", "rdrop", "sam", "cbs05", "cbs2", "cbs025", "fpcd"]
allruns = []
for a in ARMS:
    names = rng(f"w5_cd_{a}_r", 101, 104); allruns += names
    go(f"re-cool {a} x4: S - L", names, names, sufA="", sufB="@last")
print("--- member-level aggregate over the 40 hard-label re-cool runs (mean over runs of per-feed S - L)")
agg = {}
for kind in ("val", "stale", "hold"):
    d = member_mean(allruns, kind, "", "@last")
    agg[kind] = ms(d)
    if kind != "hold":
        agg[kind + "_pmix_priv"] = ms(d, c.wpriv); agg[kind + "_M>=50"] = ms(d[c.sl["M>=50"]]); agg[kind + "_M<50"] = ms(d[c.sl["M<50"]])
        agg[kind + "_coldpos"] = ms(d[c.sl["coldpos"]]); agg[kind + "_warmpos"] = ms(d[c.sl["warmpos"]])
    else:
        agg["hold_late"] = ms(d[c.hl["late>=Sep13"]])
for k, v in agg.items(): print(f"   {k:22s} {f(v)}")
R.append({"label": "AGG 40 re-cool runs S-L member level", **agg})
print("--- the same aggregate over the 16 W16 runs and over 26 d128 / hard 10-epoch runs (val, holdout; no stale files)")
for lab, names in (("W16 x16", W16), ("kd05+kd05t25+kd08+kd03+kd10 d128 x20 + base x6", rng("w5_e10_kd05_s", 951, 954) + rng("w5_e10_kd05t25_s", 951, 954)
                    + rng("w5_e10_kd08_s", 951, 954) + rng("w5_e10_kd03_s", 951, 954) + rng("w5_e10_kd10_s", 951, 954) + rng("w5_e10_base_s", 951, 956))):
    a2 = {}
    for kind in ("val", "hold"):
        d = member_mean(names, kind, "", "@last"); a2[kind] = ms(d)
        if kind == "val": a2["pmix_priv"] = ms(d, c.wpriv); a2["pmix_pub"] = ms(d, c.wpub)
    print(f"   {lab}: " + " | ".join(f"{k} {f(v)}" for k, v in a2.items()))
    R.append({"label": f"AGG {lab} S-L member level", **a2})

print("##### R2. FT / TO weights of the final (w z(FT pool) + (1-w) z(TO pool)); val-only question (the holdout has no FT branch)")
P = ROOT / "preds"
for nm in ("W16kd", "W8kd", "G3", "Q2a8s3"):
    ft, to = c.z("val", np.load(P / f"ens_{nm}_ft_val.npy")), c.z("val", np.load(P / f"ens_{nm}_to_val.npy"))
    ref = c.nv(0.5 * ft + 0.5 * to)
    out = []
    for w in (0.0, 0.3, 0.7, 1.0):
        d = c.nv(w * ft + (1 - w) * to) - ref
        out.append(f"w_ft {w:.1f}: val {f(ms(d))} pmix-pub {f(ms(d, c.wpub))} pmix-priv {f(ms(d, c.wpriv))}")
        R.append({"label": f"FTW {nm} w_ft {w} minus 0.5", "val": ms(d), "pmix_pub": ms(d, c.wpub), "pmix_priv": ms(d, c.wpriv),
                  "M<50": ms(d[c.sl["M<50"]]), "M>=50": ms(d[c.sl["M>=50"]])})
    print(f"== {nm} (0.5/0.5 level {ref.mean():.5f}):\n   " + "\n   ".join(out))

print("##### R3. COOLDOWN MIX (adopted: 0.02) and cooldown length")
go("m02 x7 vs c0 (mix 0.10) x7 [r6 roots 101-107]", rng("r6_cd_m02_r", 101, 107), rng("r6_cd_c0_r", 101, 107))
go("m02 x7 vs m0 (mix 0) x7", rng("r6_cd_m02_r", 101, 107), rng("r6_cd_m0_r", 101, 107))
go("m02ref x4 vs m01 (mix 0.01) x4", rng("w5_cd_m02ref_r", 101, 104), rng("w5_cd_m01_r", 101, 104))
go("m02ref x4 vs m04 (mix 0.04) x4", rng("w5_cd_m02ref_r", 101, 104), rng("w5_cd_m04_r", 101, 104))
go("m02 x4 vs cf4_p60k (40% cooldown, matched compute) x4", rng("r6_cd_m02_r", 101, 104), rng("r6_cd_cf4_p60k_r", 101, 104))
go("m02 x4 vs e3 (3-epoch cooldown) x4", rng("r6_cd_m02_r", 101, 104), rng("r6_cd_e3_r", 101, 104))
go("10-epoch base x4 vs e20 x4 (20 epochs: cooldown twice as long)", rng("w5_e10_base_s", 951, 954), rng("w5_e10_e20_s", 951, 954))

print("##### R5. KD WEIGHT (adopted 0.5), d128 4-pools, same seeds")
K05 = rng("w5_e10_kd05_s", 951, 954)
go("kd05 x4 vs kd08 x4", K05, rng("w5_e10_kd08_s", 951, 954))
go("kd05 x4 vs kd03 x4", K05, rng("w5_e10_kd03_s", 951, 954))
go("kd05 x4 vs kd10 x4", K05, rng("w5_e10_kd10_s", 951, 954))
go("kd05 x4 vs hard-label base x4 (KD itself)", K05, rng("w5_e10_base_s", 951, 954))
print("##### R6. WIDTH 192 (adopted), same seeds")
go("kd05w192 x4 vs kd05 x4", W192[:4], K05)
go("kd05t25w192 x4 vs kd05t25 x4", T192[:4], rng("w5_e10_kd05t25_s", 951, 954))
go("W8 (8 wide) vs 8 narrow (kd05 + kd05t25)", W8, K05 + rng("w5_e10_kd05t25_s", 951, 954))
go("kd05w224 x4 vs kd05w192 x4", rng("w5_e10_kd05w224_s", 951, 954), W192[:4])

print("##### R4b. 2-epoch screens with stored GPU stale-val (2 seeds each: member-level, noise)")
AB = rng("abl_base_s", 23, 24)
for a in ("xdrop_pop", "pop1", "pop2", "time", "kv64", "xdrop_pre2", "xdrop_ctxnd", "xdrop_ctxhn_hs", "xrel_more"):
    go(f"abl {a} x2 vs abl_base x2 [@last]", rng(f"abl_{a}_s", 23, 24), AB, sufA="@last")
go("abl pop1 x2 vs abl xdrop_pop x2 [@last]", rng("abl_pop1_s", 23, 24), rng("abl_xdrop_pop_s", 23, 24), sufA="@last")
go("abl pop2 x2 vs abl xdrop_pop x2 [@last]", rng("abl_pop2_s", 23, 24), rng("abl_xdrop_pop_s", 23, 24), sufA="@last")
go("agedrop x2 vs sx_base x2 (2-epoch; no stale files)", rng("sx_agedrop_s", 23, 24), rng("sx_base_s", 23, 24))

print("##### INFO. other arms stopped with the holdout as (co-)deciding clause, 4-pool level, vs m02ref (NOT staleness family unless noted)")
REF = rng("w5_cd_m02ref_r", 101, 104)
for a in ("sam", "rdrop", "cbs05", "cbs2", "auxf", "truncp", "fpcd", "xp2", "xp2m05", "m01", "aa0", "ap1"):
    go(f"re-cool {a} x4 vs m02ref x4 [selected]", rng(f"w5_cd_{a}_r", 101, 104), REF)
go("re-cool ap1 x7 vs m02ref x7 [selected]", rng("w5_cd_ap1_r", 101, 107), rng("w5_cd_m02ref_r", 101, 107))
for a in ("hcap", "rec33", "rho70", "aa5", "aug", "d0", "d20", "pp02", "w10"):
    go(f"r6 re-cool {a} x4 vs r6 m02 x4", rng(f"r6_cd_{a}_r", 101, 104), rng("r6_cd_m02_r", 101, 104))
go("blocks3 e10 x2 vs base x2", rng("abl_e10_blocks3_s", 951, 952), rng("w5_e10_base_s", 951, 952))
go("a4u e10 x6 vs F7 x6", rng("r6_a4u_e10_s", 101, 106), rng("v17_pre2s_s", 101, 106))
go("cset e10 x6 vs F7 x6", rng("r6_cset_e10_s", 101, 106), rng("v17_pre2s_s", 101, 106))
go("m02-a4u re-cool x6 vs m02 x6", rng("r6_cd_m02_a4u_r", 101, 106), rng("r6_cd_m02_r", 101, 106))
go("kd05w192gru x4 vs kd05w192 x4", rng("w5_e10_kd05w192gru_s", 951, 954), W192[:4])

json.dump(R, open(HERE / "out/read2.json", "w"), indent=1, default=float)
