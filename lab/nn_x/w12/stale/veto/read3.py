"""veto read 3: file-level (final form 0.5 z(FT) + 0.5 z(TO)) re-reads of the popularity-view pools, the 6 public pairs with their
pmix-private column, PSEUDO-DIRECT, and the matched-control form of hzcd. Rules: PREDECLARED.txt (02:23Z). Stored predictions only.
Output: logs/read3.txt + out/read3.json
"""
import json
import sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *

c = Ctx()
R = []
P = ROOT / "preds"
rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]


def final(nm, w=0.5):
    if nm.endswith(":to"):
        return c.z("val", np.load(P / f"ens_{nm[:-3]}_to_val.npy"))
    ft, to = c.z("val", np.load(P / f"ens_{nm}_ft_val.npy")), c.z("val", np.load(P / f"ens_{nm}_to_val.npy"))
    return w * ft + (1 - w) * to


def fpair(label, a, b, public=None):
    na, nb_ = c.nv(final(a)), c.nv(final(b)); d = na - nb_
    r = {"label": label, "levelA": float(na.mean()), "levelB": float(nb_.mean()), "val": ms(d), "pmix_pub": ms(d, c.wpub), "pmix_priv": ms(d, c.wpriv)}
    for k, m in c.sl.items(): r[k] = ms(d[m])
    r["public"] = public
    print(f"== {label}\n   val {f(r['val'])} [A {r['levelA']:.5f} B {r['levelB']:.5f}] | M>=50 {f(r['M>=50'])} | M<50 {f(r['M<50'])}\n"
          f"   coldpos {f(r['coldpos'])} | warmpos {f(r['warmpos'])} | pmix-pub {f(r['pmix_pub'])} | pmix-priv {f(r['pmix_priv'])}"
          + (f" | PUBLIC {public:+.6f}" if public is not None else ""))
    R.append(r); return r


print("##### F1. the six scored public pairs, final-form files (val | pmix-public | pmix-private | PUBLIC)")
fpair("W16 - W8", "W16kd", "W8kd", -0.000003)
fpair("W8P20 - W8", "W8P20kd", "W8kd", +0.000523)
fpair("W8TO - W8", "W8kd:to", "W8kd", -0.000477)
fpair("S16 - W8", "S16kd", "W8kd", -0.000408)
fpair("C3 - G3  (Q2a8s3 - G3)", "Q2a8s3", "G3", +0.000634)
fpair("W8 - C3", "W8kd", "Q2a8s3", +0.001032)
print("##### F2. other final-form files (not scored on public unless noted)")
fpair("Q1a - G3 (pop-free x4; public +0.00007 per DATA_PIPELINE 3.2)", "Q1apopfree", "G3", +0.00007)
fpair("Q2a - G3 (pop1 x4; public +0.00039)", "Q2apop1", "G3", +0.00039)
fpair("Q2a8 - G3 (pop1 x8, single-seed FT)", "Q2a8pop1", "G3")
fpair("C3 - C2 (Q2a8s3 - G3s3: pop1 x8 at equal 3-seed FT)", "Q2a8s3", "G3s3")
fpair("G3 TO-only - G3 (public -0.00047)", "G3:to", "G3", -0.00047)
fpair("W8C5 - W8", "W8C5kd", "W8kd")
fpair("K5kd05 - W8", "K5kd05", "W8kd")
fpair("Cd - G3 (public -0.00074)", "Cd", "G3", -0.00074)
fpair("Pall - G3 (public -0.00038)", "Pall", "G3", -0.00038)
fpair("PF7x - G3 (public -0.00010)", "PF7x", "G3", -0.00010)


def go(label, A, B, **kw):
    try:
        r = read(c, label, A, B, **kw)
    except FileNotFoundError as e:
        print(f"== {label}: MISSING {e}"); return None
    show(r); R.append(r); return r


print("##### P. PSEUDO-DIRECT (row 295; pseudo rows carry NO pre-computed inputs): 2 seeds, member-level noise")
B2 = rng("w5_e10_base_s", 951, 952)
go("pseudo x2 vs base x2 [selected]", rng("w5_e10_pseudo_s", 951, 952), B2)
go("pseudo x2 vs base x2 [@last]", rng("w5_e10_pseudo_s", 951, 952), B2, sufA="@last")
go("pseudo_xpre2 x2 vs base x2 [selected]", rng("w5_e10_pseudo_xpre2_s", 951, 952), B2)

print("##### H. hzcd in pool form with a matched control (the GLOBAL_RECAP read): G3 + hzcd x4 vs G3 + m02ref x4, train-only")
F10 = rng("v23_gru_pre2s_s", 501, 506); F7x = rng("r6_cd_m02_r", 101, 107); F7x70 = rng("r6_cd_m02_p70k_r", 101, 107)
G3 = F10 + F7x + F7x70
HZ = rng("w5_cd_hzcd_r", 101, 104); REF = rng("w5_cd_m02ref_r", 101, 104)
go("G3 + hzcd x4 vs G3 + m02ref x4 [selected]", G3 + HZ, G3 + REF)
go("G3 + hzcd@last x4 vs G3 + m02ref@last x4", G3 + [v + "@last" for v in HZ], G3 + [v + "@last" for v in REF])
go("hzcd x4 + m02ref x4 (8) vs m02ref x4 [selected]: hz as HALF of a pool", HZ + REF, REF)
print("##### H2. hz as an admixture to the student pool (hard-label hzcd members; information on the view only)")
W192 = rng("w5_e10_kd05w192_s", 951, 958); T192 = rng("w5_e10_kd05t25w192_s", 951, 958); W16 = W192 + T192
go("0.8 z(W16) + 0.2 z(hzcd x4) vs 0.8 z(W16) + 0.2 z(m02ref x4)", [(0.8, W16), (0.2, HZ)], [(0.8, W16), (0.2, REF)])

print("##### S. sub-family checks of the popularity-light admixture: which students carry the gain (0.2 weight on W16)")
P1KD = rng("abl_e10_pop1kd05t25_s", 951, 954); P1KDW = rng("abl_e10_pop1kd05w192_s", 951, 954)
POP1 = rng("abl_e10_pop1_s", 951, 958); POPF = rng("abl_e10_xdrop_pop_s", 951, 954); POP2 = rng("abl_e10_pop2_s", 951, 954)
go("0.8 z(W16) + 0.2 z(hard pop1 x8) vs W16", [(0.8, W16), (0.2, POP1)], W16)
go("0.8 z(W16) + 0.2 z(hard pop-free x4) vs W16", [(0.8, W16), (0.2, POPF)], W16)
go("0.8 z(W16) + 0.2 z(hard pop2 x4) vs W16", [(0.8, W16), (0.2, POP2)], W16)
go("CONTROL 0.8 z(W16) + 0.2 z(hard base x6) vs W16", [(0.8, W16), (0.2, rng("w5_e10_base_s", 951, 956))], W16)
go("CONTROL 0.8 z(W16) + 0.2 z(narrow kd05t25 x4) vs W16", [(0.8, W16), (0.2, rng("w5_e10_kd05t25_s", 951, 954))], W16)
go("0.8 z(W16) + 0.2 z(all 16 pop-light: pop1kd x8 + hard pop1 x8) vs W16", [(0.8, W16), (0.2, P1KD + P1KDW + POP1)], W16)

json.dump(R, open(HERE / "out/read3.json", "w"), indent=1, default=float)
