"""veto read 1: STALENESS-FAMILY items re-read from stored predictions (rules: PREDECLARED.txt, written 02:23Z before this file ran).
  cd <repo> && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 ... .venv/bin/python nn_x/w12/stale/veto/read1.py
Output: logs/read1.txt (stdout) + out/read1.json
"""
import json
import sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *

c = Ctx()
R = []
def go(label, A, B, **kw):
    r = read(c, label, A, B, **kw); show(r); R.append(r); return r

rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]
F10 = rng("v23_gru_pre2s_s", 501, 506); F7x = rng("r6_cd_m02_r", 101, 107); F7x70 = rng("r6_cd_m02_p70k_r", 101, 107)
G3 = F10 + F7x + F7x70
POPFREE = rng("abl_e10_xdrop_pop_s", 951, 954); POP1 = rng("abl_e10_pop1_s", 951, 954); POP1b = rng("abl_e10_pop1_s", 955, 958)
POP2 = rng("abl_e10_pop2_s", 951, 954); BASE4 = rng("w5_e10_base_s", 951, 954); BASE6 = rng("w5_e10_base_s", 951, 956)
W192 = rng("w5_e10_kd05w192_s", 951, 958); T192 = rng("w5_e10_kd05t25w192_s", 951, 958)
W8 = W192[:4] + T192[:4]; W16 = W192 + T192
P1KD = rng("abl_e10_pop1kd05t25_s", 951, 954); P1KDW = rng("abl_e10_pop1kd05w192_s", 951, 954)
KD05T25 = rng("w5_e10_kd05t25_s", 951, 954)
HZ = rng("w5_cd_hzcd_r", 101, 104); REF = rng("w5_cd_m02ref_r", 101, 104)

print("##### 1. hzcd (row 144): 4-pool hz re-cools vs 4-pool m02ref re-cools, hard labels")
go("hzcd x4 vs m02ref x4 [selected]", HZ, REF)
go("hzcd x4 vs m02ref x4 [@last]", HZ, REF, sufA="@last")

print("##### 2. popularity-view families as TRAIN-ONLY pool members (row 217, 208): member-equal G3 + family vs G3")
go("Q1a-TO  G3 + pop-free x4 vs G3", G3 + POPFREE, G3)
go("Q2a-TO  G3 + pop1 x4 (951-954) vs G3", G3 + POP1, G3)
go("Q2a'-TO G3 + pop1 x4 (955-958) vs G3", G3 + POP1b, G3)
go("Q2a8-TO G3 + pop1 x8 vs G3", G3 + POP1 + POP1b, G3)
go("Q3-TO   G3 + pop2 x4 vs G3", G3 + POP2, G3)
go("Q3b-TO  G3 + pop1 x4 + pop2 x4 vs G3", G3 + POP1 + POP2, G3)
print("##### 2b. matched control: the same number of SAME-VIEW members added (G3 + base x4 / x6 vs G3)")
go("ctrl    G3 + base x4 vs G3", G3 + BASE4, G3)
go("pop1 x4 vs base x4 as the ADDED block (G3 + pop1 x4 vs G3 + base x4)", G3 + POP1, G3 + BASE4)
go("pop-free x4 vs base x4 as the ADDED block", G3 + POPFREE, G3 + BASE4)
go("pop2 x4 vs base x4 as the ADDED block", G3 + POP2, G3 + BASE4)

print("##### 3. student pools: pop-light admixture at 0.2 (train-only level)")
go("0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8", [(0.8, W8), (0.2, P1KD)], W8)
go("0.8 z(W16) + 0.2 z(pop1kd d128 x4) vs W16", [(0.8, W16), (0.2, P1KD)], W16)
go("0.8 z(W16) + 0.2 z(pop1kd w192 x4) vs W16", [(0.8, W16), (0.2, P1KDW)], W16)
go("0.8 z(W16) + 0.2 z(pop1kd d128+w192 x8) vs W16", [(0.8, W16), (0.2, P1KD + P1KDW)], W16)
for w in (0.1, 0.3, 0.4, 0.5):
    go(f"{1 - w:.1f} z(W16) + {w:.1f} z(pop1kd d128+w192 x8) vs W16", [(1 - w, W16), (w, P1KD + P1KDW)], W16)
print("##### 3b. row 302 / round-2 addendum: pop1 students as member-equal partners of same-teacher students")
go("[kd05t25 x4 + pop1kd x4] vs kd05t25 x4", KD05T25 + P1KD, KD05T25)
go("[W16 + pop1kd x4 member-equal] vs W16", W16 + P1KD, W16)
go("[W16 + pop1kd x8 member-equal] vs W16", W16 + P1KD + P1KDW, W16)

print("##### 4. family level (reverse check of the 34-input preset): popularity-light members against full-view members, same seeds")
go("pop1 x4 (hard) vs base x4 (hard)", POP1, BASE4)
go("pop-free x4 (hard) vs base x4 (hard)", POPFREE, BASE4)
go("pop2 x4 (hard) vs base x4 (hard)", POP2, BASE4)
go("pop1kd w192 x4 vs kd05w192 x4 (students, d192)", P1KDW, W192[:4])
go("pop1kd d128 x4 vs kd05t25 d128 x4 (students, d128, same teacher)", P1KD, KD05T25)

json.dump(R, open(HERE / "out/read1.json", "w"), indent=1)
