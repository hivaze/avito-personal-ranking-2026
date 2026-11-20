"""veto read 11 (resume run): checkpoint-kind robustness of the first run's PASS-P reads: the same pool pairs at @last (both sides @last where
the files exist). Information; no new rule. Output: logs/read11.txt + out/read11.json"""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/stale/veto")
from vlib import *
c = Ctx(); R = []
rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]
W192 = rng("w5_e10_kd05w192_s", 951, 958); T192 = rng("w5_e10_kd05t25w192_s", 951, 958); W8 = W192[:4] + T192[:4]; W16 = W192 + T192
K5 = rng("w5_e10_kd05_s", 951, 954); K5T = rng("w5_e10_kd05t25_s", 951, 954)
P1KD = rng("abl_e10_pop1kd05t25_s", 951, 954); P1KDW = rng("abl_e10_pop1kd05w192_s", 951, 954)
def go(label, A, B, **kw):
    try:
        r = read(c, label, A, B, kinds=("val", "hold"), **kw)
    except FileNotFoundError as e:
        print(f"== {label}: MISSING {e}"); return
    show(r); R.append(r)
for suf in ("", "@last"):
    t = suf or "selected"
    go(f"teacher_mean25 students x12 vs teacher_mean11 twins x12 [{t}]", K5T + T192, K5 + W192, sufA=suf)
    go(f"kd05t25w192 x8 vs kd05w192 x8 [{t}]", T192, W192, sufA=suf)
    go(f"0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8 [{t}]", [(0.8, W8), (0.2, P1KD)], W8, sufA=suf)
    go(f"0.8 z(W16) + 0.2 z(pop1kd d128+w192 x8) vs W16 [{t}]", [(0.8, W16), (0.2, P1KD + P1KDW)], W16, sufA=suf)
    go(f"W8 vs 8 narrow students [{t}]", W8, K5 + K5T, sufA=suf)
    go(f"blocks3 e10 x2 vs base x2 [{t}]", rng("abl_e10_blocks3_s", 951, 952), rng("w5_e10_base_s", 951, 952), sufA=suf)
json.dump(R, open(HERE / "out/read11.json", "w"), indent=1, default=float)
