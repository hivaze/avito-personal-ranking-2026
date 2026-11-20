"""PREDECLARED.txt R7: turn the measured in-sample gaps (logs/gap_<member>.json, insample_gap.py) into the A2 bars in force.
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/fulldata/a2_calib.py
Writes logs/a2_calib.json / .txt (read by accept_twins.py)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBERS = ("w5_e10_kd05w192_s951", "abl_e10_pop1kd05t25_s951")
OUT = open(f"{HERE}/logs/a2_calib.txt", "w")


def P(s):
    print(s, flush=True)
    OUT.write(s + "\n")


facts = json.load(open(f"{HERE}/logs/facts.json"))
hl = np.array([r["hold_last"] for r in facts["members"][:16]])
s_null = float(np.sqrt(2) * hl.std(ddof=1))
P(f"== R7 A2 calibration   [{os.popen('date -u +%FT%TZ').read().strip()}]")
P(f"   null noise of the A2 statistic: SD of the 16 W16 originals' @last holdout NDCG {hl.std(ddof=1):.5f} (range {hl.min():.5f}-{hl.max():.5f}); "
  f"s_null = sqrt(2) x SD = {s_null:.5f}; floor 3 s_null = {3 * s_null:.5f}")
gaps, void = [], False
for v in MEMBERS:
    p = f"{HERE}/logs/gap_{v}.json"
    if not os.path.exists(p):
        P(f"   {v}: no measurement -> VOID")
        void = True
        continue
    g = json.load(open(p))
    pc = g["path_check"]
    P(f"   {v}@last: in sample {g['ndcg_in'][0]:.5f} +- {g['ndcg_in'][1]:.5f} (n {g['n_in']}) | out of sample (holdout, saved file) {g['ndcg_oos_saved_file'][0]:.5f} +- "
      f"{g['ndcg_oos_saved_file'][1]:.5f} (n {g['n_oos']}) | G_s = {g['G_s']:+.5f} +- {g['G_s_se']:.5f}")
    P(f"      path check on {pc['n']} holdout feeds: CPU {pc['ndcg_cpu'][0]:.5f} vs saved GPU file {pc['ndcg_gpu_file'][0]:.5f} (diff {pc['diff'][0]:+.5f} +- {pc['diff'][1]:.5f}), "
      f"centred corr {pc['corr_centred']:.6f} -> {'ok' if pc['ok'] else 'VOID'}")
    t = g.get("teacher")
    if t:
        P(f"      its teacher {t['file']}: in sample {t['ndcg_in'][0]:.5f}, holdout {t['ndcg_oos'][0]:.5f}, G_t = {t['G_t']:+.5f}; student - teacher: in sample "
          f"{t['student_minus_teacher_in'][0]:+.5f} +- {t['student_minus_teacher_in'][1]:.5f}, holdout {t['student_minus_teacher_oos'][0]:+.5f} +- {t['student_minus_teacher_oos'][1]:.5f}")
    void |= not pc["ok"]
    gaps.append(g["G_s"])
res = dict(void=bool(void), s_null=s_null, floor=3 * s_null, r4=dict(full=0.005, recool=0.002))
if void:
    res.update(bar_full=0.005, bar_recool=0.002, note="measurement void or absent: R4's bars in force")
else:
    G = float(min(gaps))
    res.update(G_s=G, G_s_members=dict(zip(MEMBERS, gaps)), bar_full=float(max(3 * s_null, min(0.005, 0.25 * G))),
               bar_recool=float(max(3 * s_null, min(0.002, 0.10 * G))), expected_full_lift_I=0.5 * G)
    P(f"   G_s (smaller of the two) = {G:+.5f}; expected lift of a full-retrain twin on the holdout [I] about 0.5 G_s = {0.5 * G:+.5f}")
P(f"   A2 bars in force: full retrain {res['bar_full']:.5f} (R4: 0.005), re-cool {res['bar_recool']:.5f} (R4: 0.002)" + ("   [VOID: R4]" if void else ""))
json.dump(res, open(f"{HERE}/logs/a2_calib.json", "w"), indent=1)
