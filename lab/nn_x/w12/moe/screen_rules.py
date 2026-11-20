"""w12/moe screen rules (RULE_predeclared.txt; CPU, read-only on the run logs / jsons).

  es      <arm> [seeds csv = 951,952,953,954]   early-stop status from the in-training evals of the running / finished arm runs
  decide  <arm> [seeds csv = 951,952,953,954]   the pre-declared read (needs perm8_moe.py's json; run arm_read8_moe.py for the report)

  cd <repo> && .venv/bin/python lab/nn_x/w12/moe/screen_rules.py es kd05w192moeffn
  cd <repo> && .venv/bin/python lab/nn_x/w12/moe/screen_rules.py decide kd05w192moeffn

ES1 (one run, any eval after step 10,000): loss non-finite or val_ndcg < 0.385 -> STOP that run; >= 2 stopped seeds of an arm -> KILL the arm.
ES2 (whole arm, at step 60,000): mean over the 9 evals at steps 40,000-60,000 of (arm val - reference val at the same seed and step); below
    -0.003 on >= 3 of 4 seeds -> STOP the arm.
ES3 (flag only, arms with routed modules): load entropy < 0.6 or > 25% dead experts in any routed layer for 3 consecutive evals after step
    10,000 -> 'routing collapsed' (a null with a collapsed router is inconclusive; a null with a healthy router closes the design).
READ: PROMOTE iff 4-pool val delta >= +0.0003 AND 4-pool holdout delta >= 0 AND perm8 p_perm <= 0.2; KILL iff val delta < 0; else HOLD
    (val >= +0.0003 and holdout >= 0 but p > 0.2: extending to seeds 955-958 is an owner decision).
Run logs are looked up in lab/nn_x/w12/moe/out/logs (lane_moe.sh MOE_OUT=moe) and then nn_x/w5/logs; references w5_e10_kd05w192_s<seed>.json.
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path("<repo>")
LOGDIRS = [ROOT / "lab/nn_x/w12/moe/out/logs", ROOT / "nn_x/w5/logs"]
REF = "kd05w192"
STEP = re.compile(r"^step (\d+)/(\d+) loss (\S+) val_ndcg (\S+) holdout_ndcg (\S+)")
MOE = re.compile(r"^w12/moe step (\d+): min load entropy (\S+), max dead share (\S+)")


def find(v, ext):
    for d in LOGDIRS:
        if (d / f"{v}{ext}").exists():
            return d / f"{v}{ext}"
    return None


def parse_log(v):
    p = find(v, ".log")
    if p is None:
        return None, None
    ev, moe = {}, {}
    for ln in open(p):
        m = STEP.match(ln)
        if m:
            ev[int(m.group(1))] = dict(total=int(m.group(2)), loss=float(m.group(3)), val=float(m.group(4)), hold=float(m.group(5)))
        m = MOE.match(ln)
        if m:
            moe[int(m.group(1))] = (float(m.group(2).rstrip(",")), float(m.group(3)))
    return ev, moe


def ref_hist(seed):
    p = find(f"w5_e10_{REF}_s{seed}", ".json")
    return {h["step"]: h["val_ndcg"] for h in json.load(open(p))["history"]} if p else {}


def es(arm, seeds):
    stopped, es2_bad, es2_n = 0, 0, 0
    print(f"== ES status of {arm} vs {REF} (seeds {seeds})")
    for s in seeds:
        v = f"w5_e10_{arm}_s{s}"
        ev, moe = parse_log(v)
        if not ev:
            print(f"  s{s}: no eval yet ({'no log' if ev is None else 'log without evals'})")
            continue
        last = max(ev)
        es1 = [k for k, e in ev.items() if k > 10000 and (not math.isfinite(e["loss"]) or not math.isfinite(e["val"]) or e["val"] < 0.385)]
        stopped += bool(es1)
        rh = ref_hist(s)
        win = [k for k in range(40000, 60001, 2500)]
        es2 = None
        if all(k in ev and k in rh for k in win):
            es2 = sum(ev[k]["val"] - rh[k] for k in win) / len(win)
            es2_n += 1
            es2_bad += es2 < -0.003
        run, col = 0, False
        for k in sorted(moe):
            ent, dead = moe[k]
            run = run + 1 if (k > 10000 and (ent < 0.6 or dead > 0.25)) else 0
            col = col or run >= 3
        cur = [ev[k]["val"] - rh[k] for k in sorted(ev) if k in rh][-3:]
        print(f"  s{s}: step {last}/{ev[last]['total']} val {ev[last]['val']:.5f} | ES1 {'STOP at ' + str(es1[0]) if es1 else 'ok'} | ES2 window "
              f"{'n/a (before step 60,000)' if es2 is None else f'{es2:+.5f}' + (' < -0.003' if es2 < -0.003 else '')} | last 3 val deltas "
              f"{[round(x, 5) for x in cur]} | ES3 {'n/a' if not moe else ('ROUTING COLLAPSED' if col else f'ok (min entropy {min(e for e, _ in moe.values()):.3f})')}")
    verdict = []
    if stopped >= 2:
        verdict.append(f"ES1: {stopped} seeds stopped -> KILL the arm")
    if es2_n >= 3 and es2_bad >= 3:
        verdict.append(f"ES2: {es2_bad} of {es2_n} seeds below -0.003 -> STOP the arm")
    print("  ARM: " + ("; ".join(verdict) if verdict else f"continue (ES1 stops {stopped}, ES2 below -0.003 on {es2_bad} of {es2_n} seeds read)"))


def decide(arm, seeds):
    tag = "-".join(str(s) for s in seeds)
    p = ROOT / f"lab/nn_x/w12/moe/logs/perm8_{arm}_vs_{REF}_{tag}.json"
    if not p.exists():
        sys.exit(f"run perm8_moe.py {arm} {REF} {','.join(map(str, seeds))} first ({p} missing)")
    r = json.load(open(p))
    dv, dh, pp = r["val"][0], r["holdout"][0], r["p_perm"]
    if dv >= 0.0003 and dh >= 0 and pp <= 0.2:
        d = "PROMOTE (all-data fd10 build, seeds 951-954; confirms only if val delta vs w5_fd10_kd05w192_s951-954 >= +0.0003)"
    elif dv < 0:
        d = "KILL"
    elif dv >= 0.0003 and dh >= 0:
        d = "HOLD (val and holdout clear the bar but p_perm > 0.2: extending to seeds 955-958 is an owner decision)"
    else:
        d = "HOLD"
    print(f"== {arm} vs {REF} seeds {seeds}: 4-pool val {dv:+.5f} +- {r['val'][1]:.5f}, holdout {dh:+.5f} +- {r['holdout'][1]:.5f}, "
          f"pooled z {r['pooled_z']:+.2f}, p_perm {pp:.4f} ({r['patterns']} patterns) -> {d}")
    es(arm, seeds)


if __name__ == "__main__":
    cmd, arm = sys.argv[1], sys.argv[2]
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [951, 952, 953, 954]
    {"es": es, "decide": decide}[cmd](arm, seeds)
