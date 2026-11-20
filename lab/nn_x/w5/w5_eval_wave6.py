"""Read-out of the nn_x/w5 STAGE-B WAVE-6 arms aa0 / ap1 / m01 / m04 (reports/ABLATION_PLAN_1113.md section 3.3 rows
'no_aux load-bearing -> aux_anneal 0 and aux_pos 1.0 listwise in the CD harness' and 'always -> m01 / m04 in the CD
harness'). CPU only, read-only on every prediction / log file. A thin wrapper over the w5_eval.py / nn_x/r6/r6_eval.py
functions (imported read-only; w5_eval.py's own stage rules are NOT used and NOT changed): paired per-feed deltas vs
w5_cd_m02ref_r<root>, pooled over roots 101-104, SE = sd / sqrt(n_feeds).

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w5/w5_eval_wave6.py [arm ...]
        -> nn_x/w5/logs/w5_eval_B_wave6.{txt,json} (OUT=<path> overrides)
  ... w5_eval_wave6.py selftest    plumbing check on the EXISTING stage-B arms truncp / auxf / hzcd (verdicts are
        meaningless re-reads of closed arms)        -> logs/w5_eval_B_wave6_selftest.{txt,json}

RUNS (lane.sh stage B, CD harness: 2-epoch re-cool of nn/ckpt/v17_pre2s_s<root>@77500.pt, B_BASE + HYG, CDSEED 900 + root,
roots 101-104; arm flag appended last): aa0 = --aux_anneal 0 (K9 aux weight 0.5 -> 0 over the re-cool; m02ref 0.5 -> 0.1),
ap1 = --aux_pos 1.0 (listwise aux at double weight), m01 = --cooldown_mix 0.01, m04 = --cooldown_mix 0.04.

PRE-DECLARED RULE (logs/B_wave6_predeclared.txt, frozen 2026-11-13 03:51:47 UTC before the first wave-6 run; copied here
verbatim in substance): per arm, pooled over roots 101-104, paired per val feed vs w5_cd_m02ref_r<root> on the SELECTED
checkpoint (holdout-best at step >= 12k; the LAST checkpoint is information only):
  PASS iff (1) pooled val delta > +2 SE AND (2) pooled holdout delta >= 0 AND (3) no val slice among M<50, M>=50, Tue, Thu
  < -2 SE AND (4) src/pmix.py private-mix pooled delta >= -1 SE.   STOP otherwise (wave 2-4 stage-1 futility rule).
  Diagnostics (never gates): coldpos / warmpos, the 4-root pool (member-equal z-mean ensemble) deltas, per-root deltas,
  same-root correlation, holdout late20%, the last-checkpoint read, levers / cliplog, the plain futility note (val < 0 or
  holdout < 0). INCOMPLETE if an arm misses a run (not gated).
"""
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w5_eval as E  # noqa: E402  (read-only import; its stage functions are not called)

R, P, W5 = E.R, E.P, E.W5
ARMS = ("aa0", "ap1", "m01", "m04")
# W6_ROOTS (11-13 07:35 UTC, AP1 family roots): comma list overriding the roots for INFORMATION read-outs (e.g. "105,106,107" or
# "101,...,107"); the wave-6 verdict is the default 101-104 read of logs/w5_eval_B_wave6.txt and is never re-written by an override.
ROOTS = tuple(int(x) for x in os.environ.get("W6_ROOTS", "101,102,103,104").split(","))
GATE_SLICES = ("M<50", "M>=50", "Tue", "Thu")
FLAGS = {"aa0": "--aux_anneal 0", "ap1": "--aux_pos 1.0", "m01": "--cooldown_mix 0.01", "m04": "--cooldown_mix 0.04",
         "truncp": "--trunc_off 1.0 (selftest stand-in)", "auxf": "--aux_norm feed (selftest stand-in)",
         "hzcd": "--hzcd (selftest stand-in)"}


def read_arm(v, h, arm, ref, name):
    pairs = {rd: [(E.L(ref(r), rd), E.L(name(arm, r), rd)) for r in ROOTS] for rd in ("sel", "last")}
    miss = [n for p in pairs["sel"] for n in p if not (os.path.exists(v["path"](n)) and os.path.exists(h["path"](n)))]
    if miss:
        P(f"\n== {arm}: INCOMPLETE on roots {ROOTS} (missing selected val/holdout preds: "
          f"{' '.join(R.split_name(n)[1] for n in miss)}): not gated")
        return dict(verdict="INCOMPLETE", missing=[R.split_name(n)[1] for n in miss])
    P(f"\n== {arm} ({FLAGS.get(arm, '?')}) vs m02ref (B = arm); PRIMARY = selected checkpoints")
    rv, rh = E.read_pairs(v, h, pairs["sel"], "selected (PRIMARY)")
    sv, sh = rv["pooled"], rh["pooled"]
    res = dict(flag=FLAGS.get(arm), val=sv, hold=sh, per_root_val=[p["slices"]["all"]["delta"] for p in rv["pairs"]],
               per_root_hold=[p["slices"]["all"]["delta"] for p in rh["pairs"]])
    if E.have((v, h), [n for p in pairs["last"] for n in p]):
        lv, lh = E.read_pairs(v, h, pairs["last"], "last (info)")
        res["last"] = dict(val=lv["pooled"], hold=lh["pooled"])
    else:
        P("  [last (info)]: last-checkpoint preds missing for at least one run")
        res["last"] = None
    res["ens"] = E.ens_block(v, h, [a for a, _ in pairs["sel"]], [b for _, b in pairs["sel"]])
    res["corr_same_root"] = [R.corr(v, a, b) for a, b in pairs["sel"]]
    P("  same-root corr arm vs m02ref (val, within-feed z, selected): " + " ".join(f"{x:.4f}" for x in res["corr_same_root"]))
    res["pmix"] = E.pmix_block(v, pairs["sel"])
    res["levers"] = E.lever_block(v, pairs["sel"])
    res["cliplog"] = E.cliplog_block([ref(ROOTS[0]), name(arm, ROOTS[0])])
    dv, dh = sv["all"]["delta"], sh["all"]["delta"]
    pm = res["pmix"]
    gates = {"val > +2 SE": sv["all"]["z"] > 2, "holdout >= 0": dh >= 0,
             "no val slice (M<50, M>=50, Tue, Thu) < -2 SE": all(sv[k]["z"] >= -2 for k in GATE_SLICES),
             "pmix >= -1 SE": pm is not None and not pm["below_1se"]}
    if pm is None:
        P("  NOTE: pmix unavailable -> gate (4) cannot be evaluated: counted as NOT met")
    verdict = "PASS" if all(gates.values()) else "STOP"
    futility = (dv < 0) or (dh < 0)
    res.update(gates=gates, verdict=verdict, futility_note=futility)
    P("  gates: " + "; ".join(f"{k} {'OK' if x else 'NO'}" for k, x in gates.items()))
    P(f"  slices (selected): M<50 {sv['M<50']['delta']:+.5f}+-{sv['M<50']['se']:.5f} (z {sv['M<50']['z']:+.2f})  "
      f"M>=50 {sv['M>=50']['delta']:+.5f} (z {sv['M>=50']['z']:+.2f})  Tue {sv['Tue']['delta']:+.5f} (z {sv['Tue']['z']:+.2f})  "
      f"Thu {sv['Thu']['delta']:+.5f} (z {sv['Thu']['z']:+.2f})  coldpos {sv['coldpos']['delta']:+.5f} (z {sv['coldpos']['z']:+.2f})  "
      f"warmpos {sv['warmpos']['delta']:+.5f} (z {sv['warmpos']['z']:+.2f})  | holdout late20% {sh['late20%']['delta']:+.5f}")
    P(f"  futility note (wave 2-4 stage-1 'val < 0 or holdout < 0'): {'fires' if futility else 'does not fire'}")
    P(f"  ==> {arm}: pooled val {dv:+.5f} +- {sv['all']['se']:.5f} (z {sv['all']['z']:+.2f}), holdout {dh:+.5f} +- "
      f"{sh['all']['se']:.5f} (z {sh['all']['z']:+.2f}) -> {verdict}")
    return res


def main():
    args = sys.argv[1:]
    selftest = bool(args) and args[0] == "selftest"
    arms = ("truncp", "auxf", "hzcd") if selftest else (tuple(args) or ARMS)
    ref, name = E.CFG["B_ref"], E.CFG["B_arm"]
    v, h = R.setup("val"), R.setup("hold")
    P(("#### SELFTEST on existing closed arms (verdicts meaningless)\n" if selftest else "")
      + f"== WAVE-6 STAGE B (CD re-cool, roots {ROOTS}, reference w5_cd_m02ref_r<root>, CDSEED 900 + root; PRIMARY = "
        f"SELECTED checkpoint, last = info). Rule (logs/B_wave6_predeclared.txt, 2026-11-13 03:51:47 UTC): PASS iff val > +2 SE "
        f"AND holdout >= 0 AND no val slice (M<50, M>=50, Tue, Thu) < -2 SE AND pmix >= -1 SE; STOP otherwise")
    runs = E.runs_block([ref(r) for r in ROOTS] + [name(a, r) for a in arms for r in ROOTS])
    out = {"runs": runs, "arms": {}}
    for arm in arms:
        out["arms"][arm] = read_arm(v, h, arm, ref, name)
    P("\n== summary (wave 6, stage B, SELECTED checkpoints, pooled over roots 101-104 vs m02ref)")
    for arm, r in out["arms"].items():
        if r["verdict"] == "INCOMPLETE":
            P(f"  {arm:5s} INCOMPLETE (missing {' '.join(r['missing'])})")
            continue
        s, hh, e, pm = r["val"], r["hold"], r["ens"], r["pmix"]
        P(f"  {arm:5s} val {s['all']['delta']:+.5f}+-{s['all']['se']:.5f} (z {s['all']['z']:+.2f}) hold "
          f"{hh['all']['delta']:+.5f}+-{hh['all']['se']:.5f} (z {hh['all']['z']:+.2f}) | M<50 {s['M<50']['delta']:+.5f} "
          f"M>=50 {s['M>=50']['delta']:+.5f} Tue {s['Tue']['delta']:+.5f} Thu {s['Thu']['delta']:+.5f} coldpos "
          f"{s['coldpos']['delta']:+.5f} warmpos {s['warmpos']['delta']:+.5f} | pmix "
          f"{pm['delta'] if pm else float('nan'):+.5f}+-{pm['se'] if pm else float('nan'):.5f} | {len(ROOTS)}-root pool val "
          f"{e['VAL']['delta']:+.5f} hold {e['HOLD']['delta']:+.5f}"
          + (f" | last val {r['last']['val']['all']['delta']:+.5f} hold {r['last']['hold']['all']['delta']:+.5f}" if r["last"] else "")
          + f" -> {r['verdict']}")
    P("\n== registry-ready one-liners")
    for arm, r in out["arms"].items():
        if r["verdict"] == "INCOMPLETE":
            continue
        s, hh, e, pm = r["val"], r["hold"], r["ens"], r["pmix"]
        P(f"  w5_cd_{arm}_r101-104 ({r['flag']}) vs w5_cd_m02ref (CD harness, CDSEED 900+root, selected ckpt, pre-declared "
          f"03:51 UTC rule): val {s['all']['delta']:+.5f} +- {s['all']['se']:.5f} (z {s['all']['z']:+.1f}), holdout "
          f"{hh['all']['delta']:+.5f} +- {hh['all']['se']:.5f} (z {hh['all']['z']:+.1f}), pmix "
          f"{pm['delta'] if pm else float('nan'):+.5f}, coldpos {s['coldpos']['delta']:+.5f} / warmpos "
          f"{s['warmpos']['delta']:+.5f}, {len(ROOTS)}-root pool val {e['VAL']['delta']:+.5f} / hold {e['HOLD']['delta']:+.5f}"
          + (f", @last val {r['last']['val']['all']['delta']:+.5f} / hold {r['last']['hold']['all']['delta']:+.5f}" if r["last"] else "")
          + f" -> {r['verdict']} (nn_x/w5/logs/w5_eval_B_wave6.txt)")
    out_f = os.environ.get("OUT") or f"{W5}/logs/w5_eval_B_wave6{'_selftest' if selftest else ''}.txt"
    open(out_f, "w").write("\n".join(R.OUT) + "\n")
    json.dump(out, open(re.sub(r"\.txt$", "", out_f) + ".json", "w"), indent=1, default=float)
    print(f"wrote {out_f}")


if __name__ == "__main__":
    main()
