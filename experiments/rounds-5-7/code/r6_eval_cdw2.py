# Round 6: read-out of the cooldown arms against the control cooldown, in two stages. Stage 1 (4 checkpoints)
# stops an arm that is negative on val or on the holdout. Stage 2 (7 checkpoints) adopts an arm only if the
# member gain on val exceeds 2 standard errors, the holdout is not negative, and a 7-member ensemble of the arm
# is not worse than a 7-member ensemble of the control. The ensemble condition was added after the weight-
# averaging result. It uses compare() from r6_eval.py.
# Run as (docstring): CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/r6_eval_cdw2.py stage2 <arm>
# Read-outs: ../results/round6_cooldown_*.txt.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/r6/r6_eval_cdw2.py
"""Read-out of the r6 Q1 WAVE-2 cooldown-continuation (CD) arms vs the c0 control (reports/R6_PLAN.md section 4.1).
CPU only (light analysis of saved predictions; no NN is run), read-only on every prediction / log file.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/r6_eval_cdw2.py
        -> stage 1 for every wave-2 arm (m0 aa5 d0 d20 pp02 aug) on roots 101-104 -> nn_x/r6/logs/r6_eval_cd_w2.{txt,json}
  ... r6_eval_cdw2.py stage1 d0 pp02        (a subset; arms whose 4 runs are not all finished are listed as incomplete)
  ... r6_eval_cdw2.py stage2 <arm> ...      PASS-CD on roots 101-107 -> nn_x/r6/logs/r6_eval_cd_w2_stage2.{txt,json}
  OUT=<path.txt> overrides the output path (used to verify this script on the stage-1 arms m02 / w10).

Runs: arm r6:r6_cd_<arm>_r<root>, control r6:r6_cd_c0_r<root> (same harness: F7 v17_pre2s_s<root>@77500 re-cooled,
2 epochs, cooldown_frac 1.0, seed 811, loss_norm feed, holdout-selected; nn_x/r6/cd_lane.sh). Train-only predictions.

PRE-DECLARED RULES (frozen 2026-11-12 08:43 UTC, BEFORE the first wave-2 d0/d20/pp02/aug launch at 08:43:38; CURRENT_STATUS Log):
  Stage 1 (futility only, roots 101-104): an arm CONTINUES iff its pooled member delta vs c0 is >= 0 on val AND >= 0 on
      the holdout (per-feed mean of the 4 paired deltas, as r6_eval.compare). Otherwise STOP.
  Stage 2: continuing arms get roots 105-107 (c0 105-107 exist from Q1 stage 2). PASS-CD on the 7 roots requires ALL of:
      pooled member val delta > 2 SE; holdout delta >= 0; the 7-member arm ensemble vs the 7-member c0 ensemble >= 0 on
      the holdout and >= -1 SE on val (guard against de-noising-only gains, the avg3 lesson); no val slice < -2 SE.
      A coldpos slice < -1 SE triggers the stale-val replay before adoption.
  Diagnostics (reported, not gates): every run's selected (best) step, end - selected decline on val and holdout, the
      arm-vs-c0 decline difference, same-root arm-c0 correlation, the 4-member ensemble deltas at stage 1.
"""
import json
import os
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r6_eval as R  # noqa: E402

WAVE2 = ["m0", "aa5", "d0", "d20", "pp02", "aug"]
ROOTS1 = (101, 102, 103, 104)
ROOTS2 = (101, 102, 103, 104, 105, 106, 107)
C = lambda r: f"r6:r6_cd_c0_r{r}"  # noqa: E731


def name(arm):
    return lambda r: f"r6:r6_cd_{arm}_r{r}"


def complete(v, h, arm, roots):
    return all(os.path.exists(v["path"](name(arm)(r))) and os.path.exists(h["path"](name(arm)(r))) for r in roots)


def runs_block(arms, roots):
    """Selected step and end - selected decline of every run (log json), plus the per-arm mean decline."""
    R.P("== runs: selected (holdout-best) step / total | selected val, holdout | end - selected (decline) val, holdout")
    dec = {}
    for arm in ["c0"] + list(arms):
        rows = []
        for r in roots:
            n = C(r) if arm == "c0" else name(arm)(r)
            try:
                x = R.log_readout(n)
            except FileNotFoundError:
                R.P(f"  {n:24s} (missing)")
                continue
            dv, dh = x["final_val"] - x["sel_val"], x["final_hold"] - x["sel_hold"]
            rows.append((x["best_step"], dv, dh))
            R.P(f"  {n:24s} best {x['best_step']:>6}/{x['total']} | sel val {x['sel_val']:.5f} hold {x['sel_hold']:.5f} | "
                f"end-sel val {dv:+.5f} hold {dh:+.5f} | {x['minutes']:.0f} min")
        if rows:
            a = np.array(rows, dtype=float)
            dec[arm] = dict(n=len(rows), best_steps=[int(s) for s in a[:, 0]], mean_best=float(a[:, 0].mean()),
                            dec_val=float(a[:, 1].mean()), dec_hold=float(a[:, 2].mean()))
            R.P(f"  -> {arm:5s} mean best step {a[:, 0].mean():7.0f}  mean end-sel val {a[:, 1].mean():+.5f} "
                f"hold {a[:, 2].mean():+.5f}  (n {len(rows)})")
    return dec


def arm_vs_c0(v, h, arm, roots):
    A = name(arm)
    R.P(f"\n== {arm} vs c0, roots {roots[0]}-{roots[-1]} (B = arm)")
    res = R.arm_block(v, h, C, A, roots, rules=False)
    spec_c, spec_a = "+".join(C(r) for r in roots), "+".join(A(r) for r in roots)
    ens = {}
    for nm, ctx in (("VAL", v), ("HOLD", h)):
        e = R.compare(ctx, [(spec_c, spec_a)])
        ens[nm] = e["pooled"]["all"]
        R.P(f"  ENS {nm} {len(roots)}-member arm vs {len(roots)}-member c0: " + R.fmt({"all": e["pooled"]["all"]})
            + f"  (c0 ens {e['pairs'][0]['A_mean']:.5f}, arm ens {e['pairs'][0]['B_mean']:.5f})")
    cr = [R.corr(v, A(r), C(r)) for r in roots]
    R.P("  same-root corr arm vs c0 (val, within-feed z): " + " ".join(f"{x:.4f}" for x in cr))
    res.update(ens=ens, corr_same_root=cr)
    return res


def stage1(arms):
    v, h = R.setup("val"), R.setup("hold")
    R.P(f"== r6 Q1 wave 2, STAGE 1 (futility), roots {ROOTS1}: arm CONTINUES iff pooled member delta vs c0 >= 0 on val "
        f"AND >= 0 on holdout")
    done = [a for a in arms if complete(v, h, a, ROOTS1)]
    todo = [a for a in arms if a not in done]
    if not all(os.path.exists(v["path"](C(r))) and os.path.exists(h["path"](C(r))) for r in ROOTS1):
        R.P("  c0 control incomplete: nothing to read")
        return {}
    dec = runs_block(arms, ROOTS1)
    out = {"decline": dec}
    for a in done:
        res = arm_vs_c0(v, h, a, ROOTS1)
        dv, dh = res["val"]["pooled"]["all"], res["hold"]["pooled"]["all"]
        cont = dv["delta"] >= 0 and dh["delta"] >= 0
        verdict = ("CONTINUE -> stage 2 (roots 105-107; PASS-CD on 7 roots)" if cont else "STOP (stage-1 futility)")
        dd = (dec[a]["dec_val"] - dec["c0"]["dec_val"], dec[a]["dec_hold"] - dec["c0"]["dec_hold"]) if a in dec else None
        R.P(f"  decline diagnostic (arm - c0, end-sel): val {dd[0]:+.5f} hold {dd[1]:+.5f}" if dd else "")
        R.P(f"  ==> {a}: member val {dv['delta']:+.5f} (z {dv['z']:+.2f}), holdout {dh['delta']:+.5f} (z {dh['z']:+.2f})"
            f" -> {verdict}")
        out[a] = dict(val=dv, hold=dh, slices_val=res["val"]["pooled"], slices_hold=res["hold"]["pooled"],
                      ens=res["ens"], corr_same_root=res["corr_same_root"], continue_=cont, verdict=verdict,
                      decline_vs_c0=dd)
    R.P("\n== summary (stage 1)")
    for a in done:
        o = out[a]
        R.P(f"  {a:5s} val {o['val']['delta']:+.5f} (z {o['val']['z']:+.2f})  hold {o['hold']['delta']:+.5f} "
            f"(z {o['hold']['z']:+.2f})  ENS val {o['ens']['VAL']['delta']:+.5f} hold {o['ens']['HOLD']['delta']:+.5f}"
            f"  best steps {dec.get(a, {}).get('best_steps')}  -> {o['verdict']}")
    for a in todo:
        R.P(f"  {a:5s} incomplete (needs r6_cd_{a}_r101-104 train-only val + holdout preds): not read")
    cont = [a for a in done if out[a]["continue_"]]
    if cont:
        J2 = " ".join(f"{a}:{r}" for r in (105, 106, 107) for a in cont)
        R.P(f"\n  stage-2 jobs (c0 105-107 already exist from Q1 stage 2): J2=\"{J2}\"")
        R.P(f"  then: CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/r6_eval_cdw2.py stage2 "
            f"{' '.join(cont)}")
    out["incomplete"] = todo
    return out


def stage2(arms):
    v, h = R.setup("val"), R.setup("hold")
    R.P(f"== r6 Q1 wave 2, STAGE 2 PASS-CD on roots {ROOTS2}: member val > 2 SE AND holdout >= 0 AND 7-member ensemble "
        f"vs c0 holdout >= 0 and val >= -1 SE AND no val slice < -2 SE (coldpos < -1 SE -> stale-val replay)")
    if not all(os.path.exists(v["path"](C(r))) and os.path.exists(h["path"](C(r))) for r in ROOTS2):
        R.P("  c0 roots 101-107 incomplete: nothing to read")
        return {}
    dec = runs_block(arms, ROOTS2)
    out = {"decline": dec}
    for a in arms:
        if not complete(v, h, a, ROOTS2):
            R.P(f"\n== {a}: incomplete on roots 101-107, not read")
            continue
        res = arm_vs_c0(v, h, a, ROOTS2)
        sv, sh = res["val"]["pooled"], res["hold"]["pooled"]
        ev, eh = res["ens"]["VAL"], res["ens"]["HOLD"]
        gates = {"member val > 2 SE": sv["all"]["z"] > 2, "member holdout >= 0": sh["all"]["delta"] >= 0,
                 "ensemble holdout >= 0": eh["delta"] >= 0, "ensemble val >= -1 SE": ev["delta"] >= -ev["se"],
                 "no val slice < -2 SE": all(s["z"] >= -2 for s in sv.values())}
        stale = sv["coldpos"]["z"] < -1
        ok = all(gates.values())
        verdict = "PASS-CD" if ok else "FAIL"
        R.P("  gates: " + "; ".join(f"{k} {'OK' if x else 'NO'}" for k, x in gates.items()))
        R.P(f"  ==> {a}: {verdict}" + ("  (coldpos < -1 SE: stale-val replay required before adoption)" if stale else ""))
        out[a] = dict(val=sv, hold=sh, ens=res["ens"], gates=gates, verdict=verdict, stale_val_required=stale,
                      corr_same_root=res["corr_same_root"])
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    arms = sys.argv[2:] or (WAVE2 if mode == "stage1" else [])
    assert mode in ("stage1", "stage2") and arms, "usage: r6_eval_cdw2.py [stage1 [arm ...] | stage2 arm ...]"
    res = stage1(arms) if mode == "stage1" else stage2(arms)
    out = os.environ.get("OUT") or f"{R.R6}/logs/r6_eval_cd_w2{'' if mode == 'stage1' else '_stage2'}.txt"
    open(out, "w").write("\n".join(R.OUT) + "\n")
    json.dump(res, open(out[:-4] + ".json", "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
