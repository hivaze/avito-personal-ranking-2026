"""Read-out of the r6 WAVE-4 cooldown-DATA arms vs the m02 reference (RESEARCH_SWEEP_1112 P5 / P4 / P3; workflow
wx9m2kck8). CPU only, read-only on every prediction / log file. Same approach as r6_eval_cdw3.py: r6_eval_cdw2's code
with the reference c0 -> m02 (r6_cd_m02_r<root>: F7 v17_pre2s_s<root>@77500 re-cooled 2 epochs with --cooldown_mix
0.02, seed 811, holdout-selected); every wave-4 arm = m02 + ONE change, same roots, same seed 811 (nn_x/r6/cd_lane.sh):
  hcap  (P5) --hcap 1:110,4:117,6:41,8:42,10:40: val/test history-format caps (newest kept) on every cooldown training
        batch (pool and non-pool); the SELECTION holdout (in-training eval) uses the same capped histories. The
        read-out's holdout preds (holdout_eval.py) use the stored histories, exactly as for m02. Caps: hcap_caps.py.
  rec33 (P4) --rec_q 0.33: 33% of NON-pool cooldown batches from the last 3 train days (Sep 17-19 MSK, holdout
        excluded); m02 pool share unchanged.
  rho70 (P3) --rho_file rho/excess_r<root>.npz --rho_keep 0.7: NON-pool cooldown batches sample the top 70% of the
        production feeds by excess = L_snap(@77500) - L_ref(pool-only reference r6_rhoref_pool_s1) + the pool-eligible
        feeds; m02 pool share unchanged. Launched only after rho_preflight.py PASS (ref val >= 0.36; Spearman(excess,
        L_snap) < 0.9; kept feeds have a lower top-position (0-2) positive share than dropped feeds).

PRE-DECLARED RULES (frozen 2026-11-12 16:01 UTC, `date -u`, BEFORE any wave-4 arm run started; the reference ranker
and the rho70 scoring pass had started, no arm number existed):
  All deltas = pooled member delta vs m02 on the same roots (per-feed mean of the paired deltas; r6_eval.compare).
  Stage 1 (roots 101-104), arm-specific kill criteria (an arm CONTINUES iff none fires):
    hcap : STOP iff member val delta < 0 OR stale-val member delta < 0 (stale-val = val re-predicted under the
           project's private-like 'both_pre2both' inputs, stale_pred.py, both arms; pending until those preds exist).
    rec33: STOP iff member val delta < 0 OR member holdout delta < 0 OR Thu val delta <= Tue val delta.
    rho70: STOP iff member val delta < 0 OR member holdout delta < 0.
  Stage 2 (roots 105-107 added; m02 105-107 exist), as r6_eval_cdw3 / R6_PLAN 4.1: PASS-CD iff member val > 2 SE;
      member holdout >= 0; 7-member arm ensemble vs 7-member m02 ensemble holdout >= 0 and val >= -1 SE; no val slice
      < -2 SE; plus the pmix veto (src/pmix.py: private-mix member delta < -1 SE blocks adoption). A coldpos slice
      < -1 SE requires the stale-val replay before adoption.
  Diagnostics (reported, never gates): every run's selected step and end - selected decline, same-root correlation, the
      4-member ensemble deltas, stale-val for every arm, private-mix (pmix) delta, Thu - Tue difference.
  Adoption: a PASS-CD arm becomes the re-cool / cooldown recipe for new members (F7 roots first); rho70 needs its
      reference + scoring pass per new trunk.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/r6_eval_cdw4.py
        [stage1 [arm ...] | stage2 arm ...]  -> nn_x/r6/logs/r6_eval_cd_w4[_stage2].{txt,json}
  stale-val preds (GPU, before the hcap verdict): cd nn_x/r6 && CUDA_VISIBLE_DEVICES=<g> ../../.venv/bin/python
        stale_pred.py r6_cd_m02_r101 ... r6_cd_m02_r104 r6_cd_hcap_r101 ... r6_cd_hcap_r104 [other arms]
"""
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r6_eval_cdw2 as W  # noqa: E402

R = W.R
ARMS = ["hcap", "rec33", "rho70"]
W.WAVE2 = ARMS
W.C = lambda r: f"r6:r6_cd_m02_r{r}"  # noqa: E731  the reference replaces c0 everywhere in r6_eval_cdw2
STALE = f"{R.R6}/preds/stale"


def stale_path(n):
    return f"{STALE}/nn_{R.split_name(n)[1]}_staleval.npy"


def stale_ctx():
    s = R.setup("val")
    s["path"], s["cache"] = stale_path, {}
    return s


def stale_block(sctx, arm, roots):
    names = [W.C(r) for r in roots] + [W.name(arm)(r) for r in roots]
    miss = [n for n in names if not os.path.exists(stale_path(n))]
    if miss:
        R.P(f"  STALE-VAL {arm}: pending (missing {len(miss)} stale preds: {' '.join(R.split_name(n)[1] for n in miss)})")
        return None
    res = R.compare(sctx, [(W.C(r), W.name(arm)(r)) for r in roots])
    R.P(f"  STALE-VAL {arm} vs m02 (pooled member, private-like inputs): {R.fmt(res['pooled'])}")
    return res["pooled"]


def pmix_block(v, arm, roots):
    try:
        sys.path.insert(0, "<repo>/src")
        import pmix  # noqa: E402
        Wt = pmix.weights()
    except Exception as e:  # noqa: BLE001
        R.P(f"  PMIX {arm}: unavailable ({type(e).__name__}: {e})")
        return None
    d = np.mean([R.per_feed(v, W.name(arm)(r)) - R.per_feed(v, W.C(r)) for r in roots], 0)
    m, se = pmix.wmean_se(d, Wt["private"])
    R.P(f"  PMIX {arm}: private-mix member delta {m:+.5f} +- {se:.5f} (z {m / se:+.2f}); veto iff < -1 SE: "
        f"{'VETO' if m < -se else 'no veto'}")
    return dict(delta=float(m), se=float(se), veto=bool(m < -se))


def stage1(arms):
    v, h = R.setup("val"), R.setup("hold")
    sctx = stale_ctx()
    R.P(f"== r6 WAVE 4 (cooldown data) STAGE 1, roots {W.ROOTS1}, reference m02; kill: hcap val < 0 or stale-val < 0 | "
        f"rec33 val < 0 or holdout < 0 or Thu <= Tue | rho70 val < 0 or holdout < 0")
    if not all(os.path.exists(v["path"](W.C(r))) and os.path.exists(h["path"](W.C(r))) for r in W.ROOTS1):
        R.P("  m02 reference incomplete: nothing to read")
        return {}
    done = [a for a in arms if W.complete(v, h, a, W.ROOTS1)]
    todo = [a for a in arms if a not in done]
    dec = W.runs_block(arms, W.ROOTS1)
    out = {"decline": dec}
    for a in done:
        res = W.arm_vs_c0(v, h, a, W.ROOTS1)
        sv, sh = res["val"]["pooled"], res["hold"]["pooled"]
        dv, dh = sv["all"]["delta"], sh["all"]["delta"]
        st = stale_block(sctx, a, W.ROOTS1)
        pm = pmix_block(v, a, W.ROOTS1)
        tue, thu = sv["Tue"]["delta"], sv["Thu"]["delta"]
        R.P(f"  Tue {tue:+.5f}  Thu {thu:+.5f}  (Thu - Tue {thu - tue:+.5f})")
        if a == "hcap":
            kill = dv < 0 or (st is not None and st["all"]["delta"] < 0)
            pending = st is None and not kill
            why = f"val {dv:+.5f}, stale-val {st['all']['delta']:+.5f}" if st else f"val {dv:+.5f}, stale-val pending"
        elif a == "rec33":
            kill, pending = dv < 0 or dh < 0 or thu <= tue, False
            why = f"val {dv:+.5f}, holdout {dh:+.5f}, Thu {thu:+.5f} vs Tue {tue:+.5f}"
        else:
            kill, pending = dv < 0 or dh < 0, False
            why = f"val {dv:+.5f}, holdout {dh:+.5f}"
        verdict = ("STOP (stage-1 kill: " + why + ")" if kill else
                   "PENDING (stale-val preds missing; val >= 0 so far)" if pending else
                   "CONTINUE -> stage 2 (roots 105-107; PASS-CD on 7 roots)")
        dd = (dec[a]["dec_val"] - dec["c0"]["dec_val"], dec[a]["dec_hold"] - dec["c0"]["dec_hold"]) if a in dec else None
        R.P(f"  decline diagnostic (arm - m02, end-sel): val {dd[0]:+.5f} hold {dd[1]:+.5f}" if dd else "")
        R.P(f"  ==> {a}: member val {dv:+.5f} (z {sv['all']['z']:+.2f}), holdout {dh:+.5f} (z {sh['all']['z']:+.2f})"
            f" -> {verdict}")
        out[a] = dict(val=sv, hold=sh, stale=st, pmix=pm, tue=tue, thu=thu, ens=res["ens"],
                      corr_same_root=res["corr_same_root"], kill=kill, pending=pending, verdict=verdict,
                      decline_vs_ref=dd)
    R.P("\n== summary (stage 1, reference m02)")
    for a in done:
        o = out[a]
        R.P(f"  {a:6s} val {o['val']['all']['delta']:+.5f} (z {o['val']['all']['z']:+.2f})  hold "
            f"{o['hold']['all']['delta']:+.5f} (z {o['hold']['all']['z']:+.2f})  stale "
            + (f"{o['stale']['all']['delta']:+.5f}" if o["stale"] else "pending")
            + f"  Thu-Tue {o['thu'] - o['tue']:+.5f}  ENS val {o['ens']['VAL']['delta']:+.5f} hold "
              f"{o['ens']['HOLD']['delta']:+.5f}  -> {o['verdict']}")
    for a in todo:
        R.P(f"  {a:6s} incomplete (needs r6_cd_{a}_r101-104 train-only val + holdout preds): not read")
    cont = [a for a in done if not out[a]["kill"] and not out[a]["pending"]]
    if cont:
        J2 = " ".join(f"{a}:{r}" for r in (105, 106, 107) for a in cont)
        R.P(f"\n  stage-2 jobs (m02 105-107 exist): J2=\"{J2}\"")
        R.P(f"  then: CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=12 .venv/bin/python nn_x/r6/r6_eval_cdw4.py stage2 "
            f"{' '.join(cont)}")
    out["incomplete"] = todo
    return out


def stage2(arms):
    out = W.stage2(arms)
    v = R.setup("val")
    sctx = stale_ctx()
    for a in arms:
        if a not in out:
            continue
        pm = pmix_block(v, a, W.ROOTS2)
        st = stale_block(sctx, a, W.ROOTS2)
        out[a]["pmix"], out[a]["stale"] = pm, st
        final = out[a]["verdict"] == "PASS-CD" and not (pm and pm["veto"])
        out[a]["final"] = "PASS-CD" if final else ("FAIL (pmix veto)" if out[a]["verdict"] == "PASS-CD" else "FAIL")
        R.P(f"  ==> {a}: FINAL {out[a]['final']}")
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    arms = sys.argv[2:] or (ARMS if mode == "stage1" else [])
    assert mode in ("stage1", "stage2") and arms and all(a in ARMS for a in arms), \
        "usage: r6_eval_cdw4.py [stage1 [arm ...] | stage2 arm ...]"
    res = stage1(arms) if mode == "stage1" else stage2(arms)
    out = os.environ.get("OUT") or f"{R.R6}/logs/r6_eval_cd_w4{'' if mode == 'stage1' else '_stage2'}.txt"
    txt = "\n".join(R.OUT) + "\n"
    open(out, "w").write("REFERENCE = m02 (r6_cd_m02_r<root>); every 'c0' label below means the m02 reference\n"
                         + re.sub(r"\bc0\b", "c0[=m02]", txt))
    json.dump(res, open(out[:-4] + ".json", "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
