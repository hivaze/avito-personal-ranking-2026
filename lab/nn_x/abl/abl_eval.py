"""Read-out of the nn_x/abl ABLATION SCREEN (ledger nn_x/abl/ledger/ledger.md section 4: 12 arms, 2-epoch paired screen on
the current recipe, seeds 23 / 24, base = nn_x/w5 lane.sh A_BASE + HYG = r7 BASE2 with --cooldown_mix 0.02).
CPU only, read-only on every prediction / log file. nn_x/r6/r6_eval.py (paired per-feed machinery: R.compare = per-feed
mean of the paired NDCG deltas, SE = sd / sqrt(n_feeds)) and nn_x/w5/w5_eval.py (pmix / stale-val / 08_levers / CLIPLOG /
runs blocks) are imported read-only.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/abl_eval.py screen
        -> nn_x/abl/logs/abl_eval.{txt,json}   (OUT=<path> overrides the .txt; the .json sits next to it)
  ... abl_eval.py screen <arm ...>   only those arms
  ... abl_eval.py selftest           plumbing check on EXISTING runs (nn_x/r7: r7_dwa vs r7_base, seeds 23/24, selected
        checkpoints; the verdict printed there is meaningless)        -> nn_x/abl/logs/abl_eval_selftest.txt

Runs (abl_lane.sh): abl_<arm>_s<seed>; every run writes the SELECTED checkpoint (holdout-best among evals at step >=
12000) as <v> and the LAST checkpoint (step 19,670) as <v>@last, each with train-only val preds and holdout preds
(preds/holdout/nn_<v>[@last]_holdout.npy; the fixed 30K random-like train holdout).

PRE-DECLARED RULES (frozen 2026-11-13 00:08 UTC, `date -u`, BEFORE any abl run; only the harness tests had run):
  Primary read = the LAST checkpoint (k = 19,670 >= 12k: the holdout is then an unselected judge). Secondary read
      (reported, never a gate) = the selected checkpoint.
  Delta = arm - base on the SAME seed, per feed; pooled = per-feed mean over the two seeds; SE_feed = sd / sqrt(n_feeds);
      SE_cal = sqrt(SE_feed^2 + sigma_run^2 / 2) with sigma_run 0.00035 (val) / 0.00015 (holdout) (nn_x/r7 A/A pairs).
      Slices (val): M<50, M>=50, Tue, Thu, coldpos (all positives cold), warmpos (all positives warm); holdout: all,
      late20%. Every slice uses its own SE_feed.
  ABLATION arms (no_aux, no_retr, xdrop_pop, xdrop_ctxnd, no_sim, no_domain, lnorm_batch, no_set): the component's
      WORTH on the current recipe = -delta (val and holdout).
      LOAD-BEARING  iff pooled val delta < -2 SE_cal(val) AND pooled holdout delta < 0.
      DISPENSABLE   iff |pooled val delta| <= 1 SE_cal(val) AND pooled holdout delta >= -1 SE_cal(hold) AND no val slice
                    < -2 SE_feed(slice)   (candidate simplification / cost lever; confirm with a 10-epoch pair first).
      INCONCLUSIVE  otherwise (the sign and the slices are reported).
  SCALE-UP arms (retr32, retr32u, xrel_more, blocks3):
      PASS iff pooled val delta > +1 SE_cal(val) AND pooled holdout delta >= 0  -> a 10-epoch pair with E_BASE + HYG (the
           wave-5 stage-E base runs w5_e10_base_s951-954 are the exact flags-off references: nn_x/abl is proven
           bit-identical to nn_x/w5 at flags off, tests/test_abl.py), judged by the unified rule (val > 2 SE, holdout >= 0,
           no slice < -2 SE) plus the ensemble-level check.
      Info (not a gate): STRICT-PROMOTE (r7 stage-1 rule) iff val > 1.5 SE_cal AND holdout >= 0 AND (both seeds' val
           delta > 0 OR both holdout delta > 0); COLD/WARM TRADE flag iff |coldpos - warmpos| > 2 sqrt(SE_c^2 + SE_w^2)
           (then the stale-val and pmix reads are required before any adoption).
  Diagnostics (reported, never gates): per-seed deltas; selected read; private-mix pooled delta (src/pmix.py); 08_levers
      per-positive-class loss table; CLIPLOG; train minutes vs base (cost levers); for the input arms xdrop_pop /
      xdrop_ctxnd / xrel_more the STALE-VAL delta (val re-predicted under the private-like 'both_pre2both' inputs,
      stale_pred.py, selected and last checkpoints of arm and base) and PRIVATE TRADE = stale-val delta - val delta
      (> 0: the arm loses less than the base under the private-like coverage shift).

POP1 ARM (pre-declared in ledger section 7 BEFORE the first pop1 run; nn_x/codes/followup/RESULT.txt section (4); the
frozen rules above are NOT changed, one comparison block is ADDED):
  pop1 = A_BASE + HYG + --xt_exclude <the 8 popularity columns of xdrop_pop> --xt_add xp_imp_c50 (type SCALE-UP).
  PRIMARY comparison = pop1 vs its REFERENCE ARM xdrop_pop (REFS below), same machinery as every arm-vs-base read: LAST
      checkpoints, delta = pop1 - xdrop_pop on the same seed per feed, pooled over seeds 23 / 24, SE_cal as above.
      PASS iff pooled val delta > +1 SE_cal(val) AND pooled holdout delta >= 0 -> (if the final pop-free FT read confirms the
      deficit) rerun the 10-epoch pop-free family with the same flags (E_BASE + HYG, seeds 951-954) + decide9 / decide6;
      FAIL -> the pop-free family stays as it is. STRICT-PROMOTE / COLD-WARM TRADE flags reported as information.
  pop1 vs base = INFORMATION only (the remaining gap; its SCALE-UP verdict is printed as 'would-PASS' / 'would-FAIL', never a
      gate), with the same diagnostics (slices, pmix, stale-val + PRIVATE TRADE, levers, CLIPLOG, minutes) in both blocks.

POP2 ARM (pre-declared in ledger section 8 BEFORE the first pop2 run; POPULARITY-VIEW FAMILIES follow-up, 2026-11-13 ~14:30 UTC;
the frozen rules are NOT changed, the pop1 block form is reused with REFS[pop2] = xdrop_pop):
  pop2 = A_BASE + HYG + --xt_exclude <the 8 popularity columns of xdrop_pop> --xt_add xp_ctr_c (type SCALE-UP), with
      xp_ctr_c = clip(log(tte_it_ctr / 0.0159), -2, 2) (nn_x/abl/pop2/build_xp_ctr.py). PRIMARY = pop2 vs xdrop_pop, LAST
      checkpoints, PASS iff pooled val delta > +1 SE_cal(val) AND pooled holdout delta >= 0 -> the 10-epoch family
      abl_e10_pop2_s951-954 (E_BASE + HYG + the pop2 flags) + std FT + the Q3 / Q3b pool read vs G3 (ledger section 8);
      FAIL -> Task B stops. pop2 vs base = INFORMATION only.

TIME ARM (pre-declared in ledger section 9 BEFORE the first time run; CURRENT_STATUS Log 19:08 "TIME BLIND SPOT"; the frozen
rules are NOT changed; only the ARMS entry is added here, REFERENCE = base = the default comparison, no REFS entry):
  time = A_BASE + HYG + --xt_add tm_weekend,tm_hsin,tm_hcos (type SCALE-UP; nn_x/abl/time/build_tm.py: the feed's Moscow weekend
      flag and sin / cos of its hour-of-day fraction, broadcast to every candidate). The PRE-DECLARED PRIMARY READ is the holdout
      WEEKEND slice (val has 0 weekend feeds, so the weekend x category interaction is invisible on val): it is produced by the
      COPY nn_x/abl/time/abl_eval_time.py (this module's screen() with the holdout slices weekend / weekday added), rule there and
      in ledger section 9. The standard SCALE-UP verdict printed by this module for the time arm is INFORMATION only.
"""
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
import numpy as np  # noqa: E402

ROOT = "<repo>"
ABL = f"{ROOT}/nn_x/abl"
sys.path.insert(0, f"{ROOT}/nn_x/r6")
sys.path.insert(0, f"{ROOT}/nn_x/w5")
import r6_eval as R  # noqa: E402  (read-only import of the r6 paired-evaluation helpers)
import w5_eval as W  # noqa: E402  (read-only import of the wave-5 read-out blocks; registers DIRS r7 / w5)

R.DIRS["abl"] = (f"{ABL}/preds", f"{ABL}/logs")
P = R.P
SIG_RUN = W.SIG_RUN
SEEDS = (23, 24)
# arm -> (type, component / change, stale-val read?)
ARMS = {
    "no_aux": ("ABLATION", "K9 production-order aux head (--aux_pos 0)", False),
    "no_retr": ("ABLATION", "history retrieval retr_m 16 + attn_gate (dropped)", False),
    "retr32": ("SCALE-UP", "retrieval depth 32 (--retr_m 32)", False),
    "retr32u": ("SCALE-UP", "retrieval depth 32 over distinct items + type counts (--retr_m 32 --retr_uniq --retr_tcnt)", False),
    "xdrop_pop": ("ABLATION", "8 popularity / co-visitation xt columns (--xt_exclude)", True),
    "xdrop_ctxnd": ("ABLATION", "10 ctx_nd xt columns (--xt_exclude)", True),
    "no_sim": ("ABLATION", "sim_bias cosine attention prior (dropped)", False),
    "no_domain": ("ABLATION", "traffic-domain token (--domain dropped)", False),
    "xrel_more": ("SCALE-UP", "within-feed rank/z for 4 more inputs (--xt_rel_extra)", True),
    "lnorm_batch": ("ABLATION", "K1 per-feed loss normalisation (--loss_norm batch)", False),
    "blocks3": ("SCALE-UP", "3 cross + set blocks (--blocks 3)", False),
    "no_set": ("ABLATION", "candidate-set self-attention (--no_set)", False),
    # second wave (2026-11-13 ~04:00 UTC, ABLATION_PLAN_1113 section 3.3; ledger section 6): the frozen rules above apply unchanged
    "retr_cat": ("SCALE-UP", "second HistRetrieval over the candidate's same-category history items, zero-init Z (--retr_cat 16)", False),
    "xdrop_ctxhn_hs": ("ABLATION", "6 ctx_hn + 4 ctx_hs xt columns (--xt_exclude)", True),
    "xdrop_pre2": ("ABLATION", "6 pre2* pre-trained-encoder xt columns incl. has:pre2_fe (--xt_exclude; = the cvnd inputs)", True),
    # pop1 (2026-11-13 ~06:10 UTC, ledger section 7; docstring block POP1 ARM): primary comparison vs REFS[arm], vs base = info
    "pop1": ("SCALE-UP", "exposure input xp_imp_c50 = log1p(min(tte_it_imp, 50)) on the pop-free inputs (--xt_exclude <8 pop> "
             "--xt_add xp_imp_c50)", True),
    # pop2 (2026-11-13 ~14:30 UTC, ledger section 8; docstring block POP2 ARM): primary comparison vs REFS[arm], vs base = info
    "pop2": ("SCALE-UP", "click-rate input xp_ctr_c = clip(log(tte_it_ctr / 0.0159), -2, 2) on the pop-free inputs (--xt_exclude <8 pop> "
             "--xt_add xp_ctr_c)", True),
    # time (2026-11-13 ~19:15 UTC, ledger section 9; docstring block TIME ARM): reference = base (default); the pre-declared primary
    # read (holdout weekend slice) is printed by nn_x/abl/time/abl_eval_time.py; the SCALE-UP verdict printed here is information
    "time": ("SCALE-UP", "feed time inputs tm_weekend, tm_hsin, tm_hcos (Moscow weekend flag + sin / cos hour-of-day; --xt_add "
             "tm_weekend,tm_hsin,tm_hcos on the full base inputs)", True),
}
REFS = {"pop1": "xdrop_pop", "pop2": "xdrop_pop"}  # arm -> pre-declared reference arm of its PRIMARY comparison (pop1 / pop2 vs xdrop_pop)
REF_PASS = {"pop1": "PASS -> rerun the 10-epoch pop-free family with the same flags (E_BASE + HYG, seeds 951-954) + decide9 "
                    "(if the final pop-free FT read confirms the deficit)",
            "pop2": "PASS -> 10-epoch family abl_e10_pop2_s951-954 (E_BASE + HYG + the pop2 flags) + std FT + Q3 / Q3b pools vs G3 "
                    "(ledger section 8)"}
_PASS_STD = "PASS -> 10-epoch pair (E_BASE + HYG vs w5_e10_base)"
CFG = dict(base=lambda s: f"abl:abl_base_s{s}", arm=lambda arm, s: f"abl:abl_{arm}_s{s}", last="@last", tag="")


def L(n, read):
    return n + CFG["last"] if read == "last" else n


def have(ctxs, names):
    return all(os.path.exists(c["path"](n)) for c in ctxs for n in names)


def minutes(n):
    (pdir, ldir), v = R.split_name(n)
    f = f"{ldir}/{v}.json"
    return json.load(open(f))["train_minutes"] if os.path.exists(f) else float("nan")


def verdict(kind, rv, rh):
    a, h = rv["pooled"]["all"], rh["pooled"]["all"]
    dv, dh = a["delta"], h["delta"]
    scv, sch = a["se_cal"], h["se_cal"]
    sl = {k: s_ for k, s_ in rv["pooled"].items() if k != "all"}
    worst = min(sl.items(), key=lambda kv: kv[1]["z"])
    out = dict(delta_val=dv, delta_hold=dh, se_cal_val=scv, se_cal_hold=sch, worst_slice=[worst[0], worst[1]["z"]])
    if kind == "ABLATION":
        out["worth_val"], out["worth_hold"] = -dv, -dh
        if dv < -2 * scv and dh < 0:
            v = "LOAD-BEARING"
        elif abs(dv) <= scv and dh >= -sch and all(s_["z"] >= -2 for s_ in sl.values()):
            v = "DISPENSABLE (confirm at 10 ep before adopting)"
        else:
            v = "INCONCLUSIVE"
        out["verdict"] = f"{v} (worth val {-dv:+.5f}, holdout {-dh:+.5f})"
    else:
        ok = dv > scv and dh >= 0
        strict = dv > 1.5 * scv and dh >= 0 and (all(x > 0 for x in rv["per_seed"]) or all(x > 0 for x in rh["per_seed"]))
        c, w_ = rv["pooled"]["coldpos"], rv["pooled"]["warmpos"]
        trade = abs(c["delta"] - w_["delta"]) > 2 * np.sqrt(c["se"] ** 2 + w_["se"] ** 2)
        out.update(strict_promote=bool(strict), cold_warm_trade=bool(trade))
        out["verdict"] = (("PASS -> 10-epoch pair (E_BASE + HYG vs w5_e10_base)" if ok else "FAIL")
                          + (" [strict-promote]" if strict else "") + (" [COLD/WARM TRADE: stale-val + pmix required]" if trade else ""))
    return out


def read_arm(arm, kind, what, stale, v, h, sctx, ref=None):
    A = CFG["arm"]
    base = CFG["base"] if ref is None else (lambda s: A(ref, s))  # ref: the pre-declared reference arm (REFS) replaces base
    bname = "base" if ref is None else ref
    pairs = {rd: [(L(base(s), rd), L(A(arm, s), rd)) for s in SEEDS] for rd in ("last", "sel")}
    if not have((v, h), [n for p in pairs["last"] for n in p]):
        P(f"\n== {arm} ({kind}: {what}): incomplete (needs {bname} + arm, seeds {SEEDS}, last-checkpoint val + holdout preds)")
        return None
    tag = (" [PRIMARY pre-declared comparison]" if ref is not None else
           " [INFORMATION only: the primary comparison is vs " + REFS[arm] + "]" if arm in REFS else "")
    P(f"\n== {arm} ({kind}: {what}) vs {bname} (B = arm, delta = arm - {bname}){tag}")
    rv, rh = W.read_pairs(v, h, pairs["last"], "last", se_cal=True)
    rv["per_seed"] = [p["slices"]["all"]["delta"] for p in rv["pairs"]]
    rh["per_seed"] = [p["slices"]["all"]["delta"] for p in rh["pairs"]]
    res = dict(kind=kind, what=what, val=rv["pooled"], hold=rh["pooled"], per_seed_val=rv["per_seed"], per_seed_hold=rh["per_seed"])
    if have((v, h), [n for p in pairs["sel"] for n in p]):
        sv, sh = W.read_pairs(v, h, pairs["sel"], "selected (info)")
        res["sel"] = dict(val=sv["pooled"]["all"], hold=sh["pooled"]["all"])
    mb = [minutes(base(s)) for s in SEEDS]
    ma = [minutes(A(arm, s)) for s in SEEDS]
    res["minutes"] = dict(base=mb, arm=ma, ratio=float(np.nanmean(ma) / np.nanmean(mb)) if np.nanmean(mb) > 0 else float("nan"))
    P(f"  MINUTES base {np.round(mb, 1).tolist()} arm {np.round(ma, 1).tolist()} -> ratio {res['minutes']['ratio']:.2f}")
    res["pmix"] = W.pmix_block(v, pairs["last"])
    if stale:
        st = W.stale_block(sctx, pairs["last"])
        res["stale"] = st
        if st is not None:
            tr = st["all"]["delta"] - rv["pooled"]["all"]["delta"]
            res["private_trade"] = float(tr)
            P(f"  PRIVATE TRADE (stale-val delta - val delta): {tr:+.5f} ({'arm loses LESS' if tr > 0 else 'arm loses MORE'} "
              f"than the base under the private-like coverage shift)")
        sel_pairs = pairs["sel"]
        if not any(os.path.exists(sctx["path"](n)) for p in sel_pairs for n in p):
            pass
        else:
            P("  stale-val (selected checkpoints, info):")
            res["stale_sel"] = W.stale_block(sctx, sel_pairs)
    res["levers"] = W.lever_block(v, pairs["last"])
    res["cliplog"] = W.cliplog_block([base(s) for s in SEEDS] + [A(arm, s) for s in SEEDS])
    res["readout"] = verdict(kind, rv, rh)
    if ref is not None:  # pre-declared reference-arm block: the frozen SCALE-UP numbers, its own consequence text
        res["readout"]["verdict"] = res["readout"]["verdict"].replace(_PASS_STD, REF_PASS.get(arm, _PASS_STD))
        res["ref"] = ref
    elif arm in REFS:  # arm vs base = information only for an arm with a reference arm
        res["readout"]["verdict"] = "INFO (not a gate): " + res["readout"]["verdict"].replace(_PASS_STD, "would-PASS").replace(
            "FAIL", "would-FAIL")
        res["info_only"] = True
    a = rv["pooled"]["all"]
    P(f"  ==> {arm}{'' if ref is None else ' vs ' + ref}: val {a['delta']:+.5f} (SE_cal {a['se_cal']:.5f}, z_cal {a['z_cal']:+.2f}; seeds "
      f"{' '.join(f'{x:+.5f}' for x in rv['per_seed'])}), holdout {rh['pooled']['all']['delta']:+.5f} (z_cal "
      f"{rh['pooled']['all']['z_cal']:+.2f}; seeds {' '.join(f'{x:+.5f}' for x in rh['per_seed'])}) -> {res['readout']['verdict']}")
    return res


def screen(arms=None):
    arms = arms or list(ARMS)
    v, h = R.setup("val"), R.setup("hold")
    sctx = W.stale_ctx()
    base, A = CFG["base"], CFG["arm"]
    P(f"== ABLATION SCREEN (2-epoch from-scratch, base = A_BASE + HYG, seeds {SEEDS}, primary = last checkpoint; rules in "
      f"the abl_eval.py docstring, frozen 2026-11-13 00:08 UTC)")
    refs = sorted({REFS[a] for a in arms if a in REFS and REFS[a] not in arms})
    W.runs_block([base(s) for s in SEEDS] + [A(a, s) for a in refs + list(arms) for s in SEEDS])
    out = {}
    for arm in arms:
        kind, what, stale = ARMS[arm]
        if arm in REFS:  # pre-declared PRIMARY comparison vs the reference arm first (pop1 vs xdrop_pop)
            r = read_arm(arm, kind, what, stale, v, h, sctx, ref=REFS[arm])
            if r is not None:
                out[f"{arm}_vs_{REFS[arm]}"] = r
        r = read_arm(arm, kind, what, stale, v, h, sctx)
        if r is not None:
            out[arm] = r
    P("\n== SUMMARY (last checkpoints; delta = arm - base, pooled over seeds; ABLATION worth = -delta)")
    if any("ref" in r for r in out.values()):
        P("  (rows <arm>_vs_<ref>: delta = arm - the pre-declared reference arm REFS[arm]; the plain <arm> row vs base is information)")
    P("  arm          type      val d (z_cal)        hold d (z_cal)      M<50     M>=50    Tue      Thu      coldpos  warmpos  "
      "late20%  pmix     stale    min  -> verdict")
    for arm, r in out.items():
        s, hh = r["val"], r["hold"]
        pm = r["pmix"]["delta"] if r.get("pmix") else float("nan")
        st = r["stale"]["all"]["delta"] if r.get("stale") else float("nan")
        P(f"  {arm:12s} {r['kind']:9s} {s['all']['delta']:+.5f} ({s['all']['z_cal']:+.2f})  {hh['all']['delta']:+.5f} "
          f"({hh['all']['z_cal']:+.2f})  {s['M<50']['delta']:+.5f} {s['M>=50']['delta']:+.5f} {s['Tue']['delta']:+.5f} "
          f"{s['Thu']['delta']:+.5f} {s['coldpos']['delta']:+.5f} {s['warmpos']['delta']:+.5f} {hh['late20%']['delta']:+.5f} "
          f"{pm:+.5f} {st:+.5f} {r['minutes']['ratio']:.2f} -> {r['readout']['verdict']}")
    P("\n== COMPONENT WORTH on the current recipe (ablation arms; -delta; val / holdout; +SE_cal)")
    for arm, r in out.items():
        if r["kind"] == "ABLATION":
            P(f"  {arm:12s} {r['what']:70s} worth val {r['readout']['worth_val']:+.5f} +- {r['readout']['se_cal_val']:.5f}  "
              f"holdout {r['readout']['worth_hold']:+.5f} +- {r['readout']['se_cal_hold']:.5f}")
    passed = [a for a, r in out.items() if r["kind"] == "SCALE-UP" and r["readout"]["verdict"].startswith("PASS") and "ref" not in r]
    for a, r in out.items():
        if "ref" in r:
            P(f"\n  PRE-DECLARED {a}: {r['readout']['verdict']}")
    if passed:
        P(f"\n  SCALE-UP PASS -> 10-epoch pairs (E_BASE + HYG, seeds 951/952, vs w5_e10_base_s951/952): {passed}")
    return out


def selftest():
    """Plumbing check on existing nn_x/r7 runs (r7_dwa vs r7_base, seeds 23/24, stored selected checkpoints)."""
    global ARMS
    CFG.update(base=lambda s: f"r7:r7_base_s{s}", arm=lambda arm, s: f"r7:r7_{arm}_s{s}", last="", tag="selftest")
    ARMS = {"dwa": ("SCALE-UP", "r7 dwa stand-in (selftest)", True), "nope": ("ABLATION", "missing stand-in", False)}
    P("#### SELFTEST (stand-in runs; 'last' read = the stored selected checkpoint; verdicts meaningless)")
    out = screen()
    ok = "dwa" in out and "nope" not in out and out["dwa"]["levers"] is not None and out["dwa"]["pmix"] is not None
    P(f"#### SELFTEST {'OK' if ok else 'INCOMPLETE'}: read {list(out)}")
    return dict(screen=out, ok=ok)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    assert mode in ("screen", "selftest"), "usage: abl_eval.py screen [arm ...] | selftest"
    res = screen(sys.argv[2:] or None) if mode == "screen" else selftest()
    out = os.environ.get("OUT") or f"{ABL}/logs/abl_eval{'' if mode == 'screen' else '_selftest'}.txt"
    open(out, "w").write("\n".join(R.OUT) + "\n")
    json.dump(res, open(re.sub(r"\.txt$", "", out) + ".json", "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
