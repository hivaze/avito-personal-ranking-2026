"""Read-out of the nn_x/w10/fpcd WAVE-10 arm FPCD-lite (w5_cd_fpcd_r101-104 vs the existing nn_x/w5 w5_cd_m02ref_r<root>; rule
nn_x/w5/logs/B_fpcd_predeclared.txt, frozen 2026-11-13T16:53:48Z before any arm run).
Derived 2026-11-13 ~16:55 UTC from nn_x/w5/w5_eval_wave8.py: ARMS=("fpcd",), the single-arm rule (m = 1, no Holm), gates (2) holdout and
(6) stale-val at -1 SE (the skeptics' disclosed E-regime handicap), the holdout EARLY (Sep 6-8) / LATE (Sep 9-19) INFO slices and the
FPCD log lines (realised feeds per step, pool share, split events). CPU only, read-only on every prediction / log file. A thin wrapper
over the nn_x/w5/w5_eval.py / nn_x/r6/r6_eval.py functions (imported read-only; their own stage rules are NOT used and NOT changed).
The arm's files are read under the w5: names, i.e. from nn_x/w5/{preds,logs} after the cp --update=none copy from nn_x/w10/fpcd.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/fpcd/w5_eval_wave10_fpcd.py
        -> nn_x/w5/logs/w5_eval_B_fpcd.{txt,json} (OUT=<path> overrides)

PRE-DECLARED RULE (in substance): pooled over roots 101-104, paired per val feed vs w5_cd_m02ref_r<root> on the SELECTED checkpoint
(last = info): PASS iff (1) pooled member val delta > +2 SE; (2) pooled FULL-holdout delta >= -1 SE; (3) no val slice among M<50,
M>=50, Tue, Thu < -2 SE; (4) pmix private-mix delta >= -1 SE; (5) the 4-root pool holdout delta >= +2 SE; (6) stale-val pooled delta
>= -1 SE (missing -> PENDING, never PASS). STOP otherwise; INCOMPLETE if a run is missing. INFO: holdout early / late / dE-dL, per-root
deltas, same-root corr, coldpos / warmpos, levers, cliplog, @last read, pool val, FPCD log lines, wall time. Information arm: a PASS
needs the generational bar, the TRICKS 4.2 clause and a 7-root confirmation under a new pre-declaration before any pick.
"""
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
sys.path.insert(0, "<repo>/nn_x/w5")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.stats import norm  # noqa: E402

import w5_eval as E  # noqa: E402  (read-only import of nn_x/w5/w5_eval.py; its stage functions are not called)

R, P, W5 = E.R, E.P, E.W5
ROOT = E.ROOT
ARMS = ("fpcd",)
ROOTS = (101, 102, 103, 104)
GATE_SLICES = ("M<50", "M>=50", "Tue", "Thu")
ALPHA = float(norm.sf(2.0))  # one-sided P(z > 2) = 0.02275
FLAGS = {"fpcd": "--cd_exclude_days 0,1,2", "hzcd": "--hzcd (selftest stand-in)", "auxf": "--aux_norm feed (selftest stand-in)"}
RULE = "nn_x/w5/logs/B_fpcd_predeclared.txt"


def holdout_days():
    """Moscow day of every holdout feed (train.py's holdout construction replicated, as r6_eval.setup('hold'))."""
    tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
    pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    trM = tr["M"].to_numpy()
    pr[trM < 50] = 0.0
    cand = np.where((pr > 0.9) & (trM >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    ts = tr["timestamp"].to_numpy().astype(np.int64)[hold]
    return (ts + 3 * 3600) // 86400


def add_holdout_info_slices(h):
    day = holdout_days()
    assert len(day) == len(h["M"])
    h["slices"]["early Sep6-8"] = day <= 20704
    h["slices"]["late Sep9-19"] = day >= 20705
    P(f"  holdout INFO slices: early Sep 6-8 (days <= 20704) {int((day <= 20704).sum())} feeds, late Sep 9-19 (days >= 20705) "
      f"{int((day >= 20705).sum())} feeds (of {len(day)})")


def parse_logs(arm):
    """Per root: the last FPCD summary line (realised feeds / step, pool share, split events) and the setup line, from logs/<v>.log."""
    out = {}
    for r in ROOTS:
        f = f"{W5}/logs/w5_cd_{arm}_r{r}.log"
        if not os.path.exists(f):
            continue
        txt = open(f).read()
        s = re.findall(r"wave-10 FPCD step (\d+): cooldown steps (\d+) \(regular (\d+), pool (\d+), pool share ([\d.]+)\); realised feeds/step "
                       r"regular ([\d.]+) \(unmasked ([\d.]+), ratio ([\d.]+)\) pool ([\d.]+) \(unmasked ([\d.]+), ratio ([\d.]+)\); pool passes started "
                       r"(\d+), pool passes per allowed pool feed ([\d.]+); split events regular (\d+) pool (\d+)", txt)
        setup = re.findall(r"wave-10 FPCD-lite: .*", txt)
        tm = re.findall(r"train time ([\d.]+) min", txt)
        if s:
            x = s[-1]
            out[r] = dict(step=int(x[0]), cd_steps=int(x[1]), reg=int(x[2]), pool=int(x[3]), pool_share=float(x[4]), reg_feeds=float(x[5]),
                          reg_unmasked=float(x[6]), reg_ratio=float(x[7]), pool_feeds=float(x[8]), pool_unmasked=float(x[9]),
                          pool_ratio=float(x[10]), pool_passes=int(x[11]), pool_passes_per_feed=float(x[12]), split_reg=int(x[13]),
                          split_pool=int(x[14]), setup=setup[0] if setup else None, minutes=float(tm[-1]) if tm else None)
    return out


def read_arm(v, h, sctx, arm, ref, name):
    pairs = {rd: [(E.L(ref(r), rd), E.L(name(arm, r), rd)) for r in ROOTS] for rd in ("sel", "last")}
    miss = [n for p in pairs["sel"] for n in p if not (os.path.exists(v["path"](n)) and os.path.exists(h["path"](n)))]
    if miss:
        P(f"\n== {arm}: INCOMPLETE on roots {ROOTS} (missing selected val/holdout preds: {' '.join(R.split_name(n)[1] for n in miss)}): not gated")
        return dict(verdict="INCOMPLETE", missing=[R.split_name(n)[1] for n in miss])
    P(f"\n== {arm} ({FLAGS.get(arm, '?')}) vs m02ref (B = arm); PRIMARY = selected checkpoints")
    rv, rh = E.read_pairs(v, h, pairs["sel"], "selected (PRIMARY)")
    sv, sh = rv["pooled"], rh["pooled"]
    res = dict(flag=FLAGS.get(arm), val=sv, hold=sh, per_root_val=[p["slices"]["all"]["delta"] for p in rv["pairs"]],
               per_root_hold=[p["slices"]["all"]["delta"] for p in rh["pairs"]],
               per_root_hold_early=[p["slices"]["early Sep6-8"]["delta"] for p in rh["pairs"]],
               per_root_hold_late=[p["slices"]["late Sep9-19"]["delta"] for p in rh["pairs"]])
    e_, l_ = sh["early Sep6-8"], sh["late Sep9-19"]
    res["dE_minus_dL"] = dict(delta=float(e_["delta"] - l_["delta"]), se=float(np.sqrt(e_["se"] ** 2 + l_["se"] ** 2)))
    P(f"  HOLDOUT INFO (selected): early Sep 6-8 {e_['delta']:+.5f}+-{e_['se']:.5f} (z {e_['z']:+.2f})  late Sep 9-19 {l_['delta']:+.5f}+-{l_['se']:.5f} "
      f"(z {l_['z']:+.2f})  dE-dL {res['dE_minus_dL']['delta']:+.5f}+-{res['dE_minus_dL']['se']:.5f}; per-root early "
      + " ".join(f"{x:+.5f}" for x in res["per_root_hold_early"]) + "; per-root late " + " ".join(f"{x:+.5f}" for x in res["per_root_hold_late"]))
    if E.have((v, h), [n for p in pairs["last"] for n in p]):
        lv, lh = E.read_pairs(v, h, pairs["last"], "last (info)")
        res["last"] = dict(val=lv["pooled"], hold=lh["pooled"])
    else:
        P("  [last (info)]: last-checkpoint preds missing for at least one run")
        res["last"] = None
    res["ens"] = E.ens_block(v, h, [a for a, _ in pairs["sel"]], [b for _, b in pairs["sel"]])
    res["corr_same_root"] = [R.corr(v, a, b) for a, b in pairs["sel"]]
    P("  same-root corr arm vs m02ref (val, within-feed z, selected): " + " ".join(f"{x:.4f}" for x in res["corr_same_root"])
      + f" (mean {np.mean(res['corr_same_root']):.4f})")
    res["pmix"] = E.pmix_block(v, pairs["sel"])
    res["stale"] = E.stale_block(sctx, pairs["sel"])
    res["levers"] = E.lever_block(v, pairs["sel"])
    res["cliplog"] = E.cliplog_block([ref(ROOTS[0]), name(arm, ROOTS[0])])
    res["logs"] = parse_logs(arm)
    for r, x in res["logs"].items():
        P(f"  log r{r}: FPCD step {x['step']}: cooldown steps {x['cd_steps']} (regular {x['reg']}, pool {x['pool']}, pool share {x['pool_share']:.4f}); "
          f"realised feeds/step regular {x['reg_feeds']:.1f} (unmasked {x['reg_unmasked']:.1f}, ratio {x['reg_ratio']:.4f}) pool {x['pool_feeds']:.1f} "
          f"(ratio {x['pool_ratio']:.4f}); pool passes {x['pool_passes']} ({x['pool_passes_per_feed']:.3f} per allowed pool feed); split events "
          f"regular {x['split_reg']} pool {x['split_pool']}; train time {x['minutes']} min")
    dv, dh = sv["all"]["delta"], sh["all"]["delta"]
    pm, st, eh = res["pmix"], res["stale"], res["ens"]["HOLD"]
    res["z_val"] = float(sv["all"]["z"])
    res["p_val"] = float(norm.sf(res["z_val"]))
    gates = {"val > +2 SE": res["z_val"] > 2.0,
             "holdout >= -1 SE": dh >= -sh["all"]["se"],
             "no val slice (M<50, M>=50, Tue, Thu) < -2 SE": all(sv[k]["z"] >= -2 for k in GATE_SLICES),
             "pmix >= -1 SE": pm is not None and not pm["below_1se"],
             "4-root pool holdout >= +2 SE": eh["z"] >= 2,
             "stale-val >= -1 SE": st is not None and st["all"]["delta"] >= -st["all"]["se"]}
    res["gates"] = gates
    res["stale_pending"] = st is None
    res["futility_note"] = (dv < 0) or (dh < 0)
    P(f"  slices (selected): M<50 {sv['M<50']['delta']:+.5f}+-{sv['M<50']['se']:.5f} (z {sv['M<50']['z']:+.2f})  M>=50 {sv['M>=50']['delta']:+.5f} "
      f"(z {sv['M>=50']['z']:+.2f})  Tue {sv['Tue']['delta']:+.5f} (z {sv['Tue']['z']:+.2f})  Thu {sv['Thu']['delta']:+.5f} (z {sv['Thu']['z']:+.2f})  "
      f"coldpos {sv['coldpos']['delta']:+.5f} (z {sv['coldpos']['z']:+.2f})  warmpos {sv['warmpos']['delta']:+.5f} (z {sv['warmpos']['z']:+.2f})  "
      f"| holdout late20% {sh['late20%']['delta']:+.5f}")
    P(f"  val gate input: z {res['z_val']:+.3f} (one-sided p {res['p_val']:.4f}); futility note (val < 0 or holdout < 0): "
      f"{'fires' if res['futility_note'] else 'does not fire'}")
    P("  gates: " + "; ".join(f"{k} {'OK' if x else 'NO'}" for k, x in gates.items()) + ("; stale-val PENDING" if st is None else ""))
    return res


def main():
    args = sys.argv[1:]
    selftest = bool(args) and args[0] == "selftest"
    arms = ("hzcd", "auxf") if selftest else (tuple(args) or ARMS)
    ref, name = E.CFG["B_ref"], E.CFG["B_arm"]
    v, h = R.setup("val"), R.setup("hold")
    add_holdout_info_slices(h)
    sctx = E.stale_ctx()
    P(("#### SELFTEST on existing closed arms (verdicts meaningless)\n" if selftest else "")
      + f"== WAVE-10 FPCD-lite STAGE B (CD re-cool, roots {ROOTS}, reference w5_cd_m02ref_r<root>, CDSEED 900 + root; PRIMARY = SELECTED checkpoint, "
        f"last = info). Rule ({RULE}, single arm): PASS iff val > +2 SE AND holdout >= -1 SE AND no val slice (M<50, M>=50, Tue, Thu) < -2 SE "
        f"AND pmix >= -1 SE AND 4-root pool holdout >= +2 SE AND stale-val >= -1 SE; STOP otherwise. INFORMATION ARM (grey-zone pre-gate): "
        f"holdout early / late / dE-dL are info columns; a PASS needs the generational bar + TRICKS 4.2 + a 7-root confirmation before any pick")
    runs = E.runs_block([ref(r) for r in ROOTS] + [name(a, r) for a in arms for r in ROOTS])
    out = {"runs": runs, "arms": {}, "rule": RULE}
    for arm in arms:
        out["arms"][arm] = read_arm(v, h, sctx, arm, ref, name)
    P("\n== summary (wave 10 FPCD-lite, stage B, SELECTED checkpoints, pooled over roots 101-104 vs m02ref)")
    for arm, r in out["arms"].items():
        if r.get("verdict") == "INCOMPLETE":
            P(f"  {arm:5s} INCOMPLETE (missing {' '.join(r['missing'])})")
            continue
        g = r["gates"]
        if all(g.values()):
            verdict = "PASS (information arm: generational bar + TRICKS 4.2 + 7-root confirmation needed before any pick)"
        elif r["stale_pending"] and all(x for k, x in g.items() if not k.startswith("stale")):
            verdict = "PENDING (stale-val preds missing; every other gate holds)"
        else:
            verdict = "STOP (" + ", ".join(k for k, x in g.items() if not x) + ")"
        r["verdict"] = verdict
        s, hh, e, pm, st = r["val"], r["hold"], r["ens"], r["pmix"], r["stale"]
        P(f"  {arm:5s} val {s['all']['delta']:+.5f}+-{s['all']['se']:.5f} (z {s['all']['z']:+.2f}) hold {hh['all']['delta']:+.5f}+-{hh['all']['se']:.5f} "
          f"(z {hh['all']['z']:+.2f}) [early {hh['early Sep6-8']['delta']:+.5f} late {hh['late Sep9-19']['delta']:+.5f} dE-dL {r['dE_minus_dL']['delta']:+.5f}"
          f"+-{r['dE_minus_dL']['se']:.5f}] | M<50 {s['M<50']['delta']:+.5f} M>=50 {s['M>=50']['delta']:+.5f} Tue {s['Tue']['delta']:+.5f} Thu "
          f"{s['Thu']['delta']:+.5f} coldpos {s['coldpos']['delta']:+.5f} warmpos {s['warmpos']['delta']:+.5f} | pmix "
          f"{pm['delta'] if pm else float('nan'):+.5f}+-{pm['se'] if pm else float('nan'):.5f} | pool val {e['VAL']['delta']:+.5f} hold "
          f"{e['HOLD']['delta']:+.5f}+-{e['HOLD']['se']:.5f} (z {e['HOLD']['z']:+.2f}) | stale "
          + (f"{st['all']['delta']:+.5f}+-{st['all']['se']:.5f}" if st else "pending") + f" | corr {np.mean(r['corr_same_root']):.4f}"
          + (f" | last val {r['last']['val']['all']['delta']:+.5f} hold {r['last']['hold']['all']['delta']:+.5f}" if r["last"] else "")
          + f" -> {verdict}")
    P("\n== registry-ready one-liners")
    for arm, r in out["arms"].items():
        if r["verdict"] == "INCOMPLETE":
            continue
        s, hh, e, pm, st = r["val"], r["hold"], r["ens"], r["pmix"], r["stale"]
        lg = r["logs"]
        extra = ""
        if lg:
            rr = [x["reg_ratio"] for x in lg.values()]
            ps = [x["pool_share"] for x in lg.values()]
            sp = [x["split_reg"] + x["split_pool"] for x in lg.values()]
            mins = [x["minutes"] for x in lg.values() if x["minutes"]]
            extra = (f", realised regular batch ratio {min(rr):.4f}-{max(rr):.4f}, pool share {min(ps):.4f}-{max(ps):.4f}, split events {min(sp)}-{max(sp)}"
                     + (f", {min(mins):.0f}-{max(mins):.0f} min per root" if mins else ""))
        P(f"  w5_cd_{arm}_r101-104 ({r['flag']}: Sep 6-8 pre2-blank feeds masked out of every cooldown batch, same 19,670 steps) vs w5_cd_m02ref "
          f"(CD harness, CDSEED 900+root, selected ckpt, pre-declared single-arm rule {RULE}): val {s['all']['delta']:+.5f} +- {s['all']['se']:.5f} "
          f"(z {s['all']['z']:+.1f}), holdout {hh['all']['delta']:+.5f} +- {hh['all']['se']:.5f} (z {hh['all']['z']:+.1f}; early Sep 6-8 "
          f"{hh['early Sep6-8']['delta']:+.5f}, late Sep 9-19 {hh['late Sep9-19']['delta']:+.5f}, dE-dL {r['dE_minus_dL']['delta']:+.5f} +- {r['dE_minus_dL']['se']:.5f}), "
          f"pool hold {e['HOLD']['delta']:+.5f} (z {e['HOLD']['z']:+.1f}), pool val {e['VAL']['delta']:+.5f}, pmix {pm['delta'] if pm else float('nan'):+.5f}, "
          f"stale " + (f"{st['all']['delta']:+.5f}" if st else "pending") + f", M<50 {s['M<50']['delta']:+.5f} / M>=50 {s['M>=50']['delta']:+.5f} / Tue "
          f"{s['Tue']['delta']:+.5f} / Thu {s['Thu']['delta']:+.5f}, coldpos {s['coldpos']['delta']:+.5f} / warmpos {s['warmpos']['delta']:+.5f}, "
          f"same-root corr {np.mean(r['corr_same_root']):.4f}{extra}"
          + (f", @last val {r['last']['val']['all']['delta']:+.5f} / hold {r['last']['hold']['all']['delta']:+.5f}" if r["last"] else "")
          + f" -> {r['verdict']} (nn_x/w5/logs/w5_eval_B_fpcd.txt)")
    out_f = os.environ.get("OUT") or f"{W5}/logs/w5_eval_B_fpcd{'_selftest' if selftest else ''}.txt"
    open(out_f, "w").write("\n".join(R.OUT) + "\n")
    json.dump(out, open(re.sub(r"\.txt$", "", out_f) + ".json", "w"), indent=1, default=float)
    print(f"wrote {out_f}")


if __name__ == "__main__":
    main()
