"""Read-out of the nn_x/w5 STAGE-B WAVE-7 arms rdrop / sam (cooldown regularisers, cd_arms8.py; KL probe
nn_x/tricks/rdrop_cd_kl/RESULT.txt). CPU only, read-only on every prediction / log file. A thin wrapper over the w5_eval.py /
nn_x/r6/r6_eval.py functions (imported read-only; w5_eval.py's own stage rules are NOT used and NOT changed).

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w5/w5_eval_wave7.py [arm ...]
        -> nn_x/w5/logs/w5_eval_B_wave7.{txt,json} (OUT=<path> overrides)
  ... w5_eval_wave7.py selftest    plumbing check on the EXISTING closed arms hzcd / auxf (verdicts meaningless)
        -> logs/w5_eval_B_wave7_selftest.{txt,json}

PRE-DECLARED RULE (logs/B_wave7_predeclared.txt, frozen before the first wave-7 run; in substance): per arm, pooled over roots
101-104, paired per val feed vs w5_cd_m02ref_r<root> on the SELECTED checkpoint (last = info): PASS iff (1) pooled member val
delta > +2 SE, Holm-corrected over the two arms (family alpha = P(z > 2) = 0.02275 one-sided; smaller p <= alpha/2 i.e. z >=
2.2776, then the other at alpha; primary arm rdrop reported first, its unadjusted status as info); (2) pooled holdout delta >= 0;
(3) no val slice among M<50, M>=50, Tue, Thu < -2 SE; (4) pmix private-mix delta >= -1 SE; (5) the 4-root pool holdout delta
>= +2 SE; (6) stale-val pooled delta >= 0 (stale_pred.py 'both_pre2both', selected checkpoints; missing -> PENDING, never PASS);
(7) SAM only: ||eps||/||w|| <= 0.02 in all 4 run logs, else INVALID. STOP otherwise; INCOMPLETE if a run is missing.
Diagnostics: same-root within-feed corr to m02ref, coldpos / warmpos, per-root deltas, pool val delta, last read, levers,
cliplog, logged RDROP symKL per eval, SAM ratios, the unadjusted z > 2 status, the futility note.
"""
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

import w5_eval as E  # noqa: E402  (read-only import; its stage functions are not called)

R, P, W5 = E.R, E.P, E.W5
ARMS = ("rdrop", "sam")
ROOTS = (101, 102, 103, 104)
GATE_SLICES = ("M<50", "M>=50", "Tue", "Thu")
ALPHA = float(norm.sf(2.0))  # one-sided P(z > 2) = 0.02275
FLAGS = {"rdrop": "--rdrop_cd 1.0", "sam": "--sam_cd 0.05", "hzcd": "--hzcd (selftest stand-in)", "auxf": "--aux_norm feed (selftest stand-in)"}


def parse_logs(arm):
    """Per root: RDROP mean symKL per eval (last / mean over evals) and the SAM first-step ratio / validity, from logs/<v>.log."""
    out = {}
    for r in ROOTS:
        f = f"{W5}/logs/w5_cd_{arm}_r{r}.log"
        if not os.path.exists(f):
            continue
        txt = open(f).read()
        kl = [float(x) for x in re.findall(r"wave-7 RDROP step \d+: mean per-feed symKL ([\d.]+) nats", txt)]
        m = re.search(r"wave-7 SAM first cooldown step \d+: .*?\|\|eps\|\|/\|\|w\|\| ([\d.]+) -> (VALID|INVALID)", txt)
        out[r] = dict(kl_last=kl[-1] if kl else None, kl_mean=float(np.mean(kl)) if kl else None, n_kl=len(kl),
                      sam_ratio=float(m.group(1)) if m else None, sam_valid=(m.group(2) == "VALID") if m else None)
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
               per_root_hold=[p["slices"]["all"]["delta"] for p in rh["pairs"]])
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
        bits = []
        if x["n_kl"]:
            bits.append(f"RDROP symKL last eval {x['kl_last']:.6f} / mean over {x['n_kl']} evals {x['kl_mean']:.6f} nats")
        if x["sam_ratio"] is not None:
            bits.append(f"SAM ||eps||/||w|| {x['sam_ratio']:.6f} ({'VALID' if x['sam_valid'] else 'INVALID'})")
        if bits:
            P(f"  log r{r}: " + "; ".join(bits))
    dv, dh = sv["all"]["delta"], sh["all"]["delta"]
    pm, st, eh = res["pmix"], res["stale"], res["ens"]["HOLD"]
    res["z_val"] = float(sv["all"]["z"])
    res["p_val"] = float(norm.sf(res["z_val"]))
    gates = {"holdout >= 0": dh >= 0,
             "no val slice (M<50, M>=50, Tue, Thu) < -2 SE": all(sv[k]["z"] >= -2 for k in GATE_SLICES),
             "pmix >= -1 SE": pm is not None and not pm["below_1se"],
             "4-root pool holdout >= +2 SE": eh["z"] >= 2,
             "stale-val >= 0": st is not None and st["all"]["delta"] >= 0}
    if arm == "sam" or any(x["sam_ratio"] is not None for x in res["logs"].values()):
        ok = [x["sam_valid"] for x in res["logs"].values()]
        gates["SAM ||eps||/||w|| <= 0.02 on all roots"] = len(ok) == 4 and all(ok)
    res["gates"] = gates
    res["stale_pending"] = st is None
    res["unadjusted_val_gt_2se"] = res["z_val"] > 2
    res["futility_note"] = (dv < 0) or (dh < 0)
    P(f"  slices (selected): M<50 {sv['M<50']['delta']:+.5f}+-{sv['M<50']['se']:.5f} (z {sv['M<50']['z']:+.2f})  M>=50 {sv['M>=50']['delta']:+.5f} "
      f"(z {sv['M>=50']['z']:+.2f})  Tue {sv['Tue']['delta']:+.5f} (z {sv['Tue']['z']:+.2f})  Thu {sv['Thu']['delta']:+.5f} (z {sv['Thu']['z']:+.2f})  "
      f"coldpos {sv['coldpos']['delta']:+.5f} (z {sv['coldpos']['z']:+.2f})  warmpos {sv['warmpos']['delta']:+.5f} (z {sv['warmpos']['z']:+.2f})  "
      f"| holdout late20% {sh['late20%']['delta']:+.5f}")
    P(f"  val gate input: z {res['z_val']:+.3f} (one-sided p {res['p_val']:.4f}); unadjusted z > 2: {'yes' if res['unadjusted_val_gt_2se'] else 'no'}; "
      f"futility note (val < 0 or holdout < 0): {'fires' if res['futility_note'] else 'does not fire'}")
    P("  other gates: " + "; ".join(f"{k} {'OK' if x else 'NO'}" for k, x in gates.items()) + ("; stale-val PENDING" if st is None else ""))
    return res


def holm(res):
    """Holm step-down over the readable arms' one-sided val p-values at family alpha = P(z > 2); sets gate (1) per arm."""
    arms = [a for a, r in res.items() if r.get("verdict") != "INCOMPLETE"]
    m = len(arms)
    if m == 0:
        return
    order = sorted(arms, key=lambda a: (res[a]["p_val"], 0 if a == "rdrop" else 1))  # ties -> rdrop first
    passed_prev = True
    adj_prev = 0.0
    for k, a in enumerate(order):
        p = res[a]["p_val"]
        adj = max(adj_prev, min(1.0, (m - k) * p))
        ok = passed_prev and adj <= ALPHA
        res[a]["p_holm"], res[a]["gate_val_holm"] = float(adj), bool(ok)
        res[a]["gates"] = {"val > +2 SE (Holm over %d arm%s)" % (m, "s" if m > 1 else ""): ok, **res[a]["gates"]}
        passed_prev, adj_prev = ok, adj
    P(f"\n== Holm step-down over {m} arm(s), family alpha {ALPHA:.5f} (= P(z > 2)); order by p: {order}; thresholds: first "
      f"p <= {ALPHA / m:.5f} (z >= {norm.isf(ALPHA / m):.4f})" + (f", then p <= {ALPHA:.5f} (z >= 2.0)" if m > 1 else ""))
    for a in order:
        r = res[a]
        P(f"  {a:5s} z {r['z_val']:+.3f} p {r['p_val']:.4f} Holm-adjusted p {r['p_holm']:.4f} -> val gate {'OK' if r['gate_val_holm'] else 'NO'}"
          + (" (unadjusted z > 2: yes)" if r["unadjusted_val_gt_2se"] else ""))


def main():
    args = sys.argv[1:]
    selftest = bool(args) and args[0] == "selftest"
    arms = ("hzcd", "auxf") if selftest else (tuple(args) or ARMS)
    ref, name = E.CFG["B_ref"], E.CFG["B_arm"]
    v, h = R.setup("val"), R.setup("hold")
    sctx = E.stale_ctx()
    P(("#### SELFTEST on existing closed arms (verdicts meaningless)\n" if selftest else "")
      + f"== WAVE-7 STAGE B (CD re-cool, roots {ROOTS}, reference w5_cd_m02ref_r<root>, CDSEED 900 + root; PRIMARY = SELECTED checkpoint, "
        f"last = info). Rule (logs/B_wave7_predeclared.txt): PASS iff val > +2 SE (Holm over the two arms, primary rdrop) AND holdout >= 0 "
        f"AND no val slice (M<50, M>=50, Tue, Thu) < -2 SE AND pmix >= -1 SE AND 4-root pool holdout >= +2 SE AND stale-val >= 0 "
        f"(AND SAM ||eps||/||w|| <= 0.02); STOP otherwise")
    runs = E.runs_block([ref(r) for r in ROOTS] + [name(a, r) for a in arms for r in ROOTS])
    out = {"runs": runs, "arms": {}}
    for arm in arms:
        out["arms"][arm] = read_arm(v, h, sctx, arm, ref, name)
    holm(out["arms"])
    P("\n== summary (wave 7, stage B, SELECTED checkpoints, pooled over roots 101-104 vs m02ref)")
    for arm, r in out["arms"].items():
        if r.get("verdict") == "INCOMPLETE":
            P(f"  {arm:5s} INCOMPLETE (missing {' '.join(r['missing'])})")
            continue
        g = r["gates"]
        sam_bad = any(k.startswith("SAM") and not x for k, x in g.items())
        if sam_bad:
            verdict = "INVALID (||eps||/||w|| > 0.02 or ratio missing)"
        elif all(g.values()):
            verdict = "PASS"
        elif r["stale_pending"] and all(x for k, x in g.items() if not k.startswith("stale")):
            verdict = "PENDING (stale-val preds missing; every other gate holds)"
        else:
            verdict = "STOP (" + ", ".join(k for k, x in g.items() if not x) + ")"
        r["verdict"] = verdict
        s, hh, e, pm, st = r["val"], r["hold"], r["ens"], r["pmix"], r["stale"]
        P(f"  {arm:5s} val {s['all']['delta']:+.5f}+-{s['all']['se']:.5f} (z {s['all']['z']:+.2f}, Holm p {r['p_holm']:.4f}) hold "
          f"{hh['all']['delta']:+.5f}+-{hh['all']['se']:.5f} (z {hh['all']['z']:+.2f}) | M<50 {s['M<50']['delta']:+.5f} M>=50 {s['M>=50']['delta']:+.5f} "
          f"Tue {s['Tue']['delta']:+.5f} Thu {s['Thu']['delta']:+.5f} coldpos {s['coldpos']['delta']:+.5f} warmpos {s['warmpos']['delta']:+.5f} | pmix "
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
        if arm == "rdrop" and lg:
            kls = [x["kl_last"] for x in lg.values() if x["kl_last"] is not None]
            extra = f", logged symKL at the last eval {np.mean(kls):.4f} nats" if kls else ""
        if arm == "sam" and lg:
            rs = [x["sam_ratio"] for x in lg.values() if x["sam_ratio"] is not None]
            extra = f", ||eps||/||w|| {min(rs):.5f}-{max(rs):.5f}" if rs else ""
        P(f"  w5_cd_{arm}_r101-104 ({r['flag']}) vs w5_cd_m02ref (CD harness, CDSEED 900+root, selected ckpt, pre-declared wave-7 rule with Holm): "
          f"val {s['all']['delta']:+.5f} +- {s['all']['se']:.5f} (z {s['all']['z']:+.1f}, Holm p {r['p_holm']:.3f}), holdout {hh['all']['delta']:+.5f} +- "
          f"{hh['all']['se']:.5f} (z {hh['all']['z']:+.1f}), pool hold {e['HOLD']['delta']:+.5f} (z {e['HOLD']['z']:+.1f}), pmix {pm['delta'] if pm else float('nan'):+.5f}, "
          f"stale " + (f"{st['all']['delta']:+.5f}" if st else "pending") + f", coldpos {s['coldpos']['delta']:+.5f} / warmpos {s['warmpos']['delta']:+.5f}, "
          f"same-root corr {np.mean(r['corr_same_root']):.4f}{extra}"
          + (f", @last val {r['last']['val']['all']['delta']:+.5f} / hold {r['last']['hold']['all']['delta']:+.5f}" if r["last"] else "")
          + f" -> {r['verdict']} (nn_x/w5/logs/w5_eval_B_wave7.txt)")
    out_f = os.environ.get("OUT") or f"{W5}/logs/w5_eval_B_wave7{'_selftest' if selftest else ''}.txt"
    open(out_f, "w").write("\n".join(R.OUT) + "\n")
    json.dump(out, open(re.sub(r"\.txt$", "", out_f) + ".json", "w"), indent=1, default=float)
    print(f"wrote {out_f}")


if __name__ == "__main__":
    main()
