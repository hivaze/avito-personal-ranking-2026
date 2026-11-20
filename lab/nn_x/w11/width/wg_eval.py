"""Read-out of WIDTH-GATE Step 1 (reports/SOLVER_POWER_1113.md 6.3 item 1; rule nn_x/w5/logs/B_widthgate_predeclared.txt, frozen
2026-11-14T00:08Z). Derived from nn_x/w5/w5_eval_wave8.py (arms, roots, pairing and the GO rule changed; the paired per-feed mechanics of
nn_x/r6/r6_eval.py / nn_x/w5/w5_eval.py are imported read-only). CPU only, read-only on every prediction / log file.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w11/width/wg_eval.py
        -> nn_x/w5/logs/w5_eval_B_widthgate.{txt,json} (OUT=<path> overrides)

Runs: w11_wg_d192_<arm>_r<root> (arms m02 | drop02 | mup, roots 711-715, CDSEED 900 + root) and the matched control w11_wg_d128_m02_r<root>
(roots 901-905), all 2-epoch CD re-cools of the @77500 snapshots (nn_x/w11/width/lane_wg.sh). PRIMARY = the SELECTED checkpoint
(holdout-best at step >= 12,000); @last = information. Pairing 711<->901 ... 715<->905 (independent seeds: the pairing only shapes the
per-feed SE; the pooled point estimate = difference of the 5-member means); the across-root Welch SE is reported next to it.
GO iff best d192 arm (highest pooled val delta) - matched d128 re-cool >= +0.0008 val AND >= +0.0004 holdout AND no rise in member corr
to the base (F7x = r6_cd_m02_r101-107 z-mean; arm mean corr - control mean corr <= +1 Welch SE); otherwise CLOSE ranker width.
Information: late (Sep 17-19) / early (Sep 6-8) / late20% holdout columns, val slices, pmix, stale-val, 5-member pools, same-root (b)-(a)
and (c)-(a) at d192, the 12 full-run d128 seeds, wall times, CLIPLOG, the muP effective-lr lines.
"""
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = "<repo>"
sys.path.insert(0, f"{ROOT}/nn_x/w5")
import w5_eval as E  # noqa: E402  (read-only import; registers nn_x/r6/r6_eval as E.R)

R, P = E.R, E.P
WG = f"{ROOT}/nn_x/w11/width"
R.DIRS["w11"] = (f"{WG}/preds", f"{WG}/logs")
R.DIRS["arch"] = (f"{ROOT}/archive/preds", f"{ROOT}/nn/logs")  # the v30 full-run preds moved to archive/preds (11-13 23:16 cleanup)
ARMS = ("m02", "drop02", "mup")
FLAGS = {"m02": "(a) control, no arm flag", "drop02": "(b) --drop_cd 0.2", "mup": "(c) --mup_base 128 (Muon lr x 0.8165 on the 39 width-growing matrices)"}
D192 = (711, 712, 713, 714, 715)
D128 = (901, 902, 903, 904, 905)
BASE = [f"r6:r6_cd_m02_r{r}" for r in range(101, 108)]  # the frozen pick-1 TO branch F7x (selected checkpoints)
FULL = {"v30 d128 (6, full 10-ep runs)": [f"arch:v30_f7_m02_s{s}" for s in range(901, 907)],
        "w5_e10_base d128 (6, full 10-ep runs)": [f"w5:w5_e10_base_s{s}" for s in range(951, 957)]}
GO_VAL, GO_HOLD = 0.0008, 0.0004


def ctrl(r):
    return f"w11:w11_wg_d128_m02_r{r}"


def arm(a, r):
    return f"w11:w11_wg_d192_{a}_r{r}"


def L(n, rd):
    return n + "@last" if rd == "last" else n


def add_day_slices(h):
    """late = Sep 17-19 MSK (8,279 holdout feeds), early = Sep 6-8 (7,823), from the train timestamps of the holdout feeds."""
    tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
    pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    trM = tr["M"].to_numpy()
    pr[trM < 50] = 0.0
    cand = np.where((pr > 0.9) & (trM >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    day = (tr["timestamp"].to_numpy()[hold] + 3 * 3600) // 86400
    d0 = (tr["timestamp"].to_numpy() + 3 * 3600).min() // 86400  # Sep 6 MSK
    h["slices"]["early(Sep6-8)"] = day <= d0 + 2
    h["slices"]["late(Sep17-19)"] = day >= d0 + 11
    assert h["slices"]["late(Sep17-19)"].sum() == 8279 and h["slices"]["early(Sep6-8)"].sum() == 7823, (h["slices"]["late(Sep17-19)"].sum(), h["slices"]["early(Sep6-8)"].sum())


def base_z(v):
    return np.mean([R.load_z(v, n) for n in BASE], 0)


def corr_to_base(v, bz, n):
    return float(np.corrcoef(R.load_z(v, n), bz)[0, 1])


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    se = float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))) if len(x) > 1 and len(y) > 1 else float("nan")
    return float(x.mean() - y.mean()), se


def have(ctxs, names):
    return all(os.path.exists(c["path"](n)) for c in ctxs for n in names)


def read_vs_control(v, h, sctx, bz, a, ctrl_corr):
    pairs = {rd: [(L(ctrl(rc), rd), L(arm(a, ra), rd)) for ra, rc in zip(D192, D128)] for rd in ("sel", "last")}
    miss = [n for p in pairs["sel"] for n in p if not (os.path.exists(v["path"](n)) and os.path.exists(h["path"](n)))]
    P(f"\n== d192 {a} {FLAGS[a]} vs the matched d128 m02 re-cool (B = d192 arm; pairs 711<->901 ... 715<->905); PRIMARY = selected checkpoints")
    if miss:
        P(f"  INCOMPLETE: missing selected val/holdout preds: {' '.join(R.split_name(n)[1] for n in miss)}")
        present = [(c, b) for c, b in pairs["sel"] if os.path.exists(v["path"](b)) and os.path.exists(h["path"](b))]
        if len(present) < 2:
            return dict(verdict="INCOMPLETE", missing=[R.split_name(n)[1] for n in miss])
        pairs["sel"] = present
        P(f"  (read on the {len(present)} roots present; marked INCOMPLETE)")
    rv, rh = E.read_pairs(v, h, pairs["sel"], "selected (PRIMARY)")
    sv, sh = rv["pooled"], rh["pooled"]
    res = dict(flag=FLAGS[a], val=sv, hold=sh, per_root_val=[p["slices"]["all"]["delta"] for p in rv["pairs"]],
               per_root_hold=[p["slices"]["all"]["delta"] for p in rh["pairs"]],
               member_val=[p["B_mean"] for p in rv["pairs"]], member_hold=[p["B_mean"] for p in rh["pairs"]],
               ctrl_val=[p["A_mean"] for p in rv["pairs"]], ctrl_hold=[p["A_mean"] for p in rh["pairs"]],
               incomplete=bool(miss))
    wv, wh = welch(res["member_val"], res["ctrl_val"]), welch(res["member_hold"], res["ctrl_hold"])
    res["welch"] = dict(val=dict(delta=wv[0], se=wv[1]), hold=dict(delta=wh[0], se=wh[1]))
    P(f"  across-root Welch (5 vs 5 member means): val {wv[0]:+.5f} +- {wv[1]:.5f}, holdout {wh[0]:+.5f} +- {wh[1]:.5f}")
    P(f"  holdout columns (selected): late(Sep17-19) {sh['late(Sep17-19)']['delta']:+.5f}+-{sh['late(Sep17-19)']['se']:.5f}  early(Sep6-8) "
      f"{sh['early(Sep6-8)']['delta']:+.5f}+-{sh['early(Sep6-8)']['se']:.5f}  late20% {sh['late20%']['delta']:+.5f}")
    P(f"  val slices (selected): M<50 {sv['M<50']['delta']:+.5f} (z {sv['M<50']['z']:+.2f})  M>=50 {sv['M>=50']['delta']:+.5f} (z {sv['M>=50']['z']:+.2f})  "
      f"Tue {sv['Tue']['delta']:+.5f}  Thu {sv['Thu']['delta']:+.5f}  coldpos {sv['coldpos']['delta']:+.5f} (z {sv['coldpos']['z']:+.2f})  "
      f"warmpos {sv['warmpos']['delta']:+.5f} (z {sv['warmpos']['z']:+.2f})")
    if have((v, h), [n for p in pairs["last"] for n in p]):
        lv, lh = E.read_pairs(v, h, pairs["last"], "last (info)")
        res["last"] = dict(val=lv["pooled"], hold=lh["pooled"])
    else:
        P("  [last (info)]: @last preds missing for at least one run")
        res["last"] = None
    res["ens"] = E.ens_block(v, h, [c for c, _ in pairs["sel"]], [b for _, b in pairs["sel"]])
    res["corr_base"] = [corr_to_base(v, bz, b) for _, b in pairs["sel"]]
    cr = welch(res["corr_base"], ctrl_corr)
    res["corr_rise"] = dict(delta=cr[0], se=cr[1], no_rise=bool(cr[0] <= cr[1]))
    P(f"  member corr to the base (F7x r6_cd_m02_r101-107 z-mean, val, selected): arm " + " ".join(f"{x:.4f}" for x in res["corr_base"])
      + f" (mean {np.mean(res['corr_base']):.4f}) vs d128 control mean {np.mean(ctrl_corr):.4f}: rise {cr[0]:+.4f} +- {cr[1]:.4f} -> "
      + ("no rise (<= +1 SE)" if res["corr_rise"]["no_rise"] else "RISE (> +1 SE)"))
    res["pmix"] = E.pmix_block(v, pairs["sel"])
    res["stale"] = E.stale_block(sctx, pairs["sel"])
    res["levers"] = E.lever_block(v, pairs["sel"])
    res["cliplog"] = E.cliplog_block([pairs["sel"][0][0], pairs["sel"][0][1]])
    res["gates"] = {f"val >= {GO_VAL:+.4f}": sv["all"]["delta"] >= GO_VAL, f"holdout >= {GO_HOLD:+.4f}": sh["all"]["delta"] >= GO_HOLD,
                    "no rise in corr to the base": res["corr_rise"]["no_rise"]}
    vetoes = {"pmix >= -1 SE": res["pmix"] is None or not res["pmix"]["below_1se"],
              "stale-val >= -1 SE": res["stale"] is None or res["stale"]["all"]["delta"] >= -res["stale"]["all"]["se"],
              "no val slice (M<50, M>=50, Tue, Thu) < -2 SE": all(sv[k]["z"] >= -2 for k in ("M<50", "M>=50", "Tue", "Thu"))}
    res["vetoes"] = vetoes
    P("  GO clauses: " + "; ".join(f"{k} {'OK' if x else 'NO'}" for k, x in res["gates"].items()) + " | standard vetoes (info): "
      + "; ".join(f"{k} {'OK' if x else 'NO'}" for k, x in vetoes.items()) + ("; stale-val PENDING" if res["stale"] is None else ""))
    return res


def same_root_block(v, h, a):
    """(b) - (a) and (c) - (a) at d192 on the same root (information)."""
    pairs = [(arm("m02", r), arm(a, r)) for r in D192]
    if not have((v, h), [n for p in pairs for n in p]):
        P(f"\n== d192 {a} - d192 m02 (same root): incomplete")
        return None
    P(f"\n== d192 {a} - d192 m02 control, same root (information; selected checkpoints)")
    rv, rh = E.read_pairs(v, h, pairs, "selected")
    corr = [R.corr(v, x, y) for x, y in pairs]
    P("  same-root corr: " + " ".join(f"{c:.4f}" for c in corr) + f" (mean {np.mean(corr):.4f})")
    return dict(val=rv["pooled"]["all"], hold=rh["pooled"]["all"], late=rh["pooled"]["late(Sep17-19)"], early=rh["pooled"]["early(Sep6-8)"], corr=corr)


def full_runs_block(v, h, arm_means):
    P("\n== information: the 12 full-run d128 seeds (selected checkpoints, member means) vs the d192 re-cool arms and the d128 re-cools")
    out = {}
    for name, names in FULL.items():
        ok = [n for n in names if have((v, h), [n])]
        if len(ok) < 2:
            P(f"  {name}: preds missing ({len(ok)} of {len(names)} present)")
            continue
        mv = [float(R.per_feed(v, n).mean()) for n in ok]
        mh = [float(R.per_feed(h, n).mean()) for n in ok]
        out[name] = dict(n=len(ok), val=mv, hold=mh)
        P(f"  {name}: val {np.mean(mv):.5f} +- {np.std(mv, ddof=1) / np.sqrt(len(mv)):.5f}, holdout {np.mean(mh):.5f} +- {np.std(mh, ddof=1) / np.sqrt(len(mh)):.5f} (n {len(ok)})")
    allv = [x for o in out.values() for x in o["val"]]
    allh = [x for o in out.values() for x in o["hold"]]
    if allv:
        P(f"  all full-run d128 (n {len(allv)}): val {np.mean(allv):.5f} +- {np.std(allv, ddof=1) / np.sqrt(len(allv)):.5f}, holdout {np.mean(allh):.5f} +- {np.std(allh, ddof=1) / np.sqrt(len(allh)):.5f}")
        for k, (mv, mh) in arm_means.items():
            wv, wh = welch(mv, allv), welch(mh, allh)
            P(f"    {k:16s} - full-run d128: val {wv[0]:+.5f} +- {wv[1]:.5f}, holdout {wh[0]:+.5f} +- {wh[1]:.5f}")
            out[f"{k} - full d128"] = dict(val=wv, hold=wh)
    return out


def mup_lines():
    out = {}
    for r in D192:
        f = f"{WG}/logs/w11_wg_d192_mup_r{r}.log"
        if not os.path.exists(f):
            continue
        txt = open(f).read()
        s = re.findall(r"wave-11 muP step (\d+): effective Muon lr hidden \S+ ([\d.e+-]+) \(mult ([\d.]+), decay factor [\d.]+\), input \S+ ([\d.e+-]+)", txt)
        setup = [ln for ln in txt.splitlines() if ln.startswith("wave-11 WIDTH-GATE muP")]
        out[r] = dict(n_summary=len(s), first=s[0] if s else None, last=s[-1] if s else None, setup=setup[0][:160] if setup else None)
    if out:
        r0 = min(out)
        P(f"\n== muP effective-lr lines (root {r0}): {out[r0]['setup']}")
        P(f"  first summary (step, hidden lr, mult, input lr): {out[r0]['first']}; last: {out[r0]['last']}; {out[r0]['n_summary']} summaries")
    return out


def main():
    v, h = R.setup("val"), R.setup("hold")
    add_day_slices(h)
    sctx = E.stale_ctx()
    P(f"== WIDTH-GATE STEP 1 READ-OUT (rule nn_x/w5/logs/B_widthgate_predeclared.txt, 2026-11-14T00:08Z): d192 re-cools (v29 roots 711-715) x arms "
      f"m02 / drop02 / mup vs the matched d128 m02 re-cools (v30 roots 901-905), CD harness, CDSEED 900 + root, SELECTED checkpoint primary, @last info. "
      f"GO iff best d192 arm - d128 re-cool >= {GO_VAL:+.4f} val AND >= {GO_HOLD:+.4f} holdout AND no rise in member corr to the base; otherwise CLOSE ranker width.")
    names = [ctrl(r) for r in D128] + [arm(a, r) for a in ARMS for r in D192]
    runs = E.runs_block(names)
    out = {"runs": runs, "arms": {}, "same_root": {}}
    bz = base_z(v)
    ctrl_ok = [ctrl(r) for r in D128 if have((v, h), [ctrl(r)])]
    ctrl_corr = [corr_to_base(v, bz, n) for n in ctrl_ok]
    P(f"\n== d128 m02 re-cools: member corr to the base (F7x z-mean, val): " + " ".join(f"{x:.4f}" for x in ctrl_corr) + (f" (mean {np.mean(ctrl_corr):.4f})" if ctrl_corr else " (none)"))
    P("   (same-recipe different-seed members of the base itself, r6_cd_m02 r101 vs the 7-mean: %.4f)" % corr_to_base(v, bz, BASE[0]))
    for a in ARMS:
        out["arms"][a] = read_vs_control(v, h, sctx, bz, a, ctrl_corr)
    for a in ("drop02", "mup"):
        out["same_root"][a] = same_root_block(v, h, a)
    arm_means = {f"d192 {a}": (r["member_val"], r["member_hold"]) for a, r in out["arms"].items() if r.get("verdict") != "INCOMPLETE"}
    if ctrl_ok:
        arm_means["d128 re-cool"] = ([float(R.per_feed(v, n).mean()) for n in ctrl_ok], [float(R.per_feed(h, n).mean()) for n in ctrl_ok])
    out["full_runs"] = full_runs_block(v, h, arm_means)
    out["mup_lines"] = mup_lines()
    # verdict
    readable = {a: r for a, r in out["arms"].items() if r.get("verdict") != "INCOMPLETE"}
    P("\n== summary (SELECTED checkpoints, pooled over the 5 root pairs vs the matched d128 re-cool)")
    for a, r in out["arms"].items():
        if r.get("verdict") == "INCOMPLETE":
            P(f"  {a:6s} INCOMPLETE (missing {' '.join(r['missing'])})")
            continue
        s, hh, e, pm, st = r["val"], r["hold"], r["ens"], r["pmix"], r["stale"]
        P(f"  {a:6s} val {s['all']['delta']:+.5f}+-{s['all']['se']:.5f} (Welch {r['welch']['val']['se']:.5f}; z_feed {s['all']['z']:+.2f}) hold {hh['all']['delta']:+.5f}+-{hh['all']['se']:.5f} "
          f"(Welch {r['welch']['hold']['se']:.5f}) | late {hh['late(Sep17-19)']['delta']:+.5f} early {hh['early(Sep6-8)']['delta']:+.5f} | M<50 {s['M<50']['delta']:+.5f} M>=50 {s['M>=50']['delta']:+.5f} "
          f"coldpos {s['coldpos']['delta']:+.5f} warmpos {s['warmpos']['delta']:+.5f} | pmix {pm['delta'] if pm else float('nan'):+.5f} | pool val {e['VAL']['delta']:+.5f} hold {e['HOLD']['delta']:+.5f} | "
          f"stale " + (f"{st['all']['delta']:+.5f}" if st else "pending") + f" | corr-to-base {np.mean(r['corr_base']):.4f} (rise {r['corr_rise']['delta']:+.4f}+-{r['corr_rise']['se']:.4f})"
          + (f" | @last val {r['last']['val']['all']['delta']:+.5f} hold {r['last']['hold']['all']['delta']:+.5f}" if r["last"] else "")
          + (" | INCOMPLETE roots" if r["incomplete"] else ""))
    if readable:
        best = max(readable, key=lambda a: readable[a]["val"]["all"]["delta"])
        rb = readable[best]
        go = all(rb["gates"].values())
        bad_veto = [k for k, x in rb["vetoes"].items() if not x]
        verdict = ("GO" + (f" with veto {bad_veto}" if bad_veto else "") if go else "CLOSE (ranker width closed for good, incl. BIGPOOL)")
        if any(r["incomplete"] for r in readable.values()):
            verdict += " [INCOMPLETE: some roots missing]"
        out["best_arm"], out["verdict"] = best, verdict
        P(f"\n== VERDICT: best d192 arm by pooled val = {best} ({FLAGS[best]}): val {rb['val']['all']['delta']:+.5f} (need >= {GO_VAL:+.4f}), holdout "
          f"{rb['hold']['all']['delta']:+.5f} (need >= {GO_HOLD:+.4f}), corr rise {rb['corr_rise']['delta']:+.4f} +- {rb['corr_rise']['se']:.4f} (need <= +1 SE) -> {verdict}")
        P("\n== registry-ready one-liners")
        for a, r in readable.items():
            s, hh, e, st = r["val"], r["hold"], r["ens"], r["stale"]
            P(f"  w11_wg_d192_{a}_r711-715 ({FLAGS[a]}) vs w11_wg_d128_m02_r901-905 (CD harness, CDSEED 900+root, selected ckpt, WIDTH-GATE rule): val "
              f"{s['all']['delta']:+.5f} +- {s['all']['se']:.5f} (Welch {r['welch']['val']['se']:.5f}), holdout {hh['all']['delta']:+.5f} +- {hh['all']['se']:.5f} (Welch {r['welch']['hold']['se']:.5f}), "
              f"late {hh['late(Sep17-19)']['delta']:+.5f} / early {hh['early(Sep6-8)']['delta']:+.5f}, pool val {e['VAL']['delta']:+.5f} / hold {e['HOLD']['delta']:+.5f}, stale "
              + (f"{st['all']['delta']:+.5f}" if st else "pending") + f", corr-to-base {np.mean(r['corr_base']):.4f} vs control {np.mean(ctrl_corr):.4f}"
              + (f", @last val {r['last']['val']['all']['delta']:+.5f} / hold {r['last']['hold']['all']['delta']:+.5f}" if r["last"] else "")
              + f" -> {'BEST, ' + verdict if a == best else 'not best'} (nn_x/w5/logs/w5_eval_B_widthgate.txt)")
    out_f = os.environ.get("OUT") or f"{ROOT}/nn_x/w5/logs/w5_eval_B_widthgate.txt"
    open(out_f, "w").write("\n".join(R.OUT) + "\n")
    json.dump(out, open(re.sub(r"\.txt$", "", out_f) + ".json", "w"), indent=1, default=float)
    print(f"wrote {out_f}")


if __name__ == "__main__":
    main()
