"""ENCODER-DOSE (A) read-out: the pre2noid arm against BOTH abl_base and abl_xdrop_pre2, and the pre-declared KILL ARITHMETIC
(ledger nn_x/abl/ledger/ledger.md section 10.A, written 2026-11-14T00:03:07Z before any run).

A wrapper of nn_x/abl/abl_eval.py (imported read-only; its frozen rules and its file are untouched; the arm is registered in THIS
process's copy of ARMS only). Three standard read_arm blocks (LAST checkpoints primary, per-feed paired deltas on the same seed
pooled over seeds 23 / 24, SE_cal = sqrt(SE_feed^2 + sigma_run^2 / 2), slices, selected read, pmix, levers, CLIPLOG, minutes):
    xdrop_pre2 vs base   (ABLATION; x - b; recomputes the channel's worth W = b - x; must equal the frozen +0.00128 val)
    pre2noid   vs base   (n - b; the LOSS of the swap is L = b - n)
    pre2noid   vs xdrop_pre2 (n - x; "closer to base than to xdrop_pre2")
then the pre-declared rule, verbatim from ledger 10.A:
    W = mean(b - x); L = mean(b - n); f = L / W.
    KILL encoder scaling iff f < 0.40, i.e. L < 0.40 x W = 0.40 x 0.00128 = +0.000512 val  (equivalently n - b > -0.000512, or
    n - x > 0.60 x W = +0.000768). A swap that GAINS (L <= 0) kills.
    NOT KILLED iff L >= 0.40 x W (the read is ~1 SE at the bar: it can close, not confirm).
    Decision on the point estimate; reported: SE_cal(L), z = (L - 0.40 W) / SE_cal(L), per-seed L, the holdout analogue
    (W_hold frozen +0.00054, bar +0.000216; information), the selected-checkpoint read, slices, pmix, levers, CLIPLOG, minutes,
    the standard SCALE-UP / ABLATION verdict texts (information only). Stale-val is not available for pre2noid (no counterfactual
    files for the pre2n_* names) and is skipped.
CPU only, read-only on every prediction / log file.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/encdose/abl_eval_encdose.py
        -> nn_x/abl/logs/abl_eval_encdose.{txt,json}   (OUT=<path> overrides the .txt; the .json sits next to it)
"""
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
import numpy as np  # noqa: E402

ROOT = "<repo>"
ABL = f"{ROOT}/nn_x/abl"
sys.path.insert(0, ABL)
import abl_eval as E  # noqa: E402  (read-only import; registers the r6 / w5 helpers and DIRS['abl'])

R, W, P = E.R, E.W, E.P
W_FROZEN_VAL, W_FROZEN_HOLD, FRAC = 0.00128, 0.00054, 0.40
WHAT = ("ID-free encoder columns pre2n_* in place of the 6 pre2 columns (--xt_exclude <6 pre2> --xt_add <6 pre2n> "
        "--xt_add_relonly pre2n_fe_s,pre2n_new_s,pre2n_hmax32; same layout as the base)")
E.ARMS["pre2noid"] = ("SCALE-UP", WHAT, False)  # this process only; abl_eval.py on disk is unchanged


def main():
    v, h = R.setup("val"), R.setup("hold")
    sctx = W.stale_ctx()
    base, A = E.CFG["base"], E.CFG["arm"]
    P("== ENCODER-DOSE (A) IN-RANKER DOSE read-out (ledger section 10.A, pre-declared 2026-11-14T00:03:07Z): abl_pre2noid_s23/24 vs "
      "abl_base_s23/24 and vs abl_xdrop_pre2_s23/24; LAST checkpoints primary; frozen section-4 machinery (abl_eval.read_arm)")
    W.runs_block([base(s) for s in E.SEEDS] + [A(a, s) for a in ("xdrop_pre2", "pre2noid") for s in E.SEEDS])
    out = {}
    kx, wx, sx = E.ARMS["xdrop_pre2"]
    out["xdrop_pre2_vs_base"] = E.read_arm("xdrop_pre2", kx, wx, sx, v, h, sctx)
    out["pre2noid_vs_base"] = E.read_arm("pre2noid", "SCALE-UP", WHAT, False, v, h, sctx)
    out["pre2noid_vs_xdrop_pre2"] = E.read_arm("pre2noid", "SCALE-UP", WHAT, False, v, h, sctx, ref="xdrop_pre2")
    P("\n== PRE-DECLARED KILL ARITHMETIC (ledger 10.A; val, LAST checkpoints, pooled over seeds 23 / 24)")
    rx, rn, rnx = out["xdrop_pre2_vs_base"], out["pre2noid_vs_base"], out["pre2noid_vs_xdrop_pre2"]
    if rx is None:
        P("  xdrop_pre2 / base runs missing: nothing to read")
        return out
    Wv, Wh = -rx["val"]["all"]["delta"], -rx["hold"]["all"]["delta"]
    P(f"  W (channel worth = b - x): val {Wv:+.5f} (SE_cal {rx['readout']['se_cal_val']:.5f}; frozen read {W_FROZEN_VAL:+.5f}, "
      f"|diff| {abs(Wv - W_FROZEN_VAL):.6f}), holdout {Wh:+.5f} (frozen {W_FROZEN_HOLD:+.5f}); per seed val "
      f"{' '.join(f'{-x:+.5f}' for x in rx['per_seed_val'])}")
    bar_v, bar_h = FRAC * W_FROZEN_VAL, FRAC * W_FROZEN_HOLD
    res = dict(W_val=Wv, W_hold=Wh, W_frozen_val=W_FROZEN_VAL, W_frozen_hold=W_FROZEN_HOLD, bar_val=bar_v, bar_hold=bar_h,
               W_matches_frozen=bool(abs(Wv - W_FROZEN_VAL) < 1e-5))
    if rn is None:
        P(f"  pre2noid runs incomplete: the kill rule cannot be evaluated yet (bar L < {bar_v:+.6f} val)")
        out["kill"] = res
        return out
    Lv, Lh = -rn["val"]["all"]["delta"], -rn["hold"]["all"]["delta"]
    se_v, se_h = rn["readout"]["se_cal_val"], rn["readout"]["se_cal_hold"]
    fv, fh = Lv / Wv, (Lh / Wh if Wh != 0 else float("nan"))
    kill = Lv < bar_v
    zv = (Lv - bar_v) / se_v
    nx_v = rnx["val"]["all"]["delta"] if rnx else float("nan")
    res.update(L_val=Lv, L_hold=Lh, se_cal_L_val=se_v, se_cal_L_hold=se_h, frac_val=fv, frac_hold=fh, z_val_vs_bar=zv,
               per_seed_L_val=[-x for x in rn["per_seed_val"]], per_seed_L_hold=[-x for x in rn["per_seed_hold"]],
               n_minus_x_val=nx_v, n_minus_x_bar=(1 - FRAC) * W_FROZEN_VAL, kill=bool(kill),
               kill_hold_info=bool(Lh < bar_h), sel_L_val=(-rn["sel"]["val"]["delta"] if rn.get("sel") else None),
               sel_L_hold=(-rn["sel"]["hold"]["delta"] if rn.get("sel") else None))
    P(f"  L (loss of the swap = b - n): val {Lv:+.5f} (SE_cal {se_v:.5f}; per seed {' '.join(f'{-x:+.5f}' for x in rn['per_seed_val'])}), "
      f"holdout {Lh:+.5f} (SE_cal {se_h:.5f}; per seed {' '.join(f'{-x:+.5f}' for x in rn['per_seed_hold'])})")
    P(f"  f = L / W: val {fv:+.3f} (bar {FRAC:.2f}), holdout {fh:+.3f} (information)")
    P(f"  bar: L < {FRAC:.2f} x W_frozen = {bar_v:+.6f} val  <=>  n - b > {-bar_v:+.6f}  <=>  n - x > {(1 - FRAC) * W_FROZEN_VAL:+.6f} "
      f"(observed n - x {nx_v:+.5f})")
    P(f"  z = (L - {FRAC:.2f} W) / SE_cal(L) = {zv:+.2f}  (the read is ~1 SE at the bar: it can close, not confirm)")
    if rn.get("sel"):
        P(f"  selected-checkpoint read (information): L val {res['sel_L_val']:+.5f}, holdout {res['sel_L_hold']:+.5f}")
    P(f"  holdout analogue (information): L_hold {Lh:+.5f} vs bar {bar_h:+.6f} -> {'would KILL' if Lh < bar_h else 'would NOT kill'}")
    verdict = ("KILL encoder scaling: the ID-free swap loses < 40% of the pre2 channel's worth (zero-shot quality does not drive ranker value)"
               if kill else "NOT KILLED: the swap loses >= 40% of the channel's worth (inconclusive at ~1 SE; escalation stays subject to (B) / (C))")
    res["verdict"] = verdict
    P(f"\n  ==> VERDICT (A): {verdict}")
    P(f"  registry: abl_pre2noid_s23/24 (ID-free pre2 columns) vs abl_base_s23/24 / abl_xdrop_pre2_s23/24, 2-epoch LAST, pre-declared ledger 10.A: "
      f"W val {Wv:+.5f} / hold {Wh:+.5f}; L val {Lv:+.5f} +- {se_v:.5f} (seeds {' '.join(f'{-x:+.5f}' for x in rn['per_seed_val'])}), "
      f"hold {Lh:+.5f} +- {se_h:.5f}; f {fv:+.2f} val / {fh:+.2f} hold; n - x {nx_v:+.5f}; z vs bar {zv:+.2f} -> {'KILL' if kill else 'NOT KILLED'}")
    out["kill"] = res
    return out


if __name__ == "__main__":
    res = main()
    out = os.environ.get("OUT") or f"{ABL}/logs/abl_eval_encdose.txt"
    open(out, "w").write("\n".join(R.OUT) + "\n")
    json.dump(res, open(re.sub(r"\.txt$", "", out) + ".json", "w"), indent=1, default=float)
    print(f"wrote {out}")
