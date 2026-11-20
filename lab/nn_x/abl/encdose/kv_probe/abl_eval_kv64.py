"""ENCODER-DOSE (C) read-out: arm kv64 (abl_kv64_s23 / s24 = A_BASE + HYG + --kv64 1) vs abl_base_s23 / s24 with the frozen section-4
machinery (abl_eval.read_arm imported read-only; the arm is registered in this process's copy of ARMS only) and the pre-declared
rule of ledger nn_x/abl/ledger/ledger.md section 10.C (SOLVER_POWER 6.3 (C), verbatim):
    GO only if val >= +0.0015 with holdout > 0 and stale-val / pmix vetoes >= -1 SE.
LAST checkpoints, per-feed paired deltas on the same seed pooled over seeds 23 / 24, SE_cal as in section 4. The stale-val veto needs
stale_pred.py to rebuild the model with the adapter and the states (Ranker(**cfg) fails on cfg['kv64']): until a kv64-aware
stale_pred exists the veto is reported as NOT AVAILABLE and GO cannot be declared (the time-arm precedent: an unavailable veto -> no PASS).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/encdose/kv_probe/abl_eval_kv64.py  -> nn_x/abl/logs/abl_eval_kv64.{txt,json}
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
import abl_eval as E  # noqa: E402

R, W, P = E.R, E.W, E.P
WHAT = "frozen pre2 encoder (ckpt_w1r_v3, ID-zeroed) 64-d PCA states of every history event as a zero-init gated K/V adapter in the 2 cross blocks (--kv64 1)"
E.ARMS["kv64"] = ("SCALE-UP", WHAT, True)  # stale-val read from preds/stale (kv_probe/stale_pred_kv64.py)


def main():
    v, h = R.setup("val"), R.setup("hold")
    sctx = W.stale_ctx()
    base, A = E.CFG["base"], E.CFG["arm"]
    P("== ENCODER-DOSE (C) K/V PROBE read-out (ledger section 10.C, pre-declared 2026-11-14T00:03:07Z): abl_kv64_s23/24 vs abl_base_s23/24; LAST "
      "checkpoints primary; frozen section-4 machinery. GO only if val >= +0.0015 AND holdout > 0 AND stale-val >= -1 SE AND pmix >= -1 SE")
    W.runs_block([base(s) for s in E.SEEDS] + [A("kv64", s) for s in E.SEEDS])
    r = E.read_arm("kv64", "SCALE-UP", WHAT, True, v, h, sctx)  # stale = True: preds/stale written by stale_pred_kv64.py
    out = {"kv64_vs_base": r}
    if r is None:
        P("  kv64 runs incomplete: nothing to read")
        return out
    dv, dh = r["val"]["all"]["delta"], r["hold"]["all"]["delta"]
    pm = r["pmix"]
    g = {"val >= +0.0015": bool(dv >= 0.0015), "holdout > 0": bool(dh > 0),
         "pmix >= -1 SE": (bool(pm["delta"] >= -pm["se"]) if pm else None), "stale-val >= -1 SE": None}
    st = r.get("stale")
    if st is not None:
        g["stale-val >= -1 SE"] = bool(st["all"]["delta"] >= -st["all"]["se"])
    go = all(x is True for x in g.values())
    P("\n== PRE-DECLARED RULE (ledger 10.C): " + "; ".join(f"{k}: {'OK' if x else ('NOT AVAILABLE' if x is None else 'NO')}" for k, x in g.items()))
    verdict = "GO" if go else ("NO-GO (a veto is not available: stale-val needs a kv64-aware stale_pred)" if any(x is None for x in g.values()) and
                               all(x is not False for x in g.values()) else "NO-GO")
    P(f"  ==> VERDICT (C): {verdict} (val {dv:+.5f} +- {r['readout']['se_cal_val']:.5f}, holdout {dh:+.5f} +- {r['readout']['se_cal_hold']:.5f}, "
      f"pmix {pm['delta']:+.5f} +- {pm['se']:.5f})" if pm else f"  ==> VERDICT (C): {verdict}")
    out["rule"] = dict(gates=g, go=bool(go), verdict=verdict, delta_val=dv, delta_hold=dh)
    return out


if __name__ == "__main__":
    res = main()
    out = os.environ.get("OUT") or f"{ABL}/logs/abl_eval_kv64.txt"
    open(out, "w").write("\n".join(R.OUT) + "\n")
    json.dump(res, open(re.sub(r"\.txt$", "", out) + ".json", "w"), indent=1, default=float)
    print(f"wrote {out}")
