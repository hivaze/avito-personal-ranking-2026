"""TIME arm read-out (ledger nn_x/abl/ledger/ledger.md section 9; CURRENT_STATUS Log 19:08 "TIME BLIND SPOT"): a COPY-style
wrapper of nn_x/abl/abl_eval.py (imported read-only, its frozen rules untouched) that
  (1) adds the HOLDOUT slices 'weekend' (the feed's Moscow weekday is Sat / Sun; 9,632 of the 30K in-period holdout feeds) and
      'weekday' (20,368) to the r6 holdout context (R.setup('hold') is wrapped; the val context is unchanged),
  (2) runs abl_eval.screen(['time']) (the standard blocks: runs, LAST-checkpoint pairs, SE_cal, slices, selected read, pmix,
      stale-val if present, levers, CLIPLOG, minutes; the SCALE-UP verdict printed there is INFORMATION for this arm), and
  (3) prints the PRE-DECLARED PRIMARY verdict of ledger section 9 (last checkpoints, delta = abl_time_s<seed>@last -
      abl_base_s<seed>@last per feed, pooled over seeds 23 / 24):
        PASS iff  holdout WEEKEND-slice pooled delta > +1 SE_feed(weekend)      (SE_feed = sd / sqrt(n_weekend), the frozen
                                                                                 slice convention; SE_cal shown as information)
             AND  holdout overall pooled delta >= 0
             AND  val pooled delta >= -1 SE_cal(val)
             AND  pmix pooled delta >= -1 SE(pmix)        (pmix unavailable -> the rule cannot be evaluated -> no PASS)
      Information: the holdout weekday slice, per-seed weekend / weekday deltas, the selected-checkpoint weekend read, val
      slices, stale-val (if nn_x/abl/time/wait_time.sh produced it), the standard verdict.
CPU only, read-only on every prediction / log file.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/time/abl_eval_time.py
        -> nn_x/abl/logs/abl_eval_time.{txt,json}   (OUT=<path> overrides the .txt; the .json sits next to it)
"""
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only (read-out of saved predictions)"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = "<repo>"
ABL = f"{ROOT}/nn_x/abl"
sys.path.insert(0, ABL)
import abl_eval as E  # noqa: E402  (read-only import; registers the r6 / w5 helpers and DIRS['abl'])

R, W, P = E.R, E.W, E.P
MSK = 3 * 3600
N_WEEKEND = 9632  # nn_x/abl/time/out/tm_parity.json: weekend feeds of the rng-12345 30K holdout
_setup = R.setup


def holdout_weekend():
    """Weekend mask of the 30K holdout feeds, derived exactly like r6_eval.setup('hold') derives the feed set."""
    tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
    pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    trM = tr["M"].to_numpy()
    pr[trM < 50] = 0.0
    cand = np.where((pr > 0.9) & (trM >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    ts = tr["timestamp"].to_numpy().astype(np.int64)[hold]
    wd = ((ts + MSK) // 86400 + 3) % 7  # Mon = 0 (epoch day 0 was a Thursday)
    return trM[hold], wd >= 5


def setup_time(mode):
    ctx = _setup(mode)
    if mode != "val":
        M, we = holdout_weekend()
        assert np.array_equal(M, ctx["M"]) and int(we.sum()) == N_WEEKEND, (int(we.sum()), N_WEEKEND)
        ctx["slices"]["weekend"] = we
        ctx["slices"]["weekday"] = ~we
    return ctx


R.setup = setup_time  # abl_eval.screen() builds its contexts through R.setup -> the holdout gets the two extra slices


def main():
    out = E.screen(["time"])
    res = {"screen": out}
    P("\n== TIME ARM: PRE-DECLARED PRIMARY READ (ledger section 9; holdout WEEKEND slice; last checkpoints; delta = time - base)")
    if "time" not in out:
        P("  incomplete: the time arm needs abl_time_s23 / s24 and abl_base_s23 / s24 last-checkpoint val + holdout preds")
        res["verdict"] = "INCOMPLETE"
        return res
    r = out["time"]
    h, v, pm = r["hold"], r["val"]["all"], r.get("pmix")
    we, wk, ha = h["weekend"], h["weekday"], h["all"]
    # per-seed weekend / weekday deltas (information) and the selected-checkpoint weekend read (information)
    hc = R.setup("hold")
    base, A = E.CFG["base"], E.CFG["arm"]
    per_seed = {}
    for rd, tag in (("last", "@last"), ("sel", "")):
        pairs = [(base(s) + tag, A("time", s) + tag) for s in E.SEEDS]
        if not all(os.path.exists(hc["path"](n)) for p_ in pairs for n in p_):
            continue
        c = R.compare(hc, pairs)
        per_seed[rd] = dict(weekend=[p_["slices"]["weekend"]["delta"] for p_ in c["pairs"]],
                            weekday=[p_["slices"]["weekday"]["delta"] for p_ in c["pairs"]],
                            pooled_weekend=c["pooled"]["weekend"], pooled_weekday=c["pooled"]["weekday"])
        P(f"  [{rd}] weekend per seed {' '.join(f'{x:+.5f}' for x in per_seed[rd]['weekend'])} | weekday per seed "
          f"{' '.join(f'{x:+.5f}' for x in per_seed[rd]['weekday'])}")
    se_cal_we = float(np.sqrt(we["se"] ** 2 + W.SIG_RUN["hold"] ** 2 / 2))
    c1, c2, c3 = we["delta"] > we["se"], ha["delta"] >= 0, v["delta"] >= -v["se_cal"]
    c4 = (pm is not None) and (pm["delta"] >= -pm["se"])
    P(f"  holdout WEEKEND slice (n {we['n']}): {we['delta']:+.5f} +- {we['se']:.5f} (z {we['z']:+.2f}; z with SE_cal {se_cal_we:.5f}: "
      f"{we['delta'] / se_cal_we:+.2f})  -> > +1 SE_feed: {c1}")
    P(f"  holdout WEEKDAY slice (n {wk['n']}): {wk['delta']:+.5f} +- {wk['se']:.5f} (z {wk['z']:+.2f})   [information]")
    P(f"  holdout overall: {ha['delta']:+.5f} +- {ha['se']:.5f} (z_cal {ha['z_cal']:+.2f})  -> >= 0: {c2}")
    P(f"  val overall: {v['delta']:+.5f} (SE_cal {v['se_cal']:.5f}, z_cal {v['z_cal']:+.2f})  -> >= -1 SE_cal: {c3}")
    if pm is None:
        P("  pmix: unavailable -> the rule cannot be evaluated (no PASS)")
    else:
        P(f"  pmix: {pm['delta']:+.5f} +- {pm['se']:.5f} (z {pm['delta'] / pm['se']:+.2f})  -> >= -1 SE: {c4}")
    st = r.get("stale")
    if st is not None:
        P(f"  stale-val (information): {st['all']['delta']:+.5f} +- {st['all']['se']:.5f} (z {st['all']['z']:+.2f})")
    ok = bool(c1 and c2 and c3 and c4)
    verdict = ("PASS -> 10-epoch family abl_e10_time_s951-954 (E_BASE + HYG + --xt_add tm_weekend,tm_hsin,tm_hcos) + std n2cpv3 FT "
               "+ pool read Q4 = G3 + time(4) vs G3 (nn_x/abl/time/decide_time.py; ledger section 9)" if ok else
               "FAIL -> the TIME arm stops; nothing else is run")
    P(f"  ==> TIME ARM PRIMARY VERDICT: {verdict}")
    P(f"      (standard SCALE-UP verdict of abl_eval.py, information: {r['readout']['verdict']})")
    res.update(verdict=verdict, clauses=dict(weekend_gt_1se=bool(c1), holdout_ge_0=bool(c2), val_ge_m1secal=bool(c3),
                                             pmix_ge_m1se=bool(c4)),
               weekend=we, weekday=wk, hold_all=ha, val_all=v, pmix=pm, se_cal_weekend=se_cal_we, per_seed=per_seed, stale=st)
    return res


if __name__ == "__main__":
    res = main()
    out = os.environ.get("OUT") or f"{ABL}/logs/abl_eval_time.txt"
    open(out, "w").write("\n".join(R.OUT) + "\n")
    json.dump(res, open(re.sub(r"\.txt$", "", out) + ".json", "w"), indent=1, default=float)
    print(f"wrote {out}")
