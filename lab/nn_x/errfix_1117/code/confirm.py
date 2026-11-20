"""Lead only: run submitted rule files once on the sealed samples and apply the pre-declared pass rule.

usage: confirm.py OUT.json RULE.py [RULE.py ...]

Pass rule (written in the ledger before this script was first run):
  CONFIRM (validation, the other half of the users): delta >= +0.0002 and one-sided z at or above the Holm
  threshold for the family of submitted rules at alpha 0.05.
  HOLDOUT (30,000 random-order train feeds, W8P20 train-only score): reported; a rule fails if its holdout z
  is <= -2.5, unless it reads only popularity columns (it_hist_*, seq_cv_*, xp_imp_c50), where the holdout's
  fresh inputs make it uninformative.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kit  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SAMPLES = {"confirm": HERE / "private/confirm.parquet", "holdout": HERE / "private/holdout.parquet"}


def ties_created(df: pl.DataFrame, new: np.ndarray) -> int:
    off, sz = kit.feed_layout(df)
    base = df["score"].to_numpy()
    n = 0
    for o, m in zip(off, sz):
        if len(np.unique(new[o:o + m])) < len(np.unique(base[o:o + m])):
            n += 1
    return n


def main():
    out_path, rules = sys.argv[1], sys.argv[2:]
    data = {k: pl.read_parquet(p) for k, p in SAMPLES.items()}
    res = {}
    for path in rules:
        name = f"{Path(path).parent.parent.name}/{Path(path).stem}"
        res[name] = {"path": path}
        try:
            adjust = kit.load_rule(path)
        except Exception:
            res[name]["error"] = traceback.format_exc()[-800:]
            continue
        for k, df in data.items():
            t = time.time()
            try:
                new = np.asarray(adjust(df.drop("label")), dtype=np.float64)
                assert new.shape == (df.height,), f"shape {new.shape}"
                if not np.isfinite(new).all():
                    res[name][k] = {"error": f"{(~np.isfinite(new)).sum()} non-finite scores"}
                    continue
                r = kit.evaluate(df, new)
                r["ties_created_feeds"] = ties_created(df, new)
                r["seconds"] = round(time.time() - t, 1)
                res[name][k] = r
            except Exception:
                res[name][k] = {"error": traceback.format_exc()[-800:]}
        print(name, {k: (round(v["delta"], 5), round(v["z"], 2)) if isinstance(v, dict) and "delta" in v else v
                     for k, v in res[name].items() if k in SAMPLES}, flush=True)
    # Holm over the rules that ran on CONFIRM
    ok = [(n, r["confirm"]["z"]) for n, r in res.items() if isinstance(r.get("confirm"), dict) and "z" in r["confirm"]]
    K = len(ok)
    order = sorted(ok, key=lambda x: -x[1])
    still = True
    for i, (n, z) in enumerate(order):
        p = 1 - norm.cdf(z)
        thr_p = 0.05 / (K - i)
        z_thr = norm.ppf(1 - thr_p)
        holm = still and p <= thr_p
        still = holm
        c, h = res[n]["confirm"], res[n].get("holdout", {})
        pop_only = False  # set by the lead from the analyst's declaration when reading
        hold_fail = isinstance(h, dict) and h.get("z", 0) <= -2.5
        res[n]["holm_z_threshold"] = z_thr
        res[n]["pass_confirm"] = bool(holm and c["delta"] >= 0.0002)
        res[n]["holdout_veto"] = bool(hold_fail)
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\n{'rule':55s} {'confirm delta':>14s} {'z':>6s} {'Holm z':>7s} {'pass':>5s} | {'holdout delta':>14s} {'z':>6s}")
    for n, r in sorted(res.items(), key=lambda kv: -(kv[1].get("confirm", {}) or {}).get("z", -99)):
        c, h = r.get("confirm", {}), r.get("holdout", {})
        if "delta" not in c:
            print(f"{n:55s} ERROR {str(c.get('error', r.get('error')))[-120:]}")
            continue
        hs = f"{h['delta']:+14.5f} {h['z']:6.2f}" if "delta" in h else f"{'error':>14s}"
        print(f"{n:55s} {c['delta']:+14.5f} {c['z']:6.2f} {r.get('holm_z_threshold', 0):7.2f} "
              f"{'YES' if r.get('pass_confirm') else 'no':>5s} | {hs}{'  HOLDOUT VETO?' if r.get('holdout_veto') else ''}")


if __name__ == "__main__":
    main()
