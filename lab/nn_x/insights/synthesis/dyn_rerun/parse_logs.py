"""Parse training step logs (nn/logs, nn_x/r6/logs) into one tidy table of eval curves.

Read-only on the logs. For each log the LAST run segment (after the last '==== ... args:' header) is used, since a
restarted run starts from scratch. Output: curves.csv (one row per eval) and runs.csv (one row per run, args + BEST).
CPU only, no model code.
"""
import ast
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("<repo>")
OUT = ROOT / "nn_x/insights/synthesis/dyn_rerun"
STEP = re.compile(r"^step (\d+)/(\d+) loss ([\d.na]+) val_ndcg ([\d.na]+) holdout_ndcg ([\d.na]+) lr ([\d.e+-]+) "
                  r"([\d.]+) min(?: cd_loss_reg ([\d.]+) cd_loss_pool ([\d.]+))?")
BEST = re.compile(r"^BEST step (\d+) val_ndcg ([\d.]+)")
KEEP_ARGS = ["lr", "epochs", "warmup", "eval_every", "seed", "cooldown_frac", "cooldown_mix", "aux_anneal", "aux_pos",
             "arch", "d", "heads", "xt", "init", "drop_cd", "pool_p", "aug_cand", "hcap", "rec_q", "rho_file", "wd",
             "select_on", "keep", "loss_norm", "sched", "muon"]


def parse(path):
    txt = path.read_text(errors="replace").splitlines()
    starts = [i for i, l in enumerate(txt) if l.startswith("==== ") and "args:" in l]
    if not starts:
        return None, []
    seg = txt[starts[-1]:]
    try:
        args = ast.literal_eval(seg[0].split("args: ", 1)[1])
    except Exception:
        args = {}
    rows, best = [], (None, None)
    for l in seg[1:]:
        m = STEP.match(l)
        if m:
            rows.append(dict(step=int(m[1]), total=int(m[2]), loss=float(m[3]), val=float(m[4]), hold=float(m[5]),
                             lr=float(m[6]), minutes=float(m[7]),
                             cd_reg=float(m[8]) if m[8] else float("nan"), cd_pool=float(m[9]) if m[9] else float("nan")))
            continue
        m = BEST.match(l)
        if m:
            best = (int(m[1]), float(m[2]))
    run = dict(run=path.stem, src=path.parent.parent.name if path.parent.name == "logs" else path.parent.name,
               n_eval=len(rows), total=rows[-1]["total"] if rows else None, last_step=rows[-1]["step"] if rows else None,
               best_step=best[0], best_val=best[1], complete=best[0] is not None)
    for k in KEEP_ARGS:
        run["a_" + k] = args.get(k)
    return run, rows


def main():
    pats = [(ROOT / "nn/logs", re.compile(r"^v(1[2-9]|2\d|30)_.*\.log$")),
            (ROOT / "nn_x/r6/logs", re.compile(r"^r6_(cd_|a4u_e10|cset_e10).*\.log$"))]
    runs, curves = [], []
    for d, pat in pats:
        for p in sorted(d.iterdir()):
            if not pat.match(p.name) or "_ft" in p.name:
                continue
            run, rows = parse(p)
            if run is None or not rows:
                continue
            runs.append(run)
            for r in rows:
                r["run"] = run["run"]
                curves.append(r)
    R = pd.DataFrame(runs)
    C = pd.DataFrame(curves)
    R.to_csv(OUT / "runs.csv", index=False)
    C.to_csv(OUT / "curves.csv", index=False)
    print(len(R), "runs", len(C), "evals", file=sys.stderr)


if __name__ == "__main__":
    main()
