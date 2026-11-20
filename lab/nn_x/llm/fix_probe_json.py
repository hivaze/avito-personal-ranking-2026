"""One-off correction (2026-11-12 04:20 UTC) of size-probe JSONs written before the n_evals fix in train_llm.size_probe:
the estimate used the probe's own --eval_every 40 (38 evals) instead of the full arm's 200 (8 evals, as pre-declared in
reports/llm_large.md). Recomputes est_hours / gate with n_evals = ceil(full_steps / 200); keeps the raw measurements."""
import json, math, sys
from pathlib import Path
for d in sys.argv[1:]:
    f = Path(d) / "probe.json"
    r = json.loads(f.read_text())
    if r.get("n_evals_fixed"):
        print(f"{f}: already fixed"); continue
    n = math.ceil(r["full_steps"] / 200)
    e = r["est_hours"]
    old_total, old_n = e["total"], r["n_evals"]
    e["evals"] = n * r["eval_s"] / 3600
    e["total"] = e["train"] + e["evals"] + e["fp32_final"] + e["setup"]
    r["n_evals"] = n
    g = r["gate"]
    mem = max(r["peak_reserved_train_gb"], r["peak_reserved_fp32_gb"])
    g["time_ok"] = e["total"] <= g["hours"]
    g["mem_ok"] = mem <= g["mem_gb"]
    g["PASS"] = bool(g["time_ok"] and g["mem_ok"])
    r["n_evals_fixed"] = dict(old_n_evals=old_n, old_total_h=old_total, note=__doc__.strip().splitlines()[0])
    f.write_text(json.dumps(r, indent=2))
    print(f"{f}: n_evals {old_n} -> {n}; est total {old_total:.2f} -> {e['total']:.2f} h; PASS {g['PASS']}")
