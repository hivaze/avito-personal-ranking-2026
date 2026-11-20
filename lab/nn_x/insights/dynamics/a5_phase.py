"""A5: epoch-phase structure of the stable-phase eval curves (10-epoch runs, eval every 2500 = ~0.254 epoch, so each
epoch is sampled at phases ~0.05, ~0.30, ~0.55, ~0.82, drifting slowly). Each run is detrended with a quadratic in step
over 20k-77.5k; residuals are averaged by phase bin. One run per data seed (runs sharing a seed share the batch order).
Also: the boundary-crossing change (phase ~0.82 -> next ~0.05) vs within-epoch changes.
Prints -> a5_phase.txt."""
import numpy as np
import pandas as pd

from fams import load, mse

EP = 9835
C, R = load()
rows = []
seen = set()
for run in sorted(C.run.unique()):
    a = R.loc[run]
    if a.a_eval_every != 2500 or a.a_epochs != 10.0:
        continue
    c = C[(C.run == run) & (C.step >= 20000) & (C.step <= 77500)].sort_values("step").copy()
    if len(c) < 20:
        continue
    seed = int(a.a_seed)
    dup = seed in seen
    seen.add(seed)
    for col in ("val", "hold"):
        p = np.polyfit(c.step / 1e4, c[col], 2)
        c[col + "_r"] = c[col] - np.polyval(p, c.step / 1e4)
    c["phase"] = (c.step % EP) / EP
    c["epoch"] = c.step // EP
    c["run"], c["dup"], c["lr"] = run, dup, a.a_lr
    rows.append(c)
D = pd.concat(rows)
U = D[~D.dup]
print(f"runs {D.run.nunique()} (unique data seeds {U.run.nunique()})")
U = U.assign(pb=pd.cut(U.phase, [0, 0.2, 0.45, 0.7, 1.0], labels=["~0.05", "~0.30", "~0.55", "~0.82"]))
g = U.groupby("pb", observed=True)
print("== A5.1 residual (vs quadratic trend) by epoch phase, unique-seed runs: val | hold")
for k, x in g:
    pv = x.groupby("run").val_r.mean()
    ph = x.groupby("run").hold_r.mean()
    print(f"   phase {k}: val {pv.mean():+.5f}+-{pv.std(ddof=1) / np.sqrt(len(pv)):.5f}  hold {ph.mean():+.5f}+-"
          f"{ph.std(ddof=1) / np.sqrt(len(ph)):.5f}  (runs {len(pv)}, evals {len(x)})")
# per-run contrast late-epoch minus early-epoch
late = U[U.phase > 0.7].groupby("run")[["val_r", "hold_r"]].mean()
early = U[U.phase < 0.2].groupby("run")[["val_r", "hold_r"]].mean()
d = (late - early).dropna()
for col in ("val_r", "hold_r"):
    m, s, n = mse(d[col])
    print(f"   per-run contrast phase~0.82 minus phase~0.05, {col[:-2]}: {m:+.5f} +- {s:.5f} (z {m / s:+.1f}, n {n}); "
          f"runs with positive contrast {int((d[col] > 0).sum())}/{n}")
for lr, x in U.groupby("lr"):
    l2 = x[x.phase > 0.7].groupby("run")[["val_r", "hold_r"]].mean() - x[x.phase < 0.2].groupby("run")[["val_r", "hold_r"]].mean()
    print(f"   lr {lr:.0e}: contrast val {l2.val_r.mean():+.5f} hold {l2.hold_r.mean():+.5f} (n {len(l2)})")

print("\n== A5.2 successive-eval changes: crossing an epoch boundary (phase ~0.82 -> ~0.05) vs within an epoch (raw, "
      "not detrended; the trend adds ~+0.00004 per eval)")
ch = []
for run, c in D[~D.dup].groupby("run"):
    c = c.sort_values("step")
    v, h, e = c.val.values, c.hold.values, c.epoch.values
    for i in range(1, len(c)):
        ch.append(dict(run=run, cross=e[i] != e[i - 1], dv=v[i] - v[i - 1], dh=h[i] - h[i - 1]))
ch = pd.DataFrame(ch)
for cross, x in ch.groupby("cross"):
    pv, ph = x.groupby("run").dv.mean(), x.groupby("run").dh.mean()
    print(f"   {'boundary' if cross else 'within  '}: val {pv.mean():+.5f}+-{pv.std(ddof=1) / np.sqrt(len(pv)):.5f}  hold "
          f"{ph.mean():+.5f}+-{ph.std(ddof=1) / np.sqrt(len(ph)):.5f}  (runs {len(pv)}, steps {len(x)})")

print("\n== A5.3 consequences for snapshots / selection")
for s, name in [(60000, "@60000 (cf4 snapshot)"), (70000, "@70000 (F7x70 snapshot)"), (77500, "@77500 (F7x snapshot)"),
                (87500, "87500 (most-selected step of m02 scratch runs)")]:
    print(f"   {name}: epoch phase {(s % EP) / EP:.2f}")
