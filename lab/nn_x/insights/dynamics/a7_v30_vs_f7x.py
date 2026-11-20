"""A7 (re-runnable as v30 seeds finish): from-scratch m02 (v30_f7_m02 = F7 recipe + cooldown mix 0.02) vs the m02
RE-COOLS of the F7 roots (F7x) at matched cooldown position k. Both are the same recipe; F7x differs only by the
re-cool procedure (fresh optimizer at @77500, new data order, 1000-step eval grid). Cooldown gain is measured against
each run's own trunk plateau mean P (F7x: the F7 original's P). Unpaired (different trunks): SE from seed spread.
Usage: python a7_v30_vs_f7x.py  (re-run parse_logs.py first)."""
import numpy as np
import pandas as pd
from fams import FAM10, RECOOL, T_COOL, curve, load, mse, orig_of, plateau

C, R = load()
KO = [1320, 3820, 6320, 8820, 11320, 13820, 16320, 18820, 19670]
rows = []
for run in FAM10["F7 lr2 m02 scratch (v30)"]:
    if run not in R.index:
        continue
    c = curve(C, run)
    if c.step.max() < 98350:
        continue
    Pv, Ph, _ = plateau(c)
    cc = c[c.step > T_COOL]
    rows.append(dict(src="v30 scratch", run=run, **{f"v{k}": v - Pv for k, v in zip(KO, cc.val)},
                     **{f"h{k}": h - Ph for k, h in zip(KO, cc.hold)},
                     sel_v=c.val.values[c.hold.values.argmax()] - Pv, sel_h=c.hold.values.max() - Ph,
                     last_v=c.val.values[-1] - Pv, last_h=c.hold.values[-1] - Ph, sel_k=int(c.step.values[c.hold.values.argmax()]) - T_COOL))
for run in RECOOL["m02 (cs811) =F7x"][0]:
    c, o = curve(C, run), curve(C, orig_of(run))
    Pv, Ph, _ = plateau(o)
    k = c.step.values.astype(float)
    rows.append(dict(src="F7x re-cool", run=run, **{f"v{kk}": np.interp(kk, k, c.val.values) - Pv for kk in KO},
                     **{f"h{kk}": np.interp(kk, k, c.hold.values) - Ph for kk in KO},
                     sel_v=c.val.values[c.hold.values.argmax()] - Pv, sel_h=c.hold.values.max() - Ph,
                     last_v=c.val.values[-1] - Pv, last_h=c.hold.values[-1] - Ph, sel_k=int(c.step.values[c.hold.values.argmax()])))
D = pd.DataFrame(rows)
a, b = D[D.src == "v30 scratch"], D[D.src == "F7x re-cool"]
print(f"v30 complete seeds: {len(a)} ({', '.join(a.run)}); F7x: {len(b)}")
if len(a) == 0:
    raise SystemExit
def diff(col):
    d = b[col].mean() - a[col].mean()
    se = np.sqrt(b[col].var(ddof=1) / len(b) + (a[col].var(ddof=1) / len(a) if len(a) > 1 else np.nan))
    return d, se
print("gain over own plateau P, val | hold, by cooldown position k  (v30 scratch  vs  F7x re-cool  ->  F7x - v30 gap(v-h))")
for k in KO:
    gv = (b[f"v{k}"] - b[f"h{k}"]).mean() - (a[f"v{k}"] - a[f"h{k}"]).mean()
    print(f"  k {k:5d}: v30 {a[f'v{k}'].mean():+.5f}|{a[f'h{k}'].mean():+.5f}   F7x {b[f'v{k}'].mean():+.5f}|{b[f'h{k}'].mean():+.5f}   gap diff {gv:+.5f}")
late = lambda d, p: d[[f"{p}{k}" for k in KO[-3:]]].mean(axis=1)
gl = (late(b, "v") - late(b, "h")).mean() - (late(a, "v") - late(a, "h")).mean()
se = np.sqrt((late(b, "v") - late(b, "h")).var(ddof=1) / len(b) + ((late(a, "v") - late(a, "h")).var(ddof=1) / len(a) if len(a) > 1 else np.nan))
print(f"late-cooldown (k>=16320) val-minus-hold gap, F7x minus v30: {gl:+.5f} +- {se:.5f}  "
      "(>= +0.0006 => re-cool-specific excess; <= +0.0002 => shared late-cooldown effect of m02)")
for col in ("sel_v", "sel_h", "last_v", "last_h"):
    d, s = diff(col)
    print(f"  {col}: v30 {a[col].mean():+.5f}  F7x {b[col].mean():+.5f}  F7x - v30 {d:+.5f} +- {s:.5f}")
print(f"  selected k: v30 {a.sel_k.mean():.0f}  F7x {b.sel_k.mean():.0f}")
