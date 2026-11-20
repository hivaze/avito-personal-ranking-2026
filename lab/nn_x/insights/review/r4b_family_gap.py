"""Review R4b: late-cooldown val-minus-holdout gain over own plateau (same definition as dynamics a7) for every
CONTINUOUS (from-scratch, no re-cool) 10-epoch family: mix .1 families (F7, F9, F10, F12) vs m02 families (V27-V30).
If m02-from-scratch families all show a lower gap than mix-.1-from-scratch families, the v30-vs-F7x 'excess' is
mostly an m02/continuous effect, not a re-cool effect."""
import re, numpy as np
from pathlib import Path
R = Path("<repo>")
pat = re.compile(r"step (\d+)/(\d+) .*?val_ndcg ([\d.]+) holdout_ndcg ([\d.]+)")
KO = np.array([16320, 18820, 19670])
def gap(run):
    p = R / f"nn/logs/{run}.log"
    if not p.exists(): return None
    a = np.array([(int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))) for m in map(pat.search, open(p)) if m])
    if len(a) == 0 or a[-1, 0] < a[-1, 1] or a[-1, 1] != 98350: return None
    s, v, h = a[:, 0], a[:, 2], a[:, 3]
    w = (s >= 67500) & (s <= 77500); Pv, Ph = v[w].mean(), h[w].mean()
    k = s - 78680; mk = k > 0
    lv, lh = np.interp(KO, k[mk], v[mk]).mean(), np.interp(KO, k[mk], h[mk]).mean()
    return lv - Pv, lh - Ph
FAMS = {"F7 mix.1": [f"v17_pre2s_s{i}" for i in range(101, 108)], "F9 lr1 mix.1": [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)],
        "F10 gru mix.1": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)], "F12 d192 mix.1": [f"v26_d192_pre2s_s{i}" for i in range(701, 708)],
        "V30 f7 m02": [f"v30_f7_m02_s{i}" for i in range(901, 907)], "V28 lr1 m02": [f"v28_lr1_m02_s{i}" for i in range(411, 419)],
        "V27 gru m02": [f"v27_gru_m02_s{i}" for i in range(511, 519)], "V29 d192 m02": [f"v29_d192_m02_s{i}" for i in range(711, 716)]}
res = {}
for f, runs in FAMS.items():
    g = [x for x in (gap(r) for r in runs) if x]
    if not g: print(f, "none complete"); continue
    g = np.array(g); d = g[:, 0] - g[:, 1]; res[f] = d
    print(f"{f:16s} n {len(g)}  val gain {g[:,0].mean():+.5f}  hold gain {g[:,1].mean():+.5f}  gap {d.mean():+.5f} +- {d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else np.nan:.5f}")
m1 = np.concatenate([res[k] for k in res if "mix.1" in k]); m2 = np.concatenate([res[k] for k in res if "m02" in k])
print(f"\nall mix.1 continuous: gap {m1.mean():+.5f} (n {len(m1)}) | all m02 continuous: gap {m2.mean():+.5f} (n {len(m2)}) | diff {m1.mean()-m2.mean():+.5f} +- {np.sqrt(m1.var(ddof=1)/len(m1)+m2.var(ddof=1)/len(m2)):.5f}")
for a_, b_ in [("F7 mix.1", "V30 f7 m02"), ("F9 lr1 mix.1", "V28 lr1 m02"), ("F10 gru mix.1", "V27 gru m02"), ("F12 d192 mix.1", "V29 d192 m02")]:
    if a_ in res and b_ in res: print(f"  {a_} - {b_}: {res[a_].mean()-res[b_].mean():+.5f}")
