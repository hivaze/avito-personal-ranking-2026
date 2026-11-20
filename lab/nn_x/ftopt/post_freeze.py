"""After the freeze ONLY: TUNE vs CONFIRM for every finished trial (descriptive), plus all-val numbers.

Columns per trial: pool delta vs n2cpv3 on TUNE / CONFIRM / ALL val, pool gain over the TO-only pool on each half.
Writes post_freeze.csv and prints the TUNE-CONFIRM correlation across trials.
"""
import csv
import json

import numpy as np

from ftlib import FT_DIR, SET_A, Val, load_base_ft, load_to

v = Val()
rows = {json.loads(l)["trial"]: json.loads(l) for l in open(FT_DIR / "trials.jsonl")}
to_z = v.zmean([load_to(m) for m in SET_A])
nb = v.nd(v.pool([load_base_ft(m) for m in SET_A], to_z))
nto = v.nd(to_z)
out = []
allm = np.ones(v.n, bool)
for t in sorted(rows):
    ft = [np.load(FT_DIR / "out" / f"t{t}" / f"nn_{m}_t{t}_val_oof.npy").astype(np.float64) for m in SET_A]
    nd = v.nd(v.pool(ft, to_z))
    rec = dict(trial=t)
    for nm, mk in (("tune", v.tune), ("confirm", v.confirm), ("all", allm)):
        mu, se, _, _ = v.stat(nd - nb, mk)
        rec[f"d_{nm}"], rec[f"se_{nm}"] = mu, se
        rec[f"overTO_{nm}"] = v.stat(nd - nto, mk)[0]
    for w in (0.3, 0.7):
        ndw = v.nd(v.pool(ft, to_z, w))
        rec[f"d_confirm_w{w}"] = v.stat(ndw - nb, v.confirm)[0]
        rec[f"d_all_w{w}"] = v.stat(ndw - nb, allm)[0]
    p = rows[t]["params"]
    rec.update({k: p.get(k, "") for k in ("lr", "epochs", "decay", "ema", "ema_start", "wd", "drop", "loss_norm",
                                           "preset", "extra_lr_mult", "seeds")})
    out.append(rec)
base = dict(trial="n2cpv3", d_tune=0, d_confirm=0, d_all=0, overTO_tune=v.stat(nb - nto, v.tune)[0],
            overTO_confirm=v.stat(nb - nto, v.confirm)[0], overTO_all=v.stat(nb - nto, allm)[0])
print(f"n2cpv3 baseline: pool over TO-only pool TUNE {base['overTO_tune']:+.5f} CONFIRM {base['overTO_confirm']:+.5f} "
      f"ALL {base['overTO_all']:+.5f}")
print(" t   d_TUNE   d_CONF (se)        d_ALL (se)        | overTO tune  conf    all   | conf w0.3 w0.7 | preset  ln   ep sd")
for r in sorted(out, key=lambda r: -r["d_tune"]):
    print(f"{r['trial']:2d} {r['d_tune']:+.5f} {r['d_confirm']:+.5f} ({r['se_confirm']:.5f}) {r['d_all']:+.5f} "
          f"({r['se_all']:.5f}) | {r['overTO_tune']:+.5f} {r['overTO_confirm']:+.5f} {r['overTO_all']:+.5f} | "
          f"{r['d_confirm_w0.3']:+.5f} {r['d_confirm_w0.7']:+.5f} | {r['preset']:8s} {r['loss_norm']:5s} "
          f"{r['epochs']} {r['seeds']}")
a = np.array([r["d_tune"] for r in out])
b = np.array([r["d_confirm"] for r in out])
print(f"across {len(out)} trials: corr(d_TUNE, d_CONFIRM) {np.corrcoef(a, b)[0, 1]:+.3f}; mean d_TUNE {a.mean():+.5f}, "
      f"mean d_CONFIRM {b.mean():+.5f}; mean over-TO TUNE {np.mean([r['overTO_tune'] for r in out]):+.5f}, "
      f"CONFIRM {np.mean([r['overTO_confirm'] for r in out]):+.5f}")
with open(FT_DIR / "post_freeze.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
