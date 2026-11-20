"""Print the study table from trials.jsonl (sorted by TUNE objective) and write trials.csv."""
import json
import csv
from ftlib import FT_DIR
rows = [json.loads(l) for l in open(FT_DIR / "trials.jsonl")]
keys = ["lr", "epochs", "decay", "ema", "ema_start", "wd", "drop", "loss_norm", "preset", "extra_lr_mult", "seeds"]
with open(FT_DIR / "trials.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    mk = []  # tune_mix_* of the first driver are member-misaligned: recomputed by analyze.py
    w.writerow(["trial", "value", "tune_se", "w0.3", "w0.7", "ftonly", "minutes"] + keys + mk)
    for r in rows:
        w.writerow([r["trial"], r["value"], r["tune_se"], r["tune_delta_w0.3"], r["tune_delta_w0.7"],
                    r["tune_delta_ftonly"], r["minutes"]] + [r["params"].get(k, "") for k in keys] + [r[k] for k in mk])
print(f"{len(rows)} trials")
print(" t   TUNE d    se      w0.3     w0.7    ftonly | lr       ep dec   ema   es   wd     drop ln    preset   elm  sd")
for r in sorted(rows, key=lambda r: -r["value"]):
    p = r["params"]
    print(f"{r['trial']:2d} {r['value']:+.5f} {r['tune_se']:.5f} {r['tune_delta_w0.3']:+.5f} {r['tune_delta_w0.7']:+.5f} "
          f"{r['tune_delta_ftonly']:+.5f} | {p['lr']:.2e} {p['epochs']} {p['decay']:5s} {p['ema']:5s} "
          f"{str(p.get('ema_start', '-')):4s} {p['wd']:<6} {p['drop']:<4} {p['loss_norm']:5s} {p['preset']:8s} "
          f"{str(p.get('extra_lr_mult', '-')):4s} {p['seeds']}")
