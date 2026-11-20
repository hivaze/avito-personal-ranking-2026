"""Blend test of a per-candidate score file against reference bases, overall and on the clean subset.
Usage: python pretrain/blend_report.py --score pre_final"""
import argparse, sys
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/pretrain")
from metrics import ndcg_per_feed
from evaluate import load_base, blend_per_feed, blend_eval, clean_mask, meta, PREDS, FLAT

ap = argparse.ArgumentParser(); ap.add_argument("--score", required=True); a = ap.parse_args()
f, M, off = meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
users = f["user_id"].to_numpy()
clean = clean_mask()
x = np.load(PREDS / f"{a.score}_val.npy")
pf = ndcg_per_feed(x, y, off, M)
print(f"{a.score}: zero-shot {pf.mean():.4f}; later-user feeds ({(~clean).sum()}) {pf[~clean].mean():.4f} vs others {pf[clean].mean():.4f}")
for base in ["nn_ens_stage1_val_oof", "best"]:
    b = load_base(base)
    pb = ndcg_per_feed(b, y, off, M)
    _, res, cv = blend_eval(b, x, y, off, M, users)
    row = f"  base {base} ({pb.mean():.4f}):"
    for w in (0.1, 0.2):
        d = blend_per_feed(b, x, y, off, M, w)
        row += f" | w={w}: {d.mean():+.5f} ({d.std()/np.sqrt(len(d)):.5f}), clean {d[clean].mean():+.5f} ({d[clean].std()/np.sqrt(clean.sum()):.5f})"
    row += f" | CV-w {cv[0]:+.5f} ({cv[1]:.5f})"
    print(row)
