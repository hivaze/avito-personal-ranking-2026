"""winner_model recheck, check 1: structural data facts (CPU, mmap, chunked).
 - label values present in train / val; feeds without a positive; positives per feed
 - M range, feeds below the val/test minimum M
 - exploration share by M (old detector p_random and v2), production position of positives
Usage: CUDA_VISIBLE_DEVICES="" python c01_data_facts.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path("<repo>")
FLAT = ROOT / "data" / "flat"

for split in ("train", "val", "test"):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    H = f["H"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    print(f"== {split}: feeds {len(M)}, cands {M.sum()}, M min {M.min()} max {M.max()} mean {M.mean():.1f}; "
          f"M<5 {int((M < 5).sum())}, M<10 {(M < 10).mean():.4f}, M<50 {(M < 50).mean():.4f}; H=0 {(H == 0).mean():.4f}")
    p = FLAT / split / "cand_target.npy"
    if not p.exists():
        continue
    y = np.load(p, mmap_mode="r")
    vals = {}
    npos = np.zeros(len(M), np.int64)
    CH = 20_000_000
    fid_bounds = np.searchsorted(off, np.arange(0, len(y) + CH, CH))
    for i, s in enumerate(range(0, len(y), CH)):
        yy = np.asarray(y[s:s + CH])
        u, c = np.unique(yy, return_counts=True)
        for a, b in zip(u, c):
            vals[int(a)] = vals.get(int(a), 0) + int(b)
        fid = np.searchsorted(off, np.arange(s, s + len(yy)), side="right") - 1
        npos += np.bincount(fid, weights=(yy > 0), minlength=len(M)).astype(np.int64)
    print(f"   target dtype {y.dtype}, values {vals}")
    print(f"   feeds with 0 positives {int((npos == 0).sum())}; 1 pos {(npos == 1).mean():.4f}; 2 {(npos == 2).mean():.4f}; "
          f">=3 {(npos >= 3).mean():.4f}; mean {npos.mean():.3f}; all-positive feeds {int((npos == M).sum())}")
    if split == "train":
        pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
        p1 = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
        p1[M < 50] = 0
        v2 = ROOT / "eda/labels_position/out/train_prandom_v2.npz"
        p2 = np.nan_to_num(np.load(v2)["p_random"].astype(np.float32), nan=0.0) if v2.exists() else None
        print(f"   exploration feeds: old detector p>0.5 {int((p1 > 0.5).sum())} ({(p1 > 0.5).mean():.4f})"
              + (f"; v2 p>0.5 {int((p2 > 0.5).sum())} ({(p2 > 0.5).mean():.4f}), of which M<50 {int(((p2 > 0.5) & (M < 50)).sum())}" if p2 is not None else ""))
        if p2 is not None:
            for lo, hi in ((5, 20), (20, 50), (50, 100), (100, 200), (200, 400)):
                m = (M >= lo) & (M < hi)
                print(f"     M [{lo},{hi}): feeds {int(m.sum())}, v2 exploration share {(p2[m] > 0.5).mean():.4f}, "
                      f"mean positives {npos[m].mean():.3f}")
            # candidate rows in exploration vs production feeds (how much of the training signal is in-format)
            ex = p2 > 0.5
            print(f"   candidate rows: exploration {int(M[ex].sum())} ({M[ex].sum() / M.sum():.4f}), production {int(M[~ex].sum())}")
