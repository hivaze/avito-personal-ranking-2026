"""Parity-fixed id_contrib for the OLD pre_ / prew_ families, written under NEW names (live files stay untouched):
  data/feats/{val,test}/pre_id_contrib_v2.npy, data/feats/{train,val,test}/prew_id_contrib_v2.npy
id_contrib_v2 = mean over configs (v3, v4) of a_fe * (fe_raw - fe_noid_raw), a_fe = the fe moment-alignment slope used
for that model (val: v3A/v4A -> v3/v4 on val; test: 1; train windows: w<k>_<c> -> <c> on the 60K-feed window sample,
the section-8 models). Rows without an item ID are exactly 0 on every split. prew_ = NaN where it_hist_all_luo == 0.
Usage: python pretrain/make_idcontrib_v2.py"""
import sys
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path("<repo>"); P = ROOT / "preds"; F = ROOT / "data/feats"; PT = ROOT / "data/pretrain"; WS = PT / "winscores"
FLAT = ROOT / "data/flat"


def slope(x, ref):
    mx = (x > -2) & np.isfinite(x); mr = (ref > -2) & np.isfinite(ref)
    return ref[mr].astype(np.float64).std() / x[mx].astype(np.float64).std()


def idc(fe, noid, a):
    return a * (fe.astype(np.float64) - noid.astype(np.float64))


out = {}
# val: A models aligned to their test-side twins (old content table, as for pre_*)
L = lambda t, k, sp: np.load(P / f"pre_{t}_{k}_{sp}.npy").astype(np.float32)
acc = []
for a_, b_ in [("v3A", "v3"), ("v4A", "v4")]:
    a = slope(L(a_, "fe", "val"), L(b_, "fe", "val"))
    print(f"val slope {a_}->{b_}: {a:.4f}")
    acc.append(idc(L(a_, "fe", "val"), L(a_, "fe_noid", "val"), a))
out["val"] = np.mean(acc, 0).astype(np.float32)
out["test"] = np.mean([idc(L(t, "fe", "test"), L(t, "fe_noid", "test"), 1.0) for t in ["v3", "v4"]], 0).astype(np.float32)
for sp in ["val", "test"]:
    np.save(F / sp / "pre_id_contrib_v2.npy", out[sp])
    cold = np.load(F / sp / "it_hist_all_luo.npy") == 0
    w = out[sp].copy(); w[cold] = np.nan
    np.save(F / sp / "prew_id_contrib_v2.npy", w)
    print(sp, "written", flush=True)
# train (prew only): section-8 window models w<k>_<c>, aligned on the 60K sample to the full-corpus <c>
N = len(np.load(FLAT / "train/cand_fidx.npy", mmap_mode="r"))
tr = np.full(N, np.nan, np.float32)
for k in [1, 2, 3]:
    rows = np.load(PT / f"win{k}/cand_rows.npy"); s_rows = np.load(PT / f"win{k}/cand_rows_s60000.npy")
    pos = np.searchsorted(rows, s_rows); assert (rows[pos] == s_rows).all()
    acc = []
    for c in ["v3", "v4"]:
        fe = np.load(WS / f"pre_w{k}_{c}_fe_win{k}.npy"); noid = np.load(WS / f"pre_w{k}_{c}_fe_noid_win{k}.npy")
        a = slope(fe[pos], np.load(WS / f"pre_{c}_fe_win{k}_s60000.npy"))
        print(f"train W{k} slope w{k}_{c}->{c}: {a:.4f}", flush=True)
        acc.append(idc(fe, noid, a))
    tr[rows] = np.mean(acc, 0).astype(np.float32)
cold = np.load(F / "train/it_hist_all_luo.npy") == 0
tr[cold] = np.nan
np.save(F / "train" / "prew_id_contrib_v2.npy", tr)
print("train written", flush=True)
