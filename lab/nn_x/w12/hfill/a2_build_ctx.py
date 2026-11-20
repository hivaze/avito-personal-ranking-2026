"""hfill A2 (CPU): rebuild the history-similarity xt columns of the FD16P20 members with the filled history.

The members read 20 ctx_* columns through their xt list. Of these, the ones that use history vectors are built by kernels that
skip events with emb_idx < 0 (lab/feats_v2/ctx_/build_ctx.py _nd_kernel via maxsim32 (lab/src/features.py _emb_feats) and h10,
build_ctx3.py _hn_kernel, build_ctx4.py _hs_kernel). pre2_* are NOT rebuilt: their encoder looks items up by item_id in a
global content table (experiments/pretraining/code/build_corpus.py, item_tab), so a val/test-masked item already has its vector
there. seq_cv_* / it_hist_* / xp_imp_c50 are item-id counts (no vectors).

mode orig : same inputs as the stored builds -> must equal lab/nn/feat_cache/<split>/<name>.npy (pre-check, printed)
mode fill : hist_emb_idx filled + the split's table extended (hfill_lib.load_fill), maxsim32 rebuilt from the filled history
Output: feats/<mode>/<split>/<name>.npy (float32, flat cand order). CPU only, numba threads = NUMBA_NUM_THREADS (<= 12).
Usage: python a2_build_ctx.py val,test orig,fill
"""
import json
import os
import sys
import time
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctx_kernels import _emb_feats, _hn_kernel, _hs_kernel, _nd_kernel  # noqa: E402
from hfill_lib import FLAT, HF, item_table, load_fill  # noqa: E402

FEATS = Path("<repo>/data/feats")
CACHE = Path("<repo>/lab/nn/feat_cache")
ND_NAMES = ["ctx_nd_max", "ctx_nd_c80", "ctx_nd_c90", "ctx_nd_c95", "ctx_nd_mean", "ctx_nd_top5u", "ctx_nd_top5h",
            "ctx_nd_knn_u", "ctx_nd_knn_h", "ctx_nd_knn_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_f1", "ctx_nd_dupb_u",
            "ctx_nd_ucent", "ctx_nd_hcent", "ctx_nd_knn_u_dev", "ctx_nd_knn_h_dev", None]
HN_NAMES = ["ctx_hn_top1", "ctx_hn_top1w", "ctx_hn_top3w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin",
            "ctx_hn_soft05", "ctx_hn_soft02"]
HN_VARIANTS = {"": (32, 0.85, 0, [0, 1, 2, 3, 4, 5]), "soft": (32, 0.85, 0, [6, 7]), "_e1": (32, 0.85, 1, [1, 3, 5])}
HS_VARIANTS = {"ctx_hs_t10": (32, 0.85, 0.10, 0), "ctx_hs_k128": (128, 0.95, 0.05, 0), "ctx_hs_e1": (32, 0.85, 0.05, 1)}
# the 20 ctx columns of the FD16P20 xt lists (w5 members: all 20; abl pop1 members: the same 20)
WANT = ["ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max",
        "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin",
        "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def cmp(a, b):
    """bit-identical incl. NaN positions; else max abs diff over finite pairs and NaN-pattern mismatches."""
    na, nb_ = np.isnan(a), np.isnan(b)
    same = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    fin = ~na & ~nb_
    md = float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0
    return same, md, int((na != nb_).sum()), int((a[fin] != b[fin]).sum())


def build(split, mode):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    H = f["H"].to_numpy().astype(np.int64)
    co = f["cand_off"].to_numpy().astype(np.int64)
    ho = f["hist_off"].to_numpy().astype(np.int64)
    table = np.ascontiguousarray(item_table(split))
    cemb32 = np.load(FLAT / split / "cand_emb_idx.npy")
    hemb32 = np.load(FLAT / split / "hist_emb_idx.npy")
    if mode == "fill":
        hemb32, ext, info, _ = load_fill(split, len(table))
        table = np.ascontiguousarray(np.concatenate([table, ext], 0))
        log(split, "fill", json.dumps(info))
    het = np.load(FLAT / split / "hist_event_type.npy")
    ue = np.load(FLAT / split / "user_emb.npy")
    N = len(cemb32)
    res = {}
    # maxsim32 (input of the nd kernel): lab/src/features.py driver call convention
    out = np.empty((N, 21), np.float32)
    _emb_feats(f["cand_off"].to_numpy(), M, f["hist_off"].to_numpy(), H, cemb32, hemb32, het, ue, table, out)
    maxsim_new = np.ascontiguousarray(out[:, 16])
    del out
    maxsim_stored = np.load(FEATS / split / "maxsim32.npy").astype(np.float32)
    if mode == "orig":
        log(split, "maxsim32 orig vs stored (same, maxdiff, nan-mismatch, n-diff):", cmp(maxsim_new, maxsim_stored))
        maxsim = maxsim_stored  # exactly what the stored nd build read
    else:
        maxsim = maxsim_new
        log(split, "maxsim32 fill: changed rows", int((maxsim_new.view(np.uint32) != maxsim_stored.view(np.uint32)).sum()))
    cemb = cemb32.astype(np.int64)
    hemb = hemb32.astype(np.int64)
    cos_u = np.load(FEATS / split / "cos_u.npy").astype(np.float32)
    f1 = np.load(FEATS / split / "f1.npy").astype(np.float32)
    out = np.empty((N, len(ND_NAMES)), np.float32)
    fo = np.empty((len(M), 4), np.float32)
    _nd_kernel(co, M, ho, H, cemb, hemb, table, ue, cos_u, maxsim, f1, fo, out)
    for i, nm in enumerate(ND_NAMES):
        if nm in WANT:
            res[nm] = out[:, i].copy()
    log(split, mode, "nd done")
    for v, (Kv, decay, ev, cols) in HN_VARIANTS.items():
        hmask = np.ones(len(hemb), np.bool_) if ev == 0 else (het == ev)
        out = np.empty((N, len(HN_NAMES)), np.float32)
        _hn_kernel(co, M, ho, H, cemb, hemb, hmask, table, Kv, decay, out)
        for i in cols:
            nm = HN_NAMES[i].replace("ctx_hn_", f"ctx_hn{v}_") if v and v.startswith("_") else HN_NAMES[i]
            if nm in WANT:
                res[nm] = out[:, i].copy()
    log(split, mode, "hn done")
    for nm, (K, decay, tau, ev) in HS_VARIANTS.items():
        hmask = np.ones(len(hemb), np.bool_) if ev == 0 else (het == ev)
        mass = np.empty(N, np.float32)
        mut = np.empty(N, np.float32)
        _hs_kernel(co, M, ho, H, cemb, hemb, hmask, table, K, decay, tau, mass, mut)
        res[nm] = mass
        if nm == "ctx_hs_t10":
            res["ctx_hs_mutual"] = mut
    log(split, mode, "hs done")
    assert sorted(res) == sorted(WANT), sorted(set(WANT) ^ set(res))
    d = HF / "feats" / mode / split
    d.mkdir(parents=True, exist_ok=True)
    summary = {}
    for nm in WANT:
        np.save(d / f"{nm}.npy", np.ascontiguousarray(res[nm], dtype=np.float32))
        stored = np.load(CACHE / split / f"{nm}.npy").astype(np.float32)
        summary[nm] = cmp(res[nm].astype(np.float32), stored)
        log(f"  {split} {mode} {nm:18s} vs feat_cache: bit-identical {summary[nm][0]}, maxdiff {summary[nm][1]:.3g}, "
            f"nan-mismatch {summary[nm][2]}, rows differing {summary[nm][3]} ({summary[nm][3] / N:.4f})")
    (HF / "out" / f"a2_{split}_{mode}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    nb.set_num_threads(min(12, int(os.environ.get("NUMBA_NUM_THREADS", "12"))))
    for s in sys.argv[1].split(","):
        for m in sys.argv[2].split(","):
            build(s, m)
    log("done")
