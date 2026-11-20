"""Input pipeline audit, step 1: rebuild the EXACT 104-d dense input of the live G3 members on CPU.

Uses nn_x/w5/model.py dense_features (read-only import) and a subset-SplitData that reproduces nn_x/w5/data.py
gather() + attach_xt() for a subset of feeds. Writes (scratch only):
  out/<sub>_X.npy      [n_cand, 104] float32 pre-standardisation dense input (what feature_stats sees)
  out/<sub>_raw.npy    [n_cand, 34]  float32 raw feat_cache columns (NaN kept) before attach_xt
  out/<sub>_meta.npz   M, off (local), fidx (global), user, ts, y
subsets: trn (20K random training feeds, holdout excluded), hold (the 30K holdout, train.py draw), val, test.
CPU only.  python s01_extract.py <sub>
"""
import os, sys, time, json
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, polars as pl, torch
torch.set_num_threads(12)
sys.path.insert(0, "<repo>/nn_x/w5")
from model import dense_features, N_DENSE  # read-only import
from data import SplitData  # read-only import (class reused, __init__ bypassed)

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FC = ROOT / "nn/feat_cache"
OUT = Path(__file__).resolve().parent / "out_unused"
OUT.mkdir(exist_ok=True)
CK = torch.load(ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt", map_location="cpu", weights_only=False)
CFG = CK["cfg"]
XT = CFG["xt_names"]


def subset_feeds(sub):
    f = pl.read_parquet(FLAT / ("train" if sub in ("trn", "hold") else sub) / "feeds.parquet")
    n = len(f)
    if sub in ("val", "test"):
        return sub, np.arange(n)
    M = f["M"].to_numpy()
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(n)).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    if sub == "hold":
        return "train", hold
    rest = np.setdiff1d(np.arange(n), hold)
    return "train", np.sort(np.random.default_rng(777).choice(rest, 20000, replace=False))


def build(sub):
    split, fe = subset_feeds(sub)
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)[fe]; H = f["H"].to_numpy().astype(np.int64)[fe]
    co = f["cand_off"].to_numpy().astype(np.int64)[fe]; ho = f["hist_off"].to_numpy().astype(np.int64)[fe]
    crow = np.concatenate([np.arange(o, o + m) for o, m in zip(co, M)])
    hrow = np.concatenate([np.arange(o, o + h) for o, h in zip(ho, H)]) if H.sum() else np.zeros(0, np.int64)
    ld = lambda name, rows: np.asarray(np.load(FLAT / split / f"{name}.npy", mmap_mode="r")[rows])
    sd = SplitData.__new__(SplitData)
    sd.split, sd.device, sd.max_hist = split, torch.device("cpu"), 256
    sd.n = len(fe); sd.M_np = M; sd.H_np = H
    sd.cand_off_np = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    hoff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
    sd.user_id = f["user_id"].to_numpy()[fe]; sd.dom = None
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    sd.M, sd.H, sd.cand_off, sd.hist_off = g(M), g(H), g(sd.cand_off_np), g(hoff)
    sd.user_emb = g(np.asarray(np.load(FLAT / split / "user_emb.npy", mmap_mode="r")[fe]))
    sd.user0 = g(sd.user_id == 0)
    cemb = ld("cand_emb_idx", crow).astype(np.int64); hemb = ld("hist_emb_idx", hrow).astype(np.int64)
    used = np.unique(np.concatenate([cemb[cemb >= 0], hemb[hemb >= 0]]))
    d = ROOT / "data/competition" / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    tab = np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
    t0 = time.time(); small = np.asarray(tab[used]); print(f"table rows {len(used)} {time.time()-t0:.0f}s", flush=True)
    rm = lambda e: np.where(e >= 0, np.searchsorted(used, np.maximum(e, 0)), -1).astype(np.int32)
    sd.c_emb = g(rm(cemb)); sd.h_emb = g(rm(hemb)); sd.table = g(small)
    sd.c_item = None; sd.h_item = None; sd.xt = None
    sd.n_cand = len(crow); sd.n_hist = max(len(hrow), 1)
    sd.c_cat = g(ld("cand_category_id", crow).astype(np.int64)).to(torch.int16)
    sd.c_f = torch.stack([g(ld(f"cand_f{i}", crow)) for i in range(1, 6)], 1)
    sd.has_target = (FLAT / split / "cand_target.npy").exists()
    y = ld("cand_target", crow).astype(np.float32) if sd.has_target else None
    sd.target_np = y; sd.c_tgt = g(y.astype(np.int8)) if sd.has_target else None
    sd.has_pos = False; sd.c_pos = None
    sd.h_cat = g(ld("hist_category_id", hrow).astype(np.int16)); sd.h_et = g(ld("hist_event_type", hrow))
    sd.h_es = g(ld("hist_event_source", hrow))
    if len(hrow) == 0:
        sd.h_emb = g(np.full(1, -1, np.int32)); sd.h_cat = g(np.zeros(1, np.int16))
        sd.h_et = g(np.zeros(1, np.int8)); sd.h_es = g(np.zeros(1, np.int8))
    # raw xt + exact attach_xt transform (nn_x/w5/data.py attach_xt)
    raw, cols = [], []
    for n in XT:
        src = n[4:] if n.startswith("has:") else n
        x = np.asarray(np.load(FC / split / f"{src}.npy", mmap_mode="r")[crow]).astype(np.float32)
        if n.startswith("has:"):
            x = (~np.isnan(x)).astype(np.float32)
        raw.append(x.copy())
        if n.startswith("has:") or n.endswith("_rk"):
            pass
        elif n.endswith("_z"):
            x = np.clip(x, -5, 5)
        elif n.startswith(("seq_cv", "it_hist")):
            x = np.log1p(np.maximum(x, 0))
        else:
            x = np.clip(x, -10, 10)
        cols.append(torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16))
    sd.xt = torch.stack(cols, 1)
    raw = np.stack(raw, 1)
    return sd, fe, f, raw, y


def main(sub):
    t0 = time.time()
    sd, fe, f, raw, y = build(sub)
    print(f"{sub}: {sd.n} feeds {sd.n_cand} cands built {time.time()-t0:.0f}s", flush=True)
    X = np.zeros((sd.n_cand, CFG["n_dense"]), np.float32)
    bs = sd.batches(np.arange(sd.n), shuffle=False)
    for i, fb in enumerate(bs):
        b = sd.gather(fb)
        Xb = dense_features(b, CFG["n_dense"], CFG.get("xt_rel"), 0.0, False, CFG.get("xt_idcols"), CFG.get("xt_relonly"))
        m = b["cmask"]
        X[b["ci"][m].numpy()] = Xb[m].numpy()
        if i % 50 == 0:
            print(f"  batch {i}/{len(bs)} {time.time()-t0:.0f}s", flush=True)
    np.save(OUT / f"{sub}_X.npy", X)
    np.save(OUT / f"{sub}_raw.npy", raw)
    np.savez(OUT / f"{sub}_meta.npz", M=sd.M_np, off=sd.cand_off_np, fidx=fe, user=sd.user_id,
             ts=f["timestamp"].to_numpy()[fe], H=sd.H_np, y=y if y is not None else np.zeros(0))
    print(f"{sub} DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
