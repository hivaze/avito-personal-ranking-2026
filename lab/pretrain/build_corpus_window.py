"""Time-windowed corpora so that train feeds can get leak-free pre_* features (mirrors the val/test construction).

Window W_k = [T_k, E_k) (MSK days). corpus_k =
  * background: for each non-zero user the LATEST train snapshot with feed time < T_k, plus every user-0 train feed < T_k;
  * in-window: for each non-zero user the EARLIEST train snapshot inside W_k, plus every user-0 train feed inside W_k.
No snapshot of a user taken after any of that user's in-window feeds' earliest one is used, so a window feed never sees
its own or a later same-user snapshot except its own (earliest) one, exactly like val/test.
Outputs data/pretrain/win<k>/ with the same arrays as build_corpus.py (+ tgt_fe_multi, item counts) and
window_feeds.npy (train fidx of the feeds to score).
Usage: python pretrain/build_corpus_window.py
"""
import datetime as dt, json, sys, time
from pathlib import Path
import numpy as np, polars as pl, numba as nb
sys.path.insert(0, "<repo>/pretrain")
from build_corpus import gather_rev, targets
from build_multi_targets import multi

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
PT = ROOT / "data/pretrain"
nb.set_num_threads(32)
MSK = dt.timezone(dt.timedelta(hours=3))
T = lambda d: int(dt.datetime.fromisoformat(d + "T00:00:00").replace(tzinfo=MSK).timestamp())
# W2 is closed up to Sep 17 (no data Sep 15-16 apart from 42 boundary feeds) and W3 open-ended (34 feeds at Sep 20 00:00)
WINDOWS = {1: (T("2026-09-09"), T("2026-09-12")), 2: (T("2026-09-12"), T("2026-09-17")), 3: (T("2026-09-17"), 2**31 - 1)}


def build(k, f, src, G):
    t0 = time.time()
    a, e = WINDOWS[k]
    out = PT / f"win{k}"
    out.mkdir(parents=True, exist_ok=True)
    nz = pl.col("user_id") != 0
    bg = f.filter(pl.col("timestamp") < a)
    bg_ids = np.concatenate([bg.filter(nz).sort(["timestamp", "fidx"]).group_by("user_id").agg(pl.col("fidx").last())["fidx"].to_numpy(),
                             bg.filter(~nz)["fidx"].to_numpy()])
    w = f.filter((pl.col("timestamp") >= a) & (pl.col("timestamp") < e))
    win_ids = np.concatenate([w.filter(nz).sort(["timestamp", "fidx"]).group_by("user_id").agg(pl.col("fidx").first())["fidx"].to_numpy(),
                              w.filter(~nz)["fidx"].to_numpy()])
    np.save(out / "window_feeds.npy", np.sort(w["fidx"].to_numpy()).astype(np.int64))
    H_all = f["H"].to_numpy().astype(np.int64); hoff_all = f["hist_off"].to_numpy().astype(np.int64)
    parts = {c: [] for c in src}
    lens, split = [], []
    for sid, ids in enumerate([np.sort(bg_ids), np.sort(win_ids)]):
        H, hoff = H_all[ids], hoff_all[ids]
        keep = H > 0
        H, hoff = H[keep], hoff[keep]
        ooff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
        for c, arr in src.items():
            dst = np.empty(int(H.sum()), arr.dtype)
            gather_rev(hoff, H, arr, ooff, dst)
            parts[c].append(dst)
        lens.append(H); split.append(np.full(len(H), sid, np.int8))
    lens = np.concatenate(lens)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tok = {c: np.concatenate(v) for c, v in parts.items()}
    tf = np.empty(len(tok["item_id"]), np.int32); tn = np.empty(len(tok["item_id"]), np.int32)
    targets(off, tok["item_id"], tok["event_type"], tok["event_source"], tf, tn)
    tm = np.empty((len(tf), 4), np.int32)
    multi(off, tok["item_id"], tok["event_type"], tok["event_source"], tm)
    np.save(out / "seq_off.npy", off); np.save(out / "seq_split.npy", np.concatenate(split))
    for c, n in [("item_id", "tok_item"), ("category_id", "tok_cat"), ("event_type", "tok_et"), ("event_source", "tok_es")]:
        np.save(out / f"{n}.npy", tok[c])
    np.save(out / "tgt_fe.npy", tf); np.save(out / "tgt_new.npy", tn); np.save(out / "tgt_fe_multi.npy", tm)
    np.save(out / "item_cnt.npy", np.bincount(tok["item_id"], minlength=G).astype(np.int32))
    np.save(out / "item_cnt_fe.npy", np.bincount(tf[tf >= 0], minlength=G).astype(np.int32))
    np.save(out / "item_cnt_new.npy", np.bincount(tn[tn >= 0], minlength=G).astype(np.int32))
    cnt = np.bincount(tok["item_id"], minlength=G)
    info = dict(window=k, start=a, end=e, n_window_feeds=len(w), n_bg_seqs=int((np.concatenate(split) == 0).sum()),
                n_win_seqs=int((np.concatenate(split) == 1).sum()), tokens=int(off[-1]), id_items_ge3=int((cnt >= 3).sum()),
                sec=round(time.time() - t0))
    (out / "info.json").write_text(json.dumps(info))
    print(info, flush=True)


def main():
    f = pl.read_parquet(FLAT / "train/feeds.parquet").select("fidx", "timestamp", "user_id", "H", "hist_off")
    src = {c: np.load(FLAT / "train" / f"hist_{c}.npy") for c in ["item_id", "category_id", "event_type", "event_source"]}
    G = len(np.load(PT / "item_has.npy", mmap_mode="r"))
    for k in (map(int, sys.argv[1].split(",")) if len(sys.argv) > 1 else WINDOWS):
        build(k, f, src, G)


if __name__ == "__main__":
    main()
