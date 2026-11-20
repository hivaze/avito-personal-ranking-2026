"""Build the self-supervised pretraining corpus from ALL history snapshots (train + val + test).

Leak-safe selection of snapshots (one sequence per user, user_id 0 feeds = separate sequences):
  * train: latest train snapshot per non-zero user (+ every user-0 train feed). Leaky for train rows -> features are
    only produced for val/test.
  * val/test: per non-zero user, ONLY the earliest snapshot over val U test (a later snapshot would contain the
    positives of the earlier labelled val feed; applied to test as well for val/test consistency) + every user-0 feed.
Sequences are stored old -> new. Targets per position t (computed within the 256-event window):
  * tgt_fe[t]  = item of the first event j>t with event_type 1 and event_source 1 (feed-originated open, the closest
                 proxy of the competition target) whose item does not occur in events <= t ("new item"), else -1
  * tgt_new[t] = item of the first event j>t whose item does not occur in events <= t, else -1
Also builds the global content table indexed by item_id (int8 [max_item+1,128], zero rows = missing) and item counts.
"""
import json, time
from pathlib import Path
import numpy as np, polars as pl, numba as nb

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
COMP = ROOT / "data/competition"
OUT = ROOT / "data/pretrain"
nb.set_num_threads(32)


@nb.njit(parallel=True, cache=True)
def gather_rev(hist_off, H, src_list, out_off, dst):
    # copy hist rows of each selected feed reversed (newest-first -> old->new)
    for s in nb.prange(len(H)):
        o, h, d = hist_off[s], H[s], out_off[s]
        for k in range(h):
            dst[d + k] = src_list[o + h - 1 - k]


@nb.njit(parallel=True, cache=True)
def targets(off, item, et, es, tgt_fe, tgt_new):
    for s in nb.prange(len(off) - 1):
        a, b = off[s], off[s + 1]
        L = b - a
        fo = np.empty(L, np.int32)
        for j in range(L):
            fo[j] = j
            x = item[a + j]
            for i in range(j):
                if item[a + i] == x:
                    fo[j] = i
                    break
        for t in range(L):
            tgt_fe[a + t] = -1
            tgt_new[a + t] = -1
            for j in range(t + 1, L):
                if fo[j] == j:
                    tgt_new[a + t] = item[a + j]
                    break
            for j in range(t + 1, L):
                if et[a + j] == 1 and es[a + j] == 1 and fo[j] > t:
                    tgt_fe[a + t] = item[a + j]
                    break


@nb.njit(cache=True)
def first_src(items, embs, has, src):
    for k in range(len(items)):
        e = embs[k]
        if e >= 0:
            it = items[k]
            if not has[it]:
                has[it] = True
                src[it] = e


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fe = {s: pl.read_parquet(FLAT / s / "feeds.parquet") for s in ["train", "val", "test"]}
    # --- snapshot selection
    tr = fe["train"]
    sel = {}
    tr_last = tr.filter(pl.col("user_id") != 0).sort("timestamp").group_by("user_id").agg(pl.col("fidx").last())
    sel["train"] = np.sort(np.concatenate([tr_last["fidx"].to_numpy(), tr.filter(pl.col("user_id") == 0)["fidx"].to_numpy()]))
    vt = pl.concat([fe["val"].with_columns(pl.lit(0).alias("sp")), fe["test"].with_columns(pl.lit(1).alias("sp"))])
    first = vt.filter(pl.col("user_id") != 0).sort(["timestamp", "sp", "fidx"]).group_by("user_id").agg(pl.col("fidx").first(), pl.col("sp").first())
    for k, s in enumerate(["val", "test"]):
        f = fe[s]
        a = first.filter(pl.col("sp") == k)["fidx"].to_numpy()
        b = f.filter(pl.col("user_id") == 0)["fidx"].to_numpy()
        sel[s] = np.sort(np.concatenate([a, b]))
    for s in sel:
        print(f"[sel] {s}: {len(sel[s])} of {len(fe[s])} snapshots", flush=True)
    # --- gather tokens
    cols = ["item_id", "category_id", "event_type", "event_source"]
    parts = {c: [] for c in cols}
    lens, split_of_seq, fidx_of_seq = [], [], []
    for sid, s in enumerate(["train", "val", "test"]):
        f = fe[s]
        ids = sel[s]
        hoff = f["hist_off"].to_numpy()[ids].astype(np.int64)
        H = f["H"].to_numpy()[ids].astype(np.int64)
        keep = H > 0
        ids, hoff, H = ids[keep], hoff[keep], H[keep]
        ooff = np.concatenate([[0], np.cumsum(H)[:-1]])
        for c in cols:
            src = np.load(FLAT / s / f"hist_{c}.npy", mmap_mode="r")
            src = np.ascontiguousarray(src)
            dst = np.empty(int(H.sum()), src.dtype)
            gather_rev(hoff, H, src, ooff, dst)
            parts[c].append(dst)
            del src
        lens.append(H); split_of_seq.append(np.full(len(H), sid, np.int8)); fidx_of_seq.append(ids.astype(np.int32))
        print(f"[tok] {s}: {len(H)} seqs, {H.sum()} tokens, {time.time()-t0:.0f}s", flush=True)
    lens = np.concatenate(lens)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tok = {c: np.concatenate(parts[c]) for c in cols}
    del parts
    tgt_fe = np.empty(len(tok["item_id"]), np.int32)
    tgt_new = np.empty(len(tok["item_id"]), np.int32)
    targets(off, tok["item_id"], tok["event_type"], tok["event_source"], tgt_fe, tgt_new)
    print(f"[tgt] valid fe {np.mean(tgt_fe >= 0):.3f} new {np.mean(tgt_new >= 0):.3f}, {time.time()-t0:.0f}s", flush=True)
    np.save(OUT / "seq_off.npy", off)
    np.save(OUT / "seq_split.npy", np.concatenate(split_of_seq))
    np.save(OUT / "seq_fidx.npy", np.concatenate(fidx_of_seq))
    np.save(OUT / "tok_item.npy", tok["item_id"])
    np.save(OUT / "tok_cat.npy", tok["category_id"])
    np.save(OUT / "tok_et.npy", tok["event_type"])
    np.save(OUT / "tok_es.npy", tok["event_source"])
    np.save(OUT / "tgt_fe.npy", tgt_fe)
    np.save(OUT / "tgt_new.npy", tgt_new)
    # --- item statistics over the corpus
    G = 0
    for s in ["train", "val", "test"]:
        for n in ["hist_item_id", "cand_item_id"]:
            G = max(G, int(np.load(FLAT / s / f"{n}.npy", mmap_mode="r").max()) + 1)
    cnt = np.bincount(tok["item_id"], minlength=G).astype(np.int32)
    cnt_fe = np.bincount(tgt_fe[tgt_fe >= 0], minlength=G).astype(np.int32)
    cnt_new = np.bincount(tgt_new[tgt_new >= 0], minlength=G).astype(np.int32)
    np.save(OUT / "item_cnt.npy", cnt); np.save(OUT / "item_cnt_fe.npy", cnt_fe); np.save(OUT / "item_cnt_new.npy", cnt_new)
    print(f"[cnt] G={G}, items in corpus {np.sum(cnt>0)}, >=2 {np.sum(cnt>=2)}, >=3 {np.sum(cnt>=3)}, >=5 {np.sum(cnt>=5)}, >=10 {np.sum(cnt>=10)}", flush=True)
    # --- global content table indexed by item_id (newest split first so val/test candidates use their own rows)
    has = np.zeros(G, np.bool_)
    tab = np.zeros((G, 128), np.int8)
    for s in ["test", "val", "train"]:
        meta = json.loads((COMP / s / "item_embeddings/data.meta").read_text())
        T = np.memmap(COMP / s / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
        src = np.full(G, -1, np.int64)
        has_s = has.copy()
        for kind in ["cand", "hist"]:
            p = FLAT / s / f"{kind}_item_id.npy"
            if kind == "cand" and s == "train":
                continue
            first_src(np.load(p), np.load(FLAT / s / f"{kind}_emb_idx.npy"), has_s, src)
        new = np.where(has_s & ~has)[0]
        tab[new] = np.asarray(T)[src[new]] if s != "train" else T[src[new]]
        has = has_s
        print(f"[tab] {s}: +{len(new)} rows, {time.time()-t0:.0f}s", flush=True)
    np.save(OUT / "item_tab.npy", tab)
    np.save(OUT / "item_has.npy", has)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
