"""Chunked scoring of a set of feeds with a pretrained encoder checkpoint (same exports as train.infer).

  python pretrain/score_window.py --tag w1_v3 --window 1              # all train feeds of window 1
  python pretrain/score_window.py --tag v3 --window 1 --sample 60000  # reference (full-corpus) model on a feed sample
  python pretrain/score_window.py --tag v3 --split val                # consistency check vs preds/pre_v3_*_val.npy
Outputs data/pretrain/winscores/pre_<tag>_<key>_win<k>[_s<sample>].npy aligned with
data/pretrain/win<k>/cand_rows[_s<sample>].npy (global train candidate rows).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, polars as pl, torch

sys.path.insert(0, "<repo>/pretrain")
sys.path.insert(0, "<repo>/src")
from seqenc import SeqEncoder
import train as T
from build_corpus import gather_rev
from metrics import ndcg

ROOT = Path("<repo>")
FLAT, PT = ROOT / "data/flat", ROOT / "data/pretrain"
OUT = PT / "winscores"


class Chunk:
    """FeedSplit-compatible view of a contiguous chunk of feeds (history already reversed, chunk-local offsets)."""

    def __init__(self, dev, H, item, cat, et, es, M, c_item):
        self.n = len(H)
        self.H_np = H
        self.M = M
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        self.item, self.cat, self.et, self.es = g(item), g(cat), g(et), g(es)
        self.off = g(np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64))
        self.lens = g(H.astype(np.int64))
        self.c_item = g(c_item)
        self.c_fidx = g(np.repeat(np.arange(self.n), M).astype(np.int64))
        self.off_c = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
        self.y = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=30000)
    ap.add_argument("--item_tab", default="", help="override the checkpoint's content table (e.g. item_tab2)")
    ap.add_argument("--out_tag", default="", help="output tag (default = --tag)")
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda:0")
    ck = torch.load(PT / f"ckpt_{a.tag}.pt", map_location="cpu"); ca = ck["args"]
    T.set_corpus_dir(ca.get("corpus_dir", ""))
    counts = T.counts_excluding_test() if ca.get("exclude_test") else None
    items = T.Items(dev, ca["min_cnt"], counts, tab_name=a.item_tab or ca.get("item_tab", "item_tab"))
    heads = ["fe", "new"] if ca["w_new"] > 0 else ["fe"]
    model = SeqEncoder(items.n_id, d=ca["d"], nl=ca["nl"], nh=ca["nh"], d_id=ca["d_id"], drop=ca["drop"], heads=heads,
                       id_bias=bool(ca.get("id_bias", 0)))
    model.load_state_dict(ck["model"]); model.to(dev)
    split = a.split
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    if split == "train":
        feeds = np.load(PT / f"win{a.window}/window_feeds.npy")
        if a.sample:
            feeds = np.sort(np.random.default_rng(0).choice(feeds, min(a.sample, len(feeds)), replace=False))
    else:
        feeds = np.arange(len(f))
    H = f["H"].to_numpy().astype(np.int64)[feeds]
    hoff = f["hist_off"].to_numpy().astype(np.int64)[feeds]
    M = f["M"].to_numpy().astype(np.int64)[feeds]
    coff = f["cand_off"].to_numpy().astype(np.int64)[feeds]
    ooff = np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64)
    hist = {}
    for c in ["item_id", "category_id", "event_type", "event_source"]:
        src = np.load(FLAT / split / f"hist_{c}.npy", mmap_mode="r" if split != "train" else None)
        dst = np.empty(int(H.sum()), src.dtype)
        gather_rev(hoff, H, np.ascontiguousarray(src), ooff, dst)
        hist[c] = dst
        del src
    rows = np.repeat(coff - np.concatenate([[0], np.cumsum(M)[:-1]]), M) + np.arange(M.sum())
    c_item_all = np.load(FLAT / split / "cand_item_id.npy", mmap_mode="r")[rows]
    print(f"[data] {len(feeds)} feeds, {len(rows)} cands, {int(H.sum())} hist, {time.time()-t0:.0f}s", flush=True)
    out = {}
    cpos = np.concatenate([[0], np.cumsum(M)]).astype(np.int64)
    hpos = np.concatenate([[0], np.cumsum(H)]).astype(np.int64)
    for i in range(0, len(feeds), a.chunk):
        j = min(i + a.chunk, len(feeds))
        hs, he, cs, ce = hpos[i], hpos[j], cpos[i], cpos[j]
        ch = Chunk(dev, H[i:j], hist["item_id"][hs:he], hist["category_id"][hs:he], hist["event_type"][hs:he],
                   hist["event_source"][hs:he], M[i:j], c_item_all[cs:ce])
        sc = T.infer(model, items, ch, heads)
        for k, v in sc.items():
            if k not in out:
                out[k] = np.empty(len(rows), np.float32)
            out[k][cs:ce] = v
        del ch
    if split in ("val", "test"):
        y = np.load(FLAT / "val/cand_target.npy") if split == "val" else None
        for k, v in out.items():
            ref = np.load(ROOT / f"preds/pre_{a.tag}_{k}_{split}.npy")
            msg = f"  {split} {k}: max |diff| vs saved {np.abs(v - ref).max():.2e} corr {np.corrcoef(v, ref)[0, 1]:.6f}"
            if y is not None:
                msg += f" ndcg {ndcg(v, y, f['cand_off'].to_numpy(), f['M'].to_numpy()):.4f} (saved {ndcg(ref, y, f['cand_off'].to_numpy(), f['M'].to_numpy()):.4f})"
            print(msg, flush=True)
            if a.out_tag:
                np.save(ROOT / f"preds/pre_{a.out_tag}_{k}_{split}.npy", v)
        return
    OUT.mkdir(exist_ok=True)
    sfx = f"win{a.window}" + (f"_s{a.sample}" if a.sample else "")
    np.save(PT / (f"win{a.window}/cand_rows" + (f"_s{a.sample}" if a.sample else "") + ".npy"), rows.astype(np.int64))
    for k, v in out.items():
        np.save(OUT / f"pre_{a.out_tag or a.tag}_{k}_{sfx}.npy", v)
    print(f"[done] {a.out_tag or a.tag} {sfx}: {len(rows)} cands, keys {list(out)}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
