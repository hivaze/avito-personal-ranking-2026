"""ENCODER-DOSE (A): hmax32 of the pre2 encoders with the ID slot ZEROED (ledger nn_x/abl/ledger/ledger.md section 10.A).

pretrain/train.py infer() computes hmax32 = max over the newest 32 history items of <out_vec(item_rep(cand)), out_vec(item_rep(hist))>
(-2.0 when the history is empty; item reps under bf16 autocast, out_vec in fp32) with ID-AWARE item reps. The stored fe_noid /
new_noid scores are the same heads with the candidate's ID slot zeroed, but no ID-free hmax32 was ever exported. This script
reproduces the hmax32 arithmetic with the ID slot zeroed for the candidate AND the history items (SeqEncoder built with n_id = 0,
so item_rep has no ID term at all = exactly slot 0), from the same checkpoints and the same content table (item_tab2), for the
same feed sets as the live pre2_* construction:

  window models   w{k}r_{v3,v4}  on their window feeds (data/pretrain/win<k>/window_feeds.npy)      -> winscores/pre_<tag>_hmax32_noid_win<k>.npy
  references      v3 / v4 (tags v3t2 / v4t2) on the 60K window samples (rng(0), score_window.py)   -> winscores/pre_<tag>_hmax32_noid_win<k>_s60000.npy
  val             v3A / v4A (tags v3At2 / v4At2) and v3 / v4 (tags v3t2 / v4t2)                    -> preds/pre_<tag>_hmax32_noid_val.npy
  test            v3 / v4 (tags v3t2 / v4t2)                                                       -> preds/pre_<tag>_hmax32_noid_test.npy

All outputs are NEW names (exclusive create; nothing existing is overwritten). The candidate row order equals the stored
cand_rows[_s60000].npy of the window (asserted) / the split's candidate order.

  CUDA_VISIBLE_DEVICES=<4|5> OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/encdose/score_hmax_noid.py --mode noid [--jobs all|<tag>:<split>[:k] ...]
  ... --mode id --jobs v3At2:val      VERIFICATION: ID-aware reps (the checkpoint's ID table + slot map) -> compared with the stored
                                      preds/pre_v3At2_hmax32_val.npy (max |diff|, corr); written to the scratchpad only.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "<repo>/pretrain")
from seqenc import SeqEncoder  # noqa: E402  (pretrain/train.py itself is NOT imported: its build_corpus import forces 32 numba threads)

ROOT = Path("<repo>")
PT, FLAT = ROOT / "data/pretrain", ROOT / "data/flat"
WS, PREDS = PT / "winscores", ROOT / "preds"
SCR = Path(os.environ.get("ENCDOSE_SCRATCH", "/tmp/scratchpad/encdose"))
LAST_K = 32
# tag -> (checkpoint, content table)
TAGS = {"w1r_v3": ("w1r_v3", "item_tab2"), "w1r_v4": ("w1r_v4", "item_tab2"), "w2r_v3": ("w2r_v3", "item_tab2"),
        "w2r_v4": ("w2r_v4", "item_tab2"), "w3r_v3": ("w3r_v3", "item_tab2"), "w3r_v4": ("w3r_v4", "item_tab2"),
        "v3t2": ("v3", "item_tab2"), "v4t2": ("v4", "item_tab2"), "v3At2": ("v3A", "item_tab2"), "v4At2": ("v4A", "item_tab2")}
ALL_JOBS = ([f"w{k}r_{c}:train:{k}" for k in (1, 2, 3) for c in ("v3", "v4")]
            + [f"{c}t2:train:{k}:s60000" for k in (1, 2, 3) for c in ("v3", "v4")]
            + ["v3At2:val", "v4At2:val", "v3t2:val", "v4t2:val", "v3t2:test", "v4t2:test"])


def counts_excluding_test(ptc):
    """pretrain/train.counts_excluding_test, verbatim, with the corpus dir as an argument (slot map of the exclude_test models)."""
    sp = np.load(ptc / "seq_split.npy"); off = np.load(ptc / "seq_off.npy")
    keep = np.repeat(sp != 2, np.diff(off))
    G = len(np.load(PT / "item_has.npy", mmap_mode="r"))
    ti = np.load(ptc / "tok_item.npy")[keep]; tf = np.load(ptc / "tgt_fe.npy")[keep]; tn = np.load(ptc / "tgt_new.npy")[keep]
    return (np.bincount(ti, minlength=G).astype(np.int32), np.bincount(tf[tf >= 0], minlength=G).astype(np.int32),
            np.bincount(tn[tn >= 0], minlength=G).astype(np.int32))


class Tables:
    """item_tab2 / item_has2 on the GPU (global), category per corpus_dir (as pretrain/train.Items builds it), slot in id mode."""

    def __init__(self, dev, tab_name):
        t0 = time.time()
        tab = np.load(PT / f"{tab_name}.npy", mmap_mode="r")
        self.G = tab.shape[0]
        self.tab = torch.empty((self.G, 128), dtype=torch.int8, device=dev)
        for a in range(0, self.G, 8_000_000):
            self.tab[a:a + 8_000_000] = torch.from_numpy(np.ascontiguousarray(tab[a:a + 8_000_000])).to(dev)
        self.has = torch.from_numpy(np.load(PT / f"{tab_name.replace('_tab', '_has')}.npy")).to(dev)
        self.dev, self.cat_dir, self.cat, self.slot = dev, None, None, None
        self.cand_cat = [(np.load(FLAT / s / "cand_item_id.npy"), np.load(FLAT / s / "cand_category_id.npy")) for s in ("train", "val", "test")]
        print(f"[tables] {tab_name} G={self.G} {time.time() - t0:.0f}s", flush=True)

    def set_corpus(self, corpus_dir):
        """items.cat exactly as train.Items: corpus tokens' categories, then the candidates' categories of every split."""
        if self.cat_dir == corpus_dir:
            return
        ptc = Path(corpus_dir) if corpus_dir else PT
        if not (ptc / "tok_cat.npy").exists():  # the window corpora (win<k>/tok_*.npy) were removed after the 8c scoring: the
            print(f"[tables] {ptc}/tok_cat.npy absent -> full-corpus token categories (the history items of window feeds are in both corpora; "
                  "candidates take their split's category below)", flush=True)
            ptc = PT
        cat = np.zeros(self.G, np.int8)
        cat[np.asarray(np.load(ptc / "tok_item.npy", mmap_mode="r"))] = np.asarray(np.load(ptc / "tok_cat.npy", mmap_mode="r"))
        for ci, cc in self.cand_cat:
            cat[ci] = cc
        self.cat = torch.from_numpy(cat).to(self.dev)
        self.cat_dir = corpus_dir

    def set_slot(self, min_cnt, counts, ptc):
        cnt = np.load(ptc / "item_cnt.npy") if counts is None else counts[0]
        slot = np.zeros(self.G, np.int32)
        ids = np.where(cnt >= min_cnt)[0]
        slot[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
        self.slot = torch.from_numpy(slot).to(self.dev)
        return len(ids)


def load_model(ckpt, dev, mode, tables):
    ck = torch.load(PT / f"ckpt_{ckpt}.pt", map_location="cpu", mmap=True, weights_only=False)
    ca, sd = ck["args"], ck["model"]
    heads = ["fe", "new"] if ca["w_new"] > 0 else ["fe"]
    tables.set_corpus(ca.get("corpus_dir", ""))
    if mode == "noid":
        model = SeqEncoder(0, d=ca["d"], nl=ca["nl"], nh=ca["nh"], d_id=ca["d_id"], drop=ca["drop"], heads=heads, id_bias=False)
        sd = {k: v for k, v in sd.items() if not k.startswith(("id_emb.", "id_proj.", "id_bias."))}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected and not missing, (missing, unexpected)
        n_id = 0
    else:
        ptc = Path(ca["corpus_dir"]) if ca.get("corpus_dir") else PT
        counts = counts_excluding_test(ptc) if ca.get("exclude_test") else None
        n_id = tables.set_slot(ca["min_cnt"], counts, ptc)
        assert sd["id_emb.weight"].shape[0] == n_id + 1, (sd["id_emb.weight"].shape, n_id)
        model = SeqEncoder(n_id, d=ca["d"], nl=ca["nl"], nh=ca["nh"], d_id=ca["d_id"], drop=ca["drop"], heads=heads,
                           id_bias=bool(ca.get("id_bias", 0)))
        model.load_state_dict(sd)
    model.to(dev).eval()
    return model, ca, n_id


def feed_set(split, k, sample):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    if split == "train":
        feeds = np.load(PT / f"win{k}/window_feeds.npy")
        if sample:
            feeds = np.sort(np.random.default_rng(0).choice(feeds, min(sample, len(feeds)), replace=False))
    else:
        feeds = np.arange(len(f))
    H = f["H"].to_numpy().astype(np.int64)[feeds]
    hoff = f["hist_off"].to_numpy().astype(np.int64)[feeds]
    M = f["M"].to_numpy().astype(np.int64)[feeds]
    coff = f["cand_off"].to_numpy().astype(np.int64)[feeds]
    rows = np.repeat(coff - np.concatenate([[0], np.cumsum(M)[:-1]]), M) + np.arange(M.sum())
    if split == "train":  # the stored alignment of the live construction
        ref = np.load(PT / f"win{k}/cand_rows{'_s' + str(sample) if sample else ''}.npy")
        assert np.array_equal(rows, ref), "candidate rows differ from the stored cand_rows"
    return feeds, H, hoff, M, rows


@torch.no_grad()
def score(model, tables, split, H, hoff, M, rows, dev, mode, chunk=20_000):
    hist = np.load(FLAT / split / "hist_item_id.npy", mmap_mode="r" if split != "train" else None)
    c_item_all = np.load(FLAT / split / "cand_item_id.npy", mmap_mode="r")
    out = np.empty(len(rows), np.float32)
    cpos = np.concatenate([[0], np.cumsum(M)]).astype(np.int64)
    kk = np.arange(LAST_K)
    zero_slot = None
    for i in range(0, len(H), chunk):
        j = min(i + chunk, len(H))
        h, ho = H[i:j], hoff[i:j]
        hm = kk[None, :] < h[:, None]                                        # newest LAST_K history items (raw storage is newest-first)
        hidx = np.where(hm, ho[:, None] + kk[None, :], 0)
        hit = torch.from_numpy(np.asarray(hist[hidx.ravel()]).reshape(hidx.shape).astype(np.int64)).to(dev)
        hmask = torch.from_numpy(hm).to(dev)
        hit = torch.where(hmask, hit, 0)
        c_item = torch.from_numpy(np.asarray(c_item_all[rows[cpos[i]:cpos[j]]]).astype(np.int64)).to(dev)
        fi = torch.from_numpy(np.repeat(np.arange(j - i), M[i:j])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "noid":
                if zero_slot is None or zero_slot.shape != hit.shape:
                    zero_slot = torch.zeros_like(hit)
                rep_h = model.item_rep(tables.tab[hit], tables.has[hit] & hmask, tables.cat[hit].long(), zero_slot)
                rep_c = model.item_rep(tables.tab[c_item], tables.has[c_item], tables.cat[c_item].long(), torch.zeros_like(c_item))
            else:
                rep_h = model.item_rep(tables.tab[hit], tables.has[hit] & hmask, tables.cat[hit].long(), tables.slot[hit].long())
                rep_c = model.item_rep(tables.tab[c_item], tables.has[c_item], tables.cat[c_item].long(), tables.slot[c_item].long())
        HV = model.out_vec(rep_h.float()) * hmask[..., None]
        v = model.out_vec(rep_c.float())
        s = torch.einsum("nd,nkd->nk", v, HV[fi]).masked_fill(~hmask[fi], -2.0)
        out[cpos[i]:cpos[j]] = s.max(1).values.cpu().numpy().astype(np.float32)
    return out


def save_new(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        np.save(f, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="noid", choices=["noid", "id"])
    ap.add_argument("--jobs", nargs="+", default=["all"], help="tag:split[:k[:s60000]] ... or all")
    ap.add_argument("--chunk", type=int, default=20_000)
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("4", "5"), "ENCODER-DOSE: GPUs N only, CUDA_VISIBLE_DEVICES = one id"
    jobs = ALL_JOBS if a.jobs == ["all"] else a.jobs
    dev = torch.device("cuda:0")
    tables = Tables(dev, "item_tab2")
    # group by checkpoint (one load each)
    by_ck = {}
    for jb in jobs:
        parts = jb.split(":")
        tag, split = parts[0], parts[1]
        k = int(parts[2]) if len(parts) > 2 else 0
        sample = int(parts[3][1:]) if len(parts) > 3 else 0
        by_ck.setdefault(TAGS[tag][0], []).append((tag, split, k, sample))
    t_all = time.time()
    for ckpt, jl in by_ck.items():
        t0 = time.time()
        model, ca, n_id = load_model(ckpt, dev, a.mode, tables)
        print(f"[model] ckpt_{ckpt} mode {a.mode} n_id {n_id} corpus {ca.get('corpus_dir')} {time.time() - t0:.0f}s", flush=True)
        for tag, split, k, sample in jl:
            t1 = time.time()
            feeds, H, hoff, M, rows = feed_set(split, k, sample)
            x = score(model, tables, split, H, hoff, M, rows, dev, a.mode, a.chunk)
            assert np.isfinite(x).all() and x.min() >= -2.0 and x.max() <= 1.0 + 1e-4, (x.min(), x.max())
            sfx = (f"win{k}" + (f"_s{sample}" if sample else "")) if split == "train" else split
            name = f"pre_{tag}_hmax32{'_noid' if a.mode == 'noid' else '_idcheck'}_{sfx}.npy"
            if a.mode == "noid":
                dst = (WS if split == "train" else PREDS) / name
                save_new(dst, x)
            else:  # verification against the stored ID-aware hmax32 of the same tag
                dst = SCR / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.save(dst, x)
                ref_p = (WS / f"pre_{tag}_hmax32_{sfx}.npy") if split == "train" else (PREDS / f"pre_{tag}_hmax32_{sfx}.npy")
                ref = np.load(ref_p)
                d = np.abs(x - ref)
                print(f"  VERIFY {tag} {sfx}: max |diff| {d.max():.3e} mean |diff| {d.mean():.3e} corr {np.corrcoef(x, ref)[0, 1]:.6f} "
                      f"share |diff| > 1e-2: {(d > 1e-2).mean():.5f}; sentinel -2 agree {((x == -2) == (ref == -2)).mean():.6f}", flush=True)
            print(f"  [done] {tag} {sfx}: {len(feeds)} feeds, {len(rows)} cands, mean {x[x > -2].mean():.4f}, no-history {np.mean(x == -2):.4f}, "
                  f"{time.time() - t1:.0f}s -> {dst}", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"[all done] {len(jobs)} jobs in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
