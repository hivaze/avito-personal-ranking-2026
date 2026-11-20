"""GPU-resident data pipeline for the neural ranker.

Every split (train / val / test) is loaded once into GPU memory:
  * per-candidate arrays (emb_idx, category, f1..f5, target, pos)
  * per-history-event arrays (emb_idx, category, event_type, event_source)
  * per-feed arrays (offsets, M, H, user embedding)
  * the split-specific int8 item-embedding table (train: 81.6M x 128 = 10.4 GB)

A batch is a set of feed indices; `SplitData.gather` builds padded [B, Mmax] candidate
and [B, Kmax] history tensors directly on the GPU (no CPU dataloader at all).
History is stored newest-first, so the first K events of a feed are its K most recent ones.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path("<repo>")
FLAT = ROOT / "data" / "flat"
COMP = ROOT / "data" / "competition"
FEATS = Path(os.environ.get("NNX_FEATS", "<repo>/data/feats"))  # fine-tune extras (frozen snapshot)


def _np(split, name):
    return np.load(FLAT / split / f"{name}.npy")


def load_item_table(split):
    d = COMP / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.fromfile(d / "data.memmap", dtype=np.int8).reshape(meta["shape"])


class SplitData:
    """All tensors of one split on `device`. Candidate/history order == flat file order."""

    def __init__(self, split, device, max_hist=256, verbose=True):
        t0 = time.time()
        self.split = split
        self.device = device
        self.max_hist = max_hist
        f = pl.read_parquet(FLAT / split / "feeds.parquet")
        self.feeds = f
        self.n = len(f)
        self.M_np = f["M"].to_numpy().astype(np.int64)
        self.H_np = f["H"].to_numpy().astype(np.int64)
        self.cand_off_np = f["cand_off"].to_numpy().astype(np.int64)
        self.user_id = f["user_id"].to_numpy()
        self.dom = None  # optional per-feed domain mixture [n,3] (set by train.py / finetune.py)
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.M = g(self.M_np)
        self.H = g(self.H_np)
        self.cand_off = g(self.cand_off_np)
        self.hist_off = g(f["hist_off"].to_numpy().astype(np.int64))
        self.user_emb = g(_np(split, "user_emb"))  # int8 [n,128]
        self.user0 = g(self.user_id == 0)  # pooled unknown-user bucket
        # candidates
        self.c_emb = g(_np(split, "cand_emb_idx"))  # int32
        self.c_item = None  # global item ids (loaded on demand by load_item_ids, for the item-ID residual)
        self.h_item = None
        self.xt = None  # optional train-stage extra features [n_cand, F] fp16 (attach_xt)
        self.n_cand = len(self.c_emb)
        self.c_cat = g(_np(split, "cand_category_id").astype(np.int64)).to(torch.int16)
        self.c_f = torch.stack([g(_np(split, f"cand_f{i}")) for i in range(1, 6)], dim=1)  # fp32 [N,5]
        self.has_target = (FLAT / split / "cand_target.npy").exists()
        self.target_np = _np(split, "cand_target").astype(np.float32) if self.has_target else None
        self.c_tgt = g(self.target_np.astype(np.int8)) if self.has_target else None
        self.has_pos = (FLAT / split / "cand_pos.npy").exists()
        self.c_pos = g(_np(split, "cand_pos")) if self.has_pos else None  # int16
        # history
        self.h_emb = g(_np(split, "hist_emb_idx"))  # int32
        self.h_cat = g(_np(split, "hist_category_id").astype(np.int16))
        self.h_et = g(_np(split, "hist_event_type"))  # int8 1..10
        self.h_es = g(_np(split, "hist_event_source"))  # int8 0..26
        self.n_hist = len(self.h_emb)
        # item table
        self.table = g(load_item_table(split))  # int8 [n_items,128]
        torch.cuda.synchronize(device)
        if verbose:
            print(f"[data] {split}: {self.n} feeds, {self.n_cand} cands, {self.n_hist} hist, "
                  f"table {tuple(self.table.shape)}, {time.time() - t0:.0f}s, "
                  f"gpu mem {torch.cuda.memory_allocated(device) / 2**30:.1f} GB", flush=True)

    # ------------------------------------------------------------------ batching
    def batches(self, feed_ids, cand_budget=32768, max_feeds=512, shuffle=True, rng=None, mpad=16):
        """Split feeds into batches of similar M. Feeds are bucketed by M rounded up to a multiple of `mpad`;
        each bucket is cut into batches of B = min(max_feeds, cand_budget // Mpad) feeds. Fixed shapes per
        bucket matter: every new (B, M, K) shape costs ~0.25 s of kernel selection on the GPU."""
        feed_ids = np.asarray(feed_ids)
        M = self.M_np[feed_ids]
        Mp = (M + mpad - 1) // mpad * mpad
        out = []
        for mp in np.unique(Mp):
            ids = feed_ids[Mp == mp]
            if shuffle:
                ids = rng.permutation(ids)
            B = int(max(1, min(max_feeds, cand_budget // mp)))
            out += [ids[i:i + B] for i in range(0, len(ids), B)]
        if shuffle:
            perm = rng.permutation(len(out))
            out = [out[k] for k in perm]
        return out

    @torch.no_grad()
    def gather(self, fidx_np, with_pos=False):
        """Build padded GPU tensors for a batch of feeds (numpy int array of feed indices)."""
        dev = self.device
        fidx = torch.from_numpy(np.asarray(fidx_np, dtype=np.int64)).to(dev, non_blocking=True)
        M = self.M[fidx]
        Mx = int(-(-self.M_np[fidx_np].max() // 16) * 16)  # pad to a multiple of 16 (fewer distinct shapes)
        ar = torch.arange(Mx, device=dev)
        cmask = ar[None, :] < M[:, None]
        ci = self.cand_off[fidx][:, None] + ar[None, :]
        ci = torch.where(cmask, ci, self.cand_off[fidx][:, None])
        cemb_idx = self.c_emb[ci]
        cmiss = cemb_idx < 0
        ce_raw = self.table[cemb_idx.clamp_min(0).long()].float()  # [B,M,128]
        ce_raw[cmiss] = 0.0
        H = self.H[fidx].clamp_max(self.max_hist)
        Hx = int(np.minimum(self.H_np[fidx_np], self.max_hist).max())
        Hx = min(max(-(-Hx // 64) * 64, 64), self.max_hist)  # pad to a multiple of 64
        ah = torch.arange(Hx, device=dev)
        hmask = ah[None, :] < H[:, None]
        hi = self.hist_off[fidx][:, None] + ah[None, :]
        hi = torch.where(hmask, hi, self.hist_off[fidx][:, None]).clamp_max(self.n_hist - 1)
        hemb_idx = self.h_emb[hi]
        hmiss = (hemb_idx < 0) | ~hmask
        he_raw = self.table[hemb_idx.clamp_min(0).long()].float()
        he_raw[hmiss] = 0.0
        b = dict(user0=self.user0[fidx],
            fidx=fidx, M=M, H=H, cmask=cmask, ci=ci, hmask=hmask,
            ce_raw=ce_raw, cmiss=cmiss, ccat=self.c_cat[ci].long(), cf=self.c_f[ci],
            he_raw=he_raw, hemb_ok=~hmiss, hcat=self.h_cat[hi].long(),
            het=self.h_et[hi].long(), hes=self.h_es[hi].long(),
            ue_raw=self.user_emb[fidx].float(),
        )
        if self.has_target:
            b["y"] = self.c_tgt[ci].float() * cmask
        if self.dom is not None:
            b["dom"] = self.dom[fidx]
        if self.xt is not None:
            b["xt"] = self.xt[ci].float()
        if self.c_item is not None:
            b["citem"] = self.c_item[ci]
            b["hitem"] = self.h_item[hi]
        if getattr(self, "xe", None) is not None:  # optional extra precomputed features [n_cand, F]
            b["xe"] = self.xe[ci]
        if with_pos and self.has_pos:
            b["pos"] = self.c_pos[ci].long()
        return b

    def load_item_ids(self):
        """Global item ids of candidates and history events (for the hashed item-ID residual)."""
        self.c_item = torch.from_numpy(_np(self.split, "cand_item_id")).to(self.device)
        self.h_item = torch.from_numpy(_np(self.split, "hist_item_id")).to(self.device)

    def attach_xt(self, names):
        """Train-stage extra per-candidate features (must exist, leak-safe, for train/val/test), stored as fp16.
        Fixed split-independent transform: *_rk raw, *_z clipped to +-5, seq_cv*/it_hist* counts log1p(max(x, 0)),
        others clipped to +-10; "has:<name>" = non-NaN indicator of <name>; NaN -> 0."""
        cols = []
        for n in names:
            if n.startswith("has:"):  # availability indicator of another feature (non-NaN)
                x = (~np.isnan(np.load(ROOT / "data" / "feats" / self.split / f"{n[4:]}.npy"))).astype(np.float32)
            else:
                x = np.load(ROOT / "data" / "feats" / self.split / f"{n}.npy").astype(np.float32)
            if n.startswith("has:") or n.endswith("_rk"):
                pass
            elif n.endswith("_z"):
                x = np.clip(x, -5, 5)
            elif n.startswith(("seq_cv", "it_hist")):  # counts
                x = np.log1p(np.maximum(x, 0))
            else:  # already on a sane scale (e.g. ctx_ix_* means / logs / differences)
                x = np.clip(x, -10, 10)
            cols.append(torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(self.device, torch.float16))
            del x
        self.xt = torch.stack(cols, 1)

    def attach_extra(self, names, stats=None):
        """Load precomputed per-candidate features data/feats/<split>/<name>.npy onto the GPU as self.xe.
        Count-like features (min>=0, max>20) get log1p; then standardization with `stats` (mean, std, logmask)
        computed on this split if not given. Returns the stats so other splits can reuse them."""
        cols = [np.load(FEATS / self.split / f"{n}.npy").astype(np.float32) for n in names]
        X = np.stack(cols, 1)
        if stats is None:
            logm = (np.nanmin(X, 0) >= 0) & (np.nanmax(X, 0) > 20)
        else:
            logm = stats[2]
        X[:, logm] = np.log1p(np.maximum(X[:, logm], 0))
        if stats is None:
            stats = (np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6, logm)
        X = np.clip((X - stats[0]) / stats[1], -6, 6)
        X = np.nan_to_num(X, nan=0.0)
        self.xe = torch.from_numpy(X.astype(np.float32)).to(self.device)
        return stats

    def scatter(self, preds_flat, b, logits):
        """Write padded logits [B,M] back into a flat (cand-aligned) torch buffer."""
        m = b["cmask"]
        preds_flat[b["ci"][m]] = logits[m].float()
