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
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path("<repo>")
FLAT = ROOT / "data" / "flat"
COMP = ROOT / "data" / "competition"


FEAT_CACHE = ROOT / "nn_x" / "w5" / "feat_cache"  # symlink -> ../../nn/feat_cache (same files, read-only use)


def feat_path(split, name):
    """Path of a precomputed feature in the shared hard-linked snapshot nn/feat_cache/<split>/ (via the w5 symlink).
    nn_x/w5 copy: read-only, never creates links or copies inside the shared feat_cache."""
    c = FEAT_CACHE / split / f"{name}.npy"
    if not c.exists():
        raise FileNotFoundError(f"{c} missing (nn_x/w5 never writes into the shared feat_cache)")
    return c


def feat_exists(split, name):
    return (FEAT_CACHE / split / f"{name}.npy").exists()


def _np(split, name):
    return np.load(FLAT / split / f"{name}.npy")


def load_item_table(split):
    d = COMP / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.fromfile(d / "data.memmap", dtype=np.int8).reshape(meta["shape"])


HIST_EXT_DIR = "<repo>/artifacts/data/feats_new/histext_1117"  # gap1117 lifelong (s3_build_arrays.py)


def hist_ext_kwargs(cfg):
    """gap1117 lifelong: SplitData keyword arguments for a model cfg ({} when the model has no extension tokens)."""
    if not cfg.get("hist_ext"):
        return {}
    src = cfg.get("hist_ext_src") or {}
    return dict(hist_ext=int(cfg["hist_ext"]), hist_ext_dir=src.get("dir", HIST_EXT_DIR), hist_ext_var=src.get("var", ""),
                hist_ext_uncapped=bool(src.get("uncapped", False)))


class SplitData:
    """All tensors of one split on `device`. Candidate/history order == flat file order.
    gap1117 lifelong (OFF unless hist_ext > 0): per feed up to hist_ext events OLDER than the feed's oldest stored event,
    recovered from the same user's most recent strictly earlier snapshot (leak-safe build: lab/nn_x/w12/gap1117/lifelong/);
    gather() then returns them as separate x* keys (xe_raw, xemb_ok, xmask, xcat, xet, xes, xidx, Hx) placed at positions
    max_hist .. max_hist + Kx - 1; the stored keys (he_raw, hmask, H, ...) and hence every dense input are unchanged."""

    def __init__(self, split, device, max_hist=256, verbose=True, hist_ext=0, hist_ext_dir=HIST_EXT_DIR, hist_ext_var="",
                 hist_ext_uncapped=False):
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
        self.hist_ext = 0
        if hist_ext:  # gap1117 lifelong; nothing below runs when it is off
            self._load_hist_ext(hist_ext, hist_ext_dir, hist_ext_var, hist_ext_uncapped, g)
        if torch.device(device).type == "cuda":  # nn_x/r6: CPU-safe (unit tests build splits on the CPU)
            torch.cuda.synchronize(device)
        if verbose:
            print(f"[data] {split}: {self.n} feeds, {self.n_cand} cands, {self.n_hist} hist, "
                  f"table {tuple(self.table.shape)}, {time.time() - t0:.0f}s, "
                  f"gpu mem {torch.cuda.memory_allocated(device) / 2**30 if torch.device(device).type == 'cuda' else 0:.1f} GB",
                  flush=True)

    def _load_hist_ext(self, kmax, root, var, uncapped, g):
        """gap1117 lifelong: extension arrays of this split (feed level ext_n / ext_off, event level ext_*; ext_table.npy rows
        are appended to the item table: ext_emb_idx >= n_table points at them)."""
        assert self.max_hist == 256, "--hist_ext needs the stored history at its full 256 slots (max_hist 256)"
        d = Path(root) / self.split / var if var else Path(root) / self.split
        meta = json.loads((d / "meta.json").read_text())
        n = np.load(d / "ext_n.npy").astype(np.int64)
        assert len(n) == self.n and meta["n_table"] == len(self.table), (meta, self.n, len(self.table))
        if not uncapped:  # default: extension only behind a full stored history (H == 256): the "beyond the cap" events
            n = np.where(self.H_np == self.max_hist, n, 0)
        n = np.minimum(n, kmax)
        self.x_n_np = n
        self.x_n = g(n)
        self.x_off = g(np.load(d / "ext_off.npy").astype(np.int64))
        self.x_emb = g(np.load(d / "ext_emb_idx.npy"))  # int32
        self.x_cat = g(np.load(d / "ext_category_id.npy").astype(np.int16))
        self.x_et = g(np.load(d / "ext_event_type.npy"))
        self.x_es = g(np.load(d / "ext_event_source.npy"))
        self.n_x = max(len(self.x_emb), 1)
        if len(self.x_emb) == 0:  # keep gathers valid on an empty split
            self.x_emb, self.x_cat = g(np.full(1, -1, np.int32)), g(np.zeros(1, np.int16))
            self.x_et, self.x_es = g(np.zeros(1, np.int8)), g(np.zeros(1, np.int8))
        ext_tab = np.load(Path(root) / self.split / "ext_table.npy")
        if len(ext_tab):  # items absent from this split's table: the source split's vectors
            self.table = torch.cat([self.table, g(ext_tab)], 0)
        self.hist_ext = int(kmax)
        print(f"[data] {self.split}: hist_ext {kmax} from {d}: {int((n > 0).sum())} feeds extended "
              f"({(n > 0).mean():.4f}), {int(n.sum())} events, +{len(ext_tab)} item rows", flush=True)

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
        Kx = 0
        if self.hist_ext:  # gap1117 lifelong: extension tokens sit at positions max_hist.., so the stored block is full width
            Kx = int(self.x_n_np[fidx_np].max())
            if Kx > 0:
                Hx = self.max_hist
                Kx = min(-(-Kx // 64) * 64, self.hist_ext)  # pad to a multiple of 64 (fewer distinct shapes)
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
            hidx=hemb_idx,  # nn_x/r6: item-table row of each history event (-1 = none); read only by --retr_uniq
        )
        if Kx > 0:  # gap1117 lifelong: the extension block, newest first (event 0 = just older than the stored 256th)
            xn = self.x_n[fidx]
            ax = torch.arange(Kx, device=dev)
            xmask = ax[None, :] < xn[:, None]
            xi = self.x_off[fidx][:, None] + ax[None, :]
            xi = torch.where(xmask, xi, torch.zeros_like(xi)).clamp_max(self.n_x - 1)
            xemb_idx = self.x_emb[xi]
            xmiss = (xemb_idx < 0) | ~xmask
            xe_raw = self.table[xemb_idx.clamp_min(0).long()].float()
            xe_raw[xmiss] = 0.0
            b.update(xe_raw=xe_raw, xemb_ok=~xmiss, xmask=xmask, xcat=self.x_cat[xi].long(), xet=self.x_et[xi].long(),
                     xes=self.x_es[xi].long(), xidx=torch.where(xmask, xemb_idx, torch.full_like(xemb_idx, -1)), Hx=xn)
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
                x = (~np.isnan(np.load(feat_path(self.split, n[4:])))).astype(np.float32)
            else:
                x = np.load(feat_path(self.split, n)).astype(np.float32)
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
        cols = [np.load(feat_path(self.split, n)).astype(np.float32) for n in names]
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
