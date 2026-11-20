"""CPU-only miniature of nn_x/r6/data.SplitData for unit tests: the same tensors and the same gather/batches/scatter
methods (bound from SplitData), built by memory-mapped slicing of a few hundred feeds, so a real train batch costs seconds
and a few hundred MB instead of loading the 247M-candidate split. Read-only on data/flat, data/competition, nn/feat_cache.
"""
import os
import sys

import numpy as np
import polars as pl
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.dirname(HERE)
sys.path.insert(0, R6)
import data as D  # noqa: E402  (nn_x/r6/data.py)

ROOT = "<repo>"


def _mm(split, name):
    return np.load(f"{ROOT}/data/flat/{split}/{name}.npy", mmap_mode="r")


def _xt_col(split, n, rows):
    """Same fixed transform as data.SplitData.attach_xt, on the selected candidate rows only."""
    if n.startswith("has:"):
        x = (~np.isnan(np.load(D.feat_path(split, n[4:]), mmap_mode="r")[rows])).astype(np.float32)
    else:
        x = np.asarray(np.load(D.feat_path(split, n), mmap_mode="r")[rows]).astype(np.float32)
    if n.startswith("has:") or n.endswith("_rk"):
        pass
    elif n.endswith("_z"):
        x = np.clip(x, -5, 5)
    elif n.startswith(("seq_cv", "it_hist")):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16)


class MiniSplit:
    gather = D.SplitData.gather
    batches = D.SplitData.batches
    scatter = D.SplitData.scatter

    def __init__(self, split, feeds, xt_names=None, max_hist=256, with_domain=True):
        self.split, self.device, self.max_hist = split, torch.device("cpu"), max_hist
        f = pl.read_parquet(f"{ROOT}/data/flat/{split}/feeds.parquet")
        feeds = np.sort(np.asarray(feeds))
        self.src_feeds = feeds
        M = f["M"].to_numpy().astype(np.int64)[feeds]
        H = f["H"].to_numpy().astype(np.int64)[feeds]
        coff = f["cand_off"].to_numpy().astype(np.int64)[feeds]
        hoff = f["hist_off"].to_numpy().astype(np.int64)[feeds]
        Hk = np.minimum(H, max_hist)  # gather never reads past the first max_hist events of a feed
        crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, M)])
        hrow = np.concatenate([np.arange(o, o + h) for o, h in zip(hoff, Hk)] + [np.zeros(0, np.int64)])
        if len(hrow) == 0:
            hrow = np.zeros(1, np.int64)  # keep a non-empty history array (gather clamps to n_hist - 1)
        self.n = len(feeds)
        self.M_np, self.H_np = M, H
        self.cand_off_np = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
        hist_off_np = np.concatenate([[0], np.cumsum(Hk)[:-1]]).astype(np.int64)
        uid = f["user_id"].to_numpy()[feeds]
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
        self.M, self.H, self.cand_off, self.hist_off = g(M), g(H), g(self.cand_off_np), g(hist_off_np)
        self.user_emb = g(np.asarray(_mm(split, "user_emb")[feeds]))
        self.user0 = g(uid == 0)
        cemb = np.asarray(_mm(split, "cand_emb_idx")[crow]).astype(np.int64)
        hemb = np.asarray(_mm(split, "hist_emb_idx")[hrow]).astype(np.int64)
        used = np.unique(np.concatenate([cemb[cemb >= 0], hemb[hemb >= 0]]))
        table = D.load_item_table(split)  # int8 memmap [n_items, 128]
        self.table = g(np.asarray(table[used]))
        remap = lambda e: np.where(e >= 0, np.searchsorted(used, np.maximum(e, 0)), -1).astype(np.int32)  # noqa
        self.c_emb, self.h_emb = g(remap(cemb)), g(remap(hemb))
        self.n_cand, self.n_hist = len(crow), len(hrow)
        self.c_cat = g(np.asarray(_mm(split, "cand_category_id")[crow]).astype(np.int64)).to(torch.int16)
        self.c_f = torch.stack([g(np.asarray(_mm(split, f"cand_f{i}")[crow])) for i in range(1, 6)], 1)
        self.has_target = os.path.exists(f"{ROOT}/data/flat/{split}/cand_target.npy")
        self.target_np = np.asarray(_mm(split, "cand_target")[crow]).astype(np.float32) if self.has_target else None
        self.c_tgt = g(self.target_np.astype(np.int8)) if self.has_target else None
        self.has_pos = os.path.exists(f"{ROOT}/data/flat/{split}/cand_pos.npy")
        self.c_pos = g(np.asarray(_mm(split, "cand_pos")[crow])) if self.has_pos else None
        self.h_cat = g(np.asarray(_mm(split, "hist_category_id")[hrow]).astype(np.int16))
        self.h_et = g(np.asarray(_mm(split, "hist_event_type")[hrow]))
        self.h_es = g(np.asarray(_mm(split, "hist_event_source")[hrow]))
        self.c_item = self.h_item = None
        self.xt = torch.stack([_xt_col(split, n, crow) for n in xt_names], 1) if xt_names else None
        self.dom = None
        self.prand = None
        if split == "train":  # p_random exactly as train.py (M<50 -> 0), domain mixture as train.py sets tr.dom
            pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"]
                               .astype(np.float32), nan=0.0)[feeds]
            pr[M < 50] = 0.0
            self.prand = torch.from_numpy(pr)
            if with_domain:
                self.dom = torch.stack([1 - self.prand, self.prand, torch.zeros_like(self.prand)], 1)

    def to(self, device):
        """Move every tensor to `device` (GPU-mode tests); gather/batches then build the batches on that device."""
        for k, v in list(vars(self).items()):
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        self.device = torch.device(device)
        return self


def pick_train_feeds(n_each=48, seed=0):
    """Stratified real train feeds: random-like M>=50, production M>=50, M<50, M>200, H==0 and user0 feeds."""
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M", "H", "user_id"])
    M, H, uid = f["M"].to_numpy(), f["H"].to_numpy(), f["user_id"].to_numpy()
    pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32),
                       nan=0.0)
    pr[M < 50] = 0.0
    rng = np.random.default_rng(seed)
    groups = [np.where((pr > 0.9) & (M >= 50))[0], np.where((pr < 0.1) & (M >= 50) & (M <= 200))[0],
              np.where(M < 50)[0], np.where(M > 200)[0], np.where(H == 0)[0], np.where(uid == 0)[0]]
    out = np.unique(np.concatenate([rng.choice(g_, min(n_each, len(g_)), replace=False) for g_ in groups]))
    return out
