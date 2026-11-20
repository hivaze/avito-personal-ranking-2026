"""Data layer for the soft-token LLM ranker (nn_x/llm). Device-agnostic, read-only.

It reproduces nn/data.SplitData exactly (same flat arrays, same gather() tensors, same attach_xt transform), with
three differences that matter for the LLM test:
  * it can hold an arbitrary SUBSET of a split's feeds (e.g. the 148.6K matched train feeds + the 30K holdout),
    with a compact item-embedding table, so a 150K-feed run needs ~5 GB instead of ~35-45 GB of device memory;
  * it works on CPU (no torch.cuda calls), so it can be smoke-tested with CUDA_VISIBLE_DEVICES="";
  * it NEVER writes anything (nn/data.feat_path hard-links missing features into nn/feat_cache; here a missing
    feature is an error).
nn/model.py is imported read-only (dense_features, listwise_loss, N_DENSE). nn/common.py is NOT imported (its GPU
guard raises without CUDA_VISIBLE_DEVICES in 4..7); the few helpers needed are copied below.
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path("<repo>")
FLAT = ROOT / "data" / "flat"
COMP = ROOT / "data" / "competition"
FEAT_CACHE = ROOT / "nn" / "feat_cache"
FEATS = ROOT / "data" / "feats"
N_TRAIN = 3151208
HOLDOUT_SEED = 12345

sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg, ndcg_per_feed  # noqa: E402,F401


def _import_nn_model():
    """nn/model.py as module 'nn_ranker_model' (read-only; no sys.path pollution with nn/'s data/common modules)."""
    name = "nn_ranker_model"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "nn" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nnm = _import_nn_model()
dense_features, listwise_loss, N_DENSE = nnm.dense_features, nnm.listwise_loss, nnm.N_DENSE

# --------------------------------------------------------------------------- train-stage input presets (copied)
_CVRAW = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo",
          "it_hist_et3_luo", "it_hist_et4_luo"]
_CVND = _CVRAW + ["ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u",
                  "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w",
                  "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual",
                  "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]
XT_PRESETS = {"cvnd": _CVND,
              "cvnd_pre2s": _CVND + ["pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32",
                                     "has:pre2_fe"]}
XT_REL = {"cvnd": ["seq_cv_invd20", "it_hist_all_luo"],
          "cvnd_pre2s": ["seq_cv_invd20", "it_hist_all_luo", "pre2_fe_s", "pre2_new_s", "pre2_hmax32"]}
XT_RELONLY = {"cvnd_pre2s": ["pre2_fe_s", "pre2_new_s", "pre2_hmax32"]}
# never allowed as LLM inputs (CURRENT_STATUS 2/3f/3k-P6, FEATURES_REVIEW): leaky or quarantined
BANNED_PREFIXES = ("pre_", "x_lh_", "lh_")
BANNED_SUBSTR = ("_id_contrib",)


def nn_train_presets():
    """Evaluate the XT_* dict literals of nn/train.py WITHOUT importing it (it imports nn/common.py)."""
    src = (ROOT / "nn" / "train.py").read_text()
    env = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            base = tgt.value if isinstance(tgt, ast.Subscript) else tgt
            if isinstance(base, ast.Name) and base.id.startswith("XT_"):
                exec(compile(ast.Module([node], []), "nn/train.py", "exec"), env)
    return env["XT_PRESETS"], env["XT_REL"], env["XT_RELONLY"]


def xt_config(preset):
    """(names, xt_rel idx, xt_idcols idx, xt_relonly idx, n_dense) exactly as nn/train.py builds them (with --user0)."""
    names = list(XT_PRESETS[preset])
    for n in names:
        b = n.removeprefix("has:")
        assert not b.startswith(BANNED_PREFIXES) and not any(s in b for s in BANNED_SUBSTR), f"banned input {n}"
    rel = [names.index(n) for n in XT_REL.get(preset, []) if n in names] or None
    idcols = [i for i, n in enumerate(names) if n.startswith(("seq_cv", "it_hist"))] or None
    relonly = [names.index(n) for n in XT_RELONLY.get(preset, []) if n in names] or None
    n_dense = N_DENSE + 2 + len(names) + 2 * len(rel or []) - len(relonly or [])
    return names, rel, idcols, relonly, n_dense


def feat_file(split, name):
    """Read-only lookup: the hard-linked snapshot in nn/feat_cache (what the ranker trained on), else data/feats."""
    for p in (FEAT_CACHE / split / f"{name}.npy", FEATS / split / f"{name}.npy"):
        if p.exists():
            return p
    raise FileNotFoundError(f"feature {split}/{name} not found (this module never creates feature files)")


def xt_transform(name, x):
    """nn/data.SplitData.attach_xt column transform (elementwise, so subset-then-transform == transform-then-subset)."""
    x = x.astype(np.float32)
    if name.startswith("has:") or name.endswith("_rk"):
        pass
    elif name.endswith("_z"):
        x = np.clip(x, -5, 5)
    elif name.startswith(("seq_cv", "it_hist")):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


# --------------------------------------------------------------------------- feed selection (as nn/train.py)
def train_meta():
    f = pl.read_parquet(FLAT / "train" / "feeds.parquet", columns=["M", "timestamp"])
    return f["M"].to_numpy().astype(np.int64), f["timestamp"].to_numpy()


def load_prandom(M_np):
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(len(M_np))).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[M_np < 50] = 0.0  # detector unreliable for short feeds (nn/train.py)
    return p


def holdout_feeds(prand_np, M_np, n=30000):
    cand = np.where((prand_np > 0.9) & (M_np >= 50))[0]
    return np.sort(np.random.default_rng(HOLDOUT_SEED).choice(cand, min(n, len(cand)), replace=False))


def matched_subset(seed, train_feeds, prand_np, M_np, holdout=30000):
    """Exactly nn/train.py: rng = default_rng(seed); feeds = sort(rng.choice(n, k)); feeds -= holdout.
    Returns (train feeds, holdout feeds, rng) -- rng continues as in nn/train.py (used for batch order)."""
    rng = np.random.default_rng(seed)
    feeds = np.arange(len(M_np))
    if train_feeds:
        feeds = np.sort(rng.choice(len(M_np), train_feeds, replace=False))
    hold = holdout_feeds(prand_np, M_np, holdout) if holdout else np.zeros(0, np.int64)
    feeds = np.setdiff1d(feeds, hold)
    return feeds, hold, rng


def _ranges(starts, lens):
    """Concatenation of arange(s, s+l) for all (s, l), vectorised."""
    lens = np.asarray(lens, np.int64)
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    rep = np.repeat(np.asarray(starts, np.int64) - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return rep + np.arange(tot, dtype=np.int64)


def _take(path, rows, full):
    mm = np.load(path, mmap_mode="r")
    return np.array(mm) if full else mm[rows]  # always a private writable copy (never a view of the memmap)


class FeedStore:
    """Tensors of (a subset of) one split on `device`, SplitData-compatible (gather/batches/scatter/attributes)."""

    def __init__(self, split, feed_ids=None, device="cpu", max_hist=256, xt_names=None, load_pos=True, verbose=True):
        import time
        t0 = time.time()
        self.split, self.device, self.max_hist = split, torch.device(device), max_hist
        f = pl.read_parquet(FLAT / split / "feeds.parquet")
        full = feed_ids is None
        ids = np.arange(len(f)) if full else np.sort(np.asarray(feed_ids, np.int64))
        assert full or len(np.unique(ids)) == len(ids)
        self.feed_ids, self.n = ids, len(ids)
        M = f["M"].to_numpy().astype(np.int64)[ids]
        H = f["H"].to_numpy().astype(np.int64)[ids]
        Hk = np.minimum(H, max_hist)  # gather() only ever reads the newest max_hist events
        self.M_np, self.H_np = M, H
        self.timestamp = f["timestamp"].to_numpy()[ids]
        self.user_id = f["user_id"].to_numpy()[ids]
        co = f["cand_off"].to_numpy().astype(np.int64)[ids]
        ho = f["hist_off"].to_numpy().astype(np.int64)[ids]
        crow = _ranges(co, M)  # original flat candidate rows (prediction files are aligned to these)
        hrow = _ranges(ho, Hk)
        if full:
            assert (crow == np.arange(len(crow))).all()
        self.orig_rows = crow
        self.cand_off_np = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
        hoff = np.concatenate([[0], np.cumsum(Hk)[:-1]]).astype(np.int64)
        self.n_cand = len(crow)
        D = FLAT / split
        c_emb = _take(D / "cand_emb_idx.npy", crow, full).astype(np.int64)
        h_emb = _take(D / "hist_emb_idx.npy", hrow, False).astype(np.int64)
        # compact item table (only the rows referenced by these feeds), indices remapped
        meta = json.loads((COMP / split / "item_embeddings" / "data.meta").read_text())
        tab = np.memmap(COMP / split / "item_embeddings" / "data.memmap", dtype=np.int8, mode="r",
                        shape=tuple(meta["shape"]))
        uniq = np.unique(np.concatenate([c_emb[c_emb >= 0], h_emb[h_emb >= 0]]))
        table = np.asarray(tab[uniq]) if len(uniq) else np.zeros((1, 128), np.int8)
        remap = lambda e: np.where(e >= 0, np.searchsorted(uniq, np.maximum(e, 0)), -1)  # noqa: E731
        c_emb, h_emb = remap(c_emb), remap(h_emb)
        # one dummy history row at the end: rows of padded/empty histories point inside the array (always masked)
        h_emb = np.concatenate([h_emb, [-1]])
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)  # noqa: E731
        self.M, self.H = g(M), g(H)
        self.cand_off, self.hist_off = g(self.cand_off_np), g(hoff)
        self.user_emb = g(_take(D / "user_emb.npy", ids, full))
        self.user0 = g(self.user_id == 0)
        self.c_emb = g(c_emb.astype(np.int32))
        self.c_cat = g(_take(D / "cand_category_id.npy", crow, full).astype(np.int16))
        self.c_f = torch.stack([g(_take(D / f"cand_f{i}.npy", crow, full)) for i in range(1, 6)], 1)
        self.has_target = (D / "cand_target.npy").exists()
        self.target_np = _take(D / "cand_target.npy", crow, full).astype(np.float32) if self.has_target else None
        self.c_tgt = g(self.target_np.astype(np.int8)) if self.has_target else None
        self.has_pos = load_pos and (D / "cand_pos.npy").exists()
        self.c_pos = g(_take(D / "cand_pos.npy", crow, full)) if self.has_pos else None
        self.h_emb = g(h_emb.astype(np.int32))
        pad1 = lambda a: np.concatenate([a, np.zeros(1, a.dtype)])  # noqa: E731
        self.h_cat = g(pad1(_take(D / "hist_category_id.npy", hrow, False).astype(np.int16)))
        self.h_et = g(pad1(_take(D / "hist_event_type.npy", hrow, False)))
        self.h_es = g(pad1(_take(D / "hist_event_source.npy", hrow, False)))
        self.n_hist = len(h_emb)
        self.table = g(table)
        self.dom = None
        self.xt, self.xt_names = None, None
        if xt_names:
            self.attach_xt(xt_names)
        if verbose:
            mb = sum(t.numel() * t.element_size() for t in self._tensors()) / 2**20
            print(f"[FeedStore] {split}: {self.n} feeds, {self.n_cand} cands, {self.n_hist - 1} hist, table "
                  f"{tuple(table.shape)}, {mb:.0f} MB on {self.device}, {time.time() - t0:.0f}s", flush=True)

    def _tensors(self):
        return [v for v in vars(self).values() if isinstance(v, torch.Tensor)]

    def attach_xt(self, names):
        cols = []
        for n in names:
            if n.startswith("has:"):
                x = ~np.isnan(_take(feat_file(self.split, n[4:]), self.orig_rows, False))
            else:
                x = _take(feat_file(self.split, n), self.orig_rows, False)
            cols.append(torch.from_numpy(xt_transform(n, x)).to(self.device, torch.float16))
        self.xt, self.xt_names = torch.stack(cols, 1), list(names)

    def set_domain(self, prand_split):
        """Train feeds: production soft-mixed toward random-like by p_random (nn/train.py --domain)."""
        p = torch.from_numpy(np.asarray(prand_split, np.float32)[self.feed_ids]).to(self.device)
        self.prand = p
        self.dom = torch.stack([1 - p, p, torch.zeros_like(p)], 1)

    def local(self, orig_feed_ids):
        loc = np.searchsorted(self.feed_ids, orig_feed_ids)
        assert (self.feed_ids[loc] == orig_feed_ids).all()
        return loc

    def rows_of(self, local_ids):
        local_ids = np.asarray(local_ids)
        return _ranges(self.cand_off_np[local_ids], self.M_np[local_ids])

    # ------------------------------------------------------------------ batching
    def batches(self, feed_ids, cand_budget=32768, max_feeds=512, shuffle=True, rng=None, mpad=16):
        """nn/data.SplitData.batches (copied): buckets by M rounded up to `mpad`."""
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

    def seq_len(self, feed_ids, n_pre=1):
        """Soft-token sequence length per feed: n_pre (1 user, or 2 = sink + user, amendment S1) + min(H, max_hist)
        history + M candidates."""
        return n_pre + np.minimum(self.H_np[feed_ids], self.max_hist) + self.M_np[feed_ids]

    def token_batches(self, feed_ids, token_budget=56000, max_feeds=1024, shuffle=True, rng=None, tpad=32, n_pre=1):
        """LLM batches by TOKEN budget: bucket feeds by sequence length rounded up to `tpad`, B = budget // Tpad."""
        feed_ids = np.asarray(feed_ids)
        Tp = (self.seq_len(feed_ids, n_pre) + tpad - 1) // tpad * tpad
        out = []
        for tp in np.unique(Tp):
            ids = feed_ids[Tp == tp]
            if shuffle:
                ids = rng.permutation(ids)
            B = int(max(1, min(max_feeds, token_budget // tp)))
            out += [ids[i:i + B] for i in range(0, len(ids), B)]
        if shuffle:
            perm = rng.permutation(len(out))
            out = [out[k] for k in perm]
        return out

    def token_batches_left(self, feed_ids, token_budget=56000, max_feeds=1024, shuffle=True, rng=None, n_pre=2,
                           hb=32, mb=16):
        """Token-budget batches for the LEFT-PADDED Qwen3.5 branch layout (qwen35_branch), where every row of a batch
        has n_pre + max(H) + max(M rounded up to 16) tokens (the prefix is left-padded to the batch's longest history,
        the candidate block to its widest feed). token_batches buckets by H + M only, so the branch rows of its
        batches are 1.64x the budget on the seed-23 subset (68.4M rows per epoch vs 41.7M). Here feeds are ordered by
        (H bucket of `hb`, H == max_hist its own bucket; M bucket of `mb`; random tie-break) and filled greedily while
        B x row <= token_budget; the batch order is shuffled. Same budget meaning (padded tokens of the trunk's own
        layout per batch) and the same feeds per step: seed 23 -> 757 steps/epoch, 196.3 feeds/step, 42.2M rows
        (token_batches: 758 steps, 196.0 feeds/step). Large-model arms only (reports/llm_large.md); the Qwen3 arms
        keep token_batches."""
        feed_ids = np.asarray(feed_ids)
        Hk = np.minimum(self.H_np[feed_ids], self.max_hist)
        M = self.M_np[feed_ids]
        kh = np.where(Hk >= self.max_hist, 1 << 20, (Hk + hb - 1) // hb)
        km = (M + mb - 1) // mb
        tie = rng.random(len(feed_ids)) if shuffle else np.arange(len(feed_ids))
        o = np.lexsort((tie, km, kh))
        ids, Hk, M16 = feed_ids[o], Hk[o].tolist(), (-(-M[o] // 16) * 16).tolist()
        out, start, mh, mm = [], 0, 0, 0
        for i in range(len(ids)):
            nh, nm = max(mh, Hk[i]), max(mm, M16[i])
            n = i - start + 1
            if n > 1 and (n * (n_pre + nh + nm) > token_budget or n > max_feeds):
                out.append(ids[start:i])
                start, nh, nm = i, Hk[i], M16[i]
            mh, mm = nh, nm
        if len(ids):
            out.append(ids[start:])
        if shuffle:
            perm = rng.permutation(len(out))
            out = [out[k] for k in perm]
        return out

    def row_tokens(self, feed_ids, n_pre=2):
        """Padded tokens of one left-layout batch (B x (n_pre + max H + max M rounded to 16))."""
        feed_ids = np.asarray(feed_ids)
        return int(len(feed_ids) * (n_pre + np.minimum(self.H_np[feed_ids], self.max_hist).max()
                                    + -(-self.M_np[feed_ids].max() // 16) * 16))

    @torch.no_grad()
    def gather(self, fidx_np, with_pos=False):
        """nn/data.SplitData.gather (copied, device-agnostic). fidx_np = LOCAL feed indices of this store."""
        dev = self.device
        fidx = torch.from_numpy(np.asarray(fidx_np, dtype=np.int64)).to(dev)
        M = self.M[fidx]
        Mx = int(-(-self.M_np[fidx_np].max() // 16) * 16)
        ar = torch.arange(Mx, device=dev)
        cmask = ar[None, :] < M[:, None]
        ci = self.cand_off[fidx][:, None] + ar[None, :]
        ci = torch.where(cmask, ci, self.cand_off[fidx][:, None])
        cemb_idx = self.c_emb[ci]
        cmiss = cemb_idx < 0
        ce_raw = self.table[cemb_idx.clamp_min(0).long()].float()
        ce_raw[cmiss] = 0.0
        H = self.H[fidx].clamp_max(self.max_hist)
        Hx = int(np.minimum(self.H_np[fidx_np], self.max_hist).max())
        Hx = min(max(-(-Hx // 64) * 64, 64), self.max_hist)
        ah = torch.arange(Hx, device=dev)
        hmask = ah[None, :] < H[:, None]
        hi = self.hist_off[fidx][:, None] + ah[None, :]
        hi = torch.where(hmask, hi, self.hist_off[fidx][:, None]).clamp_max(self.n_hist - 1)
        hemb_idx = self.h_emb[hi]
        hmiss = (hemb_idx < 0) | ~hmask
        he_raw = self.table[hemb_idx.clamp_min(0).long()].float()
        he_raw[hmiss] = 0.0
        b = dict(user0=self.user0[fidx], fidx=fidx, M=M, H=H, cmask=cmask, ci=ci, hmask=hmask,
                 ce_raw=ce_raw, cmiss=cmiss, ccat=self.c_cat[ci].long(), cf=self.c_f[ci],
                 he_raw=he_raw, hemb_ok=~hmiss, hcat=self.h_cat[hi].long(),
                 het=self.h_et[hi].long(), hes=self.h_es[hi].long(), ue_raw=self.user_emb[fidx].float())
        if self.has_target:
            b["y"] = self.c_tgt[ci].float() * cmask
        if self.dom is not None:
            b["dom"] = self.dom[fidx]
        if self.xt is not None:
            b["xt"] = self.xt[ci].float()
        if with_pos and self.has_pos:
            b["pos"] = self.c_pos[ci].long()
        return b

    def scatter(self, preds_flat, b, logits):
        m = b["cmask"]
        preds_flat[b["ci"][m]] = logits[m].float()


@torch.no_grad()
def feature_stats(data, feed_ids, dense_fn, n_batches=40, seed=0):
    """nn/common.feature_stats (copied): mean/std of the dense features over 40 random ranker-style batches."""
    rng = np.random.default_rng(seed)
    bs = data.batches(feed_ids, shuffle=True, rng=rng)[:n_batches]
    s = s2 = n = 0
    for fb in bs:
        b = data.gather(fb)
        X = dense_fn(b)[b["cmask"]].double()
        s = s + X.sum(0)
        s2 = s2 + (X ** 2).sum(0)
        n += X.shape[0]
    mean = s / n
    std = (s2 / n - mean ** 2).clamp_min(0).sqrt()
    return mean.float(), std.float()


def feed_z(s, M):
    """Per-feed z-score of flat scores (as nn/pcmp.fz)."""
    fid = np.repeat(np.arange(len(M)), M)
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]
