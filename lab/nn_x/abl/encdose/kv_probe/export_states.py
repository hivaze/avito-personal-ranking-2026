"""ENCODER-DOSE (C) step 1: export the frozen pre2 encoder's LAST-LAYER hidden states, 64-d after ONE fixed PCA (ledger
nn_x/abl/ledger/ledger.md section 10.C; SOLVER_POWER_1113 section 6.3 item 2 (C)).

Encoder: data/pretrain/ckpt_w1r_v3.pt (SeqEncoder d 384, 4 pre-LN RoPE blocks; its corpus = window 1, histories up to Sep 11 MSK),
run ID-ZEROED (SeqEncoder built with n_id = 0: item_rep has no ID term; the model was trained with ID-dropout 0.3 so the content
path works alone), content table item_tab2 (the checkpoint's), bf16 autocast item reps, the encoder's own forward
(pretrain/seqenc.SeqEncoder.encode: BOS + old -> new events, RoPE causal blocks, final LayerNorm). h[t + 1] = the state after
old -> new event t.

Rows and the LEAK RULE (one latent space, no later-snapshot leak):
    train   feeds with Moscow day >= Sep 12 2026 (day 20708; 1,470,065 feeds, 272.8M history rows) -> exported; feeds on or before
            Sep 11 (the corpus end) -> NOT exported (flag 0 -> the ranker's learned null token). The 30K holdout feeds are train feeds
            and follow the same rule.
    val / test  all feeds exported (they are after the corpus end by construction).
PCA: fit ONCE on a fixed sample of states from Sep 12-14 train feeds (seed 0; ~2M states), 384 -> 64, applied to every exported state
of every split; stored in kv_probe/out/pca64.npz (mean, components, explained variance).

Alignment with the ranker: the flat history arrays data/flat/<split>/hist_*.npy are NEWEST-FIRST per feed (rows hist_off .. hist_off + H);
the ranker (nn_x/abl/data.SplitData.gather) reads slot k as row hist_off + k. The encoder sees the OLD -> NEW sequence, so the state
of raw slot k is h[H - k] (k = 0 newest -> h[H]; k = H - 1 oldest -> h[1]). The export writes, per split, a COMPACT fp16 array
    data/feats/<split>/kv64_states.npy   [n_exported_hist_rows, 64]   (the exported feeds' rows, in feed order, newest-first per feed)
    data/feats/<split>/kv64_off.npy      [n_feeds] int64  offset of the feed's first row in kv64_states, -1 = not exported (null token)
    data/feats/<split>/kv64_user.npy     [n_feeds, 64] fp16 the user state h[H] after the newest event (0 where not exported; information)
by exclusive create (nothing existing is overwritten). Train: ~35 GB fp16; it fits on a GPU next to a training process (the ranker
copies it to the GPU like the other history arrays) or in pinned host RAM.

  CUDA_VISIBLE_DEVICES=<4|5> OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/encdose/kv_probe/export_states.py --fit_pca            # step 0: PCA on the Sep 12-14 sample
  ...  --split val | test | train [--budget 200000]                                                                               # step 1: export (train last; ~1 GPU-h)
  ...  --split val --sample 3000 --out_dir <scratch>                                                                               # smoke: a feed sample, outputs to a scratch dir
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
sys.path.insert(0, "<repo>/nn_x/abl/encdose")
from seqenc import SeqEncoder  # noqa: E402  (pretrain/train.py is not imported: its build_corpus import forces 32 numba threads)
from score_hmax_noid import Tables  # noqa: E402  (item_tab2 / has / category tables on the GPU)

ROOT = Path("<repo>")
PT, FLAT, FEATS = ROOT / "data/pretrain", ROOT / "data/flat", ROOT / "data/feats"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CKPT = "w1r_v3"
D_OUT = 64
DAY_FIRST = 20708  # Sep 12 2026 MSK = the first exported train day (the w1r_v3 corpus ends Sep 11)
PCA_DAYS = (20708, 20710)  # Sep 12-14: the PCA sample days
PCA_FEEDS, PCA_SEED = 15000, 0  # ~15K feeds x ~185 events = ~2.7M states


def msk_day(ts):
    return (np.asarray(ts, np.int64) + 3 * 3600) // 86400


def exported_mask(split, day):
    """The leak rule: train feeds on or after Sep 12 MSK; every val / test feed."""
    day = np.asarray(day)
    return day >= DAY_FIRST if split == "train" else np.ones(len(day), bool)


def raw_slots_from_states(h, H):
    """h [H + 1, d] encoder states of ONE feed (h[0] = BOS, h[t + 1] = after old -> new event t) -> states of the NEWEST-FIRST raw slots
    [H, d]: slot k = h[H - k]."""
    return h[1:H + 1].flip(0)


def load_encoder(dev):
    ck = torch.load(PT / f"ckpt_{CKPT}.pt", map_location="cpu", mmap=True, weights_only=False)
    ca, sd = ck["args"], ck["model"]
    heads = ["fe", "new"] if ca["w_new"] > 0 else ["fe"]
    model = SeqEncoder(0, d=ca["d"], nl=ca["nl"], nh=ca["nh"], d_id=ca["d_id"], drop=ca["drop"], heads=heads, id_bias=False)
    sd = {k: v for k, v in sd.items() if not k.startswith(("id_emb.", "id_proj.", "id_bias."))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model.to(dev).eval(), ca


class Split:
    def __init__(self, split, feeds=None):
        f = pl.read_parquet(FLAT / split / "feeds.parquet")
        self.n_all = len(f)
        self.day = msk_day(f["timestamp"].to_numpy())
        self.H_all = f["H"].to_numpy().astype(np.int64)
        self.hoff_all = f["hist_off"].to_numpy().astype(np.int64)
        self.feeds = np.arange(self.n_all) if feeds is None else np.asarray(feeds, np.int64)
        mm = "r" if split != "train" else None
        self.item = np.load(FLAT / split / "hist_item_id.npy", mmap_mode=mm)
        self.et = np.load(FLAT / split / "hist_event_type.npy", mmap_mode=mm)
        self.es = np.load(FLAT / split / "hist_event_source.npy", mmap_mode=mm)


@torch.no_grad()
def encode_feeds(model, tables, sp, feeds, dev, budget=200_000):
    """Yields (feed ids, H, h) per chunk: h [B, Lmax + 1, d] fp32 last-layer states (position 0 = BOS); feeds sorted by H inside."""
    H = sp.H_all[feeds]
    order = np.argsort(H, kind="stable")
    B = max(1, budget // 258)
    for i in range(0, len(feeds), B):
        ids = feeds[order[i:i + B]]
        Hc = sp.H_all[ids]
        L = int(max(Hc.max(), 1))
        ar = np.arange(L)
        mask_np = ar[None, :] < Hc[:, None]
        # old -> new: raw slot H - 1 - t at sequence position t
        raw = sp.hoff_all[ids][:, None] + (Hc[:, None] - 1 - ar[None, :])
        raw = np.where(mask_np, raw, sp.hoff_all[ids][:, None].clip(max=len(sp.item) - 1))
        it = torch.from_numpy(np.asarray(sp.item[raw.ravel()]).reshape(raw.shape).astype(np.int64)).to(dev)
        et = torch.from_numpy(np.asarray(sp.et[raw.ravel()]).reshape(raw.shape).astype(np.int64)).to(dev)
        es = torch.from_numpy(np.asarray(sp.es[raw.ravel()]).reshape(raw.shape).astype(np.int64)).to(dev)
        mask = torch.from_numpy(mask_np).to(dev)
        it = torch.where(mask, it, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rep = model.item_rep(tables.tab[it], tables.has[it] & mask, tables.cat[it].long(), torch.zeros_like(it))
            h = model.encode(rep, et * mask, es * mask)
        yield ids, Hc, h.float()


def fit_pca(model, tables, dev, budget):
    sp = Split("train")
    cand = np.where((sp.day >= PCA_DAYS[0]) & (sp.day <= PCA_DAYS[1]) & (sp.H_all > 0))[0]
    feeds = np.sort(np.random.default_rng(PCA_SEED).choice(cand, PCA_FEEDS, replace=False))
    n, mean, cov = 0, torch.zeros(384, device=dev, dtype=torch.float64), torch.zeros(384, 384, device=dev, dtype=torch.float64)
    t0 = time.time()
    for ids, Hc, h in encode_feeds(model, tables, sp, feeds, dev, budget):
        L = h.shape[1] - 1
        m = torch.from_numpy(np.arange(L)[None, :] < Hc[:, None]).to(dev)
        x = h[:, 1:][m].double()  # states after real events only (no BOS, no padding)
        n += x.shape[0]
        mean += x.sum(0)
        cov += x.T @ x
    mean /= n
    cov = cov / n - torch.outer(mean, mean)
    evals, evecs = torch.linalg.eigh(cov)
    idx = torch.argsort(evals, descending=True)[:D_OUT]
    comp = evecs[:, idx].T.contiguous()  # [64, 384]
    ev = evals[idx] / evals.clamp_min(0).sum()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pca64.npz", mean=mean.cpu().numpy().astype(np.float32), components=comp.cpu().numpy().astype(np.float32),
             explained=ev.cpu().numpy().astype(np.float32), n_states=n, feeds=feeds, days=np.array(PCA_DAYS), seed=PCA_SEED, ckpt=CKPT)
    print(f"[pca] {n:,} states from {len(feeds)} Sep 12-14 feeds (seed {PCA_SEED}); explained variance of 64 comps "
          f"{float(ev.sum()):.3f} (top 8 {np.round(ev[:8].cpu().numpy(), 3).tolist()}); {time.time() - t0:.0f}s -> {OUT / 'pca64.npz'}", flush=True)


def export(split, model, tables, dev, budget, sample=0, out_dir=None):
    sp = Split(split)
    pca = np.load(OUT / "pca64.npz")
    mean = torch.from_numpy(pca["mean"]).to(dev)
    comp = torch.from_numpy(pca["components"]).to(dev)  # [64, 384]
    exp_mask = exported_mask(split, sp.day)
    feeds = np.where(exp_mask)[0]
    if sample:
        feeds = np.sort(np.random.default_rng(1).choice(feeds, min(sample, len(feeds)), replace=False))
    H = sp.H_all[feeds]
    off = np.full(sp.n_all, -1, np.int64)
    off[feeds] = np.concatenate([[0], np.cumsum(H)[:-1]])
    n_rows = int(H.sum())
    out_dir = Path(out_dir) if out_dir else FEATS / split
    out_dir.mkdir(parents=True, exist_ok=True)
    p_states, p_off, p_user = out_dir / "kv64_states.npy", out_dir / "kv64_off.npy", out_dir / "kv64_user.npy"
    for p in (p_states, p_off, p_user):
        assert not p.exists(), f"{p} exists: refusing to overwrite"
    states = np.lib.format.open_memmap(p_states, mode="w+", dtype=np.float16, shape=(max(n_rows, 1), D_OUT))
    user = np.zeros((sp.n_all, D_OUT), np.float16)
    t0, done = time.time(), 0
    for ids, Hc, h in encode_feeds(model, tables, sp, feeds, dev, budget):
        z = ((h - mean) @ comp.T).half()  # [B, L + 1, 64]
        Hc_t = torch.from_numpy(Hc).to(dev)
        user[ids] = z[torch.arange(len(ids), device=dev), Hc_t].cpu().numpy()  # h[H]: after the newest event (BOS when H = 0)
        L = z.shape[1] - 1
        ar = torch.arange(L, device=dev)
        # newest-first raw slot k = position H - k  (k < H)
        pos = (Hc_t[:, None] - ar[None, :]).clamp_min(0)
        g = z.gather(1, pos[..., None].expand(-1, -1, D_OUT))  # [B, L, 64]: slot k -> z[H - k]
        m = (ar[None, :] < Hc_t[:, None]).cpu().numpy()
        rows = (off[ids][:, None] + np.arange(L)[None, :])[m]
        states[rows] = g.cpu().numpy()[m]
        done += len(ids)
        if done % (50 * max(1, budget // 258)) < len(ids):
            print(f"  [{split}] {done:,} / {len(feeds):,} feeds, {time.time() - t0:.0f}s", flush=True)
    states.flush()
    del states
    with open(p_off, "xb") as fh:
        np.save(fh, off)
    with open(p_user, "xb") as fh:
        np.save(fh, user)
    print(f"[done] {split}: {len(feeds):,} exported feeds of {sp.n_all:,} ({n_rows:,} history rows, {n_rows * D_OUT * 2 / 1e9:.1f} GB fp16), "
          f"{time.time() - t0:.0f}s -> {p_states}, {p_off}, {p_user}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_pca", action="store_true")
    ap.add_argument("--split", default="")
    ap.add_argument("--budget", type=int, default=200_000)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out_dir", default="")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("4", "5"), "ENCODER-DOSE: GPUs N only, CUDA_VISIBLE_DEVICES = one id"
    dev = torch.device("cuda:0")
    tables = Tables(dev, "item_tab2")
    model, ca = load_encoder(dev)
    tables.set_corpus(ca.get("corpus_dir", ""))
    print(f"[model] ckpt_{CKPT} ID-zeroed (n_id 0), d {model.d}, corpus {ca.get('corpus_dir')}", flush=True)
    if a.fit_pca:
        fit_pca(model, tables, dev, a.budget)
    if a.split:
        export(a.split, model, tables, dev, a.budget, a.sample, a.out_dir or None)


if __name__ == "__main__":
    main()
