"""P3 rho70 excess scoring (reports/RESEARCH_SWEEP_1112.md P3, RHO-LOSS / JEST). GPU, one process, read-only on every
checkpoint and data file; writes only nn_x/r6/rho/.

  cd <repo>/nn_x/r6 && CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 ../../.venv/bin/python rho_score.py \
      --ref r6_rhoref_pool_s1 --roots 101,102,103,104 [--wait_min 90]

Feeds scored = the PRODUCTION training feeds of the CD harness: all train feeds minus the fixed 30K holdout (rng 12345,
exactly train.py --holdout 30000) with p_random <= 0.5 (M<50 feeds have p_random 0 -> production). Pool-eligible feeds
(p_random > 0.5) are not scored (the reference trained on them) and stay in the rho70 stream unchanged. Val / test are
never read here.

Per feed: L = listwise softmax CE of the feed (train.py's per-feed listwise term) / log M, eval mode, bf16 autocast
(cd_arms4.per_feed_ce).
  L_snap(r) = under the root's pre-cooldown snapshot nn/ckpt/v17_pre2s_s<r>@77500.pt (the CD --init), with the training
              domain token of train.py (production soft-mixed to random-like by p_random), i.e. the feed's training loss
              at the start of the cooldown;
  L_ref     = under the pool-only reference ranker nn_x/r6/ckpt/<ref>.pt (train.py --only_random 0.5 --holdout 30000, F7
              inputs/architecture, aux off), with the reference's inference domain (random-like), i.e. a position-free
              relevance model's loss on the feed;
  excess    = L_snap - L_ref (RHO-LOSS reducible loss). rho70 keeps the top 70% by excess (cd_arms4.rho_stream_feeds).
Snapshots are scored first, so the pass overlaps with the reference training (--wait_min polls for logs/<ref>.json,
which train.py writes last).

Outputs: rho/L_snap_r<r>.npy, rho/L_ref_<ref>.npy (caches, aligned with rho/prod_fidx.npy), rho/excess_r<r>.npz
(fidx, L_snap, L_ref, excess, M, npos, ntop3 [positives at production positions 0-2], snap, ref), rho/score_meta.json.
"""
import argparse
import json
import time

import numpy as np
import torch

from cd_arms4 import per_feed_ce, production_feeds
from common import CKPT, LOGS, R6, ROOT
from data import SplitData
from model import Ranker

OUT = R6 / "rho"


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = Ranker(**ck["cfg"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="r6_rhoref_pool_s1")
    ap.add_argument("--roots", default="101,102,103,104")
    ap.add_argument("--wait_min", type=float, default=90.0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    dev = torch.device("cuda:0")
    roots = [int(r) for r in a.roots.split(",")]
    snaps = {r: f"{ROOT}/nn/ckpt/v17_pre2s_s{r}@77500.pt" for r in roots}
    tr = SplitData("train", dev, max_hist=256)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(tr.n)).all()
    prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    prand_np[tr.M_np < 50] = 0.0  # train.py rule
    cand = np.where((prand_np > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    feeds = np.setdiff1d(np.arange(tr.n), hold)
    prod = production_feeds(feeds, prand_np)
    assert not np.isin(prod, hold).any()
    np.save(OUT / "prod_fidx.npy", prod)
    print(f"production feeds {len(prod)} (train {tr.n}, holdout {len(hold)}, pool {len(feeds) - len(prod)})", flush=True)
    # per-feed positive counts (pre-flight: share of positives at production positions 0-2)
    fid = np.repeat(np.arange(tr.n, dtype=np.int32), tr.M_np)
    pos = np.load(ROOT / "data/flat/train/cand_pos.npy", mmap_mode="r")
    y = tr.target_np
    npos = np.bincount(fid, weights=y, minlength=tr.n)[prod]
    ntop3 = np.bincount(fid, weights=y * (np.asarray(pos) < 3), minlength=tr.n)[prod]
    del fid
    prt = torch.from_numpy(prand_np).to(dev)
    dom_train = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1)
    xt_cur = None
    meta = dict(roots=roots, ref=a.ref, n_prod=len(prod), times={})

    def attach(cfg):
        nonlocal xt_cur
        if cfg.get("xt_names") != xt_cur:
            tr.xt = None
            if cfg.get("xt_names"):
                tr.attach_xt(cfg["xt_names"])
            xt_cur = cfg.get("xt_names")

    L_snap = {}
    for r in roots:
        f = OUT / f"L_snap_r{r}.npy"
        if f.exists():
            L_snap[r] = np.load(f)
            continue
        t1 = time.time()
        m, ck = load_model(snaps[r], dev)
        attach(ck["cfg"])
        tr.dom = dom_train if ck["cfg"].get("domain") else None  # training-time domain (train.py)
        L_snap[r] = per_feed_ce(m, tr, prod)
        np.save(f, L_snap[r])
        meta["times"][f"snap_{r}"] = time.time() - t1
        print(f"L_snap r{r}: mean {np.nanmean(L_snap[r]):.4f}, NaN {int(np.isnan(L_snap[r]).sum())}, "
              f"{time.time() - t1:.0f}s", flush=True)
        del m
        torch.cuda.empty_cache()
    fref = OUT / f"L_ref_{a.ref}.npy"
    if fref.exists():
        L_ref = np.load(fref)
    else:
        js = LOGS / f"{a.ref}.json"
        while not js.exists():
            assert time.time() - t0 < a.wait_min * 60, f"reference {a.ref} not finished after {a.wait_min} min"
            time.sleep(30)
        t1 = time.time()
        m, ck = load_model(CKPT / f"{a.ref}.pt", dev)
        assert ck["args"]["only_random"] == 0.5 and ck["args"]["holdout"] == 30000, "reference must be pool-only"
        attach(ck["cfg"])
        tr.dom = None  # the reference's inference domain (random-like), as for val/test
        L_ref = per_feed_ce(m, tr, prod)
        np.save(fref, L_ref)
        meta["times"]["ref"] = time.time() - t1
        print(f"L_ref: mean {np.nanmean(L_ref):.4f}, NaN {int(np.isnan(L_ref).sum())}, {time.time() - t1:.0f}s", flush=True)
    for r in roots:
        exc = L_snap[r] - L_ref
        assert np.isfinite(exc).all(), "non-finite excess (a production feed without positives?)"
        np.savez(OUT / f"excess_r{r}.npz", fidx=prod, L_snap=L_snap[r], L_ref=L_ref, excess=exc, M=tr.M_np[prod],
                 npos=npos, ntop3=ntop3, snap=np.array(snaps[r]), ref=np.array(str(CKPT / f"{a.ref}.pt")))
        print(f"wrote rho/excess_r{r}.npz: excess mean {exc.mean():+.4f} sd {exc.std():.4f}", flush=True)
    meta["total_s"] = time.time() - t0
    json.dump(meta, open(OUT / "score_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
