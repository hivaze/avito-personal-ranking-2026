"""Predict the fixed 30K random-like train holdout (same selection as train.py --holdout) with several checkpoints and
evaluate single models and equal-weight ensembles there (member selection without looking at val labels).

  CUDA_VISIBLE_DEVICES=N python nn/holdout_eval.py s2_v4_cv s2_v4_cv_gru ...
writes preds/holdout/nn_<v>_holdout.npy (scores of the holdout candidates, holdout feeds in sorted order).
"""
import sys

import numpy as np
import torch

from common import CKPT, PREDS, ROOT, ndcg, predict  # noqa: E402  (must precede blend: both define "common" paths)
from blend import feed_z  # noqa: F401
from data import SplitData
from model import Ranker

OUT = PREDS / "holdout"
OUT.mkdir(exist_ok=True)


def main(variants):
    dev = torch.device("cuda:0")
    tr = SplitData("train", dev)
    pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[tr.M_np < 50] = 0.0
    cand = np.where((pr > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    M, off = tr.M_np[hold], tr.cand_off_np[hold]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off, M)])
    offs = np.concatenate([[0], np.cumsum(M)[:-1]])
    y = tr.target_np[rows]
    prt = torch.from_numpy(pr).to(dev)
    for v in variants:
        f = OUT / f"nn_{v}_holdout.npy"
        if not f.exists():
            ck = torch.load(CKPT / f"{v}.pt", map_location=dev, weights_only=False)
            cfg = ck["cfg"]
            assert not cfg.get("hist_ext"), "gap1117: models with extension tokens write their holdout preds in train.py (--save_holdout)"
            if cfg.get("xt_names") != getattr(tr, "_xt_names", None):  # reuse the loaded inputs when unchanged
                tr.xt = None
                if cfg.get("xt_names"):
                    tr.attach_xt(cfg["xt_names"])
                tr._xt_names = cfg.get("xt_names")
            tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1) if cfg.get("domain") else None
            m = Ranker(**cfg).to(dev)
            m.load_state_dict(ck["state_dict"])
            s = predict(m, tr, hold)
            np.save(f, s[rows].astype(np.float32))
            del m
            torch.cuda.empty_cache()
        s = np.load(f)
        print(f"{v:24s} holdout NDCG {ndcg(s, y, offs, M):.5f}", flush=True)
    np.save(OUT / "holdout_meta.npy", np.stack([M, offs]))
    np.save(OUT / "holdout_y.npy", y)


if __name__ == "__main__":
    main(sys.argv[1:])
