"""why/fit F1/F3 GPU part (read-only on every checkpoint): score selected checkpoints on fixed feed sets of the TRAIN split.
  CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/moe/why/fit/fwd_train.py <part> <n_parts> [@last]
Feed sets (train split feed indices, deterministic):
  HOLD = the 30,000 holdout feeds (p_random > 0.9, M >= 50, rng 12345; excluded from every model's training) - out of sample
  RIN  = every other p_random > 0.9 & M >= 50 train feed (random-order feeds that WERE in training) - in sample, holdout's distribution
  PIN  = 60,000 production-order train feeds (p_random < 0.1, not holdout, rng 7) - in sample, the bulk of the training objective
Same scoring path as training-time evaluation (common.predict: eval mode, bf16 autocast, fp32 head; tr.dom = p_random mix as in train.py).
Output (scratchpad): <OUT>/meta.npz (feed sets, rows, y, teacher logits) and <OUT>/scores/<name>.npz (hold, rin, pin float32 scores);
the HOLD scores are checked against the stored holdout preds of the same checkpoint (max |diff| and NDCG)."""
import os
import sys
import time

sys.path.insert(0, "<repo>/lab/nn_x/w12/moe")  # model_moe (superset of w5/model.py: loads plain w5 checkpoints too)
sys.path.insert(0, "<repo>/lab/nn_x/w5")  # common, data (unchanged w5 modules, as train.py runs them)
import numpy as np
import torch

torch.set_num_threads(12)
from common import ROOT, predict  # noqa: E402
from data import SplitData  # noqa: E402
from model_moe import Ranker  # noqa: E402

OUT = "/tmp/scratchpad/why/fit"
MOE = ROOT / "lab/nn_x/w12/moe/out"
W5 = ROOT / "nn_x/w5"


def model_list(suf=""):
    L = []
    for s in range(951, 959):
        L.append(("kd05w192", s))
    for arm in ("kd05w192moeffn", "kd05w192moevo", "kd05w192b3ff1"):
        for s in range(951, 955):
            L.append((arm, s))
    for fam, seeds in (("kd05", range(951, 955)), ("kd05w224", range(951, 955)), ("kd08w192", range(951, 955)),
                       ("kwd2", range(951, 955)), ("kw256d2", range(951, 955)), ("kwh6", range(951, 955)),
                       ("kw_ff4", (951, 952)), ("kw_hl2", (951, 952)), ("kw_drop0", (951, 952)), ("kw_e20", (951, 952)),
                       ("base", range(951, 955))):
        for s in seeds:
            L.append((fam, s))
    out = []
    for fam, s in L:
        v = f"w5_e10_{fam}_s{s}{suf}"
        if "moe" in fam or "b3ff1" in fam:
            out.append((v, MOE / "ckpt" / f"{v}.pt", MOE / "preds/holdout" / f"nn_{v}_holdout.npy"))
        else:
            out.append((v, W5 / "ckpt" / f"{v}.pt", W5 / "preds/holdout" / f"nn_{v}_holdout.npy"))
    return out


def rows_of(off, M, feeds):
    return np.concatenate([np.arange(o, o + m) for o, m in zip(off[feeds], M[feeds])])


def main():
    part, nparts = int(sys.argv[1]), int(sys.argv[2])
    suf = sys.argv[3] if len(sys.argv) > 3 else ""  # "@last": the last checkpoint (step 98,350) of every run instead of the selected one
    os.makedirs(f"{OUT}/scores", exist_ok=True)
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    tr = SplitData("train", dev)
    M, off = tr.M_np, tr.cand_off_np
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(tr.n)).all()
    prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    prand_np[M < 50] = 0.0
    prand = torch.from_numpy(prand_np).to(dev)
    cand = np.where((prand_np > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    rin = np.setdiff1d(cand, hold)
    prod = np.setdiff1d(np.where(prand_np < 0.1)[0], hold)
    pin = np.sort(np.random.default_rng(7).choice(prod, 60000, replace=False))
    sets = dict(hold=hold, rin=rin, pin=pin)
    rows = {k: rows_of(off, M, v) for k, v in sets.items()}
    hm = np.load(W5 / "preds/holdout/holdout_meta.npy")
    assert (hm[0] == M[hold]).all(), "holdout feed set differs from the stored holdout_meta"
    if part == 0 and not os.path.exists(f"{OUT}/meta.npz"):
        kd = np.load(ROOT / "nn_x/w12/kd/out/teacher_mean11.npy", mmap_mode="r")
        meta = {}
        for k in sets:
            meta[f"{k}_feeds"] = sets[k]
            meta[f"{k}_M"] = M[sets[k]]
            meta[f"{k}_off"] = np.concatenate([[0], np.cumsum(M[sets[k]])[:-1]]).astype(np.int64)
            meta[f"{k}_y"] = tr.target_np[rows[k]]
            meta[f"{k}_t"] = np.asarray(kd[rows[k]], np.float32)
            meta[f"{k}_prand"] = prand_np[sets[k]]
        np.savez(f"{OUT}/meta.npz", **meta)
        print(f"meta saved: hold {len(hold)} rin {len(rin)} pin {len(pin)} feeds; rows {[len(r) for r in rows.values()]}", flush=True)
    feed_union = np.sort(np.concatenate([hold, rin, pin]))
    tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1)
    xt = "unset"
    todo = model_list(suf)[part::nparts]
    print(f"part {part}/{nparts}: {len(todo)} models; data loaded in {time.time() - t0:.0f}s", flush=True)
    for v, ck_path, hold_path in todo:
        dst = f"{OUT}/scores/{v}.npz"
        if os.path.exists(dst):
            continue
        t1 = time.time()
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        cfg = ck["cfg"]
        if cfg.get("xt_names") != xt:
            tr.xt = None
            torch.cuda.empty_cache()
            if cfg.get("xt_names"):
                tr.attach_xt(cfg["xt_names"])
            xt = cfg.get("xt_names")
        assert cfg.get("domain"), v
        m = Ranker(**cfg).to(dev)
        m.load_state_dict(ck["state_dict"])
        s = predict(m, tr, feed_ids=feed_union)
        sc = {k: s[r].astype(np.float32) for k, r in rows.items()}
        ref_h = np.load(hold_path).astype(np.float32)
        dmax = float(np.abs(ref_h - sc["hold"]).max())
        np.savez(dst + ".tmp.npz", **sc)
        os.replace(dst + ".tmp.npz", dst)
        print(f"{v}: done in {time.time() - t1:.0f}s; hold check max|diff| vs stored {dmax:.2e} (ref sd {ref_h.std():.3f})", flush=True)
        del m, ck, s
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
