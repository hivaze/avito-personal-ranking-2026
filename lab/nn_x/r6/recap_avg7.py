"""r6 recap: plateau weight averaging (LAWA-style) of the F7 snapshots, all 7 seeds. CPU ONLY (fp32 inference).

For each seed s in 101..107: avg3 = uniform mean of nn/ckpt/v17_pre2s_s{s}@{60000,70000,77500}.pt state_dicts.
Predict val (all 14,423 feeds) and the fixed 30K random-like train holdout for avg3 AND for the holdout-selected
cooled checkpoint nn/ckpt/v17_pre2s_s{s}.pt with the same CPU fp32 pipeline (removes the GPU-bf16 vs CPU-fp32 offset).
Writes nn_x/r6/preds/nn_r6avg3_s{s}_trainonly_val.npy / _holdout.npy and nn_r6cool_s{s}_* (never preds/).

PRE-DECLARED reading (written after the s101/s102 probe of recap_swa.py, before any 7-seed look):
  * single member: avg3 - cooled, pooled per feed over the 7 seeds: CONFIRMED if val > 2 SE and holdout >= 0
    (holdout SE calibrated as sqrt(SE_feed^2 + 0.00085^2/7), nnx_VERDICT 2.2) and >= 5/7 seeds positive on val.
  * ensemble (information for the lead, no pick made here): z-mean of the 7 avg3 members vs the 7 cooled members, and
    the 14-member pool, on val and holdout; within-feed seed correlation of avg3 members vs cooled members.
Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 python nn_x/r6/recap_avg7.py 101 102 103
"""
import copy
import os
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))
torch.cuda.synchronize = lambda *a, **k: None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "<repo>/src")
import data as D  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
from model import Ranker  # noqa: E402

R = "<repo>"
OUTP = f"{HERE}/preds"


def _ro(split, name):
    c = D.FEAT_CACHE / split / f"{name}.npy"
    if not c.exists():
        raise FileNotFoundError(c)
    return c


D.feat_path = _ro


@torch.no_grad()
def predict(model, data, feed_ids):
    model.eval()
    out = torch.zeros(data.n_cand, dtype=torch.float32)
    for fb in data.batches(feed_ids, 16384, 256, shuffle=False):
        b = data.gather(fb)
        lg = model(b, use_pos=False)
        data.scatter(out, b, lg)
    return out.numpy()


def main(seeds):
    log = open(f"{HERE}/logs/recap_avg7.log", "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    cpu = torch.device("cpu")
    va = D.SplitData("val", cpu, verbose=False)
    tr = D.SplitData("train", cpu, verbose=False)
    pr = np.nan_to_num(np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[tr.M_np < 50] = 0.0
    cand = np.where((pr > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    HM, HO = np.load(f"{R}/preds/holdout/holdout_meta.npy")
    assert (tr.M_np[hold] == HM).all()
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(tr.cand_off_np[hold], tr.M_np[hold])])
    HY = np.load(f"{R}/preds/holdout/holdout_y.npy").astype(np.float64)
    prt = torch.from_numpy(pr)
    yv = va.target_np.astype(np.float64)
    cur = None
    for s in seeds:
        t0 = time.time()
        snaps = [torch.load(f"{R}/nn/ckpt/v17_pre2s_s{s}@{k}.pt", map_location="cpu", weights_only=False)
                 for k in (60000, 70000, 77500)]
        cool = torch.load(f"{R}/nn/ckpt/v17_pre2s_s{s}.pt", map_location="cpu", weights_only=False)
        cfg = cool["cfg"]
        assert all(c["cfg"] == cfg for c in snaps)
        if cfg["xt_names"] != cur:
            for sd_ in (va, tr):
                sd_.xt = None
                sd_.attach_xt(cfg["xt_names"])
            cur = cfg["xt_names"]
        tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1) if cfg.get("domain") else None
        avg = copy.deepcopy(snaps[0]["state_dict"])
        for k in avg:
            if avg[k].dtype.is_floating_point:
                avg[k] = sum(c["state_dict"][k].float() for c in snaps) / 3
        for tag, sd in ((f"r6avg3_s{s}", avg), (f"r6cool_s{s}", cool["state_dict"])):
            m = Ranker(**cfg)
            m.load_state_dict(sd)
            pv = predict(m, va, np.arange(va.n))
            np.save(f"{OUTP}/nn_{tag}_trainonly_val.npy", pv.astype(np.float32))
            ph = predict(m, tr, hold)[rows]
            np.save(f"{OUTP}/nn_{tag}_holdout.npy", ph.astype(np.float32))
            P(f"{tag:16s} val {ndcg_per_feed(pv, yv, va.cand_off_np, va.M_np).mean():.5f} "
              f"holdout {ndcg_per_feed(ph, HY, HO, HM).mean():.5f}  ({time.time() - t0:.0f}s)")
    P("done", seeds)


if __name__ == "__main__":
    main(sys.argv[1:])
