"""r6 recap: does averaging WSD-plateau snapshots act as a cooldown substitute? CPU ONLY, val inference only.

Inputs: nn/ckpt/v17_pre2s_s10x@{60000,70000,77500}.pt (F7 plateau snapshots, lr 2e-3, constant-LR phase) and the
holdout-selected cooled checkpoint nn/ckpt/v17_pre2s_s10x.pt (80-87.5k). Averaging = uniform mean of state_dicts
(LayerNorm-only model, same trajectory). Scored with src/metrics on val (train-only, no labels used for anything else).

PRE-DECLARED reading (written before running):
  * avg3 - best single snapshot > +0.0010 val (per-feed paired, both seeds positive) -> plateau weight averaging is a
    real partial cooldown substitute -> rank an in-run EMA / LAWA arm (zero-cost instrumentation) for the 10-epoch plateau.
  * avg3 >= cooled selected checkpoint - 0.0005 -> averaging could replace the over-fitting late cooldown.
  * otherwise: drop the idea (C1 stage-1 EMA stays closed).
Information only.
Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 python nn_x/r6/recap_swa.py 101 102
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


def _ro(split, name):
    c = D.FEAT_CACHE / split / f"{name}.npy"
    if not c.exists():
        raise FileNotFoundError(c)
    return c


D.feat_path = _ro


@torch.no_grad()
def predict(model, data):
    model.eval()
    out = torch.zeros(data.n_cand, dtype=torch.float32)
    for fb in data.batches(np.arange(data.n), 16384, 256, shuffle=False):
        b = data.gather(fb)
        lg = model(b, use_pos=False)
        data.scatter(out, b, lg)
    return out.numpy()


def main(seeds):
    log = open(f"{HERE}/logs/recap_swa.log", "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    va = D.SplitData("val", torch.device("cpu"), verbose=False)
    y = va.target_np.astype(np.float64)
    off, M = va.cand_off_np, va.M_np
    cur = None
    res = {}
    for s in seeds:
        names = [f"v17_pre2s_s{s}@60000", f"v17_pre2s_s{s}@70000", f"v17_pre2s_s{s}@77500", f"v17_pre2s_s{s}"]
        cks = [torch.load(f"{R}/nn/ckpt/{n}.pt", map_location="cpu", weights_only=False) for n in names]
        cfg = cks[0]["cfg"]
        if cfg["xt_names"] != cur:
            va.xt = None
            va.attach_xt(cfg["xt_names"])
            cur = cfg["xt_names"]
        variants = {n: c["state_dict"] for n, c in zip(names, cks)}
        sd3 = copy.deepcopy(cks[0]["state_dict"])
        sd2 = copy.deepcopy(cks[0]["state_dict"])
        for k in sd3:
            if sd3[k].dtype.is_floating_point:
                sd3[k] = sum(c["state_dict"][k].float() for c in cks[:3]) / 3
                sd2[k] = sum(c["state_dict"][k].float() for c in cks[1:3]) / 2
        variants[f"avg3_s{s}(60k,70k,77.5k)"] = sd3
        variants[f"avg2_s{s}(70k,77.5k)"] = sd2
        for n, sd in variants.items():
            t0 = time.time()
            m = Ranker(**cfg)
            m.load_state_dict(sd)
            pf = ndcg_per_feed(predict(m, va), y, off, M)
            res[n] = pf
            P(f"{n:34s} val {pf.mean():.5f}  ({time.time() - t0:.0f}s)")
        a3, a2 = res[f"avg3_s{s}(60k,70k,77.5k)"], res[f"avg2_s{s}(70k,77.5k)"]
        best_snap = max(names[:3], key=lambda n: res[n].mean())
        for lab, a in (("avg3", a3), ("avg2", a2)):
            for ref in (best_snap, names[2], names[3]):
                d = a - res[ref]
                P(f"  s{s} {lab} - {ref}: {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}")
    P("done")


if __name__ == "__main__":
    main(sys.argv[1:])
