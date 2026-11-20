"""r6 recap: does the 7-seed ENSEMBLE (the 'mean function') still improve along the WSD plateau, and does the cooldown
improve it or only de-noise members? CPU ONLY, val inference only (F7 snapshots @60k/@70k/@77.5k, all 7 seeds).

Motivation: recap_avg7 showed plateau weight averaging removes cross-seed diversity (avg3 seed corr 0.975 vs 0.941 for
cooled members; 7-member ensemble -0.0025 val). So member noise IS the ensemble's diversity, and what matters for the
pool is the quality of the ensemble-level function, which single-member curves cannot show.

PRE-DECLARED reading (written before running):
  * ens7(77.5k) - ens7(60k) > +0.0010 (paired per-feed, > 2 SE) -> the plateau is still improving the mean function at
    6-8 epochs -> rank a longer-plateau family (12-15 epochs) as an ensemble-level lever, judged by gate (b) only.
    <= +0.0005 -> the mean function has saturated by 6 epochs -> longer training is not a lever.
  * ens7(cooled) - ens7(77.5k): > +0.0010 -> the cooldown improves the mean function (keep it); within +-0.0005 -> the
    cooldown only de-noises members (ensemble members could stop at the plateau end, 20% cheaper; needs a gate-(b) check).
Information only. Writes nn_x/r6/preds/nn_r6snap_s{s}@{k}_trainonly_val.npy.
"""
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
def predict(model, data):
    model.eval()
    out = torch.zeros(data.n_cand, dtype=torch.float32)
    for fb in data.batches(np.arange(data.n), 16384, 256, shuffle=False):
        b = data.gather(fb)
        data.scatter(out, b, model(b, use_pos=False))
    return out.numpy()


def fz(s, fid, M):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def main():
    log = open(f"{HERE}/logs/recap_plateau_ens.log", "a")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    va = D.SplitData("val", torch.device("cpu"), verbose=False)
    y = va.target_np.astype(np.float64)
    off, M = va.cand_off_np, va.M_np
    fid = np.repeat(np.arange(len(M)), M)
    seeds, steps = list(range(101, 108)), (60000, 70000, 77500)
    cur = None
    for s in seeds:
        for k in steps:
            f = f"{OUTP}/nn_r6snap_s{s}@{k}_trainonly_val.npy"
            if os.path.exists(f):
                continue
            t0 = time.time()
            ck = torch.load(f"{R}/nn/ckpt/v17_pre2s_s{s}@{k}.pt", map_location="cpu", weights_only=False)
            if ck["cfg"]["xt_names"] != cur:
                va.xt = None
                va.attach_xt(ck["cfg"]["xt_names"])
                cur = ck["cfg"]["xt_names"]
            m = Ranker(**ck["cfg"])
            m.load_state_dict(ck["state_dict"])
            pv = predict(m, va)
            np.save(f, pv.astype(np.float32))
            P(f"s{s}@{k} val {ndcg_per_feed(pv, y, off, M).mean():.5f} ({time.time() - t0:.0f}s)")
    Z = {(s, k): fz(np.load(f"{OUTP}/nn_r6snap_s{s}@{k}_trainonly_val.npy"), fid, M) for s in seeds for k in steps}
    for s in seeds:
        Z[(s, "cool")] = fz(np.load(f"{OUTP}/nn_r6cool_s{s}_trainonly_val.npy"), fid, M)
    E = {}
    for k in list(steps) + ["cool"]:
        E[k] = ndcg_per_feed(np.mean([Z[(s, k)] for s in seeds], 0), y, off, M)
        single = np.mean([ndcg_per_feed(Z[(s, k)], y, off, M).mean() for s in seeds])
        zs = [Z[(s, k)] for s in seeds]
        corr = np.mean([np.corrcoef(zs[i], zs[j])[0, 1] for i in range(7) for j in range(i + 1, 7)])
        P(f"step {k}: mean single {single:.5f} | 7-seed ens {E[k].mean():.5f} | gain {E[k].mean() - single:+.5f} | seed corr {corr:.4f}")
    E["all21"] = ndcg_per_feed(np.mean([Z[(s, k)] for s in seeds for k in steps], 0), y, off, M)
    E["all28"] = ndcg_per_feed(np.mean([Z[(s, k)] for s in seeds for k in list(steps) + ["cool"]], 0), y, off, M)
    P(f"ens of all 21 plateau snapshots {E['all21'].mean():.5f}; all 21 + 7 cooled {E['all28'].mean():.5f}")
    for a, b in ((77500, 60000), (70000, 60000), (77500, 70000), ("cool", 77500), ("all21", "cool"), ("all28", "cool")):
        d = E[a] - E[b]
        P(f"  ens {a} - ens {b}: {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}")
    P("done")


if __name__ == "__main__":
    main()
