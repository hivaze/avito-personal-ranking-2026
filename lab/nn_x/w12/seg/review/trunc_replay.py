"""T (DECLARATION block 0): model-based truncation dose-response on CPU.
History truncated to the newest K events through SplitData.max_hist (the loader's own cap: H clamp, mask, tensors
sliced; every in-model history-derived input follows). Precomputed xt inputs stay as stored (full history).
Read-only imports of nn_x/w5/{model,data}.py; common.py (GPU guard) is NOT imported. Writes only into this directory.

  python trunc_replay.py <member> [<member> ...]      -> out/trunc/<member>~K<k>.npy (flat val scores, fp32)
  python trunc_replay.py --speed <member>             -> timing on 300 feeds, no file
"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(k, "12")
import numpy as np
import torch

torch.set_num_threads(12)
ROOT = "<repo>"
HERE = f"{ROOT}/nn_x/w12/seg/review"
sys.path.insert(0, f"{ROOT}/nn_x/w5")
import data as Dm  # noqa: E402  (read-only)
from model import Ranker  # noqa: E402

KS = [256, 192, 128, 64]
os.makedirs(f"{HERE}/out/trunc", exist_ok=True)


@torch.no_grad()
def predict_cpu(model, va, feed_ids, cand_budget=16384, max_feeds=256):
    out = torch.zeros(va.n_cand, dtype=torch.float32)
    model.eval()
    for fb in va.batches(feed_ids, cand_budget, max_feeds, shuffle=False):
        b = va.gather(fb)
        lg = model(b, use_pos=False)
        va.scatter(out, b, lg)
    return out.numpy()


def main():
    speed = sys.argv[1] == "--speed"
    members = sys.argv[2:] if speed else sys.argv[1:]
    dev = torch.device("cpu")
    va = Dm.SplitData("val", dev)
    xt_cur = None
    for v in members:
        ck = torch.load(f"{ROOT}/nn_x/w5/ckpt/{v}.pt", map_location=dev, weights_only=False)
        assert ck["args"]["max_hist"] == 256 and not ck["cfg"].get("id_buckets")
        m = Ranker(**ck["cfg"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        xt = ck["cfg"].get("xt_names") or []
        if xt != xt_cur:
            va.xt = None
            va.attach_xt(xt)
            xt_cur = xt
        if speed:
            ids = np.sort(np.random.default_rng(0).choice(va.n, 300, replace=False))
            t = time.time()
            p = predict_cpu(m, va, ids)
            print(f"speed: 300 feeds in {time.time() - t:.1f}s -> {va.n / 300 * (time.time() - t) / 60:.1f} min per full pass", flush=True)
            return
        ref = None
        for K in KS:
            f = f"{HERE}/out/trunc/{v}~K{K}.npy"
            if os.path.exists(f):
                if K == 256:
                    ref = np.load(f)
                print(v, K, "exists", flush=True)
                continue
            t = time.time()
            va.max_hist = K
            ids = np.arange(va.n) if K == 256 else np.where(va.H_np > K)[0]
            p = predict_cpu(m, va, ids)
            va.max_hist = 256
            if K == 256:
                ref = p
                st = np.load(f"{ROOT}/preds/nn_{v}_trainonly_val.npy")
                print(f"{v} ref vs stored train-only val: corr {np.corrcoef(st, p)[0, 1]:.6f} max|d| {np.abs(st - p).max():.3e}", flush=True)
            else:  # feeds with H <= K are untouched by the truncation: copy the reference scores
                rows = np.zeros(va.n_cand, bool)
                for o, mm in zip(va.cand_off_np[ids], va.M_np[ids]):
                    rows[o:o + mm] = True
                p = np.where(rows, p, ref)
            assert np.isfinite(p).all()
            np.save(f + ".tmp.npy", p.astype(np.float32))
            os.replace(f + ".tmp.npy", f)
            print(f"{time.strftime('%FT%TZ', time.gmtime())} {v} K={K}: {len(ids)} feeds scored in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
