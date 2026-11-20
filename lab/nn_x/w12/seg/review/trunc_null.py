"""TN (DECLARATION block 2): keep the stored history length, remove only the EMBEDDINGS of the events at index >= K
(exactly the loader's representation of an event without an embedding). CPU, fp32, read-only imports of nn_x/w5.
  python trunc_null.py <member> ...   -> out/trunc/<member>~N<k>.npy
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
import data as Dm  # noqa: E402
from model import Ranker  # noqa: E402

MODE = {"K": None}
_gather = Dm.SplitData.gather


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    K = MODE["K"]
    if K is not None and b["he_raw"].shape[1] > K:
        b["he_raw"][:, K:] = 0.0
        b["hemb_ok"][:, K:] = False
        b["hidx"][:, K:] = -1
    return b


Dm.SplitData.gather = gather


@torch.no_grad()
def predict_cpu(model, va, feed_ids, cand_budget=16384, max_feeds=256):
    out = torch.zeros(va.n_cand, dtype=torch.float32)
    for fb in va.batches(feed_ids, cand_budget, max_feeds, shuffle=False):
        b = va.gather(fb)
        va.scatter(out, b, model(b, use_pos=False))
    return out.numpy()


def main():
    dev = torch.device("cpu")
    va = Dm.SplitData("val", dev)
    xt_cur = None
    for v in sys.argv[1:]:
        ck = torch.load(f"{ROOT}/nn_x/w5/ckpt/{v}.pt", map_location=dev, weights_only=False)
        m = Ranker(**ck["cfg"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        xt = ck["cfg"].get("xt_names") or []
        if xt != xt_cur:
            va.xt = None
            va.attach_xt(xt)
            xt_cur = xt
        ref = np.load(f"{HERE}/out/trunc/{v}~K256.npy")
        # identity check of the hook: K = 256 changes nothing
        MODE["K"] = 256
        ids = np.where(va.H_np > 200)[0][:300]
        p = predict_cpu(m, va, ids)
        rows = np.concatenate([np.arange(o, o + mm) for o, mm in zip(va.cand_off_np[ids], va.M_np[ids])])
        print(f"{v} hook identity (K = 256, 300 feeds): max|d| vs ref {np.abs(p[rows] - ref[rows]).max():.2e}", flush=True)
        for K in (192, 128):
            f = f"{HERE}/out/trunc/{v}~N{K}.npy"
            if os.path.exists(f):
                continue
            t = time.time()
            MODE["K"] = K
            ids = np.where(va.H_np > K)[0]
            p = predict_cpu(m, va, ids)
            rows = np.zeros(va.n_cand, bool)
            for o, mm in zip(va.cand_off_np[ids], va.M_np[ids]):
                rows[o:o + mm] = True
            p = np.where(rows, p, ref)
            np.save(f + ".tmp.npy", p.astype(np.float32))
            os.replace(f + ".tmp.npy", f)
            print(f"{time.strftime('%FT%TZ', time.gmtime())} {v} N={K}: {len(ids)} feeds in {time.time() - t:.0f}s", flush=True)
        MODE["K"] = None


if __name__ == "__main__":
    main()
