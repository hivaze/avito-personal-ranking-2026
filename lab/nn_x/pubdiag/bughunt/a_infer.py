"""(a) Re-run inference of saved checkpoints through ONE code path (nn/ or nn_x/r6/) on val and test.
Usage: CUDA_VISIBLE_DEVICES=<id> python a_infer.py {nn|r6} ckpt1 ckpt2 ...
ckpt = absolute .pt path. Writes out/<path>__<ckptname>_{val,test}.npy. Read-only w.r.t. the repo (asserts that every
feature file exists in nn/feat_cache before the nn/ path could create a hard link)."""
import os
import sys
import time
from pathlib import Path

import numpy as np

path = sys.argv[1]
ckpts = sys.argv[2:]
ROOT = Path("<repo>")
code = {"nn": ROOT / "nn", "r6": ROOT / "nn_x" / "r6"}[path]
sys.path.insert(0, str(code))
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

import torch  # noqa: E402

import common  # noqa: E402  (asserts single CUDA_VISIBLE_DEVICES)
import data  # noqa: E402
import model as model_mod  # noqa: E402

torch.set_num_threads(12)
assert Path(data.__file__).parent == code and Path(model_mod.__file__).parent == code, (data.__file__, model_mod.__file__)
print("code path", path, data.__file__, model_mod.__file__, "FEAT_CACHE", data.FEAT_CACHE, flush=True)
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True  # as in train.py / finetune.py

cks = {c: torch.load(c, map_location=dev, weights_only=False) for c in ckpts}
xts = {tuple(ck["cfg"].get("xt_names") or ()) for ck in cks.values()}
mh = {ck["args"]["max_hist"] for ck in cks.values()}
assert len(xts) == 1 and len(mh) == 1, (xts, mh)
xt = list(next(iter(xts)))
for sp in ("val", "test"):
    for n in xt:
        assert (ROOT / "nn" / "feat_cache" / sp / f"{n.removeprefix('has:')}.npy").exists(), (sp, n)
maxh = next(iter(mh))
splits = {}
for sp in ("val", "test"):
    sd = data.SplitData(sp, dev, max_hist=maxh)
    if xt:
        sd.attach_xt(xt)
    splits[sp] = sd
for c, ck in cks.items():
    m = model_mod.Ranker(**ck["cfg"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    name = Path(c).stem
    for sp, sd in splits.items():
        t0 = time.time()
        p = common.predict(m, sd)
        np.save(OUT / f"{path}__{name}_{sp}.npy", p.astype(np.float32))
        print(f"{path} {name} {sp}: n={len(p)} mean {p.mean():.5f} std {p.std():.5f} nan {np.isnan(p).sum()} "
              f"{time.time() - t0:.1f}s", flush=True)
    del m
    torch.cuda.empty_cache()
