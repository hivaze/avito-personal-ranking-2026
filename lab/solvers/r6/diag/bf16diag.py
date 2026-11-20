"""Is the GPU-vs-CPU per-feed Spearman deficit bf16 numerics? CPU bf16 autocast emulation of the GPU predict path
(head / wide forced to fp32, as model.py does on CUDA) on 1500 random val feeds, member v17_pre2s_s101."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, "<repo>/solvers/r6")
import tta
import numpy as np, torch
torch.set_num_threads(4)
import torch.nn as nn
import data as D
class FP32(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        with torch.autocast("cpu", enabled=False):
            return s.m(x.float())
v = "v17_pre2s_s101"
m, ck = tta.load(v)
m.head, m.wide = FP32(m.head), FP32(m.wide)
d = D.SplitData("val", torch.device("cpu"), max_hist=256)
d.attach_xt(ck["cfg"]["xt_names"])
rng = np.random.default_rng(0)
fids = np.sort(rng.choice(d.n, 1500, replace=False))
out16 = torch.zeros(d.n_cand)
with torch.no_grad():
    for fb in d.batches(fids, 65536, 1024, shuffle=False):
        b = d.gather(fb)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lg = m(b, use_pos=False)
        d.scatter(out16, b, lg)
out16 = out16.numpy()
M, off, y, day, fid = tta.val_meta()
c = np.load(tta.pass_path(v, "det", 0)); r = np.load(f"<repo>/preds/nn_{v}_trainonly_val.npy")
sel = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fids])
offs = np.concatenate([[0], np.cumsum(M[fids])[:-1]])
for nm, (a, bb) in {"GPU vs CPU-fp32": (r, c), "CPU-bf16 vs CPU-fp32": (out16, c), "GPU vs CPU-bf16": (r, out16)}.items():
    sp = tta.per_feed_spearman(a[sel], bb[sel], offs, M[fids])
    print(f"{nm:22s} per-feed Spearman mean {sp.mean():.6f} p01 {np.quantile(sp, .01):.4f} share>=0.9999 {(sp >= 0.9999).mean():.3f} pearson {np.corrcoef(a[sel], bb[sel])[0, 1]:.7f}")
