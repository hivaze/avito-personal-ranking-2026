"""Step 3 micro-benchmark (GPU N only, synthetic data, no file read except a checkpoint cfg): training-step and inference time
of the final student architecture (cfg of w5_e10_kd05w192_s951@77500: d 192, 4 heads, 1 history layer, 2 cross/set blocks,
sim_bias 1, gated split-kv attention, retrieval top-16, 104 dense inputs) with the history at K = 256 (today) against
K = 320 / 384 (+64 / +128 extension tokens). The dense features are computed over the stored 256 events only (as the patch
specifies); the token path (history encoder, cross-attention with cosine bias, top-16 retrieval) sees all K tokens.
lab/nn_x/w5/model.py is imported read-only (no bytecode written).
  CUDA_VISIBLE_DEVICES=N python bench_attn.py > out_bench_attn.txt
"""
import os
import sys
import time

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "GPU N only"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(12)
sys.path.insert(0, "<repo>/lab/nn_x/w5")
from model import Ranker, dense_features, listwise_loss  # noqa: E402

dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
cfg = torch.load("<repo>/lab/nn_x/w5/ckpt/w5_e10_kd05w192_s951@77500.pt", map_location="cpu", weights_only=False)["cfg"]
torch.manual_seed(0)
model = Ranker(**cfg).to(dev)
d = cfg["d"]
old = model.rec.weight.data
model.rec = nn.Embedding(384, d).to(dev)  # the patch: 128 more rows (init irrelevant for timing)
model.rec.weight.data[:256] = old
model.train()
n_xt = len(cfg["xt_names"])
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)


def batch(B, M, Ks, Ke):
    g = torch.Generator(device=dev).manual_seed(B * 1000 + M + Ke)
    K = Ks + Ke
    b = {}
    b["cmask"] = torch.ones(B, M, dtype=torch.bool, device=dev)
    b["ce_raw"] = torch.randint(-60, 60, (B, M, 128), device=dev, generator=g).float()
    b["cmiss"] = torch.rand(B, M, device=dev, generator=g) < 0.05
    b["ccat"] = torch.randint(0, 54, (B, M), device=dev, generator=g)
    b["cf"] = torch.randn(B, M, 5, device=dev, generator=g)
    b["ue_raw"] = torch.randint(-60, 60, (B, 128), device=dev, generator=g).float()
    b["user0"] = torch.zeros(B, dtype=torch.bool, device=dev)
    b["M"] = torch.full((B,), M, device=dev)
    b["xt"] = torch.randn(B, M, n_xt, device=dev, generator=g)
    b["dom"] = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(B, 3)
    he = torch.randint(-60, 60, (B, K, 128), device=dev, generator=g).float()
    ok = torch.rand(B, K, device=dev, generator=g) < 0.8
    het = torch.randint(1, 11, (B, K), device=dev, generator=g)
    hes = torch.randint(0, 30, (B, K), device=dev, generator=g)
    hcat = torch.randint(0, 54, (B, K), device=dev, generator=g)
    hm = torch.ones(B, K, dtype=torch.bool, device=dev)
    # dense inputs over the stored part only
    bs = dict(b)
    bs.update(he_raw=he[:, :Ks], hemb_ok=ok[:, :Ks], het=het[:, :Ks], hes=hes[:, :Ks], hcat=hcat[:, :Ks], hmask=hm[:, :Ks],
              H=torch.full((B,), Ks, device=dev))
    y = (torch.rand(B, M, device=dev, generator=g) < 0.03).float()
    return bs, dict(he_full=he, ok=ok, het=het, hes=hes, hcat=hcat, hm=hm), y


def step(bs, full, y, train=True):
    with torch.no_grad():
        X = model.dense(bs)
    b = dict(bs)
    he = full["he_full"]
    b.update(he=he / he.norm(dim=-1, keepdim=True).clamp_min(1e-3), hemb_ok=full["ok"], het=full["het"], hes=full["hes"],
             hcat=full["hcat"], hmask=full["hm"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(b, X)
    if not train:
        return lg
    loss = listwise_loss(lg, y, b["cmask"]) + 0.5 * listwise_loss(model.aux_out, y, b["cmask"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss


def timeit(fn, n_warm=4, n=15):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ---- the real batch-shape mix of a training epoch (cand_budget 32768, max_feeds 512, M padded to 16) and of extension presence
f = pl.read_parquet("<repo>/data/flat/train/feeds.parquet")
Mtr = f["M"].to_numpy()
Mp = (Mtr + 15) // 16 * 16
ext_n = np.load("<repo>/artifacts/data/feats_new/histext_1117/train/ext_n.npy").astype(np.int64)
cap = np.load("<repo>/artifacts/data/feats_new/histext_1117/train/ext_capped.npy")
ext_n = np.where(cap, ext_n, 0)  # capped-only, as specified
rng = np.random.default_rng(0)
mix = {}
for mp in np.unique(Mp):
    ids = rng.permutation(np.where(Mp == mp)[0])
    B = int(max(1, min(512, 32768 // mp)))
    for i in range(0, len(ids), B):
        mx = ext_n[ids[i:i + B]].max()
        Ke = 0 if mx == 0 else (64 if mx <= 64 else 128)
        mix[(int(mp), B, Ke)] = mix.get((int(mp), B, Ke), 0) + 1
nb = sum(mix.values())
share_ke = {k: sum(v for (m_, b_, ke), v in mix.items() if ke == k) / nb for k in (0, 64, 128)}
print(f"training batches per epoch {nb}; with K extension tokens 0 / 64 / 128: "
      + " / ".join(f"{share_ke[k]:.3f}" for k in (0, 64, 128)), flush=True)

shapes = sorted({(mp, B) for (mp, B, _) in mix})
reps = [(32, 512), (64, 512), (128, 256), (256, 128), (512, 64)]
res = {}
print("\n| batch (Mpad x B) | K = 256 train step (ms) | K = 320 | K = 384 | 384 / 256 | inference K = 256 / 384 (ms) |")
print("|---|---|---|---|---|---|")
for mp, B in reps:
    row = {}
    for Ke in (0, 64, 128):
        bs, full, y = batch(B, mp, 256, Ke)
        row[Ke] = timeit(lambda: step(bs, full, y))
        if Ke in (0, 128):
            model.eval()
            with torch.no_grad():
                row[("inf", Ke)] = timeit(lambda: step(bs, full, y, train=False))
            model.train()
        del bs, full, y
        torch.cuda.empty_cache()
    res[(mp, B)] = row
    print(f"| {mp} x {B} | {1000 * row[0]:.1f} | {1000 * row[64]:.1f} | {1000 * row[128]:.1f} | {row[128] / row[0]:.3f} | "
          f"{1000 * row[('inf', 0)]:.1f} / {1000 * row[('inf', 128)]:.1f} |", flush=True)
# epoch-weighted estimate: nearest representative shape for each real (Mpad, B), extension K as measured on the real mix
rep_m = np.array([r[0] for r in reps])
tot0 = tot1 = 0.0
for (mp, B, Ke), cnt in mix.items():
    r = reps[int(np.argmin(np.abs(np.log(rep_m) - np.log(max(mp, 16)))))]
    scale = (mp * B) / (r[0] * r[1])  # candidate count relative to the representative (cost ~ linear in candidates)
    tot0 += cnt * res[r][0] * scale
    tot1 += cnt * res[r][Ke] * scale
print(f"\nepoch-weighted training-step time with the extension (capped feeds only, k0 arrays) / without: {tot1 / tot0:.3f} "
      f"(estimated GPU time of one epoch: {tot0 / 60:.1f} -> {tot1 / 60:.1f} min of pure forward+backward+step on this GPU)")
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
