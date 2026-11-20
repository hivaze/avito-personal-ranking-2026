"""w12/moe design check (GPU N only, synthetic batches, <= 15 min; reads only a checkpoint cfg and train feeds.parquet for the
batch-shape mix). For the reference student (cfg of w5_e10_kd05w192_s951) and the three designed arms:
  parameter counts (total, Muon / MuonBatched / AdamW split, active per candidate token),
  exact no-op at init (eval logits of A and B equal the baseline's bit for bit),
  training step time (fwd + bwd + clip + AdamW + Muon + MuonBatched) on 5 representative batch shapes and the epoch-weighted ratio,
  peak GPU memory; plus CPU checks of the quantile balancer and of MuonBatched against torch.optim.Muon.
  cd <repo> && CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
      .venv/bin/python lab/nn_x/w12/moe/bench_arms.py > lab/nn_x/w12/moe/out_bench_arms.txt
"""
import copy
import math
import os
import sys
import time

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "7", "GPU N only"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(12)
sys.path.insert(0, "<repo>/lab/nn_x/w5")
sys.path.insert(0, "<repo>/lab/nn_x/w12/moe")
from model import Ranker, listwise_loss  # noqa: E402
from proto_modules import MuonBatched, RoutedFFN, SwitchVO, quantile_shift  # noqa: E402

T0 = time.time()
dev = torch.device("cuda:0")
cfg = torch.load("<repo>/lab/nn_x/w5/ckpt/w5_e10_kd05w192_s951.pt", map_location="cpu", weights_only=False)["cfg"]
D = cfg["d"]
ROUTERS = ("router", "rv", "ro")


def build(arm):
    torch.manual_seed(951)
    m = Ranker(**cfg) if arm != "C" else Ranker(**{**cfg, "blocks": 3, "ff": 1})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(951 + 7919)
        if arm == "A":
            for blk in list(m.cross) + list(m.sets):
                blk.ff = RoutedFFN(blk.ff, D, n_exp=8, width=48, topk=2, scale=2.0, drop=cfg["drop"], xdrop=0.1, eta=0.1)
        if arm == "B":
            for blk in m.cross:
                blk.attn = SwitchVO(blk.attn, D, n_exp=4, topk=2, eta=0.1)
    return m


def split(m):
    muon = {f"{mn}.weight" for mn, mod in m.named_modules()
            if isinstance(mod, nn.Linear) and min(mod.weight.shape) > 1 and not mn.endswith(ROUTERS)}
    p3 = [(n, p) for n, p in m.named_parameters() if p.ndim == 3]
    rest = [(n, p) for n, p in m.named_parameters() if n not in muon and p.ndim != 3]
    par = lambda n: n.rsplit(".", 1)[0] if "." in n else ""
    dk = lambda n, p: p.ndim >= 2 and "emb" not in n and not par(n).endswith(ROUTERS)
    decay = [p for n, p in rest if dk(n, p)]
    nodecay = [p for n, p in rest if not dk(n, p)]
    return [p for n, p in m.named_parameters() if n in muon], [p for _, p in p3], decay, nodecay


def count(m, arm):
    tot = sum(p.numel() for p in m.parameters())
    pm, p3, dc, nd = split(m)
    act = tot
    if arm == "A":  # 6 of 8 routed experts idle per candidate token, in 4 blocks
        act = tot - 4 * 6 * 2 * 48 * D
    if arm == "B":  # 2 of 4 routed V and O experts idle per head and token, in 2 cross blocks
        act = tot - 2 * 2 * (cfg["heads"] * 2 * D * (D // cfg["heads"]))
    return tot, sum(p.numel() for p in pm), len(pm), sum(p.numel() for p in p3), len(p3), sum(p.numel() for p in dc + nd), act


n_xt = len(cfg["xt_names"])


def batch(B, M, K=256, seed=0):
    g = torch.Generator(device=dev).manual_seed(B * 1000 + M + seed)
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
    b["he_raw"] = torch.randint(-60, 60, (B, K, 128), device=dev, generator=g).float()
    b["hemb_ok"] = torch.rand(B, K, device=dev, generator=g) < 0.8
    b["het"] = torch.randint(1, 11, (B, K), device=dev, generator=g)
    b["hes"] = torch.randint(0, 30, (B, K), device=dev, generator=g)
    b["hcat"] = torch.randint(0, 54, (B, K), device=dev, generator=g)
    b["hmask"] = torch.ones(B, K, dtype=torch.bool, device=dev)
    b["H"] = torch.full((B,), K, device=dev)
    y = (torch.rand(B, M, device=dev, generator=g) < 0.03).float()
    y[:, 0] = 1.0
    return b, y


def make_opts(m):
    pm, p3, dc, nd = split(m)
    opts = [torch.optim.AdamW([{"params": dc, "weight_decay": 0.02}, {"params": nd, "weight_decay": 0.0}], lr=2e-3, betas=(0.9, 0.98)),
            torch.optim.Muon(pm, lr=2e-3, weight_decay=0.02, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")]
    if p3:
        opts.append(MuonBatched(p3, lr=2e-3, weight_decay=0.02))
    return opts


def train_step(m, opts, b, y, bal):
    X = m.dense(b)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = m(b, X, use_pos=True)
    loss = listwise_loss(lg, y, b["cmask"]) + 0.5 * listwise_loss(m.aux_out, y, b["cmask"])
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    for o in opts:
        o.step()
    for mod in bal:
        mod.update_bias()
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


# ---------------------------------------------------------------- 1. CPU checks: balancer and MuonBatched
print("== 1a. quantile balancer vs sign balancer (CPU, 20k fixed tokens, router skewed so expert 0 takes every token)")
torch.manual_seed(0)
xt_ = torch.randn(20000, D)
for mode in ("quantile eta 0.1", "sign u 1e-3", "sign u 1e-2"):
    f = RoutedFFN(nn.Identity(), D, n_exp=8, width=48, topk=2).train()
    with torch.no_grad():
        f.router.weight.mul_(0.3)
        f.router.weight[0] += 0.3  # expert 0 strongly preferred
    row = []
    for it in range(301):
        with torch.no_grad():
            s = torch.sqrt(F.softplus(f.router(xt_)))
            z = s + f.rbias[0]
            sel = z.topk(2, -1).indices
            load = F.one_hot(sel, 8).sum(1).float().mean(0) / 2
            if it in (0, 10, 30, 100, 300):
                ent = float(-(load * load.clamp_min(1e-12).log()).sum() / math.log(8))
                row.append(f"it {it}: ent {ent:.3f} max/min {float(load.max() / load.min().clamp_min(1e-6)):.1f}")
            if mode.startswith("quantile"):
                f.rbias[0] -= 0.1 * quantile_shift(z[:, None, :], 2)[0]
            else:
                u = float(mode.split()[-1])
                f.rbias[0] += u * torch.sign(1 / 8 - load)
    print(f"  {mode:18s} | " + " | ".join(row))

print("== 1b. MuonBatched [1, A, B] against torch.optim.Muon [A, B] (CPU fp32 params, one and five steps)")
for shp in ((48, 192), (192, 48)):
    torch.manual_seed(1)
    w0 = torch.randn(*shp) * 0.05
    p1, p2 = nn.Parameter(w0.clone()), nn.Parameter(w0.clone()[None])
    o1 = torch.optim.Muon([p1], lr=2e-3, weight_decay=0.02, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")
    o2 = MuonBatched([p2], lr=2e-3, weight_decay=0.02)
    for st in range(5):
        gr = torch.randn(*shp)
        p1.grad, p2.grad = gr.clone(), gr.clone()[None]
        o1.step()
        o2.step()
    print(f"  shape {shp}: max |diff| after 5 steps {float((p1 - p2[0]).abs().max()):.2e} (param scale {float(w0.abs().mean()):.3f})")

# ---------------------------------------------------------------- 2. parameter counts and exact no-op at init
print("\n== 2. parameters (total | Muon 2-D | MuonBatched 3-D | AdamW | active per candidate token)")
models = {}
for arm in ("ref", "A", "B", "C"):
    m = build(arm)
    tot, pm, npm, p3, np3, pa, act = count(m, arm)
    print(f"  {arm:3s} total {tot:,} | Muon {pm:,} ({npm} mats) | MuonBatched {p3:,} ({np3} stacks) | AdamW {pa:,} | active {act:,}"
          f" | ratio {tot / 2118848:.3f}")
    models[arm] = m
for arm in ("A", "B"):  # same baseline weights? (built from the same seed, extra modules under a forked RNG)
    sd0, sd1 = models["ref"].state_dict(), models[arm].state_dict()
    shared_keys = [k for k in sd0 if k in sd1]
    same = all(torch.equal(sd0[k], sd1[k]) for k in shared_keys)
    moved = [k for k in sd0 if k not in sd1]
    print(f"  {arm}: baseline tensors identical {same} ({len(shared_keys)} same-name tensors; renamed by wrapping: {len(moved)})")

bb, yy = batch(64, 48, seed=5)
outs = {}
for arm in ("ref", "A", "B"):
    m = models[arm].to(dev).eval()
    # wrapped modules renamed (ff -> ff.shared, attn -> attn.attn): copy the baseline weights in explicitly
    if arm != "ref":
        ref_sd = models["ref"].state_dict()
        sd = m.state_dict()
        for k in sd:
            k0 = k.replace(".ff.shared.", ".ff.").replace(".attn.attn.", ".attn.")
            if k0 in ref_sd and k0 != k:
                assert torch.equal(sd[k].cpu(), ref_sd[k0].cpu()), k
    with torch.no_grad():
        X = m.dense(bb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs[arm] = m(bb, X).float()
    m.train()
for arm in ("A", "B"):
    print(f"  {arm}: eval logits equal to the baseline bit for bit: {torch.equal(outs[arm], outs['ref'])} "
          f"(max |diff| {float((outs[arm] - outs['ref']).abs().max()):.2e})")

# ---------------------------------------------------------------- 3. step time and memory
f = pl.read_parquet("<repo>/data/flat/train/feeds.parquet")
Mp = (f["M"].to_numpy() + 15) // 16 * 16
mix = {}
for mp in np.unique(Mp):
    n = int((Mp == mp).sum())
    B = int(max(1, min(512, 32768 // mp)))
    mix[(int(mp), B)] = math.ceil(n / B)
reps = [(32, 512), (64, 512), (128, 256), (256, 128), (304, 107)]
print(f"\n== 3. training step (ms; fwd + bwd + clip + all optimisers + balancer) | batches per epoch {sum(mix.values())}")
print("| batch (Mpad x B) | ref | A moe-ffn | B switch-vo | C 3 pairs ff1 |")
print("|---|---|---|---|---|")
res = {}
peak = {}
for arm in ("ref", "A", "B", "C"):
    models[arm] = models[arm].to(dev).train()
for mp, B in reps:
    row = {}
    b, y = batch(B, mp)
    for arm in ("ref", "A", "B", "C"):
        m = models[arm]
        opts = make_opts(m)
        bal = [mod for mod in m.modules() if hasattr(mod, "update_bias")]
        torch.cuda.reset_peak_memory_stats()
        row[arm] = timeit(lambda: train_step(m, opts, b, y, bal))
        peak[arm] = max(peak.get(arm, 0), torch.cuda.max_memory_allocated() / 2**30)
        del opts
    res[(mp, B)] = row
    print(f"| {mp} x {B} | " + " | ".join(f"{1000 * row[a]:.1f}" for a in ("ref", "A", "B", "C")) + " |", flush=True)
    del b, y
    torch.cuda.empty_cache()
rep_m = np.array([r[0] for r in reps])
tot = {a: 0.0 for a in ("ref", "A", "B", "C")}
for (mp, B), cnt in mix.items():
    r = reps[int(np.argmin(np.abs(np.log(rep_m) - np.log(max(mp, 16)))))]
    sc = (mp * B) / (r[0] * r[1])
    for a in tot:
        tot[a] += cnt * res[r][a] * sc
print("\nepoch-weighted step time relative to ref: " + ", ".join(f"{a} {tot[a] / tot['ref']:.3f}" for a in tot)
      + f" | pure GPU minutes per epoch: " + ", ".join(f"{a} {tot[a] / 60:.1f}" for a in tot))
print("peak GPU memory of the model step (GB, excludes the ~21.6 GB resident train/val split): "
      + ", ".join(f"{a} {peak[a]:.1f}" for a in peak))
print(f"wall {time.time() - T0:.0f} s")
