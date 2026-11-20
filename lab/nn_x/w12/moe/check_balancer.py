"""CPU check of the arm-A/B balancer on the A5 failure mode (lab/nn_x/attn logs: sets.1 max/min expert load 70-9000 under the
1e-3 sign update): (i) a router that sends nearly every token to expert 0, (ii) low-diversity tokens (late set block: all candidates
of a batch nearly the same vector), with the router drifting every step (a fake 'gradient' push toward expert 0).
Also MuonBatched vs torch.optim.Muon after the baddbmm change.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/moe/check_balancer.py"""
import math, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.dont_write_bytecode = True
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "<repo>/lab/nn_x/w12/moe")
from proto_modules import MuonBatched, quantile_shift
torch.set_num_threads(12)
D, E, K = 192, 8, 2
def run(case, mode, steps=1000):
    g = torch.Generator().manual_seed(0)
    W = torch.randn(E, D, generator=g) * 0.07
    b = torch.zeros(E)
    base = torch.randn(D, generator=g)
    out = []
    for it in range(steps + 1):
        if case == "skew":
            x = torch.randn(25000, D, generator=g); W[0] = W[1:].mean(0) + 0.25 * x.mean(0) + 0.2
            x = x + 0.0
            lg = x @ W.t(); lg[:, 0] += 3.0
        else:  # low diversity + drift toward expert 0 (router 'learns' expert 0 by 1e-3 per step)
            x = base + 0.03 * torch.randn(25000, D, generator=g)
            lg = x @ W.t(); lg[:, 0] += 1e-3 * it
        s = torch.sqrt(F.softplus(lg)); z = s + b
        sel = z.topk(K, -1).indices
        load = F.one_hot(sel, E).sum(1).float().mean(0) / K
        if it in (0, 10, 50, 200, 1000):
            ent = float(-(load * load.clamp_min(1e-12).log()).sum() / math.log(E))
            out.append(f"it {it}: ent {ent:.3f} max/min {float(load.max() / load.min().clamp_min(1e-5)):.0f}")
        if mode == "quantile":
            b -= 0.1 * quantile_shift(z[:, None, :], K)[0]
        else:
            b += float(mode.split()[-1]) * torch.sign(1 / E - load)
    return " | ".join(out)
for case in ("skew", "lowdiv"):
    for mode in ("quantile", "sign 1e-3", "sign 1e-2"):
        print(f"{case:7s} {mode:10s} | {run(case, mode)}")
for shp in ((48, 192), (192, 48)):
    torch.manual_seed(1); w0 = torch.randn(*shp) * 0.05
    p1, p2 = nn.Parameter(w0.clone()), nn.Parameter(w0.clone()[None])
    o1 = torch.optim.Muon([p1], lr=2e-3, weight_decay=0.02, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")
    o2 = MuonBatched([p2], lr=2e-3, weight_decay=0.02)
    for _ in range(5):
        gr = torch.randn(*shp); p1.grad, p2.grad = gr.clone(), gr.clone()[None]; o1.step(); o2.step()
    print(f"MuonBatched vs torch Muon {shp}: max |diff| after 5 steps {float((p1 - p2[0]).abs().max().detach()):.2e}, "
          f"total movement {float((p1 - w0).abs().max().detach()):.2e}")
