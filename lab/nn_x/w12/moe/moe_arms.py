"""w12/moe trainer helpers (lab/nn_x/w12/moe/train.py; OFF unless --moe_ffn / --vo_moe; the model side lives in model_moe.py).

  moe_kwargs(a)  the Ranker keyword arguments of the arm flags ({} when every flag is off, so the w5 constructor call is unchanged)
  setup_moe      optimiser split of the routed modules (called only when model.moe_on):
                   routers (RoutedFFN.router, SwitchVO.rv / .ro; nn.Linear(d, E*), caught by the Moonlight rule of train.py) are taken
                   OUT of the Muon set and put in the AdamW no-decay group;
                   the 3-D expert stacks (w1, w2, wv, wo) are kept out of both AdamW groups and go to MuonBatched (per-slice Muon, same
                   lr / wd / momentum / Nesterov / 5 Newton-Schulz steps / match_rms_adamw rule as torch.optim.Muon on each [in, out]
                   expert matrix);
                 update_bias() after every optimiser step (quantile balancer of every routing site);
                 summary(step) at every eval: per routing site the normalised load entropy (min over groups), max/min load, dead-expert
                   share (load < 0.1 of uniform), selection-bias range and RMS(routed) / RMS(shared), plus the ES3 flag (entropy < 0.6 or
                   > 25% dead experts in any routed layer for 3 consecutive evals after step 10,000 -> 'ROUTING COLLAPSED').
  MuonBatched    torch.optim.Muon semantics for 3-D parameters [S, A, B] (each slice orthogonalised on its own in one batched bmm).
"""
import math

import torch

from model_moe import RoutedFFN, SwitchVO

ROUTERS = ("router", "rv", "ro")
STACKS = ("w1", "w2", "wv", "wo")


def moe_kwargs(a):
    """Ranker keyword arguments of the arm flags; {} when all are off (the w5 Ranker call)."""
    kw = {}
    if a.moe_ffn:
        kw.update(moe_ffn=a.moe_ffn, moe_w=a.moe_w, moe_k=a.moe_k, moe_scale=a.moe_scale, moe_xdrop=a.moe_xdrop, moe_eta=a.moe_eta)
    if a.vo_moe:
        kw.update(vo_moe=a.vo_moe, vo_k=a.vo_k, vo_eta=a.vo_eta)
    return kw


class MuonBatched(torch.optim.Optimizer):
    """torch.optim.Muon semantics for 3-D parameters [S, A, B] (each slice orthogonalised on its own, batched bmm):
    buf = lerp(buf, g, 1 - momentum); u = lerp(g, buf, momentum) (Nesterov); X = NS5(u / ||u||_F per slice) in bf16 (transposed when
    A > B); p *= 1 - lr wd; p -= lr 0.2 sqrt(max(A, B)) X (match_rms_adamw)."""

    def __init__(self, params, lr=1e-3, weight_decay=0.02, momentum=0.95, nesterov=True,
                 ns_coefficients=(3.4445, -4.775, 2.0315), ns_steps=5, eps=1e-7):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_coefficients=ns_coefficients, ns_steps=ns_steps, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        for gr in self.param_groups:
            a, b, c = gr["ns_coefficients"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 3, "MuonBatched: 3-D expert stacks only"
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - gr["momentum"])
                u = p.grad.lerp(buf, gr["momentum"]) if gr["nesterov"] else buf
                X = u.bfloat16()
                tr = X.size(1) > X.size(2)
                if tr:
                    X = X.transpose(1, 2)
                X = X / X.flatten(1).norm(dim=1).clamp(min=gr["eps"])[:, None, None]
                for _ in range(gr["ns_steps"]):  # same fused ops as torch's addmm form (b A + c A A, then a X + (.) X)
                    A = torch.bmm(X, X.transpose(1, 2))
                    Bm = torch.baddbmm(A, A, A, beta=b, alpha=c)
                    X = torch.baddbmm(X, Bm, X, beta=a)
                if tr:
                    X = X.transpose(1, 2)
                lr_adj = gr["lr"] * 0.2 * math.sqrt(max(p.size(1), p.size(2)))
                p.mul_(1 - gr["lr"] * gr["weight_decay"])
                p.add_(X.to(p.dtype), alpha=-lr_adj)


class MoeArms:
    def __init__(self, a, model, muon_names, log):
        assert a.muon, "w12/moe arms need --muon (expert stacks -> MuonBatched, RMS-matched like the hidden matrices)"
        self.mods = model.moe_modules()
        assert self.mods, "w12/moe: model.moe_on but no routed module"
        named = dict(model.named_parameters())
        self.router_names = {f"{n}.{r}.weight" for n, m in self.mods for r in ROUTERS if hasattr(m, r)}
        self.stack_names = {f"{n}.{s}" for n, m in self.mods for s in STACKS if hasattr(m, s)}
        assert self.router_names <= set(named) and self.stack_names <= set(named)
        assert self.router_names <= muon_names, "w12/moe: the Moonlight rule should have caught the routers"
        assert not (self.stack_names & muon_names)
        muon_names -= self.router_names  # in place: train.py builds rest / P_muon from this set
        self.stacks = [p for n, p in model.named_parameters() if n in self.stack_names]
        self.bad_run, self.collapsed, self.last = 0, False, {}
        n_tot = sum(p.numel() for p in model.parameters())
        n_idle = 0
        for _, m in self.mods:  # experts not selected for a token (per candidate / key token)
            if isinstance(m, RoutedFFN):
                n_idle += (m.n_exp - m.k) * (m.w1[0].numel() + m.w2[0].numel())
            else:
                n_idle += m.h * (m.E - m.k) * (m.wv[0].numel() + m.wo[0].numel())
        log(f"w12/moe: {len(self.mods)} routed modules ({', '.join(n for n, _ in self.mods)}); total params {n_tot:,}, active per token "
            f"{n_tot - n_idle:,}; routers {len(self.router_names)} ({sum(named[n].numel() for n in self.router_names):,} params) -> AdamW "
            f"no decay; expert stacks {len(self.stacks)} ({sum(p.numel() for p in self.stacks):,} params) -> MuonBatched; quantile balancer "
            f"after every optimiser step; cfg {dict((k, v) for k, v in model.cfg.items() if k.startswith(('moe_', 'vo_')))}")

    def adamw_groups(self, rest):
        """train.py's decay / no-decay split of `rest` (= every parameter outside the Muon set) without the expert stacks and with the
        routers in the no-decay group."""
        rest = [(n, p) for n, p in rest if n not in self.stack_names]
        dk = lambda n, p: p.ndim >= 2 and "emb" not in n and n not in self.router_names  # noqa: E731
        return [p for n, p in rest if dk(n, p)], [p for n, p in rest if not dk(n, p)]

    def optimizer(self, a):
        return MuonBatched(self.stacks, lr=a.lr, weight_decay=a.wd, momentum=0.95, nesterov=True)

    def update_bias(self):
        for _, m in self.mods:
            m.update_bias()

    def summary(self, step):
        parts, self.last = [], {}
        for n, m in self.mods:
            for gi, st in enumerate(m.sites):
                s = st.stats(m.rbias[gi] if isinstance(m, SwitchVO) else m.rbias)
                if s is None:
                    continue
                self.last[f"{n}:{st.name}"] = s
                parts.append(f"{n.replace('.attn.vo', '.vo')}:{st.name} ent {s['ent']:.3f} maxmin {s['maxmin']:.2f} dead {s['dead']:.2f} "
                             f"b [{s['b_lo']:+.3f},{s['b_hi']:+.3f}] rms {s['rms']:.3f}")
        if not self.last:
            self.diag = None
            return f"w12/moe step {step}: no training forward since the last eval"
        ent = min(s["ent"] for s in self.last.values())
        dead = max(s["dead"] for s in self.last.values())
        self.bad_run = self.bad_run + 1 if (step > 10000 and (ent < 0.6 or dead > 0.25)) else 0
        self.collapsed = self.collapsed or self.bad_run >= 3
        self.diag = dict(ent_min=ent, dead_max=dead, bad_run=self.bad_run, collapsed=self.collapsed, sites=self.last)
        return (f"w12/moe step {step}: min load entropy {ent:.3f}, max dead share {dead:.2f}"
                + (" ROUTING COLLAPSED (ES3)" if self.collapsed else "") + " | " + " | ".join(parts))


def setup_moe(a, model, muon_names, log):
    return MoeArms(a, model, muon_names, log)
