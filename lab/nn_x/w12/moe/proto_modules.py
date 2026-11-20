"""w12/moe design prototypes (reference code for the implementer; not used by any training run yet).

RoutedFFN  : arm A - shared dense FFN (the baseline Block.ff, untouched) + E routed GELU experts, top-k, DeepSeek-V4 affinity
             sqrt(softplus), renormalised over the top-k and multiplied by a routed scale, quantile-balanced selection bias,
             expert dropout; routed down-projections zero-init (exact baseline function at step 0).
SwitchVO   : arm B - SwitchHead-style routed value (routed on the key/history token) and output (routed on the query/candidate
             token) projections per head, ADDED to the dense v / o (the dense ones act as the shared expert); sigmoid gate, top-k per
             head, quantile-balanced selection bias; routed matrices zero-init (exact baseline function at step 0).
MuonBatched: Muon for 3-D expert stacks [S, in, out]: every slice is its own matrix (per-expert Newton-Schulz), same coefficients,
             steps, eps, decoupled weight decay and 'match_rms_adamw' lr rule (0.2 sqrt(max(in, out)) per slice) as torch.optim.Muon.
quantile_shift: the balancer (our implementation of the K3 "quantile balancing" idea): per expert, the shift of its selection bias
             that would give it exactly the target load k/E on the current batch; the bias moves by eta times that shift.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantile_shift(z, k):
    """z [T, G, E] biased selection scores of T real tokens (G independent routing groups, E experts each, top-k per group).
    Returns q [G, E]: expert e of group g is selected on a token iff z - theta > 0 with theta the (k+1)-th largest score of the
    token's group; q is the (1 - k/E) quantile of that margin, so b -= q would give every expert the target load k/E."""
    E = z.shape[-1]
    theta = z.topk(k + 1, -1).values[..., -1:]
    m = (z - theta).float()
    return torch.quantile(m, 1.0 - k / E, dim=0)


class _Balance:
    def _bal_init(self, groups, n_exp, k, eta):
        self.k, self.n_exp, self.groups, self.eta = k, n_exp, groups, eta
        self.register_buffer("rbias", torch.zeros(groups, n_exp))
        self._pending = None
        self.load_sum = torch.zeros(groups, n_exp)
        self.load_n = 0

    @torch.no_grad()
    def _bal_record(self, z, sel, tmask):
        """z [T, G, E] biased scores, sel [T, G, k] selected experts, tmask [T] real tokens (or None)."""
        if tmask is not None:
            z, sel = z[tmask], sel[tmask]
        if z.shape[0] < 4 * self.n_exp:
            return
        self._pending = quantile_shift(z, self.k)
        oh = F.one_hot(sel, self.n_exp).sum(-2).float()  # [T, G, E]
        self.load_sum = self.load_sum.to(oh.device) + oh.sum(0)
        self.load_n += oh.shape[0]

    @torch.no_grad()
    def update_bias(self):
        """call after every optimizer step (train.py hook); no-op when the module saw no training batch since the last call."""
        if self._pending is not None:
            self.rbias -= self.eta * self._pending
            self._pending = None

    @torch.no_grad()
    def load_stats(self, reset=True):
        """share of selections per expert since the last call; normalised entropy (1 = uniform), max/min ratio, bias range."""
        if not self.load_n:
            return {}
        p = self.load_sum / self.load_sum.sum(-1, keepdim=True).clamp_min(1)
        ent = (-(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(self.n_exp))
        out = dict(ent_min=float(ent.min()), maxmin=float((p.max(-1).values / p.min(-1).values.clamp_min(1e-9)).max()),
                   dead=float((p < 0.1 / self.n_exp).float().mean()), b_lo=float(self.rbias.min()), b_hi=float(self.rbias.max()))
        if reset:
            self.load_sum.zero_()
            self.load_n = 0
        return out


class RoutedFFN(nn.Module, _Balance):
    """Arm A. y = ff(x) + scale * sum_{i in TopK(s + b)} (s_i / sum_TopK s) * mask_i * E_i(x);  s = sqrt(softplus(W_r x)) (fp32),
    E_i(x) = Dropout(GELU(x W1_i^T)) W2_i, W2_i zero-init; mask_i = inverted expert dropout (train only)."""

    def __init__(self, shared, d, n_exp=8, width=48, topk=2, scale=2.0, drop=0.1, xdrop=0.1, eta=0.1):
        super().__init__()
        self.shared = shared
        self.width, self.scale, self.xdrop = width, scale, xdrop
        w1 = torch.empty(n_exp, width, d)
        for i in range(n_exp):  # nn.Linear default init per expert
            nn.init.kaiming_uniform_(w1[i], a=math.sqrt(5))
        self.w1 = nn.Parameter(w1)                              # [E, w, d]  (Muon slice [w, d])
        self.w2 = nn.Parameter(torch.zeros(n_exp, width, d))    # [E, w, d]  in-features first: y = h @ w2
        self.router = nn.Linear(d, n_exp, bias=False)
        self.dp = nn.Dropout(drop)
        self._bal_init(1, n_exp, topk, eta)

    def forward(self, x, tmask=None):
        y = self.shared(x)
        D = x.shape[-1]
        E, w = self.n_exp, self.width
        xf = x.reshape(-1, D)
        with torch.autocast("cuda", enabled=False):
            s = torch.sqrt(F.softplus(self.router(xf.float())))          # [T, E]
            z = s + self.rbias[0]
            sel = z.topk(self.k, -1).indices
            g = s.gather(-1, sel)
            g = self.scale * g / g.sum(-1, keepdim=True).clamp_min(1e-9)
            gate = torch.zeros_like(s).scatter(-1, sel, g)
            if self.training and self.xdrop > 0:
                gate = gate * (torch.rand_like(gate) >= self.xdrop).float() / (1 - self.xdrop)
        if self.training:
            self._bal_record(z.detach()[:, None, :], sel[:, None, :], None if tmask is None else tmask.reshape(-1))
        h = F.gelu(xf @ self.w1.reshape(E * w, D).t().to(xf.dtype)).view(-1, E, w) * gate.to(xf.dtype)[..., None]
        out = self.dp(h.reshape(-1, E * w)) @ self.w2.reshape(E * w, D).to(xf.dtype)
        return y + out.view_as(y)


class SwitchVO(nn.Module, _Balance):
    """Arm B: wraps a baseline Attn (q, k, v, o, g, bias_scale untouched). Per head h and expert e (E per head, top-k per head):
    v_h(c) += sum_{e in TopK_h} sigma(Rv c)_{h,e} c Wv_{h,e}        (routed on the key token c = lnc(ctx))
    out(x) += sum_h sum_{e in TopK_h} sigma(Ro x)_{h,e} o_h Wo_{h,e}  (routed on the query token x = ln1(candidate), o_h = gated head output)
    Wv [h*E, d, dh], Wo [h*E, dh, d], both zero-init. Two balancers (V and O) share the class machinery via two child modules."""

    def __init__(self, attn, d, n_exp=4, topk=2, eta=0.1):
        super().__init__()
        self.attn = attn
        h = attn.h
        dh = d // h
        self.h, self.dh, self.E = h, dh, n_exp
        self.wv = nn.Parameter(torch.zeros(h * n_exp, d, dh))
        self.wo = nn.Parameter(torch.zeros(h * n_exp, dh, d))
        self.rv = nn.Linear(d, h * n_exp, bias=False)
        self.ro = nn.Linear(d, h * n_exp, bias=False)
        self.k, self.eta = topk, eta
        self._pend = {}
        self.register_buffer("rbias", torch.zeros(2, h, n_exp))  # [V router, O router] x head x expert, selection only

    def _route(self, r, x, gi, tmask):
        with torch.autocast("cuda", enabled=False):
            s = torch.sigmoid(r(x.float())).view(*x.shape[:-1], self.h, self.E)   # [..., h, E]
            z = s + self.rbias[gi]
            sel = z.topk(self.k, -1).indices
            gate = torch.zeros_like(s).scatter(-1, sel, s.gather(-1, sel))
        if self.training:
            zf, sf = z.detach().reshape(-1, self.h, self.E), sel.reshape(-1, self.h, self.k)
            if tmask is not None:
                zf, sf = zf[tmask.reshape(-1)], sf[tmask.reshape(-1)]
            if zf.shape[0] >= 4 * self.E:
                q = quantile_shift(zf, self.k)                                      # [h, E]
                self._pend[gi] = q
        return gate

    @torch.no_grad()
    def update_bias(self):
        for gi in (0, 1):
            if self._pend.get(gi) is not None:
                self.rbias[gi] -= self.eta * self._pend[gi]
        self._pend = {}

    def forward(self, x, ctx, kmask, bias=None, qmask=None):
        a = self.attn
        B, N, D = x.shape
        K = ctx.shape[1]
        h, dh, E = self.h, self.dh, self.E
        q = a.q(x).view(B, N, h, dh).transpose(1, 2)
        k = a.k(ctx).view(B, K, h, dh)
        gv = self._route(self.rv, ctx, 0, kmask)                                     # [B, K, h, E]
        ve = (ctx @ self.wv.permute(1, 0, 2).reshape(D, h * E * dh).to(ctx.dtype)).view(B, K, h, E, dh)
        v = a.v(ctx).view(B, K, h, dh) + (ve * gv.to(ve.dtype)[..., None]).sum(3)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if bias is not None and hasattr(a, "bias_scale"):
            am = bias[:, None] * a.bias_scale[None, :, None, None]
            am = am.masked_fill(~kmask[:, None, None, :], float("-inf")).to(q.dtype)
        else:
            am = kmask[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=a.drop if self.training else 0.0)
        if a.gate:
            o = o * torch.sigmoid(a.g(x)).view(B, N, h, dh).transpose(1, 2)
        o = o.transpose(1, 2)                                                        # [B, N, h, dh]
        go = self._route(self.ro, x, 1, qmask)                                       # [B, N, h, E]
        oe = (o[:, :, :, None, :] * go.to(o.dtype)[..., None]).reshape(B, N, h * E * dh)
        return a.o(o.reshape(B, N, D)) + oe @ self.wo.reshape(h * E * dh, D).to(oe.dtype)


class MuonBatched(torch.optim.Optimizer):
    """torch.optim.Muon semantics for 3-D parameters [S, A, B] (each slice orthogonalised on its own, batched bmm)."""

    def __init__(self, params, lr=1e-3, weight_decay=0.02, momentum=0.95, nesterov=True,
                 ns_coefficients=(3.4445, -4.775, 2.0315), ns_steps=5, eps=1e-7):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_coefficients=ns_coefficients, ns_steps=ns_steps, eps=eps))

    @torch.no_grad()
    def step(self):
        for gr in self.param_groups:
            a, b, c = gr["ns_coefficients"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
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
