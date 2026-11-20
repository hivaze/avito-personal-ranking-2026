"""WIDTH-GATE arm (c): per-matrix muP multiplier for Muon (reports/SOLVER_POWER_1113.md section 6.3 item 1; nn_x/w11/width,
2026-11-14). Used by nn_x/w11/width/train.py behind `if a.mup_base > 0:` / `if mup is not None:` guards only: at the default
(--mup_base 0) train.py never calls anything here, so a flag-off run executes exactly the nn_x/w5 code path
(tests/test_width.py: AST proof + CPU end-to-end identity on the miniature split).

Mechanism. With adjust_lr_fn="match_rms_adamw" torch's Muon scales every update to spectral norm 0.2 * lr * sqrt(max(A, B))
(torch/optim/_muon.py, _adjust_lr), so at the same lr a d x d hidden matrix of a d192 model steps sqrt(192/128) = 1.22x
"hotter" than at d128 (1.73x at d384). Spectral muP (Yang, Simon, Bernstein 2310.17813) wants the spectral step of a hidden
matrix proportional to sqrt(fan_out / fan_in), i.e. constant across widths for the square / 2d maps of this ranker. The arm
multiplies the Muon lr of every Muon matrix whose FAN-IN grows with d by sqrt(fan_in at d_base / fan_in at d) = sqrt(d_base / d)
(0.8165 at d192, 0.5774 at d384) and leaves the 7 fixed-fan-in input projections (c_lin, c_mlp.0, retr.wk, retr.t1, retr.t2,
u_lin, h_lin; scratch skeptic_scalediag/muon_shapes.log) at 1.0. Muon's decoupled weight decay (param *= 1 - lr * wd) uses the
same group lr, so on the scaled matrices it shrinks by the same multiplier (stated, not corrected). The multiplier lives in the
Muon param group ("lr_mult") and is re-applied INSIDE the per-step loop right after train.py's per-step reset
`g["lr"] = lr_at(step)` (nn_x/w5/train.py l.438-439, which would otherwise erase it): g["lr"] = lr_at(step) * g["lr_mult"].
AdamW (embeddings, norms, biases, 1-output heads) is untouched.
"""
import math

import torch

# the 7 Muon matrices whose fan-in does not grow with d (fan-out may): never scaled
FIXED_FAN_IN = ("c_lin.weight", "c_mlp.0.weight", "retr.wk.weight", "retr.t1.weight", "retr.t2.weight", "u_lin.weight",
                "h_lin.weight")
LOG_EXAMPLES = ("cross.0.attn.q.weight", "c_lin.weight", "retr.wk.weight")  # one hidden, one input projection, one fixed


def classify(model, muon_names, base):
    """{muon matrix name: lr multiplier}. Width-growing fan-in -> sqrt(base / d); the 7 FIXED_FAN_IN names -> 1.0."""
    d = int(model.cfg["d"])
    assert base > 0 and d > 0, (base, d)
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters() if n in muon_names}
    assert len(shapes) == len(muon_names), "every Muon name must be a model parameter"
    missing = [n for n in FIXED_FAN_IN if n not in shapes]
    assert not missing, f"--mup_base: fixed-fan-in matrices not among the Muon matrices: {missing}"
    mult = math.sqrt(base / d)
    out = {}
    for n, (fo, fi) in shapes.items():
        if n in FIXED_FAN_IN:
            if d != base:  # sanity: a fixed fan-in is not a multiple of the width (c_lin's 2d / retr.t2's d coincide at d128 only)
                assert fi % d != 0, f"--mup_base: {n} fan_in {fi} looks width-dependent at d {d}"
            out[n] = 1.0
        else:
            assert fi % d == 0, f"--mup_base: {n} fan_in {fi} is not a multiple of d {d} (expected a width-growing matrix)"
            out[n] = mult
    return out


class MuP:
    def __init__(self, a, model, opts, muon_names, log):
        assert a.muon, "--mup_base needs --muon (it scales the Muon groups)"
        assert a.mup_base > 0, a.mup_base
        idx = [i for i, o in enumerate(opts) if isinstance(o, torch.optim.Muon)]
        assert len(idx) == 1, f"--mup_base: expected exactly one Muon optimizer, found {len(idx)}"
        old = opts[idx[0]]
        assert len(old.param_groups) == 1 and not old.state, "--mup_base must be set up before the first step (fresh Muon)"
        g0 = old.param_groups[0]
        self.d, self.base = int(model.cfg["d"]), int(a.mup_base)
        self.mult_by_name = classify(model, muon_names, self.base)
        name_of = {id(p): n for n, p in model.named_parameters() if n in muon_names}
        params = list(g0["params"])  # keep the original order inside each group
        scaled = [p for p in params if self.mult_by_name[name_of[id(p)]] != 1.0]
        fixed = [p for p in params if self.mult_by_name[name_of[id(p)]] == 1.0]
        assert len(scaled) + len(fixed) == len(params)
        self.mult = math.sqrt(self.base / self.d)
        groups = [{"params": scaled, "lr_mult": self.mult}, {"params": fixed, "lr_mult": 1.0}]
        groups = [g for g in groups if g["params"]]
        self.muon = torch.optim.Muon(groups, lr=g0["lr"], weight_decay=g0["weight_decay"], momentum=g0["momentum"],
                                     nesterov=g0["nesterov"], adjust_lr_fn=g0["adjust_lr_fn"])
        opts[idx[0]] = self.muon  # in place: train.py's `for opt in opts` loops see the new optimizer
        self.shape = {n: tuple(p.shape) for n, p in model.named_parameters() if n in muon_names}
        self.n_scaled, self.n_fixed = len(scaled), len(fixed)
        self.wd = float(g0["weight_decay"])
        fixed_names = sorted(n for n, m in self.mult_by_name.items() if m == 1.0)
        log(f"wave-11 WIDTH-GATE muP: --mup_base {self.base} at d {self.d}: Muon lr multiplier sqrt({self.base}/{self.d}) = "
            f"{self.mult:.4f} on {self.n_scaled} width-growing matrices ({sum(p.numel() for p in scaled) / 1e6:.3f}M params); "
            f"1.0 on {self.n_fixed} fixed-fan-in matrices {fixed_names} ({sum(p.numel() for p in fixed) / 1e6:.3f}M params); "
            f"applied inside the step loop after the per-step lr reset; Muon's decoupled weight decay (wd {self.wd}) scales by the "
            f"same multiplier on the scaled matrices; AdamW groups untouched")
        for n in LOG_EXAMPLES:
            fo, fi = self.shape[n]
            log(f"wave-11 muP example {n} shape ({fo}, {fi}): mult {self.mult_by_name[n]:.4f}; spectral step at lr "
                f"= 0.2 * lr * {self.mult_by_name[n]:.4f} * sqrt({max(fo, fi)}) = {0.2 * self.mult_by_name[n] * math.sqrt(max(fo, fi)):.4f} * lr"
                + (f" (d{self.base} reference 0.2 * sqrt({max(fo, fi) * self.base // self.d}) = "
                   f"{0.2 * math.sqrt(max(fo, fi) * self.base / self.d):.4f} * lr)" if self.mult_by_name[n] != 1.0 else ""))

    def apply(self, lr):
        """Per-step: g['lr'] = lr_at(step) * g['lr_mult'] on the Muon groups (after train.py's reset to lr_at(step))."""
        for g in self.muon.param_groups:
            g["lr"] = lr * g["lr_mult"]

    def effective(self, lr):
        return {n: lr * m for n, m in self.mult_by_name.items()}

    def summary(self, step):
        lr_now = {m: g["lr"] for g in self.muon.param_groups for m in [g["lr_mult"]]}
        eff = {n: lr_now[self.mult_by_name[n]] for n in LOG_EXAMPLES}
        return ("wave-11 muP step %d: effective Muon lr hidden %s %.3e (mult %.4f, decay factor %.6f), input %s %.3e (1.0), "
                "fixed %s %.3e (1.0)" % (step, LOG_EXAMPLES[0], eff[LOG_EXAMPLES[0]], self.mult,
                                          1 - eff[LOG_EXAMPLES[0]] * self.wd, LOG_EXAMPLES[1], eff[LOG_EXAMPLES[1]],
                                          LOG_EXAMPLES[2], eff[LOG_EXAMPLES[2]]))


def setup_mup(a, model, opts, muon_names, log):
    return MuP(a, model, opts, muon_names, log)
