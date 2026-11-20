"""nn_x/w5 WAVE-7 cooldown regularisers R-DROP (--rdrop_cd) and SAM (--sam_cd), used by nn_x/w5/train.py. Written
2026-11-13 ~05:20 UTC (sweep item 'two-pass consistency regularisers'; KL probe nn_x/tricks/rdrop_cd_kl/RESULT.txt: two-mask
dropout symKL 0.0111 nats at the m02 cooldown end). OFF at the train.py defaults (0.0): train.py builds the objects only behind
`if a.rdrop_cd > 0:` / `if a.sam_cd > 0:` and reads them only through `if rdrop is not None ...` / `if sam is not None ...`
blocks, so a flag-off run executes exactly the previous nn_x/w5 code path (tests/test_w7.py: AST proof against
tests/ref_pre_w7/train.py.txt + end-to-end batch / logit / log / checkpoint identity; the older tests strip these statements
first, tests/strip_w7.py). Imports torch and model.{listwise_loss, bce_loss} only, so the CPU unit tests call the very
functions train.py uses.

DESIGN (one design each, fixed 2026-11-13 05:15 UTC):
  --rdrop_cd ALPHA (0 = off; arm value 1.0)  R-DROP: on every cooldown step (step >= t_cool; step 0 in the CD harness) the
    SAME batch is forwarded a second time in train mode (independent dropout masks: nn.Dropout sites and the SDPA attention-
    probability dropout; X = model.dense(b) is shared, it carries no dropout in the harness), the second pass gets the same
    loss as the first (main listwise CE, feed-normalised, + K9 aux at the step's annealed weight; main_aux_loss below replicates
    train.py's statements exactly, tests/test_w7.py W1), and
        loss = 0.5 (L1 + L2) + ALPHA * sum_f valid_f symKL_f / max_feeds,
    symKL_f = 0.5 [KL(p1||p2) + KL(p2||p1)] of the two masked per-feed listwise softmaxes (fp32, nats; valid = feeds with a
    positive, exactly the feeds the main CE counts; the sum / max_feeds is --loss_norm feed's normalisation). So the K9 aux is
    averaged over both passes and the main CE is averaged over both passes. The mean per-feed symKL since the last eval is
    logged at every eval ('wave-7 RDROP' line; the probe's scale is 0.005-0.011 nats). CLIPLOG's main / aux split keeps its
    meaning: main := 0.5 (CE1 + CE2) + ALPHA KL, aux := the rest. Step time ~1.7-2x, memory ~ + one graph.
  --sam_cd RHO (0 = off; arm value 0.05)  SAM (Foret et al. 2021, one ascent step, un-normalised by layer): on every cooldown
    step, after train.py's loss.backward() has produced g1 = grad of (main + aux) at w, eps = RHO * g1 / ||g1||_2 over ALL
    trainable parameters (one global L2 norm; fp32); w <- w + eps; the gradients are dropped; a second forward (fresh dropout
    masks) + backward of the SAME loss at w + eps gives g2; w is restored EXACTLY (copies taken before the perturbation, not
    w + eps - eps); train.py then clips g2 (1.0) and steps every optimiser (AdamW and Muon) with g2, as in the flag-off path.
    ||eps|| / ||w|| (global L2 norms) is logged at the first cooldown step; the run is marked INVALID in its log if > 0.02
    (the sweep's precondition; at RHO 0.05 the expected ratio is ~1e-4-1e-3). The logged training loss stays the loss at w.
    Step time ~2x, memory + one parameter copy.
  Not combined with each other, with --loss bce / both, --intent, --val_mix (asserted at setup). --trunc_off is honoured on
  the second pass (same offset). Evaluation, selection, save_* and holdout / stale predictions are untouched (eval mode).
  lane.sh arms: B:rdrop:<root> = B:m02ref + --rdrop_cd 1.0; B:sam:<root> = B:m02ref + --sam_cd 0.05 (CDSEED 900 + root).
"""
import torch

from model import bce_loss, listwise_loss


def main_aux_loss(a, model, b, lg, aux_w_step, prand, fw, trunc):
    """Replica of train.py's loss statements for ONE forward's logits `lg` (model.aux_out was set by that forward): TRUNC
    offset, main loss (--loss), K9 / MSE aux at weight aux_w_step. Returns (loss, main, loss-input logits). tests/test_w7.py W1
    execs the exact train.py statements on the same tensors and asserts bitwise equality."""
    if trunc is not None:
        lg = lg + trunc.offset(b).to(lg.dtype)
    y, m = b["y"], b["cmask"]
    if a.loss == "softmax":
        loss = listwise_loss(lg, y, m, fw, norm=a.loss_norm, denom=a.max_feeds)
    elif a.loss == "bce":
        loss = bce_loss(lg, y, m)
    else:
        loss = listwise_loss(lg, y, m) + bce_loss(lg, y, m)
    main = loss
    if a.aux_pos > 0:
        if a.aux_mode == "mse":
            pct = b["pos"].float() / (b["M"][:, None].float() - 1).clamp_min(1)
            am = m.float() if prand is None else m * (1 - prand[b["fidx"]])[:, None]
            loss = loss + aux_w_step * (((model.aux_out - pct) ** 2) * am).sum() / am.sum().clamp_min(1)
        else:
            tgt = torch.softmax((-b["pos"].float() / a.aux_tau).masked_fill(~m, -1e4), 1) * m
            lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
            fw_aux = torch.ones_like(tgt[:, 0]) if prand is None else 1 - prand[b["fidx"]]
            loss = loss + aux_w_step * (-(tgt * lp).sum(1) * fw_aux).sum() / (
                a.max_feeds / a.aux_norm_c if a.aux_norm == "feed" else fw_aux.sum().clamp_min(1e-6))
    return loss, main, lg


def symkl_feed(lg1, lg2, m):
    """Per-feed symmetric KL 0.5 [KL(p1||p2) + KL(p2||p1)] (fp32, nats) of the masked listwise softmaxes of two logit sets
    [B, M]; masked candidates carry probability 0 and contribute nothing."""
    l1 = torch.log_softmax(lg1.float().masked_fill(~m, -1e4), 1)
    l2 = torch.log_softmax(lg2.float().masked_fill(~m, -1e4), 1)
    p1, p2 = l1.exp() * m, l2.exp() * m
    return 0.5 * ((p1 * (l1 - l2)).sum(1) + (p2 * (l2 - l1)).sum(1))


def _common_asserts(a, flag, t_cool, total):
    assert a.loss == "softmax", f"{flag}: the listwise softmax loss only (--loss {a.loss})"
    assert a.intent == 0, f"{flag}: not combined with --intent"
    assert not (getattr(a, "val_mix", 0) > 0), f"{flag}: not combined with --val_mix"
    assert t_cool < total, f"{flag}: the cooldown never starts (t_cool {t_cool} >= total steps {total})"
    assert a.sched == "wsd" or t_cool == 0, f"{flag}: cooldown steps are defined by the WSD schedule"


class RDrop:
    """Second dropout pass + symmetric KL (module docstring). Lives on the model's device; no host sync per step."""

    def __init__(self, alpha, max_feeds, dev):
        assert alpha > 0
        self.alpha, self.max_feeds = float(alpha), int(max_feeds)
        self.kl_sum = torch.zeros((), device=dev, dtype=torch.float64)
        self.n_feeds = torch.zeros((), device=dev, dtype=torch.float64)
        self.n_steps = 0
        self.rdrop = True

    def combine(self, model, a, b, X, lg, loss, aux_w_step, prand, fw, trunc, cliplog):
        """Second train-mode forward of the same batch; returns 0.5 (loss + loss2) + alpha * feed-normalised symKL.
        `lg` = the first pass's loss-input logits, `loss` = the first pass's main + aux loss (train.py's inline statements)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            lg2 = model(b, X, use_pos=True)
        loss2, main2, lg2 = main_aux_loss(a, model, b, lg2, aux_w_step, prand, fw, trunc)
        valid = (b["y"].sum(1) > 0).float()
        kl = symkl_feed(lg, lg2, b["cmask"]) * valid
        kl_term = kl.sum() / self.max_feeds
        self.kl_sum += kl.detach().sum().double()
        self.n_feeds += valid.sum().double()
        self.n_steps += 1
        if cliplog is not None and cliplog.main is not None:  # keep CLIPLOG's main / aux split meaningful
            cliplog.main = 0.5 * (cliplog.main + main2) + self.alpha * kl_term
        return 0.5 * (loss + loss2) + self.alpha * kl_term

    def summary(self, step):
        n = float(self.n_feeds)
        mean = float(self.kl_sum) / max(n, 1.0)
        s = (f"wave-7 RDROP step {step}: mean per-feed symKL {mean:.6f} nats over {int(n)} valid feeds / {self.n_steps} two-pass "
             f"steps since the last eval (alpha {self.alpha}, KL term = sum / {self.max_feeds})")
        self.kl_sum.zero_()
        self.n_feeds.zero_()
        self.n_steps = 0
        return s


class SAM:
    """One-step sharpness-aware gradient (module docstring): g2 at w + rho g1 / ||g1||, w restored exactly."""

    def __init__(self, rho, model, log):
        assert rho > 0
        self.rho, self.log = float(rho), log
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.w0 = [torch.empty_like(p) for p in self.params]  # exact-restore copies
        dev = self.params[0].device
        self.ratio = self.eps_norm = self.w_norm = self.g1_norm = None
        self.valid = None
        self.first_step = None
        self.loss1_sum = torch.zeros((), device=dev, dtype=torch.float64)
        self.loss2_sum = torch.zeros((), device=dev, dtype=torch.float64)
        self.n = 0
        self.sam = True

    def second_pass(self, model, a, b, X, loss1, aux_w_step, prand, fw, trunc, step):
        """Call right after train.py's loss.backward() (g1 in .grad) and before the gradient clipping / optimiser steps.
        Leaves g2 (the gradient at w + eps) in .grad and the parameters exactly at w."""
        ps = self.params
        gs = [p.grad for p in ps]
        gn = torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None))
        scale = self.rho / (gn + 1e-12)
        with torch.no_grad():
            for p, w0 in zip(ps, self.w0):
                w0.copy_(p)
            for p, g in zip(ps, gs):
                if g is not None:
                    p.add_(g * scale)
            if self.ratio is None:  # first cooldown step: ||eps|| / ||w|| (global L2), validity mark
                wn = torch.sqrt(sum((w.float() ** 2).sum() for w in self.w0))
                en = torch.sqrt(sum(((p - w).float() ** 2).sum() for p, w in zip(ps, self.w0)))
                self.ratio, self.eps_norm, self.w_norm, self.g1_norm = float(en / wn), float(en), float(wn), float(gn)
                self.valid, self.first_step = self.ratio <= 0.02, int(step)
                self.log(f"wave-7 SAM first cooldown step {step}: ||g1|| {self.g1_norm:.4f}, ||eps|| {self.eps_norm:.5f} (rho "
                         f"{self.rho}), ||w|| {self.w_norm:.3f}, ||eps||/||w|| {self.ratio:.6f} -> "
                         + ("VALID (<= 0.02)" if self.valid else "INVALID (> 0.02): the arm is marked INVALID"))
        for p in ps:
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            lg2 = model(b, X, use_pos=True)
        loss2, _, _ = main_aux_loss(a, model, b, lg2, aux_w_step, prand, fw, trunc)
        loss2.backward()
        with torch.no_grad():
            for p, w0 in zip(ps, self.w0):
                p.copy_(w0)
        self.loss1_sum += loss1.detach().double()
        self.loss2_sum += loss2.detach().double()
        self.n += 1

    def summary(self, step):
        n = max(self.n, 1)
        s = (f"wave-7 SAM step {step}: ||eps||/||w|| at the first cooldown step ({self.first_step}) "
             + (f"{self.ratio:.6f} ({'VALID' if self.valid else 'INVALID'})" if self.ratio is not None else "n/a")
             + f"; mean loss at w {float(self.loss1_sum) / n:.4f} vs at w + eps {float(self.loss2_sum) / n:.4f} over {self.n} "
               f"steps since the last eval (rho {self.rho})")
        self.loss1_sum.zero_()
        self.loss2_sum.zero_()
        self.n = 0
        return s


def setup_rdrop(a, model, t_cool, total, log):
    """RDrop for train.py. Called only behind `if a.rdrop_cd > 0`."""
    assert a.rdrop_cd > 0, f"--rdrop_cd must be > 0: {a.rdrop_cd}"
    assert not (a.sam_cd > 0), "--rdrop_cd is not combined with --sam_cd"
    _common_asserts(a, "--rdrop_cd", t_cool, total)
    log(f"wave-7 RDROP: from step {t_cool} (of {total}) every training step forwards its batch twice in train mode (independent "
        f"dropout masks); loss = 0.5 (L1 + L2) + {a.rdrop_cd} * sum_f symKL_f / {a.max_feeds} (fp32 symmetric KL of the masked "
        f"per-feed softmaxes; K9 aux averaged over both passes); the mean per-feed symKL is logged at every eval")
    return RDrop(a.rdrop_cd, a.max_feeds, next(model.parameters()).device)


def setup_sam(a, model, t_cool, total, log):
    """SAM for train.py. Called only behind `if a.sam_cd > 0`."""
    assert a.sam_cd > 0, f"--sam_cd must be > 0: {a.sam_cd}"
    assert not (a.rdrop_cd > 0), "--sam_cd is not combined with --rdrop_cd"
    _common_asserts(a, "--sam_cd", t_cool, total)
    sam = SAM(a.sam_cd, model, log)
    log(f"wave-7 SAM: from step {t_cool} (of {total}) every training step takes the optimiser step with the gradient at "
        f"w + {a.sam_cd} * g1 / ||g1||_2 (g1 = grad of main + aux at w over all {len(sam.params)} trainable tensors, "
        f"{sum(p.numel() for p in sam.params) / 1e6:.3f}M parameters; w restored exactly; fresh dropout masks on the second "
        f"pass); ||eps||/||w|| is logged at the first cooldown step and the run is marked INVALID if > 0.02")
    return sam
