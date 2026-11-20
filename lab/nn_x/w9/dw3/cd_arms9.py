"""nn_x/w9/dw3 WAVE-9 arm DW3 = random-order-domain-weighted PLATEAU (fresh-eye idea DW3, scratchpad w9/dw3.json, as amended by
its two skeptic reports), used by nn_x/w9/dw3/train.py. Written 2026-11-13 ~13:30 UTC. Both flags are OFF at the train.py
default: train.py builds the DW3 object only behind `if a.prand_norm or a.prand_until_cool:` and applies it only behind
`if dw3 is not None:`, so a flag-off run executes exactly the nn_x/w5 code path (tests/test_dw3.py: AST proof against
tests/ref_w5_train.py.txt + flag-off end-to-end identity).

Mechanism. The existing --prand_base B (>= 0) makes train.py weight the MAIN listwise loss per feed by fw_f = B + p_f, with
p = the label-free p_random of eda/labels_position/out/train_prandom.npz (zeroed for M<50; the 30K holdout feeds are never
trained on). Under --loss_norm feed the main loss is sum_f fw_f CE_f / max_feeds, so WITHOUT a normalisation the mean weight
B + mean(p) (0.348 at B = 0.3: mean p over the training feeds is 0.048) scales the whole main loss, its gradient relative to
the K9 aux loss and its share of the 1.0 clip norm down by ~3x. Two flags:
  --prand_norm        fw_f = (B + p_f) / mean_{training feeds, holdout excluded}(B + p): the mean per-feed weight is 1, the
                      main-loss scale of the base is kept and only the RELATIVE weight of random-like vs production feeds moves
                      (B = 0.3: p > 0.5 pool feeds 3.26x on average = 14.0% of the main-loss weight for 4.3% of the feeds;
                      production feeds 0.90x; M<50 feeds 0.86x).
  --prand_until_cool  fw = None for EVERY step >= t_cool (regular and m02 pool batches): the WSD cooldown, and hence the m02
                      pool-feed dose, is weighted exactly as a --prand_base -1 run. This is the skeptics' fix of the spec's
                      '--prand_pool_off keeps the m02 pool dose' error: the up-weighted feeds ARE the m02 pool feeds and they
                      also appear in the NON-pool cooldown batches, so weighting them through the cooldown would raise the
                      cooldown pool-feed gradient share ~2.5x into the CLOSED m04 / mix-0.1 dose range.
The K9 aux weighting (1 - p) and the domain token are the base's; nothing else changes.
"""
import numpy as np


class DW3:
    """Per-step hook on train.py's per-feed main-loss weight fw (a [B] tensor B + p, or None without --prand_base)."""

    def __init__(self, scale, until_cool, t_cool):
        self.scale, self.until_cool, self.t_cool = float(scale), bool(until_cool), int(t_cool)

    def weight(self, fw, step, is_pool):
        if fw is None:
            return None
        if self.until_cool and step >= self.t_cool:  # cooldown (regular and pool batches): the base's unweighted loss
            return None
        return fw if self.scale == 1.0 else fw * self.scale


def setup_dw3(a, tr, prand, feeds, t_cool, total, log):
    """DW3 object for train.py; logs the realised dose. Called only when --prand_norm or --prand_until_cool is on."""
    assert a.prand_base >= 0 and prand is not None, "--prand_norm / --prand_until_cool need --prand_base >= 0"
    assert a.loss == "softmax", "wave-9 DW3 weights the main listwise loss (--loss softmax)"
    p = prand.detach().cpu().numpy().astype(np.float64)[feeds]  # the training feeds only (holdout / only_random excluded)
    w = a.prand_base + p
    m = float(w.mean())
    scale = 1.0 / m if a.prand_norm else 1.0
    wn = w * scale
    pool = p > 0.5
    short = tr.M_np[feeds] < 50
    log(f"wave-9 DW3: prand_base {a.prand_base}, {len(feeds)} training feeds, mean p {p.mean():.4f}, mean(prand_base + p) "
        f"{m:.4f}, scale {scale:.4f} ({'--prand_norm: mean per-feed weight 1' if a.prand_norm else 'no normalisation'}); "
        f"realised mean weight {wn.mean():.4f}; p > 0.5 feeds {int(pool.sum())} ({pool.mean() * 100:.2f}% of feeds) carry "
        f"{wn[pool].sum() / wn.sum() * 100:.2f}% of the main-loss weight (mean {wn[pool].mean():.3f}x); production (p <= 0.5) "
        f"mean {wn[~pool].mean():.3f}x; M<50 mean {wn[short].mean():.3f}x; max {wn.max():.3f}x")
    if a.prand_until_cool:
        log(f"wave-9 DW3: --prand_until_cool: per-feed weights on steps 0-{t_cool - 1} only; fw = None (the base's loss) on "
            f"every step >= {t_cool} of {total} (regular and pool cooldown batches)")
    return DW3(scale, a.prand_until_cool, t_cool)
