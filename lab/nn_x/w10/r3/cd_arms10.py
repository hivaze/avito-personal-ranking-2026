"""WAVE 10 arm R3 (reports/EXTERNAL_SCAN_1113.md section 2.2 R3; rule nn_x/w5/logs/E_r3_predeclared.txt): objective-diverse
member family = the recipe's listwise softmax CE + a pointwise BCE term, flag --bce_w W (OFF by default, W = 0).

    loss = listwise_loss(lg, y, m, fw, norm=a.loss_norm, denom=a.max_feeds)          (unchanged main loss)
         + W * bce_feed_loss(lg, y, m, fw, norm=a.loss_norm, denom=a.max_feeds)      (this module)

bce_feed_loss = the SAME per-feed weighting and normalisation as model.listwise_loss (per-feed weight w = fw (or 1) times the
valid mask ys > 0; "feed" = sum / denom, "batch" = weighted mean, "idcg" = like "feed" with the K/IDCG(K) factor), applied to
the MEAN-per-feed binary cross-entropy of the feed's candidates: bce_f = sum_i m_i BCE(lg_i, y_i) / sum_i m_i. The existing
`--loss both` path (train.py: listwise_loss(lg, y, m) + bce_loss(lg, y, m)) drops fw, --loss_norm and the denom and takes the
candidate-weighted batch mean; it is NOT used (EXTERNAL_SCAN_1113 finding 8).
Scale (why mean per feed): a calibrated model's per-candidate BCE is ~ the label entropy (~0.08 nats at a 1.5% click rate), so
bce_f ~ 0.08 vs the listwise CE_f ~ 3; W = 5 adds ~0.4 to a per-feed loss of ~3, and its logit gradient per feed
(sum_i |sigmoid(lg_i) - y_i| / M ~ 2/M) is ~10-15% of the listwise gradient (~2 (1 - p_top)) once the logits are calibrated
(tests/test_r3.py C3 measures both on real batches of a trained base checkpoint). The BCE term is pointwise: it pins the
absolute logit level (log-odds) that the shift-invariant softmax never sees, i.e. a different objective, not a re-weighting.
"""
import torch
import torch.nn.functional as F

import model as _model


def bce_feed_loss(logit, y, mask, w=None, norm="batch", denom=None):
    """Feed-normalised pointwise BCE: per-feed MEAN BCE over the valid candidates, then the per-feed weighting / normalisation
    of model.listwise_loss (w [B] or None -> ones; times valid = ys > 0; norm batch | feed | idcg; denom for feed / idcg)."""
    lg = logit.float()
    yf = y.float()
    mf = mask.float()
    l_c = F.binary_cross_entropy_with_logits(lg, yf, reduction="none") * mf          # [B, K] per-candidate BCE, 0 on padding
    l = l_c.sum(1) / mf.sum(1).clamp_min(1)                                           # [B] mean per feed
    ys = yf.sum(1)
    valid = (ys > 0).float()
    w = torch.ones_like(l) if w is None else w.float()
    w = w * valid
    if norm == "batch":
        return (l * w).sum() / w.sum().clamp_min(1e-6)
    if norm == "idcg":
        if _model._IDCG is None or _model._IDCG.device != l.device:
            _model._IDCG = torch.cat([torch.ones(1), torch.cumsum(1.0 / torch.log2(torch.arange(2, 400).float()), 0)]).to(l.device)
        k = ys.long().clamp(0, 398)
        w = w * ys / _model._IDCG[k].clamp_min(1e-6) * (k > 0)
    return (l * w).sum() / denom


def setup_bce(a, log):
    """Called once when --bce_w > 0: guards and the log line (no state)."""
    assert a.bce_w > 0
    assert a.loss == "softmax", "--bce_w adds the BCE term to the standard listwise loss (--loss softmax); never with --loss bce|both"
    assert a.rdrop_cd <= 0 and a.sam_cd <= 0, "--bce_w is not propagated into the wave-7 R-Drop / SAM second passes"
    log(f"wave-10 R3: --bce_w {a.bce_w}: loss = listwise (fw, loss_norm {a.loss_norm}, denom {a.max_feeds}) + {a.bce_w} * "
        f"feed-normalised mean-per-feed BCE with the same fw / loss_norm / denom (cd_arms10.bce_feed_loss)")
    return float(a.bce_w)
