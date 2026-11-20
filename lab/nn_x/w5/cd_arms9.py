"""gap1117 loss options of nn_x/w5/train.py (2026-11-17; notes lab/nn_x/w12/gap1117/ENGINEER_A.txt, tests lab/nn_x/w12/gap1117/tests/).
Every option is OFF by default; train.py builds the helper only when one of them is on (`gap` stays None otherwise), so a flag-off
run executes exactly the previous code path.

The main loss is the listwise softmax cross-entropy of a feed against its target distribution t = y / max(sum y, 1) (y = clicks, mixed
with the teacher softmax under --kd_file; model.listwise_loss). It splits exactly into a between-category and a within-category part:
    CE_items(t, s) = CE_cat(t_c, S_c) + sum_c t_c * CE_within_c
  t_c = sum of t over the feed's candidates of category c, S_c = logsumexp of their scores (softmax over the S_c = the category
  probabilities), CE_within_c = cross-entropy of softmax(s restricted to c) against t restricted to c and renormalised.
  --wcl LAMBDA   adds LAMBDA * sum_c t_c CE_within_c  (only categories with >= 2 candidates and t_c > 0 contribute; a single-candidate
                 category contributes exactly 0 anyway)
  --ccl LAMBDA   adds LAMBDA * CE_cat(t_c, S_c)
  --fsw ALPHA    per-feed weight M^ALPHA (M = real candidates of the feed) on the main loss and on the two terms above (NOT on the K9
                 display-order aux), normalised to mean 1
                   --fsw_norm batch (default): over the feeds of each batch. NOTE SplitData.batches cuts batches inside M buckets of
                     width 16, so every batch is nearly M-homogeneous: this only re-weights feeds inside a bucket (train feeds:
                     weight sd 0.049 / 0.097 at ALPHA 0.25 / 0.5) and leaves the loss share of M >= 50 feeds at 0.4904 -> 0.4906.
                   --fsw_norm train: one constant divisor = mean of M^ALPHA over the run's training feeds (epoch-mean weight 1);
                     moves the loss share of M >= 50 feeds to 0.606 / 0.709 at ALPHA 0.25 / 0.5.
The two terms use the main loss's per-feed weight (prand_base weight x fsw weight x [feed has a positive]) and its normalisation
(--loss_norm feed: sum / max_feeds; batch: sum / sum of the weights). Categories = the candidate category input b["ccat"]; padding is
masked (its own segment, zero target). Every training batch goes through these statements, the cooldown pool batches included.
"""
import numpy as np
import torch

_SENT = 1 << 20  # category key of padding positions (real category ids are int16)


def category_split(lg, t, m, ccat):
    """Per-feed (CE_cat [B], sum_c t_c CE_within_c [B]) of the logits lg [B, Mx] (float) against the target t [B, Mx] (non-negative,
    zero on padding; the main loss's t = y / max(sum y, 1)), candidate mask m [B, Mx] (bool) and categories ccat [B, Mx] (int).
    Their sum equals the main loss's per-feed CE -(t * log_softmax(lg)).sum(1) up to rounding (identity, tests/test_gap1117.py).
    No host synchronisation: per-feed category segments come from a row-wise sort; segment logsumexp by scatter amax + index_add."""
    B, Mx = lg.shape
    s = lg.masked_fill(~m, -1e4)  # same masking value as model.listwise_loss
    key = ccat.long().masked_fill(~m, _SENT)  # padding: one segment of its own per row (zero target, never mixed with real rows)
    ks, order = torch.sort(key, dim=1, stable=True)
    new = torch.ones_like(ks, dtype=torch.bool)
    new[:, 1:] = ks[:, 1:] != ks[:, :-1]
    loc = torch.empty_like(ks).scatter_(1, order, new.long().cumsum(1) - 1)  # local category index of every position
    seg = (torch.arange(B, device=lg.device)[:, None] * Mx + loc).reshape(-1)
    sf = s.reshape(-1)
    mx = torch.full((B * Mx,), float("-inf"), device=lg.device, dtype=sf.dtype).scatter_reduce(
        0, seg, sf.detach(), reduce="amax", include_self=True)[seg]  # per-segment max (shift only: no gradient needed)
    tot = torch.zeros(B * Mx, device=lg.device, dtype=sf.dtype).index_add(0, seg, torch.exp(sf - mx))[seg]
    S = (torch.log(tot) + mx).view(B, Mx)  # S_c of each position's category (tot >= 1 at every used segment)
    cnt = torch.zeros(B * Mx, device=lg.device, dtype=sf.dtype).index_add(0, seg, m.reshape(-1).to(sf.dtype))[seg].view(B, Mx)
    lse = torch.logsumexp(s, 1, keepdim=True)
    tm = t * m
    ce_cat = -(tm * (S - lse)).sum(1)  # -sum_c t_c log softmax(S)_c
    ce_within = -(tm * (s - S) * (cnt >= 2)).sum(1)  # -sum_c sum_{i in c} t_i log softmax(s|c)_i = sum_c t_c CE_within_c
    return ce_cat, ce_within


def feed_reduce(l, w, norm, denom):
    """The reduction of model.listwise_loss for norm 'batch' / 'feed' (w already holds the per-feed weight x [feed has a positive])."""
    if norm == "batch":
        return (l * w).sum() / w.sum().clamp_min(1e-6)
    return (l * w).sum() / denom


class GapLoss:
    """--wcl / --ccl / --fsw (module docstring). feed_weight() before the main loss, add_terms() right after it, summary() at evals."""

    def __init__(self, a, train_M):
        self.wcl, self.ccl, self.fsw, self.fsw_norm = float(a.wcl), float(a.ccl), float(a.fsw), a.fsw_norm
        self.norm, self.denom = a.loss_norm, a.max_feeds
        self.cM = None
        if self.fsw and self.fsw_norm == "train":
            self.cM = float(np.mean(np.asarray(train_M, dtype=np.float64) ** self.fsw))
        self._w = None
        self._reset()

    def _reset(self):
        self.n, self.nf = 0, 0  # steps, feeds (host counters)
        self.acc = None  # GPU sums: [CE_cat, within, n valid feeds, wcl term, ccl term (+ fsw: sum w, sum w [M>=50], n [M>=50])]
        self.wmin, self.wmax = None, None

    def feed_weight(self, b, fw):
        """fw (None = 1) times the --fsw weight M^ALPHA / norm; unchanged when --fsw 0."""
        if not self.fsw:
            self._w = None
            return fw
        w = b["M"].double() ** self.fsw
        w = (w / (w.mean() if self.fsw_norm == "batch" else self.cM)).float()
        self._w = w
        return w if fw is None else fw * w

    def add_terms(self, loss, lg, y, m, b, fw):
        """loss + wcl * within + ccl * CE_cat (reduced like the main loss); the loss object itself when both are 0 (fsw only:
        the decomposition is still computed, without gradient, for the log)."""
        ys = y.sum(1)
        t = y / ys.clamp_min(1)[:, None]  # the main loss's per-feed target distribution
        w = (torch.ones_like(ys) if fw is None else fw) * (ys > 0).float()
        grad = self.wcl > 0 or self.ccl > 0
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            ce_cat, ce_win = category_split(lg, t, m, b["ccat"])
            tw = feed_reduce(ce_win, w, self.norm, self.denom) * self.wcl if self.wcl else None
            tc = feed_reduce(ce_cat, w, self.norm, self.denom) * self.ccl if self.ccl else None
        with torch.no_grad():  # diagnostics (no host sync): unweighted per-feed means over the valid feeds, the added terms, fsw stats
            z, v = torch.zeros((), device=lg.device), (ys > 0).float()
            row = [(ce_cat * v).sum(), (ce_win * v).sum(), v.sum(), z if tw is None else tw.detach(), z if tc is None else tc.detach()]
            if self._w is not None:
                big = (b["M"] >= 50).float()
                row += [self._w.sum(), (self._w * big).sum(), big.sum()]
                self.wmin = self._w.min() if self.wmin is None else torch.minimum(self.wmin, self._w.min())
                self.wmax = self._w.max() if self.wmax is None else torch.maximum(self.wmax, self._w.max())
            row = torch.stack([r.float() for r in row])
            self.acc = row if self.acc is None else self.acc + row
            self.n += 1
            self.nf += y.shape[0]
        if tw is not None:
            loss = loss + tw
        if tc is not None:
            loss = loss + tc
        return loss

    def summary(self, step):
        if self.acc is None:
            return f"gap1117 step {step}: no training step since the last eval"
        v = self.acc.tolist()
        s = (f"gap1117 step {step} ({self.n} steps; wcl {self.wcl} ccl {self.ccl} fsw {self.fsw} {self.fsw_norm}): per-feed CE_cat "
             f"{v[0] / max(v[2], 1e-9):.4f} within {v[1] / max(v[2], 1e-9):.4f} (sum {(v[0] + v[1]) / max(v[2], 1e-9):.4f}) | added "
             f"terms per step: wcl {v[3] / self.n:.4f} ccl {v[4] / self.n:.4f}")
        if len(v) > 5:
            s += (f" | fsw weight min {float(self.wmin):.3f} max {float(self.wmax):.3f} mean {v[5] / max(self.nf, 1):.3f}, loss-weight "
                  f"share of M>=50 feeds {v[6] / max(v[5], 1e-9):.4f} (unweighted {v[7] / max(self.nf, 1):.4f})")
        self._reset()
        return s


def setup_gap(a, train_M, log):
    """Checks + the helper. Refuses combinations whose loss is recomputed elsewhere without these terms."""
    assert a.loss == "softmax", "--wcl / --ccl / --fsw: listwise softmax loss only"
    assert a.loss_norm in ("feed", "batch"), "--wcl / --ccl / --fsw: --loss_norm feed or batch (idcg not supported)"
    assert not a.rdrop_cd and not a.sam_cd, "--wcl / --ccl / --fsw: not with --rdrop_cd / --sam_cd (they recompute the main loss)"
    assert a.wcl >= 0 and a.ccl >= 0 and np.isfinite(a.fsw), "--wcl / --ccl must be >= 0, --fsw finite"
    g = GapLoss(a, train_M)
    log(f"gap1117: wcl {g.wcl} ccl {g.ccl} fsw {g.fsw} (norm {g.fsw_norm}"
        + (f", train mean M^alpha {g.cM:.4f} over {len(train_M)} training feeds" if g.cM is not None else "")
        + f"); loss_norm {g.norm} denom {g.denom}; applied to every training batch (pool batches included), never to the K9 aux")
    return g
