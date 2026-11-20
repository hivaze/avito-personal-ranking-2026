"""Round L1117 (2026-11-17, lead; owner: asked for a composite loss aimed at the largest misses): composite failure-targeted KD loss.

Design and adversarial review: workflow a workflow run (scratchpad loss_design/; reviewer verdict "reject, fair P(pass) ~0.04, run only as one
pre-declared closure arm"). Both parts act on the KD mixture target of the main listwise loss only (never the K9 aux), are computed without
gradient, and are OFF by default (setup_kdc returns None, the trainer path is then unchanged).

  --kdgap GAMMA     per-feed weight w_f = clip((KL_f / mean KL)^GAMMA, lo, hi), renormalised to mean 1 over the batch's valid feeds,
                    KL_f = KL(softmax teacher || softmax student) of the feed: more gradient where the teacher holds structure the student
                    has not absorbed (label-free hard-feed mining; realised-loss mining would chase the label draw, probe P1).
  --kdgap_clip LO,HI  clip range of w_f (default 0.25,4).
  --robust BETA     click-surprise factor c_f = exp(-BETA * CE_click_f), renormalised to mean 1 over the valid feeds; the per-feed loss becomes
                    0.5 c_f CE(clicks/K, s) + 0.5 CE(softmax teacher, s): clicks that look like the label draw count less.
Composite per-feed loss: w_f * [0.5 c_f CE_click + 0.5 CE_teacher] (needs --kd_w 0.5 and no U20 unlabelled fold).

  --casc K          cascade correction (idle-GPU screen from workflow a workflow run, ranked item 4): on production feeds (p_random < 0.5)
                    the main listwise loss (click half AND teacher half) ignores every negative whose production position is more than K
                    below the feed's last positive (unexamined under a cascade model); random-order feeds and the K9 aux keep the full mask.
                    -1 = off.
"""
import torch


class KDComposite:
    def __init__(self, a, log):
        self.gamma, self.beta, self.casc = float(a.kdgap), float(a.robust), int(a.casc)
        self.composite = self.gamma > 0 or self.beta > 0
        self.cn, self.ckept = 0, 0.0
        lo, hi = (float(x) for x in a.kdgap_clip.split(","))
        self.lo, self.hi = lo, hi
        self.log, self.n = log, 0
        self.acc = torch.zeros(6, dtype=torch.float64)  # sum w, sum w^2, min w, max w, sum c, n feeds (host copy at log time)
        self.wmin, self.wmax, self.cmin, self.cmax = 1e9, -1e9, 1e9, -1e9
        log(f"L1117 kdc: kdgap gamma {self.gamma} clip [{lo}, {hi}], robust beta {self.beta}, casc {self.casc}; main loss only, never the K9 aux")

    def casc_mask(self, b, m, prand):
        """Main-loss mask: on production feeds with a positive, drop negatives below (last positive position + K)."""
        with torch.no_grad():
            pos = b["pos"].long()
            ypos = (b["y"] > 0) & m
            last = torch.where(ypos, pos, torch.full_like(pos, -1)).max(1).values  # -1 when no positive
            prod = (prand[b["fidx"]] < 0.5) if prand is not None else torch.ones_like(last, dtype=torch.bool)
            cut = prod & (last >= 0)
            keep = ypos | (pos <= (last + self.casc)[:, None]) | ~cut[:, None]
            mm = m & keep
            self.cn += 1
            if self.cn % 1000 == 1:
                kept = (mm.sum() / m.sum().clamp_min(1)).item()
                self.log(f"L1117 casc step-sample {self.cn}: production feeds cut {int(cut.sum().item())}/{len(cut)}, candidates kept {kept:.3f}")
        return mm

    def apply(self, lg, yc, qt, m, fw):
        """yc = click target (before the KD mixture), qt = teacher softmax (zero on padding). Returns (target, feed weight) for listwise_loss."""
        with torch.no_grad():
            lp = torch.log_softmax(lg.float().masked_fill(~m, -1e4), 1).masked_fill(~m, 0.0)
            K = yc.sum(1)
            valid = (K > 0).float()
            nv = valid.sum().clamp_min(1.0)
            w = torch.ones_like(K)
            if self.gamma > 0:
                kl = (qt * (torch.log(qt.clamp_min(1e-12)) - lp)).sum(1).clamp_min(0.0)
                mk = (kl * valid).sum() / nv
                w = ((kl / mk.clamp_min(1e-12)) ** self.gamma).clamp(self.lo, self.hi)
                w = w / ((w * valid).sum() / nv).clamp_min(1e-12)
            c = torch.ones_like(K)
            if self.beta > 0:
                ce = -(yc / K.clamp_min(1)[:, None] * lp).sum(1)
                c = torch.exp(-self.beta * ce)
                c = c / ((c * valid).sum() / nv).clamp_min(1e-12)
            y2 = 0.5 * c[:, None] * yc / K.clamp_min(1)[:, None] + 0.5 * qt * (K > 0)[:, None].float()
            s = y2.sum(1)  # = 0.5 c + 0.5 on valid feeds; listwise_loss divides the per-feed CE by it, so multiply it back
            wf = w * s * valid
            self.n += 1
            if self.n % 1000 == 1:
                v = valid > 0
                self.log(f"L1117 kdc step-sample {self.n}: w min {w[v].min().item():.3f} max {w[v].max().item():.3f} sd {w[v].std().item():.3f} | "
                         f"c min {c[v].min().item():.3f} max {c[v].max().item():.3f} sd {c[v].std().item():.3f} | valid feeds {int(nv.item())}")
        return y2, (wf if fw is None else fw * wf)


def setup_kdc(a, log):
    if float(a.kdgap) == 0.0 and float(a.robust) == 0.0 and int(a.casc) < 0:
        return None
    assert not (a.wcl or a.ccl or a.fsw), "--kdgap / --robust / --casc: not combined with the gap1117 options"
    assert a.loss == "softmax", "--kdgap / --robust: --loss softmax only"
    if float(a.kdgap) > 0 or float(a.robust) > 0:
        assert a.kd_file and abs(float(a.kd_w) - 0.5) < 1e-9, "--kdgap / --robust: need the KD mixture at --kd_w 0.5"
    assert not getattr(a, "xfold_unlab", False), "--kdgap / --robust / --casc: not with the U20 unlabelled fold"
    return KDComposite(a, log)
