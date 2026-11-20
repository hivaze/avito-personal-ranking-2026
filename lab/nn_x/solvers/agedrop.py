"""Age-conditioned ID-input dropout (review_ideas #6, round-5 agedrop arm). TRAINING ONLY; inference is unchanged.

Staleness simulation: on test (esp. private) the popularity / co-visitation inputs of YOUNG listings are much more often
zero than on train rows (train rows count dense, partly future train snapshots). During training, each candidate row's
ID-based train-stage inputs are blanked with probability p(age bucket), so that the train blank-state rate per age bucket
matches the private-test rate (measured, label-free: scratchpad round5/agedrop/measure.py, reports/r5_agedrop.md).

* Columns blanked = the xt columns whose names start with seq_cv / it_hist (exactly train.py's xt_idcols rule; for
  cvnd_pre2s: seq_cv_invd20, seq_cv_pop, seq_cv_norm, it_hist_all_luo, it_hist_et1..4_luo). Their in-model within-feed
  rank/z channels (XT_REL: seq_cv_invd20, it_hist_all_luo) are computed by model.dense_features AFTER this blanking,
  so they see the blanked values, exactly like a genuinely missing row at inference.
* "Blank" = 0.0 in the stored xt tensor. attach_xt maps these columns with log1p(max(x, 0)) + nan_to_num, so a raw NaN
  or a raw 0 count both become exactly 0.0; on val/test, it_hist_all_luo == 0 implies all 8 columns are 0 (measured:
  P(col zero | luo0) = 1.000 for all 8 on train, val and private). No has:-flag exists for these columns in cvnd_pre2s.
* Age = age(f4) hours via the life_ knots (reports/feat_life_.md sec 1: f4 is the listing's exact age at the feed time;
  same arithmetic as feats_v2/life_/post_time.py:64-68). Row-local, label-free, identical on every split. f4 >= 63/64
  (age cap, >= 4.6 years) is its own bucket.
* RNG: a dedicated torch.Generator, so the model's own dropout RNG stream is untouched (flag on or off).
* Nothing here is stored in model.cfg: checkpoints stay loadable by the stock model.py (predict, holdout_eval, finetune,
  solvers/stale replay).
"""
import numpy as np
import torch

KNOTS = "<repo>/data/feats/life_f4_age_knots.npy"
K, CAP = 64, 63 / 64 - 1e-6  # feats_v2/life_/post_time.py:33-34
F4_COL = 3  # data.SplitData.c_f stacks cand_f1..cand_f5, so f4 is column 3 of b["cf"]

# Buckets: [0,6h) [6,24h) [1,2d) [2,3d) [3,5d) [5,7d) [7,14d) [14,30d) [30,60d) [60,100d) [100,200d) [200d,cap) capped.
EDGES_H = [6, 24, 48, 72, 120, 168, 336, 720, 1440, 2400, 4800]
LABELS = ["0-6h", "6-24h", "1-2d", "2-3d", "3-5d", "5-7d", "7-14d", "14-30d", "30-60d", "60-100d", "100-200d", ">200d",
          "capped"]
# PRE-DECLARED (round 5, before any treatment run): p = clip((z_private - z_train_rl) / (1 - z_train_rl), 0, 1) per
# bucket, z = blank-state share (all 8 ID inputs zero), train_rl = train random-like rows (p_random > 0.5, holdout
# feeds excluded), private = test feeds with timestamp >= 1790497920. Numbers: round5/agedrop/measure.json.
PRESETS = {
    "prv_rl13": dict(edges_h=EDGES_H,
                     p=[0.961, 0.937, 0.898, 0.862, 0.833, 0.810, 0.578, 0.164, 0.063, 0.048, 0.051, 0.094, 0.097]),
}


def id_cols(xt_names):
    """Indices of the ID-based (popularity / co-vis) xt columns: same rule as train.py xt_idcols."""
    return [i for i, n in enumerate(xt_names or []) if n.startswith(("seq_cv", "it_hist"))]


def age_bucket(f4, knots, edges):
    """f4 [...] -> int64 bucket index in 0..len(edges) (age buckets), len(edges)+1 for capped f4."""
    x = f4.double()
    u = x * K
    k = torch.floor(u).long().clamp(1, K - 2)
    w = u - k
    a = knots[k] * (1 - w) + knots[k + 1] * w
    bk = torch.bucketize(a, edges, right=True)  # edges[i-1] <= a < edges[i] -> i
    return torch.where(x < CAP, bk, torch.full_like(bk, len(edges) + 1))


class AgeDrop:
    def __init__(self, preset, xt_names, device, seed):
        spec = PRESETS[preset]
        assert len(spec["p"]) == len(spec["edges_h"]) + 2, "one p per age bucket + one for capped f4"
        self.preset = preset
        self.cols = id_cols(xt_names)
        assert self.cols, "agedrop needs ID-based xt columns (seq_cv* / it_hist*)"
        self.names = [xt_names[i] for i in self.cols]
        self.cm = torch.zeros(len(xt_names), device=device)
        self.cm[self.cols] = 1.0
        self.ci = torch.tensor(self.cols, device=device)
        self.knots = torch.tensor(np.load(KNOTS), dtype=torch.float64, device=device)
        self.edges = torch.tensor(spec["edges_h"], dtype=torch.float64, device=device)
        self.p = torch.tensor(spec["p"], dtype=torch.float32, device=device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed))
        nb = len(spec["p"])
        self.acc = torch.zeros(4, nb, dtype=torch.float64, device=device)  # valid, blank before, dropped, blank after

    @torch.no_grad()
    def __call__(self, b):
        """Blank (in place) the ID-based xt columns of the batch's candidate rows with probability p(age bucket)."""
        xt, m = b["xt"], b["cmask"]
        g = age_bucket(b["cf"][..., F4_COL], self.knots, self.edges)
        u = torch.rand(g.shape, generator=self.gen, device=xt.device)
        drop = (u < self.p[g]) & m
        before = (xt[..., self.ci] == 0).all(-1) & m
        xt.mul_(1 - self.cm * drop[..., None].to(xt.dtype))
        after = (xt[..., self.ci] == 0).all(-1) & m
        gi = g[m]
        for r, v in enumerate((m, before, drop, after)):
            self.acc[r].index_add_(0, gi, v[m].double())
        return drop

    def report(self, reset=True):
        a = self.acc.cpu().numpy()
        n = np.maximum(a[0], 1)
        tot = max(a[0].sum(), 1)
        s = (f"agedrop[{self.preset}] rows {int(a[0].sum())} blank before {a[1].sum() / tot:.4f} after "
             f"{a[3].sum() / tot:.4f} changed {(a[3] - a[1]).sum() / tot:.4f} | after by bucket "
             + " ".join(f"{lab}:{x:.3f}" for lab, x in zip(LABELS, a[3] / n)))
        if reset:
            self.acc.zero_()
        return s
