"""nn_x/w10/r2 WAVE-10 arm R2 = XOFF, a fixed EXAMINATION offset on the exploration (random-order) training feeds only
(reports/EXTERNAL_SCAN_1113.md section 2.2 R2; x10.3), used by nn_x/w10/r2/train.py. Written 2026-11-13 ~16:55 UTC. The flag is OFF
at the train.py default (--xoff 0): train.py builds the object only behind `if a.xoff > 0:` and applies it only behind
`if xoff is not None:`, so a flag-off run executes exactly the nn_x/w5 code path (tests/test_r2.py: AST proof against
tests/ref_w5_train.py.txt + flag-off end-to-end identity).

Mechanism (PAL on random logging, Google KDD'23). On the 134,118 exploration train feeds (NN p_random > 0.5, M >= 50, 30K holdout
excluded; the m02 pool feeds) the shown order is random, so the displayed position is independent of the item and a position-blind
model (the recipe has --no_pos: no position tower, no positional encoding) learns relevance up to a constant; the examination
pattern of those rows (positive rate 3.2x the feed base at the first shown slot, 0.84x deep; premise/premise_check.txt) is label
noise it has to absorb. XOFF explains that part of the label with a fixed additive term on the MAIN listwise logits of TRAINING
batches only:
    lg[f, i] += A * 1[p_f > 0.5] * g_noend[clamp(pos_i, 0, 299)]       (A = 0.5, the arm value)
  p_f    = the NN p_random of the feed (eda/labels_position/out/train_prandom.npz, NaN -> 0, p := 0 for M < 50 = train.py's rule),
           so production feeds (96% of the training feeds), every M < 50 feed and the holdout feeds (never trained on) get 0;
  pos_i  = the raw displayed slot of the candidate (data/flat/train/cand_pos.npy; gaps = items the user had already viewed, removed
           from the dataset; train.py gathers it with with_pos=True on every training batch);
  g_noend = log of the WITHIN-FEED observed/expected positive-rate lift by raw pos on the pool rows (premise/g_noend_pos_oe.json,
           premise_g.py: O = positives at pos, E = sum of the feeds' own non-end rates; the last shown item (rkend 0, the truncation
           spike at rate 0.28) is excluded from O and E, so the table carries NO end-of-feed term (labels_loss A3: an end-of-feed
           offset on exploration feeds cost -0.0011 holdout, z -3.7); raw for pos 0-19, bins above, log1p interpolation).
           g at pos 0/1/2/5/10/20/50/100/299 = +0.97 / +0.76 / +0.50 / +0.30 / +0.26 / +0.12 / -0.06 / -0.11 / -0.17.
  Applied on every training batch that contains such feeds: the regular stream (plateau and cooldown; 4.3% of the feeds) and the
  m02 cooldown pool batches (all pool feeds). Never on the K9 aux (model.aux_out, a separate head), never at eval / holdout
  selection / val / test prediction (common.py calls the model directly and never sees the offset), never on val feeds (--val_mix
  is refused). The difference from the CLOSED position rows: TRUNC-PROD / trunc2 / truncp offset PRODUCTION rows by rank-from-end;
  the old position tower learned g(pos) on production rows (the confounded 6x curve) and the PAL proxy gated by (1 - p_random)
  also acted on production rows; XOFF touches only the exploration rows' own examination pattern.
  Leakage: train-row displayed positions + train labels of the pool feeds (a 300-entry marginal table; the same class of
  artefact as the accepted trunc_lift.json). Half strength (A = 0.5) = the spec's dose (the GBDT/MLP proxy's half-strength point).
"""
import hashlib
import json

import numpy as np
import torch

ROOT = "<repo>"
XOFF_TABLE = f"{ROOT}/nn_x/w10/r2/premise/g_noend_pos_oe.json"
POOL_P = 0.5  # NN pool / production threshold on the NN p_random (nn_x/r6 POOL_P_DEFAULT; the m02 pool rule)
MAX_POS = 300


def load_table(path):
    """(g [300] float32, md5 of the file). The table must be finite, 300 long, and have no end-of-feed term (it is indexed by pos only)."""
    tab = json.load(open(path))
    g = np.asarray(tab["g_noend"], dtype=np.float32)
    assert g.shape == (MAX_POS,) and np.isfinite(g).all(), f"--xoff_table {path}: need 300 finite entries, got {g.shape}"
    assert g[0] > 0 and g[0] == g.max() and g[-1] < 0, f"--xoff_table {path}: not an examination curve (g[0] {g[0]}, g[299] {g[-1]})"
    return g, hashlib.md5(open(path, "rb").read()).hexdigest()


def xoff_coef(a_off, prand_nn, hold=None):
    """Per-feed coefficient [n] float32: a_off on exploration feeds (NN p_random > 0.5; p is 0 for M < 50), 0 elsewhere and on the holdout."""
    coef = np.where(prand_nn > POOL_P, np.float32(a_off), np.float32(0.0)).astype(np.float32)
    if hold is not None:
        coef[hold] = 0.0
    return coef


class XOff:
    """Holds the per-feed coef table and the 300-entry g table on the device. offset(b) -> [B, Mx] float32 offsets of a gathered
    training batch (0 on padding); depends on b['fidx'], b['pos'], b['cmask'] only (the same for any history view of the batch)."""

    def __init__(self, a_off, tr, prand_nn, hold, log=print, table_path=XOFF_TABLE):
        g, md5 = load_table(table_path)
        assert prand_nn.shape == (tr.n,), f"p_random has {prand_nn.shape}, train split has {tr.n} feeds"
        assert getattr(tr, "has_pos", False), "XOFF needs the train split's cand_pos (gathered with with_pos=True)"
        coef = xoff_coef(a_off, prand_nn, hold)
        dev = tr.device
        self.coef = torch.from_numpy(coef).to(dev)
        self.g = torch.from_numpy(g).to(dev)
        self.a, self.md5, self.table_path = float(a_off), md5, table_path
        on = coef > 0
        nh = 0 if hold is None else int(on[hold].sum())
        log(f"wave-10 XOFF: a {a_off}; table {table_path} (md5 {md5}); g at pos 0/1/2/5/10/20/50/100/299 "
            f"{[round(float(g[q]), 3) for q in (0, 1, 2, 5, 10, 20, 50, 100, 299)]}; offset a*g on the exploration feeds (NN p_random > "
            f"{POOL_P}): {int(on.sum())} feeds ({on.mean() * 100:.2f}% of {tr.n}; holdout feeds with a non-zero coef {nh}); "
            f"0 on production / M<50 / holdout feeds; main listwise logits of training batches only (never K9 / eval)")

    def offset(self, b):
        c = self.coef[b["fidx"]]  # [B]
        return c[:, None] * self.g[b["pos"].clamp(0, MAX_POS - 1)] * b["cmask"]


def setup_xoff(a, tr, prand, hold, log):
    """XOff for train.py. Called only when --xoff > 0."""
    assert a.loss == "softmax", "--xoff acts on the listwise softmax loss only"
    assert hold is not None and prand is not None, "--xoff needs --holdout / p_random (--domain), as the recipe has"
    assert not (a.aug_cand or a.aug_hist), "--xoff is not combined with the wave-2 pool augmentation"
    assert a.val_mix <= 0, "--xoff is not combined with --val_mix (val feeds have no displayed position)"
    assert a.rdrop_cd <= 0 and a.sam_cd <= 0, "--xoff is not combined with the wave-7 second-pass arms (they re-build the logits without it)"
    assert not a.trunc_off, "--xoff is not combined with --trunc_off (one offset at a time)"
    prand_nn = prand.detach().cpu().numpy().astype(np.float64)
    return XOff(a.xoff, tr, prand_nn, hold, log, table_path=a.xoff_table)
