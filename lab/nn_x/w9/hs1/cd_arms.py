"""r6 wave-2 cooldown-continuation (CD) arms (reports/R6_PLAN.md section 4.1 "Wave 2"), used by nn_x/r6/train.py.
Every helper is OFF at its train.py default, and train.py calls them only behind `if <flag on>` guards, so a flag-off run
executes exactly the previous code path (tests/test_cdw2.py). Imports only numpy/torch (no GPU guard), so the unit tests
call the very functions train.py uses.

  --drop_cd P   (default -1 = off)  set_dropout: dropout P on EVERY dropout site of the Ranker from the cooldown start
                                    (step t_cool; = step 0 in the CD harness, --cooldown_frac 1.0, i.e. right after --init)
  --pool_p T    (default 0.5 = off) exploration_pool: cooldown pool = training feeds with p_random > T (holdout excluded)
  --aug_cand A  (default 0 = off)   augment_pool_batch (r6_external R2): on cooldown POOL batches only, per feed
                                    q ~ U(A, 1) and each NEGATIVE candidate kept w.p. q (every positive kept)
  --aug_hist H  (default 0 = off)   augment_pool_batch (R2): on cooldown POOL batches only, drop each history event w.p. H
"""
import numpy as np
import torch
import torch.nn as nn

POOL_P_DEFAULT = 0.5  # the threshold of nn/train.py and every earlier r6 run (p_random > 0.5)


def dropout_sites(model):
    """(nn dropout modules, attention modules with probability dropout) of an nn_x/r6 Ranker. Covers every dropout site of
    model.py: nn.Dropout in each Block (FFN hidden + residual branch `dr`) and in HistRetrieval / CandRetrieval (value
    MLP `dp`), plus the attention-probability dropout of every Attn (F.scaled_dot_product_attention dropout_p, which is
    Attn.attn_drop when set, else Attn.drop). Not dropout sites in this sense: xt_drop / id_drop (input blanking for the
    cold simulation, 0 / unused in F7) and nn.GRU (single layer, no inter-layer dropout)."""
    mods = [m for m in model.modules() if isinstance(m, nn.modules.dropout._DropoutNd)]
    attns = [m for m in model.modules() if type(m).__name__ == "Attn"]
    for m in attns:
        assert hasattr(m, "attn_drop") and hasattr(m, "drop"), "set_dropout needs the nn_x/r6 Attn (attn_drop attribute)"
    return mods, attns


def set_dropout(model, p):
    """--drop_cd: set dropout probability p on every dropout site (see dropout_sites), INCLUDING attention-probability
    dropout. Exact: dropout has no parameters or state, so weights, buffers, eval-mode outputs and model.cfg are
    unchanged (the saved checkpoint keeps cfg drop=0.1, stays loadable by nn/model.py, and a later fine-tune of it uses
    the standard recipe dropout). Returns (#dropout modules, #attention modules) changed."""
    assert 0.0 <= p < 1.0, f"--drop_cd must be in [0, 1): {p}"
    mods, attns = dropout_sites(model)
    for m in mods:
        m.p = float(p)
    for m in attns:
        m.attn_drop = float(p)  # Attn.forward: pdrop = drop if attn_drop is None else attn_drop
    return len(mods), len(attns)


def exploration_pool(feeds, prand_np, thr=POOL_P_DEFAULT, hold=None):
    """Cooldown exploration pool = the training feeds with p_random > thr (nn/train.py: thr 0.5). prand_np is 0 for M<50
    feeds (train.py zeroes it), so for any thr >= 0 the pool holds only M>=50 feeds (sol_solvers 2.6: short feeds hurt).
    `feeds` already excludes the 30K holdout (train.py); the assert is the explicit leak guard."""
    assert thr >= 0.0, f"--pool_p must be >= 0 (M<50 feeds have p_random 0): {thr}"
    pool = feeds[prand_np[feeds] > thr]
    if hold is not None:
        assert not np.isin(pool, hold).any(), "LEAK GUARD: holdout feeds in the cooldown exploration pool"
    return pool


def aug_enabled(aug_cand, aug_hist):
    return 0.0 < aug_cand < 1.0 or aug_hist > 0.0


def augment_pool_batch(b, aug_cand=0.0, aug_hist=0.0):
    """R2 (reports/r6_external.md R2), in place on a gathered TRAIN pool batch, after tr.gather and before model.dense.
    Candidate subset (0 < aug_cand < 1): per feed q ~ U(aug_cand, 1); every positive and each negative w.p. q stays.
      Labels only decide which rows stay in the loss (all positives kept, as in R2); no label value enters any input.
      A feed that would lose every candidate keeps all of them (cannot happen with positives present; NaN guard).
    History-event drop (aug_hist > 0): each event dropped w.p. aug_hist from hmask and hemb_ok; recency positions,
      b['H'] (log H feature) and the event order stay (R2).
    Every in-model input is recomputed from these masks (dense features, within-feed z/rank, attention key masks,
    retrieval, loss masks); the precomputed xt columns still describe the full feed (R2's 'stale' caveat). Only the keys
    cmask, y, hmask, hemb_ok change. Uses the global torch RNG of the batch device. Never called at eval/predict time
    (common.predict / eval_subset / holdout_eval.py gather their own batches). Returns b."""
    if 0.0 < aug_cand < 1.0:
        y, cm = b["y"], b["cmask"]
        q = aug_cand + (1.0 - aug_cand) * torch.rand(y.shape[0], 1, device=y.device)
        keep = (torch.rand_like(y) < q) | (y > 0)
        new = cm & keep
        new = torch.where(new.any(1, keepdim=True), new, cm)
        b["cmask"] = new
        b["y"] = y * new
    if aug_hist > 0.0:
        hk = torch.rand(b["hmask"].shape, device=b["hmask"].device) >= aug_hist
        b["hmask"] = b["hmask"] & hk
        b["hemb_ok"] = b["hemb_ok"] & hk
    return b
