"""nn_x/w5 WAVE-5 valcd arm: VAL-IN-COOLDOWN (idea D1 of reports/EXTERNAL_TOP1_1113.md; nn_x/audit/critique2/critique2.txt
item 3), used by nn_x/w5/train.py. Written 2026-11-12 ~23:35 UTC. OFF at the train.py default (--val_mix 0): train.py builds
the val pool only behind `if a.val_mix > 0` and reads it only through `if valcd is not None ...` blocks, so a flag-off run
executes exactly the previous nn_x/w5 code path (tests/test_valcd.py: AST proof against tests/ref_pre_valcd/train.py.txt +
end-to-end batch / logit / log / checkpoint identity; tests/test_w5.py, test_trunc_lifts.py and test_hzcd.py keep their own
proofs by stripping the valcd statements first, tests/strip_valcd.py). Imports numpy / torch / sklearn only (no GPU guard),
so the unit tests and the read-out (w5_eval_valcd.py) call the very functions train.py uses.

DESIGN (one design, chosen 2026-11-12 23:30 UTC; the alternative "replace a fraction of the m02 pool batches" was NOT chosen
because it halves the train exploration share, which the task keeps unchanged):
  --val_mix P (0 = off; arm value 0.01), --val_fold K (0-4)   ADDITIVE second exploration pool made of LABELLED VAL FEEDS.
    At every cooldown step (step >= t_cool; step 0 in the CD harness, --cooldown_frac 1.0 right after --init), AFTER the
    unchanged m02 draw (mix_rng, seed + 1: the train exploration pool batches land at exactly the same steps as in the m02ref
    run of the same CDSEED), a NON-pool step becomes a val step with probability P (own RNG stream, seed + 2): the regular
    train batch of that step is skipped and the model trains on one batch of val feeds instead. Effective val share =
    P * (1 - cooldown_mix) of the cooldown steps (~193 val batches of 19,670 at P 0.01 / mix 0.02 = ~3.8 passes over the
    11.5K cooldown val feeds; the m02 pool share 0.02 is unchanged). The m02 and regular streams are untouched: the sequence
    of train batches at non-val steps is the m02ref sequence.
    Folds: nn/finetune.py's construction EXACTLY (GroupKFold(5) over np.arange(n_val) grouped by user_id, user-0 feeds as
    singleton groups; val_folds below; sizes 2885/2885/2885/2884/2884). The run of fold K trains on the val feeds of the
    other 4 folds (fold != K) and NEVER on fold K: the fold-K feeds are its out-of-fold (OOF) read; the in-sample feeds must be
    masked out of every read-out (w5_eval_valcd.py assembles the OOF member from the 5 fold runs; test prediction = mean of
    the 5 runs' per-feed z-scores).
    Val batch = SplitData.gather of the VAL split (the val feature arrays: candidates, history, xt features of val, user_emb;
    labels cand_target; val has no cand_pos) with (ValPool.gather): b['dom'] = the RANDOM-LIKE domain token one-hot (0, 1, 0)
    = exactly what val / test are scored with (Ranker.infer_domain 1, common.predict passes no 'dom'; the train exploration
    pool feeds carry (1-p, p, 0) with p > 0.5, i.e. the same token up to the soft mix); b['pos'] = arange (the display
    position of the random order; it never reaches a loss: see next); b['fidx'] = val index + n_train, so that the loop's
    p_random lookups prand[b['fidx']] read the EXTENDED p_random vector (train p_random ++ ones(n_val)) that setup_valcd
    returns and train.py rebinds inside its guard: p = 1 for every val feed => the K9 aux weight fw_aux = 1 - p = 0 (no
    production-order target on exploration traffic, as for the random-like train feeds), the MSE aux mask 0, and
    --prand_base (unused in the harness) would give them the maximal feed weight. Batch size rule, loss (listwise, feed-
    normalised), lr schedule, aux annealing: the ones of the step they replace. is_pool is set True for val steps, so the
    cd_loss_pool diagnostic of train.py covers train-pool AND val batches (the valcd line reports the val-batch loss alone) and
    cooldown-pool augmentation / wave-4 streams (asserted off here) never touch them.
    NEVER applied to: feature_stats (model build; before --init anyway), the selection holdout (eval_subset reads tr; the
    holdout is train data, so --select_on holdout stays a label-free selection w.r.t. val: asserted), val / test predict,
    save_last / save_holdout / holdout_eval.py. The val split object itself is unchanged (no dom attached, no xt change), so
    predict(model, va) and eval_ndcg are the previous code; only the LOGGED full-val NDCG of a valcd run is inflated by its
    in-sample folds: the valcd log line adds the OOF (fold K) and in-sample NDCG at every eval and for the selected checkpoint.
    Not combined with --trunc_off / --trunc_lifts (TruncOffset indexes train arrays by fidx), --hcap / --rec_q / --rho_file
    (wave-4 streams), --aug_cand / --aug_hist (pool augmentation) or --hzcd (asserted).
    lane.sh arm: B:valcd:<root>:<fold> = B:m02ref + --val_mix 0.01 --val_fold <fold> -> variant w5_cd_valcd_r<root>_f<fold>
    (CDSEED 900 + root as every stage-B arm; roots 101-104 x folds 0-4 = 20 runs).
"""
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

N_FOLDS = 5
FOLD_SIZES = (2885, 2885, 2885, 2884, 2884)  # nn/finetune.py --u0_singleton folds of the 14,423 val feeds
DOM_VAL = (0.0, 1.0, 0.0)  # random-like domain token = Ranker.infer_domain 1, the token val / test are scored with


def val_folds(user_id, n_folds=N_FOLDS):
    """Fold id (int8) of every val feed, nn/finetune.py's construction with --u0_singleton EXACTLY: GroupKFold(n_folds).split
    (np.arange(n), groups) with groups = user_id, except user-0 feeds, which are singleton groups -1 - i. Fold k's held-out
    feeds (finetune's ho_f of iteration k) get id k."""
    uid = np.asarray(user_id).astype(np.int64)
    n = len(uid)
    groups = np.where(uid == 0, -1 - np.arange(n), uid)
    fold = np.full(n, -1, np.int8)
    for k, (_, ho) in enumerate(GroupKFold(n_splits=n_folds).split(np.arange(n), groups=groups)):
        assert (fold[ho] == -1).all()
        fold[ho] = k
    assert (fold >= 0).all()
    return fold


class ValPool:
    """The labelled val feeds of the cooldown folds (fold != k) as an additive second exploration pool of train.py."""

    def __init__(self, va, n_train, fold, k, p, cand_budget, max_feeds, seed, eval_ndcg=None):
        assert 0 <= int(k) < N_FOLDS, f"--val_fold must be 0-{N_FOLDS - 1}: {k}"
        assert 0.0 < float(p) < 1.0, f"--val_mix must be in (0, 1): {p}"
        assert len(fold) == va.n
        self.va, self.n_train, self.p, self.k = va, int(n_train), float(p), int(k)
        self.fold = np.asarray(fold)
        self.feeds = np.flatnonzero(self.fold != self.k)  # cooldown (in-sample) val feeds
        self.oof = np.flatnonzero(self.fold == self.k)  # never trained on: the OOF read of this run
        assert len(self.feeds) + len(self.oof) == va.n and not np.isin(self.feeds, self.oof).any()
        self.cand_budget, self.max_feeds = cand_budget, max_feeds
        self.rng = np.random.default_rng(seed + 2)  # own stream: the m02 draws / batches (seed + 1) stay unchanged
        self.queue, self.qi = [], 0
        self.is_val = False
        self.n_val = self.n_passes = self.n_feeds = 0
        self.loss_sum, self.loss_n = 0.0, 0
        self.dom = torch.tensor(DOM_VAL, device=va.device, dtype=torch.float32)
        self.eval_ndcg = eval_ndcg
        self.valcd = True

    def draw(self, is_pool):
        """One draw per cooldown step (after the m02 draw): True iff this NON-pool step trains on a val batch."""
        u = self.rng.random()
        self.is_val = (not is_pool) and u < self.p
        return self.is_val

    def next(self):
        """Next val batch of feed indices (fold != k), reshuffled by the own RNG when exhausted (as the m02 pool)."""
        if self.qi >= len(self.queue):
            self.queue = self.va.batches(self.feeds, self.cand_budget, self.max_feeds, shuffle=True, rng=self.rng)
            self.qi = 0
            self.n_passes += 1
        fb = self.queue[self.qi]
        self.qi += 1
        self.n_val += 1
        self.n_feeds += len(fb)
        return fb

    @torch.no_grad()
    def gather(self, fb):
        """SplitData.gather of the val split + the keys the training loop needs (see the module docstring)."""
        fb = np.asarray(fb)
        assert not np.isin(fb, self.oof).any(), "LEAK GUARD: OOF (fold k) val feeds in a cooldown batch"
        b = self.va.gather(fb, with_pos=True)
        assert "y" in b and "dom" not in b and "pos" not in b, "ValPool.gather expects the untouched val split (no dom / pos)"
        B, Mx = b["cmask"].shape
        b["dom"] = self.dom.expand(B, 3)
        b["pos"] = torch.arange(Mx, device=b["cmask"].device).expand(B, Mx)
        b["fidx"] = b["fidx"] + self.n_train
        return b

    def add_loss(self, loss):
        self.loss_sum += float(loss)
        self.loss_n += 1

    def summary(self, tag, pv):
        """Log line: val batches so far, their mean loss, and this fold's OOF vs in-sample NDCG of the val preds `pv`."""
        oof = self.eval_ndcg(pv, self.va, self.oof) if self.eval_ndcg is not None else float("nan")
        ins = self.eval_ndcg(pv, self.va, self.feeds) if self.eval_ndcg is not None else float("nan")
        return (f"wave-5 valcd [{tag}]: val batches {self.n_val} ({self.n_feeds} feeds, {self.n_passes} passes started), "
                f"mean val-batch loss {self.loss_sum / max(self.loss_n, 1):.4f}; val NDCG OOF fold {self.k} ({len(self.oof)} "
                f"feeds) {oof:.5f} | in-sample folds != {self.k} ({len(self.feeds)} feeds) {ins:.5f}")


def setup_valcd(a, tr, va, prand, hold, t_cool, total, log, eval_ndcg=None):
    """(ValPool, extended p_random) for train.py. Called only behind `if a.val_mix > 0`. The extended p_random =
    cat(train p_random, ones(n_val)) replaces `prand` in train.py's loop lookups prand[b['fidx']] (val rows: p = 1)."""
    assert 0.0 < a.val_mix < 1.0, f"--val_mix must be in (0, 1): {a.val_mix}"
    assert 0 <= a.val_fold < N_FOLDS, f"--val_fold must be 0-{N_FOLDS - 1}: {a.val_fold}"
    assert prand is not None, "--val_mix needs p_random (use --holdout or --domain)"
    assert a.select_on == "holdout", "--val_mix: checkpoint selection must not read val labels (--select_on holdout)"
    assert hold is not None, "--val_mix needs --holdout (the label-free selection judge)"
    assert t_cool < total, "--val_mix: the cooldown never starts (t_cool >= total steps)"
    assert not (getattr(a, "trunc_off", 0) or getattr(a, "trunc_lifts", "")), "--val_mix is not combined with TRUNC-PROD"
    assert not (getattr(a, "hcap", "") or getattr(a, "rec_q", 0) > 0 or getattr(a, "rho_file", "")), \
        "--val_mix is not combined with the wave-4 cooldown streams"
    assert not (getattr(a, "aug_cand", 0) or getattr(a, "aug_hist", 0)), "--val_mix is not combined with --aug_*"
    assert not getattr(a, "hzcd", False), "--val_mix is not combined with --hzcd"
    assert va.split == "val" and va.has_target and va.dom is None and not va.has_pos, "ValPool needs the untouched val split"
    assert len(prand) == tr.n
    t0 = time.time()
    fold = val_folds(va.user_id)
    sizes = tuple(int(x) for x in np.bincount(fold, minlength=N_FOLDS))
    assert sizes == FOLD_SIZES, f"val folds {sizes} != nn/finetune.py's {FOLD_SIZES}"
    pool = ValPool(va, tr.n, fold, a.val_fold, a.val_mix, a.cand_budget, a.max_feeds, a.seed, eval_ndcg)
    prand_ext = torch.cat([prand, torch.ones(va.n, device=prand.device, dtype=prand.dtype)])
    nb = len(va.batches(pool.feeds, a.cand_budget, a.max_feeds, shuffle=False))
    exp_val = a.val_mix * (1.0 - a.cooldown_mix) * (total - t_cool)
    log(f"wave-5 valcd: from step {t_cool} (of {total}) each NON-pool cooldown step trains on a batch of LABELLED VAL feeds "
        f"w.p. {a.val_mix} (own RNG seed {a.seed + 2}; the m02 draws / batches are unchanged): fold {a.val_fold} of "
        f"{N_FOLDS} held out (OOF {len(pool.oof)} feeds), cooldown val feeds {len(pool.feeds)} (folds {sizes}), "
        f"{nb} batches per pass, expected val batches {exp_val:.0f} (~{exp_val / max(nb, 1):.1f} passes); val batches carry "
        f"the random-like domain token {DOM_VAL}, p_random 1 (aux weight 0), fidx offset {tr.n}; selection on the holdout; "
        f"{time.time() - t0:.1f}s")
    return pool, prand_ext
