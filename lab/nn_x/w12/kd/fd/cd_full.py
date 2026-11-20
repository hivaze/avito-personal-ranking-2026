"""FULLDATA helper of train_full.py / train_full_abl.py (--holdout_in_train; reports/STALE_1115/fulldata.md).

OFF at the default: the train files build a FullData object only behind `if a.holdout_in_train` and use it only through
`if fd is not None` guards (or the conditional expressions `hold if fd is None else None`, `... and fd is None`), which reduce
to the original statement when fd is None, so a flag-off run executes exactly the original code path (test_full.py).

With the flag (owner 2026-11-15: to make sure that every submitted solution is trained on all data):
  * the 30,000 holdout feeds are drawn exactly as before (train.py: p_random > 0.9, M >= 50, rng 12345) -> hold / hold_rows,
    p_random, the domain token and every log line up to "holdout: ..." are unchanged;
  * put_back(): the training feed list is the list BEFORE the holdout was removed (all 3,151,208 feeds, or the --train_feeds
    subsample); every later consumer of `feeds` (feature_stats of a from-scratch run, the epoch batches, the cooldown pool)
    therefore sees the holdout feeds like any other feed. With a distillation file (--kd_file) their target is
    (1 - kd_w) y / sum(y) + kd_w softmax(teacher[ci] / T), the teacher rows of those feeds (out-of-sample predictions of the
    teacher members, which all trained with --holdout 30000: reports/RECHECK_1114/kd_audit.md; re-checked in teacher_holdout.py);
  * the cooldown exploration pool keeps its holdout feeds (train.py passes hold=None to exploration_pool, whose leak guard
    is for held-out runs): 134,118 -> 164,118 feeds at the default threshold;
  * NO checkpoint selection: a.keep = 'last' and the selection score is -inf at every eval, so the kept checkpoint is the
    last step (the holdout is in-sample; val is never a selection metric in this recipe). --time_limit is refused (an early
    break would leave no checkpoint);
  * the holdout NDCG of every eval is logged on its own line with the word IN-SAMPLE and stored in logs/<v>_fulldata.json;
    the regular `holdout_ndcg` log field / history entry is nan, so no reader can take it for an out-of-sample number;
  * holdout predictions: preds/holdout/nn_<v>_holdoutINSAMPLE.npy (kept = last checkpoint), with --save_last also
    preds/holdout/nn_<v>@last_holdoutINSAMPLE.npy, with --hzcd also nn_<v>_holdoutINSAMPLE_hz.npy. The out-of-sample names
    nn_<v>_holdout.npy / nn_<v>@last_holdout.npy / nn_<v>_holdout_hz.npy are never written.
Refused together with the flag (they use the holdout as a held-out set or re-define the training feeds; not needed for the
finals): --xfold / --xfold_unlab, --val_mix, --hcap, --rec_q, --rho_file, --trunc_off, --prand2, --only_random, --time_limit.
Imports numpy / torch only (+ cd_arms5.holdout_preds lazily), so the CPU tests call the very class the train files use.
"""
import json

import numpy as np

SUFFIX = "holdoutINSAMPLE"
N_HOLD = 30000


class FullData:
    def __init__(self, a, feeds, hold, hold_rows, tr, log):
        """Called right after hold / hold_rows are built and BEFORE train.py removes the holdout from `feeds`."""
        assert a.holdout == N_HOLD and len(hold) == N_HOLD, "--holdout_in_train: the fixed 30,000-feed holdout only (--holdout 30000)"
        bad = [k for k, on in (("xfold", getattr(a, "xfold", -1) >= 0), ("xfold_unlab", getattr(a, "xfold_unlab", False)),
                               ("val_mix", a.val_mix > 0), ("hcap", bool(a.hcap)), ("rec_q", a.rec_q > 0),
                               ("rho_file", bool(a.rho_file)), ("trunc_off", bool(a.trunc_off)),
                               ("prand2", bool(getattr(a, "prand2", ""))), ("only_random", a.only_random >= 0),
                               ("time_limit", bool(a.time_limit))) if on]
        assert not bad, f"--holdout_in_train is not combined with {bad} (see cd_full.py)"
        self.before = np.array(feeds, copy=True)
        self.hold, self.hold_rows = hold, hold_rows
        self.n_hold_in = int(np.isin(self.before, hold).sum())  # 30,000 unless --train_feeds subsamples
        self.hold_cands_in = int(tr.M_np[np.intersect1d(self.before, hold)].sum())
        self.hist, self.files, self.pool_hold, self.epoch_hold = [], [], None, None
        self.keep_before = a.keep
        a.keep = "last"  # with sel = -inf at every eval (train file) the kept checkpoint is the last step
        log(f"FULLDATA --holdout_in_train: the {len(hold)} holdout feeds stay identified but are NOT removed from training; "
            f"checkpoint = LAST step (a.keep {self.keep_before!r} -> 'last', --select_on {a.select_on!r} ignored); their NDCG is "
            f"IN-SAMPLE and never used")

    def put_back(self, feeds, log):
        """`feeds` = np.setdiff1d(before, hold) as train.py just computed it; returns the list before the removal."""
        assert np.array_equal(np.setdiff1d(self.before, self.hold), feeds), "FULLDATA: unexpected training feed list"
        assert len(self.before) == len(feeds) + self.n_hold_in and np.all(np.diff(self.before) > 0)
        log(f"FULLDATA: {self.n_hold_in} holdout feeds ({self.hold_cands_in} candidates) put back: training feeds "
            f"{len(feeds)} -> {len(self.before)}")
        return self.before

    def pool_note(self, pool, log):
        self.pool_hold = int(np.isin(pool, self.hold).sum())
        log(f"FULLDATA: cooldown exploration pool {len(pool)} feeds, of which {self.pool_hold} holdout feeds (IN the pool)")

    def steps_note(self, epoch_batches, total, a, tr, log):
        ep0 = np.concatenate(epoch_batches[0])
        self.epoch_hold = int(np.isin(ep0, self.hold).sum())
        assert self.epoch_hold == self.n_hold_in and len(ep0) == len(self.before) and len(np.unique(ep0)) == len(ep0), \
            "FULLDATA: every training feed (holdout included) must appear exactly once per epoch"
        n_without = len(tr.batches(np.setdiff1d(self.before, self.hold), a.cand_budget, a.max_feeds, shuffle=False))
        self.steps = dict(steps_per_epoch=len(epoch_batches[0]), steps_per_epoch_without_holdout=n_without, total=int(total))
        log(f"FULLDATA: each epoch holds every training feed once, {self.epoch_hold} of them holdout feeds; steps/epoch "
            f"{len(epoch_batches[0])} (the same feeds without the holdout: {n_without}), total steps {total}")

    def insample(self, step, ndh, log):
        """Log the in-sample holdout NDCG of an eval on its own line; return nan for the regular log / history field."""
        self.hist.append(dict(step=int(step), holdout_insample_ndcg=float(ndh)))
        log(f"FULLDATA step {step}: holdout IN-SAMPLE ndcg {ndh:.5f} (these feeds are trained on: not a generalisation "
            f"estimate, never used for selection; the holdout_ndcg field of the next line is nan on purpose)")
        return float("nan")

    def _write(self, name, sh, tr, preds_dir, ndcg, log, what):
        hd = preds_dir / "holdout"
        hd.mkdir(exist_ok=True)
        path = hd / f"nn_{name}.npy"
        assert SUFFIX in path.name
        np.save(path, sh)
        M_h = tr.M_np[self.hold]
        offs = np.concatenate([[0], np.cumsum(M_h)[:-1]])
        nd = float(ndcg(sh, tr.target_np[self.hold_rows], offs, M_h))
        self.files.append(dict(file=str(path), insample_ndcg=nd, what=what))
        log(f"FULLDATA saved holdout preds of {what} {sh.shape} -> {path.name} (IN-SAMPLE NDCG {nd:.5f})")
        return nd

    def save_insample(self, name, model, tr, preds_dir, predict, ndcg, log, what="the last checkpoint"):
        """Same numerics as cd_arms5.save_holdout / save_last (holdout_preds: predict without TF32), IN-SAMPLE file name."""
        from cd_arms5 import holdout_preds
        sh = holdout_preds(predict, model, tr, self.hold, self.hold_rows)
        return self._write(f"{name}_{SUFFIX}", sh, tr, preds_dir, ndcg, log, what)

    def finish(self, a, model, tr, tr_hz, best_step, total, preds_dir, logs_dir, predict, ndcg, log):
        """After the kept checkpoint is loaded and saved: its holdout preds (as built; through the hz view with --hzcd)."""
        assert best_step == total, f"FULLDATA: the kept checkpoint must be the last step ({best_step} != {total})"
        self.save_insample(a.variant, model, tr, preds_dir, predict, ndcg, log, what="the kept (= LAST) checkpoint")
        if tr_hz is not None:  # --hzcd: cd_arms6.save_holdout_hz numerics (predict through the view, no TF32), IN-SAMPLE name
            import torch
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                sh = predict(model, tr_hz, self.hold)[self.hold_rows].astype(np.float32)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev
            self._write(f"{a.variant}_{SUFFIX}_hz", sh, tr, preds_dir, ndcg, log, "the kept (= LAST) checkpoint through the hz view")
        out = dict(variant=a.variant, holdout_in_train=True, kept_step=int(best_step), total_steps=int(total),
                   n_holdout_feeds_in_training=self.n_hold_in, holdout_candidates_in_training=self.hold_cands_in,
                   n_training_feeds=int(len(self.before)), pool_holdout_feeds=self.pool_hold,
                   holdout_feeds_per_epoch=self.epoch_hold, steps=getattr(self, "steps", None),
                   keep_arg_before=self.keep_before, insample_history=self.hist, insample_files=self.files,
                   note="every holdout number of this run is IN-SAMPLE; val is the only clean witness")
        (logs_dir / f"{a.variant}_fulldata.json").write_text(json.dumps(out, indent=2))
        log(f"FULLDATA summary -> {a.variant}_fulldata.json (kept step {best_step} = last; {self.n_hold_in} holdout feeds trained on)")
