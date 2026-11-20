"""FULLVAL helper of finetune_full.py / finetune_full_abl.py (--full_val; reports/STALE_1115/fulldata.md section 5).

OFF at the default: the fine-tune copies build a FullVal object only behind `if a.full_val` and use it only through
`if fv is not None` guards or conditional expressions `X if fv is None else Y` that reduce to the original expression X,
so a flag-off run executes exactly the original finetune.py code path (test_ft_full.py: AST proof).

With the flag: "fine-tune on all of val". The production recipe (n2cpv3s3) fits 5 user-grouped folds x 3 seeds, i.e. 15
models on 80% of val each; here ONE fit set = all 14,423 val feeds and `--seeds` models (seed s: torch.manual_seed(s),
numpy rng s; the same per-epoch batching, warm-up, cosine decay, EMA, final-EMA predictions, extra-feature branch).
  * a.folds is forced to 1, so every normalisation of the original code (/ a.folds, / a.seeds, test_folds[k]) holds;
  * the loop's "held-out" feed list is ALL val feeds: the val predictions it accumulates are IN-SAMPLE. They are written
    as preds/nn_<out>_valINSAMPLE.npy, never as ..._val_oof.npy, and every "OOF" figure the original code logs after the
    loop is announced as in-sample; <out>_ft.json is rewritten with insample_* keys. Only "epoch 0" (the starting
    train-only model, before any step) is a genuine out-of-sample val number;
  * refused: --pseudo, --add_random (not part of the recipe), --select best without --ema (it would select an epoch on
    in-sample NDCG), an existing ..._val_oof.npy of the same --out (a fold run and a full-val run must not share a name).
There is NO val witness for a full-val model: the only measurements are a nested simulation inside val or the public
leaderboard. Imports numpy only.
"""
import json

import numpy as np


class FullVal:
    def __init__(self, a, out, log):
        assert not a.pseudo and not a.add_random, "--full_val is not combined with --pseudo / --add_random"
        assert a.ema > 0 or a.select == "snapshot", "--full_val: '--select best' would pick an epoch on IN-SAMPLE NDCG"
        assert "INSAMPLE" not in out and not out.endswith("_n2cpv3s3") and not out.endswith("_n2cpv3"), \
            f"--full_val: choose an --out name of its own (got {out!r}; the fold recipes own *_n2cpv3 / *_n2cpv3s3)"
        self.out, self.folds_before = out, a.folds
        self.val_name = f"nn_{out}_valINSAMPLE.npy"
        a.folds = 1
        log(f"FULLVAL --full_val: every model is fitted on ALL val feeds ({a.seeds} seeds, no folds: a.folds {self.folds_before} -> 1); "
            f"there is NO out-of-fold prediction: val predictions of this run are IN-SAMPLE -> {self.val_name}")

    def splits(self, n):
        """The loop's (fit, 'held-out') pairs: one pair, both = all val feeds."""
        allf = np.arange(n)
        yield allf, allf

    def warn(self, log):
        log("FULLVAL: the 'OOF ndcg' / 'best epoch' / 'snapshot mean' / 'final EMA weights' lines below are IN-SAMPLE "
            "(every val feed was trained on); only epoch 0 = the starting train-only model is out of sample")

    def finish(self, logs_dir, preds_dir, log):
        assert not (preds_dir / f"nn_{self.out}_val_oof.npy").exists(), "FULLVAL: a ..._val_oof.npy file of this --out exists"
        assert (preds_dir / self.val_name).exists() and (preds_dir / f"nn_{self.out}_test.npy").exists()
        p = logs_dir / f"{self.out}_ft.json"
        j = json.loads(p.read_text())
        new = {}
        for k, v in j.items():
            new[k.replace("oof_", "insample_") if k.startswith("oof_") else k] = v
        new["full_val"] = True
        new["train_only_val_ndcg_epoch0"] = j["oof_by_epoch"][0]
        new["note"] = ("FULLVAL: every model saw all val feeds; all val figures except epoch 0 are IN-SAMPLE; "
                       f"val predictions in {self.val_name}; folds argument was {self.folds_before}")
        p.write_text(json.dumps(new, indent=2))
        log(f"FULLVAL: summary rewritten with insample_* keys -> {p.name}; test predictions nn_{self.out}_test.npy = mean of the full-val models")
