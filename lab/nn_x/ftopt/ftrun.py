"""Run the UNCHANGED finetune.py of a member's code dir with outputs re-pointed to nn_x/ftopt.

Usage: CUDA_VISIBLE_DEVICES=N python nn_x/ftopt/ftrun.py --code r6|nn --outdir DIR [--no_test] -- <finetune.py args>

Only module globals are re-pointed: finetune.PREDS / finetune.LOGS -> DIR (so nothing is written to preds/ or
nn*/logs), and with --no_test the test-split predict() calls return zeros (val OOF is all the study scores;
predict() runs in eval mode and draws no RNG, so the training trajectory is unchanged).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path("<repo>")
CODE = {"nn": ROOT / "nn", "r6": ROOT / "nn_x" / "r6"}

argv = sys.argv[1:]
cut = argv.index("--") if "--" in argv else len(argv)
p = argparse.ArgumentParser()
p.add_argument("--code", required=True, choices=list(CODE))
p.add_argument("--outdir", required=True)
p.add_argument("--no_test", action="store_true")
a = p.parse_args(argv[:cut])
sys.path.insert(0, str(CODE[a.code]))

import finetune  # noqa: E402

out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
finetune.PREDS = out
finetune.LOGS = out
if a.no_test:
    _orig = finetune.predict

    def _predict(model, data, feed_ids=None, *args, **kw):
        if getattr(data, "split", None) == "test":
            return np.zeros(data.n_cand, np.float32)
        return _orig(model, data, feed_ids, *args, **kw)

    finetune.predict = _predict
sys.argv = ["finetune.py"] + argv[cut + 1:]
finetune.main()
