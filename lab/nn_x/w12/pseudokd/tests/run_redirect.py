"""Output-redirecting runner of the pseudokd plumbing smoke (lane_pk.sh PK_SMOKE=1). Runs <train file> as __main__ exactly as
`python <train file> <args>` with PYTHONPATH=nn_x/w5 would (sys.path = [dirname(train file), nn_x/w5 modules, ...]), after pointing the
nn_x/w5 common.CKPT / LOGS / PREDS at <outdir>/{ckpt,logs,preds}: train.py reads these names at its own import (`from common import
CKPT, LOGS, PREDS`) and passes them to every helper that writes (save_last / save_holdout / cliplog / pseudo head), so nothing of the
smoke is written into nn_x/w5. Nothing else is patched.
  usage: PYTHONPATH=<nn_x/w5> python tests/run_redirect.py <outdir> <train file> --variant w5_smoke_... <train.py args>
"""
import os
import runpy
import sys
from pathlib import Path

outdir, train_file, args = Path(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3:]
HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == HERE:  # this runner's directory is not part of a `python <train file>` run
    sys.path.pop(0)
sys.path.insert(0, os.path.dirname(train_file))
assert "smoke" in " ".join(args), "run_redirect.py is for smoke variants only"
import common as Cm  # noqa: E402  (nn_x/w5/common.py through PYTHONPATH)

assert os.path.realpath(os.path.dirname(Cm.__file__)) == "<repo>/lab/nn_x/w5", Cm.__file__
for sub in ("ckpt", "logs", "preds", "preds/holdout"):
    (outdir / sub).mkdir(parents=True, exist_ok=True)
Cm.CKPT, Cm.LOGS, Cm.PREDS = outdir / "ckpt", outdir / "logs", outdir / "preds"
print(f"[run_redirect] outputs -> {outdir}/{{ckpt,logs,preds}}; train file {train_file}; sys.path[:3] {sys.path[:3]}", flush=True)
sys.argv = [train_file] + args
runpy.run_path(train_file, run_name="__main__")
