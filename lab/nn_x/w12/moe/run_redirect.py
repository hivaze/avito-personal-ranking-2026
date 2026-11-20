"""Output-redirecting runner of the w12/moe lane (lane_moe.sh smoke mode, and full runs with MOE_OUT=moe). Copy of
lab/nn_x/w12/pseudokd/tests/run_redirect.py without its "smoke" assert. Runs <train file> as __main__ exactly as
`python <train file> <args>` with PYTHONPATH=lab/nn_x/w5 would (sys.path = [dirname(train file), lab/nn_x/w5 modules, ...]), after
pointing the w5 common.CKPT / LOGS / PREDS at <outdir>/{ckpt,logs,preds}: train.py reads these names at its own import (`from common
import CKPT, LOGS, PREDS`) and passes them to every helper that writes (save_last / save_holdout / cliplog), so nothing of the run is
written into lab/nn_x/w5. Nothing else is patched, except with MOE_DIAG=1 (smokes; no effect on what the trainer computes):
  common.predict / common.eval_subset -> timed (CUDA-synchronised wall time) and the CUDA peak memory of the training segment before
      each eval is read and the peak counter reset, so that the train-step peak is separated from the eval / prediction peak;
  torch.Tensor.backward -> the scalar every .backward() is called on is kept (detached, on the device; no host sync per step);
  at exit: one line '[moe-diag] {json}' with the training steps, training seconds net of evals, steps/sec, peak allocated / reserved
      memory of the training segments (GB, incl. the resident splits; peak_step_alloc_gb / step_overhead_gb / resident_gb from the pure
      training segments after the first eval; warm_steps_per_s = steps / CUDA-synchronised wall time of those segments), memory allocated at the first eval, overall peak, and the mean
      training loss over the first / last 10 steps and the last 25% of the steps (written also to <outdir>/logs/<variant>_diag.json).
  usage: PYTHONPATH=<lab/nn_x/w5> [MOE_DIAG=1] python run_redirect.py <outdir> <train file> --variant w5_... <train.py args>
"""
import atexit
import json
import os
import runpy
import sys
import time
from pathlib import Path

outdir, train_file, args = Path(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3:]
HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == HERE:  # this runner's directory is not part of a `python <train file>` run
    sys.path.pop(0)
sys.path.insert(0, os.path.dirname(train_file))
import common as Cm  # noqa: E402  (lab/nn_x/w5/common.py through PYTHONPATH)

assert os.path.realpath(os.path.dirname(Cm.__file__)) == "<repo>/lab/nn_x/w5", Cm.__file__
assert os.path.realpath(outdir) != "<repo>/lab/nn_x/w5", "run_redirect.py: the output directory must not be lab/nn_x/w5"
for sub in ("ckpt", "logs", "preds", "preds/holdout"):
    (outdir / sub).mkdir(parents=True, exist_ok=True)
Cm.CKPT, Cm.LOGS, Cm.PREDS = outdir / "ckpt", outdir / "logs", outdir / "preds"
print(f"[run_redirect] outputs -> {outdir}/{{ckpt,logs,preds}}; train file {train_file}; sys.path[:3] {sys.path[:3]}", flush=True)

if os.environ.get("MOE_DIAG") == "1":
    import torch

    G = 2 ** 30
    DG = dict(calls=[], losses=[], mem_first_eval=None)  # calls: (seconds, peak alloc / reserved of the segment before the call)
    _predict, _eval_subset, _bw = Cm.predict, Cm.eval_subset, torch.Tensor.backward

    def _timed(fn):
        def w(*a, **k):
            torch.cuda.synchronize()
            pa, pr = torch.cuda.max_memory_allocated() / G, torch.cuda.max_memory_reserved() / G
            if DG["mem_first_eval"] is None:
                DG["mem_first_eval"] = torch.cuda.memory_allocated() / G
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            t1 = time.time()
            DG["calls"].append((t1 - t0, pa, pr, torch.cuda.max_memory_allocated() / G, DG.get("base", 0.0), t0, t1))
            torch.cuda.reset_peak_memory_stats()
            DG["base"] = torch.cuda.memory_allocated() / G  # resident memory at the start of the next training segment
            return out
        return w

    def _backward(self, *a, **k):
        if self.dim() == 0:
            DG["losses"].append(self.detach().clone())
        return _bw(self, *a, **k)

    Cm.predict, Cm.eval_subset = _timed(_predict), _timed(_eval_subset)
    torch.Tensor.backward = _backward
    T_START = time.time()

    @atexit.register
    def _report():
        import numpy as np
        L = np.array([float(x) for x in DG["losses"]]) if DG["losses"] else np.zeros(0)
        n = len(L)
        v = next((args[i + 1] for i, x in enumerate(args) if x == "--variant"), "?")
        try:
            js = json.load(open(outdir / "logs" / f"{v}.json"))
        except Exception:
            js = None
        nl = 2 * len(js["history"]) if js else len(DG["calls"])  # in-loop evals: predict(val) + eval_subset(holdout) each
        loop = DG["calls"][:nl]
        out = dict(variant=v, steps=n, train_minutes=js["train_minutes"] if js else None, in_loop_eval_calls=len(loop),
                   in_loop_eval_s=round(sum(c[0] for c in loop), 1),
                   peak_train_alloc_gb=round(max([c[1] for c in loop] or [0.0]), 2),
                   peak_train_reserved_gb=round(max([c[2] for c in loop] or [0.0]), 2),
                   peak_eval_alloc_gb=round(max([c[3] for c in DG["calls"]] or [0.0]), 2),
                   # pure training segments = before the 2nd, 3rd, ... in-loop predict (the 1st segment also holds the data loading)
                   peak_step_alloc_gb=round(max([c[1] for c in loop[2::2]] or [float("nan")]), 2),
                   step_overhead_gb=round(max([c[1] - c[4] for c in loop[2::2]] or [float("nan")]), 2),
                   resident_gb=round(min([c[4] for c in loop[2::2]] or [float("nan")]), 2),
                   mem_alloc_first_eval_gb=None if DG["mem_first_eval"] is None else round(DG["mem_first_eval"], 2),
                   wall_s=round(time.time() - T_START, 1),
                   loss_first10=float(L[:10].mean()) if n else None, loss_last10=float(L[-10:].mean()) if n else None,
                   loss_last25pct=float(L[-max(1, n // 4):].mean()) if n else None, loss_all=float(L.mean()) if n else None,
                   finite=bool(np.isfinite(L).all()) if n else None)
        if js and len(loop) >= 4:  # warm rate: the training segments after the first eval (no CUDA warm-up, no data loading)
            st = [h["step"] for h in js["history"]]
            seg_t = sum(loop[2 * i][5] - loop[2 * i - 1][6] for i in range(1, len(st)))
            out["warm_steps"] = st[-1] - st[0]
            out["warm_steps_per_s"] = round((st[-1] - st[0]) / seg_t, 3) if seg_t > 0 else None
        if js:
            net = js["train_minutes"] * 60 - out["in_loop_eval_s"]
            out["train_s_net_of_evals"] = round(net, 1)
            out["steps_per_s"] = round(n / net, 3) if net > 0 else None
        print("[moe-diag] " + json.dumps(out), flush=True)
        try:
            json.dump(out, open(outdir / "logs" / f"{v}_diag.json", "w"), indent=1)
        except Exception:
            pass

sys.argv = [train_file] + args
runpy.run_path(train_file, run_name="__main__")
