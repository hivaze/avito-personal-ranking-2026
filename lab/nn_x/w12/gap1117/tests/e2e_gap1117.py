"""End-to-end wrapper of the gap1117 GPU tests (tests/test_gap1117_gpu.py). Runs a train.py FILE as __main__ with the given arguments
after (1) optionally (E2E_DET=1) switching torch to deterministic kernels (as nn_x/w5/tests/e2e_w7.py), (2) redirecting every output
of the run (common.CKPT / LOGS / PREDS) to <outdir>/{ckpt,logs,preds} so that nothing is written into nn_x/w5, and (3) wrapping,
without changing results:
  data.SplitData.gather (with_pos=True = one training gather per step) -> feed ids of every step + a synchronised wall clock at the
      start of every step (step time = difference of consecutive starts; intervals containing an eval are flagged);
  torch.Tensor.backward                -> the value of the loss of every step (the only .backward call of train.py);
  model.listwise_loss                  -> the loss-input logits of every step;
  torch.optim.AdamW.step               -> global L2 norm of all gradients at the optimiser entry (after clipping);
  cd_arms9.GapLoss.add_terms (if the file imports cd_arms9) -> call count, added term value (returned loss - input loss), finiteness;
  common.predict / eval_subset         -> counts (eval flags).
  usage: python e2e_gap1117.py <train file> <outdir> <record.npz> --variant w5_... <train.py args>
"""
import json
import os
import runpy
import sys
import time
from pathlib import Path

W5 = "<repo>/lab/nn_x/w5"
sys.path.insert(0, W5)
train_file, outdir, rec_path, args = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
DET = os.environ.get("E2E_DET", "0") == "1"
if DET:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np  # noqa: E402
import torch  # noqa: E402

if DET:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
import common as Cm  # noqa: E402
import data as D  # noqa: E402
import model as Mo  # noqa: E402

for sub in ("ckpt", "logs", "preds", "preds/holdout"):
    (outdir / sub).mkdir(parents=True, exist_ok=True)
Cm.CKPT, Cm.LOGS, Cm.PREDS = outdir / "ckpt", outdir / "logs", outdir / "preds"

REC = dict(g_fidx=[], t_start=[], ev_flag=[], loss=[], lg_loss=[], gnorm=[], add_n=0, add_val=[], add_finite=True, n_predict=0, n_eval=0)
ST = {"model": None, "evals_at_last_step": 0}
_gather, _ll, _bwd, _adam = D.SplitData.gather, Mo.listwise_loss, torch.Tensor.backward, torch.optim.AdamW.step
_predict, _eval = Cm.predict, Cm.eval_subset


def gather(self, fidx_np, with_pos=False):
    if with_pos:
        torch.cuda.synchronize()
        REC["t_start"].append(time.perf_counter())
        n_ev = REC["n_predict"] + REC["n_eval"]
        REC["ev_flag"].append(n_ev != ST["evals_at_last_step"])  # an eval ran inside the previous step's interval
        ST["evals_at_last_step"] = n_ev
        REC["g_fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
    return _gather(self, fidx_np, with_pos)


def listwise_loss(logit, *a, **k):
    REC["lg_loss"].append(logit.detach().float().cpu().numpy().ravel())
    return _ll(logit, *a, **k)


def backward(self, *a, **k):
    if self.dim() == 0 and self.requires_grad:
        REC["loss"].append(float(self.detach().double()))
    return _bwd(self, *a, **k)


def adam_step(self, *a, **k):
    with torch.no_grad():
        ps = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
        REC["gnorm"].append(float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in ps))))
    return _adam(self, *a, **k)


def predict(model, data, *a, **k):
    REC["n_predict"] += 1
    return _predict(model, data, *a, **k)


def eval_subset(model, data, *a, **k):
    REC["n_eval"] += 1
    return _eval(model, data, *a, **k)


D.SplitData.gather = gather
Mo.listwise_loss = listwise_loss
torch.Tensor.backward = backward
torch.optim.AdamW.step = adam_step
Cm.predict, Cm.eval_subset = predict, eval_subset
try:
    import cd_arms9 as C9
    _add = C9.GapLoss.add_terms

    def add_terms(self, loss, *a, **k):
        out = _add(self, loss, *a, **k)
        REC["add_n"] += 1
        d = float((out - loss).detach().double()) if out is not loss else 0.0
        REC["add_val"].append(d)
        REC["add_finite"] &= bool(np.isfinite(d)) and bool(torch.isfinite(out).all())
        return out
    C9.GapLoss.add_terms = add_terms
except ImportError:
    pass
sys.argv = ["train.py"] + args
try:
    runpy.run_path(train_file, run_name="__main__")
finally:
    torch.cuda.synchronize()
    REC["t_start"].append(time.perf_counter())
    REC["ev_flag"].append(REC["n_predict"] + REC["n_eval"] != ST["evals_at_last_step"])
    g = REC["g_fidx"]
    ts = np.array(REC["t_start"])
    np.savez(rec_path, g_fidx=np.concatenate(g) if g else np.zeros(0, np.int64),
             g_off=np.concatenate([[0], np.cumsum([len(x) for x in g])]).astype(np.int64),
             step_time=np.diff(ts), step_eval=np.array(REC["ev_flag"][1:], bool), loss=np.array(REC["loss"]),
             lg_loss=np.concatenate(REC["lg_loss"]) if REC["lg_loss"] else np.zeros(0, np.float32),
             gnorm=np.array(REC["gnorm"]), add_n=np.array(REC["add_n"]), add_val=np.array(REC["add_val"]),
             add_finite=np.array(REC["add_finite"]), n_predict=np.array(REC["n_predict"]), n_eval=np.array(REC["n_eval"]),
             meta=np.array(json.dumps(dict(det=DET, n_steps=len(g), train_file=train_file))))
