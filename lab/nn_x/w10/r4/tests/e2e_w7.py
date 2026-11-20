"""Deterministic end-to-end wrapper for tests/test_w7.py (GPU). Runs a train.py FILE as __main__ with the given arguments
after (1) switching torch to deterministic kernels (as tests/e2e_valcd.py) and (2) wrapping, without changing results:
  data.SplitData.gather   -> records the feed ids of every training gather (with_pos=True; one per training step) and
                             advances the step counter that groups the forwards below;
  model.Ranker.forward    -> TRAIN-mode forwards: the step index, the logits, the global L2 norm of all parameters and, from the
                             second forward of a step on, ||params - params at the step's first forward||_2 (SAM: must be rho)
                             and whether every parameter is bitwise equal to the first forward's (R-Drop: must be True);
                             the first forward of a step snapshots the parameters;
  model.listwise_loss     -> loss-input logits (every call: pass 1 inline, pass 2 replica);
  torch.optim.AdamW.step / torch.optim.Muon.step -> at entry: whether every parameter is bitwise equal to the step's first-
                             forward snapshot (SAM must have restored w exactly) and the global L2 norm of the gradients;
  common.predict / eval_subset -> counts.
  usage: python tests/e2e_w7.py <train file> <record.npz> --variant w5_... <train.py args>
"""
import json
import os
import runpy
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = os.path.dirname(os.path.abspath(__file__))
W5 = os.path.dirname(HERE)
sys.path.insert(0, W5)
train_file, rec_path, args = sys.argv[1], sys.argv[2], sys.argv[3:]
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
import common as Cm  # noqa: E402
import data as D  # noqa: E402
import model as Mo  # noqa: E402

REC = dict(g_fidx=[], f_step=[], f_shape=[], lg_model=[], f_pnorm=[], f_eps=[], f_same=[], lg_loss=[], o_step=[], o_same=[], o_gnorm=[],
           n_eval=0, n_predict=0)
ST = {"step": -1, "snap": None, "model": None}
_gather, _dense, _fwd, _ll, _eval, _predict = D.SplitData.gather, Mo.Ranker.dense, Mo.Ranker.forward, Mo.listwise_loss, Cm.eval_subset, Cm.predict
_adam_step, _muon_step = torch.optim.AdamW.step, torch.optim.Muon.step


def _params(model):
    return [p for p in model.parameters() if p.requires_grad]


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    if with_pos:
        REC["g_fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
        ST["step"] += 1
        ST["snap"] = None
    return b


def forward(self, *a, **k):
    if self.training:
        ST["model"] = self
        ps = _params(self)
        with torch.no_grad():
            pn = float(torch.sqrt(sum((p.float() ** 2).sum() for p in ps)))
            if ST["snap"] is None:
                ST["snap"] = [p.detach().clone() for p in ps]
                eps, same = 0.0, True
            else:
                eps = float(torch.sqrt(sum(((p - s).float() ** 2).sum() for p, s in zip(ps, ST["snap"]))))
                same = all(torch.equal(p, s) for p, s in zip(ps, ST["snap"]))
    out = _fwd(self, *a, **k)
    if self.training:
        REC["f_step"].append(ST["step"])
        REC["f_shape"].append(out.shape)
        REC["lg_model"].append(out.detach().float().cpu().numpy())
        REC["f_pnorm"].append(pn)
        REC["f_eps"].append(eps)
        REC["f_same"].append(same)
    return out


def listwise_loss(logit, *a, **k):
    REC["lg_loss"].append(logit.detach().float().cpu().numpy())
    return _ll(logit, *a, **k)


def _opt_entry(opt):
    m = ST["model"]
    if m is None or ST["snap"] is None:
        return
    ps = _params(m)
    with torch.no_grad():
        same = all(torch.equal(p, s) for p, s in zip(ps, ST["snap"]))
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None)))
    REC["o_step"].append(ST["step"])
    REC["o_same"].append(same)
    REC["o_gnorm"].append(gn)


def adam_step(self, *a, **k):
    _opt_entry(self)
    return _adam_step(self, *a, **k)


def muon_step(self, *a, **k):
    _opt_entry(self)
    return _muon_step(self, *a, **k)


def eval_subset(model, data, *a, **k):
    REC["n_eval"] += 1
    return _eval(model, data, *a, **k)


def predict(model, data, *a, **k):
    REC["n_predict"] += 1
    return _predict(model, data, *a, **k)


D.SplitData.gather = gather
Mo.Ranker.forward, Mo.listwise_loss = forward, listwise_loss
torch.optim.AdamW.step, torch.optim.Muon.step = adam_step, muon_step
Cm.eval_subset, Cm.predict = eval_subset, predict
sys.argv = ["train.py"] + args
try:
    runpy.run_path(train_file, run_name="__main__")
finally:
    flat = lambda xs: np.concatenate([x.ravel() for x in xs]) if xs else np.zeros(0, np.float32)  # noqa: E731
    g = REC["g_fidx"]
    np.savez(rec_path, g_fidx=np.concatenate(g) if g else np.zeros(0, np.int64),
             g_off=np.concatenate([[0], np.cumsum([len(x) for x in g])]).astype(np.int64),
             f_step=np.array(REC["f_step"], np.int64), f_shape=np.array(REC["f_shape"], np.int64).reshape(-1, 2),
             lg_model=flat(REC["lg_model"]).astype(np.float32), f_pnorm=np.array(REC["f_pnorm"]), f_eps=np.array(REC["f_eps"]),
             f_same=np.array(REC["f_same"], bool), lg_loss=flat(REC["lg_loss"]).astype(np.float32),
             n_loss=np.array(len(REC["lg_loss"])), o_step=np.array(REC["o_step"], np.int64), o_same=np.array(REC["o_same"], bool),
             o_gnorm=np.array(REC["o_gnorm"]), n_eval=np.array(REC["n_eval"]), n_predict=np.array(REC["n_predict"]),
             meta=np.array(json.dumps(dict(n_forward=len(REC["f_step"]), n_steps=ST["step"] + 1))))
