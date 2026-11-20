"""Deterministic end-to-end wrapper for tests/test_w5.py G3 (GPU). Runs a train.py FILE as __main__ with the given
arguments after (1) switching torch to deterministic kernels (as test_cdw4) and (2) wrapping, without changing results:
  data.SplitData.gather  -> records every TRAINING batch (calls with with_pos=True; eval / predict / feature_stats gather
                            without positions): feed ids, candidate rows ci, cmask;
  model.Ranker.forward   -> records the logits of every TRAIN-mode forward (the training batches);
  model.listwise_loss    -> records the logits that reach the main listwise loss (patched before train.py imports it);
  common.predict / common.eval_subset -> count the eval / predict calls (none of them may see an offset: they call the
                            model directly).
  usage: python tests/e2e_w5.py <train file> <record.npz> --variant w5_... <train.py args>
"""
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

REC = dict(fidx=[], ci=[], lg_model=[], lg_loss=[], n_eval=0, n_predict=0)
_gather, _fwd, _ll, _eval, _predict = D.SplitData.gather, Mo.Ranker.forward, Mo.listwise_loss, Cm.eval_subset, Cm.predict


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    if with_pos:
        REC["fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
        REC["ci"].append(torch.where(b["cmask"], b["ci"], -1).cpu().numpy().astype(np.int64))
    return b


def forward(self, *a, **k):
    out = _fwd(self, *a, **k)
    if self.training:
        REC["lg_model"].append(out.detach().float().cpu().numpy())
    return out


def listwise_loss(logit, *a, **k):
    REC["lg_loss"].append(logit.detach().float().cpu().numpy())
    return _ll(logit, *a, **k)


def eval_subset(*a, **k):
    REC["n_eval"] += 1
    return _eval(*a, **k)


def predict(*a, **k):
    REC["n_predict"] += 1
    return _predict(*a, **k)


D.SplitData.gather, Mo.Ranker.forward, Mo.listwise_loss = gather, forward, listwise_loss
Cm.eval_subset, Cm.predict = eval_subset, predict
sys.argv = ["train.py"] + args
try:
    runpy.run_path(train_file, run_name="__main__")
finally:
    f = REC["fidx"]
    flat = lambda xs: np.concatenate([x.ravel() for x in xs]) if xs else np.zeros(0)  # noqa: E731
    np.savez(rec_path, fidx=np.concatenate(f) if f else np.zeros(0, np.int64),
             off=np.concatenate([[0], np.cumsum([len(x) for x in f])]).astype(np.int64),
             shapes=np.array([x.shape for x in REC["ci"]]).reshape(-1, 2),
             ci=flat(REC["ci"]).astype(np.int64), lg_model=flat(REC["lg_model"]).astype(np.float32),
             lg_loss=flat(REC["lg_loss"]).astype(np.float32),
             n_model=np.array(len(REC["lg_model"])), n_loss=np.array(len(REC["lg_loss"])),
             n_eval=np.array(REC["n_eval"]), n_predict=np.array(REC["n_predict"]))
