"""Deterministic end-to-end wrapper for tests/test_valcd.py (GPU). Runs a train.py FILE as __main__ with the given arguments
after (1) switching torch to deterministic kernels (as tests/e2e_w5.py / e2e_hzcd.py) and (2) wrapping, without changing results:
  data.SplitData.gather     -> tags every batch with its split name (b['_split']) and source 'tr'; records the feed ids and the
                               split of every gather with with_pos=True (training gathers; ValPool.gather also calls it so);
  cd_arms7.ValPool.gather   -> tags the batch source 'val' (the module exists in both code dirs; the previous train.py never
                               calls it);
  model.Ranker.dense        -> TRAIN-mode calls (one per training step, the batch that reaches the model): records the source /
                               split tags, fidx, ci (masked -1), dom rows (NaN when absent), y, ccat, pos, xt columns 0 / 3;
                               EVAL-mode calls (feature_stats / eval_subset / predict): counts per split|src|has_dom;
  model.Ranker.forward      -> train-mode logits;  model.listwise_loss -> loss-input logits (as e2e_w5.py);
  common.predict / eval_subset -> whether the data object they read is a ValPool (attribute 'valcd').
  usage: python tests/e2e_valcd.py <train file> <record.npz> --variant w5_... <train.py args>
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
import cd_arms7 as C7  # noqa: E402
import common as Cm  # noqa: E402
import data as D  # noqa: E402
import model as Mo  # noqa: E402

REC = dict(g_fidx=[], g_split=[], src=[], split=[], fidx=[], ci=[], dom=[], y=[], ccat=[], pos=[], xt=[], lg_model=[], lg_loss=[],
           n_eval=0, n_predict=0, eval_valpool=[], predict_valpool=[])
EVAL = {}
_gather, _vgather = D.SplitData.gather, C7.ValPool.gather
_dense, _fwd, _ll, _eval, _predict = Mo.Ranker.dense, Mo.Ranker.forward, Mo.listwise_loss, Cm.eval_subset, Cm.predict


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    b["_split"], b["_src"] = self.split, "tr"
    if with_pos:
        REC["g_fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
        REC["g_split"].append(self.split)
    return b


def vgather(self, fb):
    b = _vgather(self, fb)
    b["_src"] = "val"
    return b


def dense(self, b):
    if self.training:
        REC["src"].append(b.get("_src", "?"))
        REC["split"].append(b.get("_split", "?"))
        REC["fidx"].append(b["fidx"].cpu().numpy().astype(np.int64))
        REC["ci"].append(torch.where(b["cmask"], b["ci"], -1).cpu().numpy().astype(np.int64))
        d = b.get("dom")
        REC["dom"].append(d.float().cpu().numpy() if d is not None else np.full((b["cmask"].shape[0], 3), np.nan, np.float32))
        REC["y"].append(b["y"].float().cpu().numpy())
        REC["ccat"].append(b["ccat"].cpu().numpy().astype(np.int64))
        REC["pos"].append(b["pos"].cpu().numpy().astype(np.int64) if "pos" in b else np.full(b["cmask"].shape, -1, np.int64))
        REC["xt"].append(b["xt"][..., [0, 3]].float().cpu().numpy())
    else:
        k = f"{b.get('_split', '?')}|{b.get('_src', '?')}|dom={'dom' in b}"
        EVAL[k] = EVAL.get(k, 0) + 1
    return _dense(self, b)


def forward(self, *a, **k):
    out = _fwd(self, *a, **k)
    if self.training:
        REC["lg_model"].append(out.detach().float().cpu().numpy())
    return out


def listwise_loss(logit, *a, **k):
    REC["lg_loss"].append(logit.detach().float().cpu().numpy())
    return _ll(logit, *a, **k)


def eval_subset(model, data, *a, **k):
    REC["n_eval"] += 1
    REC["eval_valpool"].append(bool(getattr(data, "valcd", False)))
    return _eval(model, data, *a, **k)


def predict(model, data, *a, **k):
    REC["n_predict"] += 1
    REC["predict_valpool"].append(bool(getattr(data, "valcd", False)))
    return _predict(model, data, *a, **k)


D.SplitData.gather, C7.ValPool.gather = gather, vgather
Mo.Ranker.dense, Mo.Ranker.forward, Mo.listwise_loss = dense, forward, listwise_loss
Cm.eval_subset, Cm.predict = eval_subset, predict
sys.argv = ["train.py"] + args
try:
    runpy.run_path(train_file, run_name="__main__")
finally:
    flat = lambda xs: np.concatenate([x.ravel() for x in xs]) if xs else np.zeros(0)  # noqa: E731
    g = REC["g_fidx"]
    np.savez(rec_path, g_fidx=np.concatenate(g) if g else np.zeros(0, np.int64),
             g_off=np.concatenate([[0], np.cumsum([len(x) for x in g])]).astype(np.int64), g_split=np.array(REC["g_split"]),
             shapes=np.array([x.shape for x in REC["ci"]]).reshape(-1, 2), src=np.array(REC["src"]), split=np.array(REC["split"]),
             fidx=flat(REC["fidx"]).astype(np.int64), fidx_off=np.concatenate([[0], np.cumsum([len(x) for x in REC["fidx"]])]).astype(np.int64),
             ci=flat(REC["ci"]).astype(np.int64), dom=(np.concatenate(REC["dom"]) if REC["dom"] else np.zeros((0, 3))).astype(np.float32),
             y=flat(REC["y"]).astype(np.float32), ccat=flat(REC["ccat"]).astype(np.int64), pos=flat(REC["pos"]).astype(np.int64),
             xt=(np.concatenate([x.reshape(-1, 2) for x in REC["xt"]]) if REC["xt"] else np.zeros((0, 2))).astype(np.float32),
             lg_model=flat(REC["lg_model"]).astype(np.float32), lg_loss=flat(REC["lg_loss"]).astype(np.float32),
             n_model=np.array(len(REC["lg_model"])), n_loss=np.array(len(REC["lg_loss"])),
             n_eval=np.array(REC["n_eval"]), n_predict=np.array(REC["n_predict"]),
             eval_valpool=np.array(REC["eval_valpool"], bool), predict_valpool=np.array(REC["predict_valpool"], bool),
             eval_counts=np.array(json.dumps(EVAL)))
