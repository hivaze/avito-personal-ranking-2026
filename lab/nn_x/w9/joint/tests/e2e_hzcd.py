"""Deterministic end-to-end wrapper for tests/test_hzcd.py (GPU). Runs a train.py FILE as __main__ with the given arguments
after (1) switching torch to deterministic kernels (as tests/e2e_w5.py) and (2) wrapping, without changing results:
  data.SplitData.gather    -> tags every batch with its split name (b['_split']) and source 'tr'; records the feed ids of
                              every TRAINING gather (with_pos=True);
  cd_arms6.HzTrainView.gather -> tags the batch source 'hz';
  model.Ranker.dense       -> TRAIN-mode calls (the training batches): records ci (masked -1), the LUO-slot columns of
                              b['xt'] exactly as they reach dense_features, the source tag and the split; EVAL-mode calls
                              (feature_stats / eval_subset / predict): compares those columns on the fly with the split's
                              nn/feat_cache luo files (source 'tr') or the data/feats_new hz files (source 'hz': the
                              save_holdout_hz predict) under the attach_xt transform; EVAL[split|src|eval] = [rows,
                              mismatches vs that reference, calls, mismatches vs luo];
  model.Ranker.forward     -> train-mode logits;  model.listwise_loss -> loss-input logits (as e2e_w5.py);
  common.predict / eval_subset -> whether the data object they read is the hz view.
  usage: python tests/e2e_hzcd.py <train file> <record.npz> --variant w5_... <train.py args>
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
import cd_arms6 as C6  # noqa: E402
import common as Cm  # noqa: E402
import data as D  # noqa: E402
import model as Mo  # noqa: E402

FC = "<repo>/nn/feat_cache"
FN = "<repo>/data/feats_new"
REC = dict(fidx=[], ci=[], xt=[], src=[], split=[], lg_model=[], lg_loss=[], n_eval=0, n_predict=0, eval_hz=[], predict_hz=[])
EVAL = {}  # (split, src, training) -> [rows, mismatches, calls]
REF = {}
_gather, _hzgather = D.SplitData.gather, C6.HzTrainView.gather
_dense, _fwd, _ll, _eval, _predict = Mo.Ranker.dense, Mo.Ranker.forward, Mo.listwise_loss, Cm.eval_subset, Cm.predict


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    b["_split"], b["_src"] = self.split, "tr"
    if with_pos:
        REC["fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
    return b


def hzgather(self, fidx_np, with_pos=False):
    b = _hzgather(self, fidx_np, with_pos)
    b["_src"] = "hz"
    return b


def ref_cols(split, cols, src="tr"):
    key = (split, src)
    if key not in REF:
        REF[key] = ([np.load(f"{FC}/{split}/{luo_n}.npy", mmap_mode="r") for _, luo_n, _ in cols] if src == "tr" else
                    [np.load(f"{FN}/{split}/{hz_n}.npy", mmap_mode="r") for _, _, hz_n in cols])
    return REF[key]


def dense(self, b):
    cols = C6.luo_cols(self.cfg["xt_names"])
    idx = [c for c, _, _ in cols]
    ci = torch.where(b["cmask"], b["ci"], -1).cpu().numpy().astype(np.int64)
    vals = b["xt"][..., idx].float().cpu().numpy()
    if self.training:
        REC["ci"].append(ci)
        REC["xt"].append(vals)
        REC["src"].append(b.get("_src", "?"))
        REC["split"].append(b.get("_split", "?"))
    else:
        split, src = b.get("_split", "?"), b.get("_src", "?")
        e = EVAL.setdefault(f"{split}|{src}|eval", [0, 0, 0, 0])
        m = ci >= 0
        rows = ci[m]
        got = vals[m]
        bad = bad_luo = 0
        for k, (_, luo_n, _) in enumerate(cols):
            exp_luo = torch.from_numpy(C6.xt_transform(luo_n, ref_cols(split, cols)[k][rows])).to(torch.float16).float().numpy()
            exp = exp_luo if src != "hz" else torch.from_numpy(
                C6.xt_transform("it_hist_hz", ref_cols(split, cols, "hz")[k][rows])).to(torch.float16).float().numpy()
            bad += int((exp != got[:, k]).sum())
            bad_luo += int((exp_luo != got[:, k]).sum())
        e[0] += int(m.sum())
        e[1] += bad
        e[2] += 1
        e[3] += bad_luo
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
    REC["eval_hz"].append(bool(getattr(data, "hzcd", False)))
    return _eval(model, data, *a, **k)


def predict(model, data, *a, **k):
    REC["n_predict"] += 1
    REC["predict_hz"].append(bool(getattr(data, "hzcd", False)))
    return _predict(model, data, *a, **k)


D.SplitData.gather, C6.HzTrainView.gather = gather, hzgather
Mo.Ranker.dense, Mo.Ranker.forward, Mo.listwise_loss = dense, forward, listwise_loss
Cm.eval_subset, Cm.predict = eval_subset, predict
sys.argv = ["train.py"] + args
try:
    runpy.run_path(train_file, run_name="__main__")
finally:
    f = REC["fidx"]
    flat = lambda xs: np.concatenate([x.ravel() for x in xs]) if xs else np.zeros(0)  # noqa: E731
    np.savez(rec_path, fidx=np.concatenate(f) if f else np.zeros(0, np.int64),
             off=np.concatenate([[0], np.cumsum([len(x) for x in f])]).astype(np.int64),
             shapes=np.array([x.shape for x in REC["ci"]]).reshape(-1, 2), ci=flat(REC["ci"]).astype(np.int64),
             xt=(np.concatenate([x.reshape(-1, x.shape[-1]) for x in REC["xt"]]) if REC["xt"] else np.zeros((0, 1))).astype(np.float32),
             src=np.array(REC["src"]), split=np.array(REC["split"]),
             lg_model=flat(REC["lg_model"]).astype(np.float32), lg_loss=flat(REC["lg_loss"]).astype(np.float32),
             n_model=np.array(len(REC["lg_model"])), n_loss=np.array(len(REC["lg_loss"])),
             n_eval=np.array(REC["n_eval"]), n_predict=np.array(REC["n_predict"]),
             eval_hz=np.array(REC["eval_hz"], bool), predict_hz=np.array(REC["predict_hz"], bool),
             eval_counts=np.array(json.dumps(EVAL)))
