"""CPU end-to-end runner for a train.py FILE of the nn_x/w5 / nn_x/abl family (the originals and the full-data copies in this
directory). The train file is executed UNCHANGED as __main__ (runpy); only the process environment is prepared so that the
GPU-only file runs on the CPU and writes nothing outside --out:

  usage: CUDA_VISIBLE_DEVICES="" python cpu_run.py <code_dir> <train_file> <out_dir> <train.py args ...>
         code_dir = <repo>/nn_x/w5 or .../nn_x/abl (common / data / model / cd_arms* are imported from there, read-only)

  1. CUDA is never touched: the process must start with CUDA_VISIBLE_DEVICES=""; torch.cuda.is_available is pinned to False
     and torch.cuda._lazy_init raises, so no tensor can reach a GPU even by mistake. The import guard of <code_dir>/common.py
     ("set CUDA_VISIBLE_DEVICES to exactly one GPU id") is satisfied by setting the variable to "0" ONLY while `import common`
     runs (no CUDA call is possible in that window) and back to "" right after.
  2. torch.device("cuda:0") (hard-coded in train.py) resolves to the CPU through a torch.device stand-in (isinstance and
     every non-cuda device keep working). torch.autocast("cuda") is disabled by torch itself when CUDA is unavailable, so
     the run is fp32 (GPU runs are bf16 autocast + TF32): numbers are NOT comparable to GPU runs, only between CPU runs.
  3. common.CKPT / LOGS / PREDS are redirected to <out_dir>/{ckpt,logs,preds} BEFORE the train file imports them.
  4. deterministic kernels (torch.use_deterministic_algorithms(True, warn_only=True)), 12 threads, no bytecode files.
  5. FD_EVAL_CAP="<val>,<holdout>,<test>" (optional, speed only): common.predict / eval_subset score only the FIRST n feeds
     of the requested feed list for val / train-holdout / test (rows of the other feeds stay 0, exactly as predict leaves
     the rows of feeds outside feed_ids). Identity checks between two runs with the same cap are unaffected; NDCG values
     of a capped run are not real NDCGs.
  6. Recorder (no effect on results) -> <out_dir>/record.npz: per listwise_loss call the split and feed ids of the batch
     gathered just before it (ll_*; used for finetune.py); for every TRAINING step of train.py (gather with_pos=True) the feed ids,
     the flat candidate rows (masked -1), the listwise target y that reaches listwise_loss, the main loss value and the
     train-mode logits checksum; counts of predict / eval_subset calls with their data split and number of feeds.
"""
import os
import runpy
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "cpu_run.py: start with CUDA_VISIBLE_DEVICES=''"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(k, "12")
    assert int(os.environ[k]) <= 12, f"{k} must be <= 12"
sys.dont_write_bytecode = True
code_dir, train_file, out_dir, args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
HERE = os.path.dirname(os.path.abspath(__file__))
assert os.path.abspath(out_dir).startswith(HERE + os.sep), "cpu_run.py writes only inside its own directory"

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(12)
assert not torch.cuda.is_available()


def _no_cuda(*a, **k):
    raise RuntimeError("cpu_run.py: CUDA use attempted in a CPU-only run")


torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = _no_cuda
torch.use_deterministic_algorithms(True, warn_only=True)

# ---- torch.device stand-in: cuda -> cpu
_Dev = torch.device


class _DevMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _Dev)

    def __call__(cls, *a, **k):
        d = _Dev(*a, **k)
        return _Dev("cpu") if d.type == "cuda" else d


class _CpuDevice(metaclass=_DevMeta):
    pass


torch.device = _CpuDevice

# ---- imports from the code directory (production order: script dir first, then PYTHONPATH = code_dir)
sys.path.insert(0, code_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(train_file)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # common.py import guard only; no CUDA call can happen (see _no_cuda)
try:
    import common as Cm  # noqa: E402
finally:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
assert not torch.cuda.is_initialized()
torch.set_num_threads(12)  # common.py sets min(16, OMP_NUM_THREADS) = 12; kept explicit
import data as D  # noqa: E402
import model as Mo  # noqa: E402

assert os.path.dirname(os.path.abspath(Cm.__file__)) == os.path.abspath(code_dir), Cm.__file__
from pathlib import Path  # noqa: E402

OUT = Path(out_dir)
Cm.CKPT, Cm.LOGS, Cm.PREDS = OUT / "ckpt", OUT / "logs", OUT / "preds"
for p in (Cm.CKPT, Cm.LOGS, Cm.PREDS):
    p.mkdir(parents=True, exist_ok=True)

# ---- optional eval cap (speed)
CAP = None
if os.environ.get("FD_EVAL_CAP"):
    cv, ch, ct = (int(x) for x in os.environ["FD_EVAL_CAP"].split(","))
    CAP = {"val": cv, "train": ch, "test": ct}

REC = dict(fidx=[], ci=[], y=[], loss=[], lgsum=[], calls=[], t=[], ll_fidx=[], ll_split=[])
STATE = dict(b=None, last=None)
_gather, _ll, _predict, _eval, _fwd = D.SplitData.gather, Mo.listwise_loss, Cm.predict, Cm.eval_subset, Mo.Ranker.forward


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    STATE["last"] = (self.split, np.asarray(fidx_np, dtype=np.int64).copy())  # the batch a following listwise_loss call trains on
    if with_pos and self.split == "train":  # a TRAINING batch of train.py (eval / predict gather without pos)
        REC["fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
        REC["ci"].append(torch.where(b["cmask"], b["ci"], -1).cpu().numpy().astype(np.int64))
        REC["t"].append(time.time())
        STATE["b"] = b
    return b


def forward(self, *a, **k):
    out = _fwd(self, *a, **k)
    if self.training:
        REC["lgsum"].append(float(out.detach().double().abs().sum()))
    return out


def listwise_loss(logit, y, m, *a, **k):
    loss = _ll(logit, y, m, *a, **k)
    REC["y"].append((y.detach().float() * m).cpu().numpy().astype(np.float32))
    REC["loss"].append(float(loss.detach().double()))
    REC["ll_split"].append(STATE["last"][0])  # (finetune.py: every loss call = one training step on a val batch)
    REC["ll_fidx"].append(STATE["last"][1])
    return loss


def _cap(data, feed_ids):
    if CAP is None:
        return feed_ids
    ids = np.arange(data.n) if feed_ids is None else np.asarray(feed_ids)
    return ids[:CAP[data.split]]


def predict(model, data, feed_ids=None, *a, **k):
    ids = _cap(data, feed_ids)
    REC["calls"].append(("predict", data.split, int(data.n if ids is None else len(ids)), bool(getattr(data, "hzcd", False))))
    return _predict(model, data, ids, *a, **k)


def eval_subset(model, data, feed_ids, rows=None):
    REC["calls"].append(("eval_subset", data.split, int(len(feed_ids)), bool(getattr(data, "hzcd", False))))
    if CAP is None:
        return _eval(model, data, feed_ids, rows)
    # capped: same code as common.eval_subset, but only the first n feeds are scored (the other rows stay 0)
    feed_ids = np.sort(feed_ids)
    M, off = data.M_np[feed_ids], data.cand_off_np[feed_ids]
    if rows is None:
        rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off, M)])
    sc = _predict(model, data, feed_ids[:CAP[data.split]])[rows]
    offs = np.concatenate([[0], np.cumsum(M)[:-1]])
    return Cm.ndcg(sc, data.target_np[rows], offs, M)


D.SplitData.gather, Mo.listwise_loss, Mo.Ranker.forward = gather, listwise_loss, forward
Cm.predict, Cm.eval_subset = predict, eval_subset
sys.argv = [train_file] + args
t0 = time.time()
ok = False
try:
    runpy.run_path(train_file, run_name="__main__")
    ok = True
finally:
    assert not torch.cuda.is_initialized()
    f = REC["fidx"]
    flat = lambda xs, dt: np.concatenate([np.asarray(x).ravel() for x in xs]).astype(dt) if xs else np.zeros(0, dt)  # noqa: E731
    np.savez(OUT / "record.npz", fidx=flat(f, np.int64), off=np.concatenate([[0], np.cumsum([len(x) for x in f])]).astype(np.int64),
             shapes=np.array([x.shape for x in REC["ci"]], np.int64).reshape(-1, 2), ci=flat(REC["ci"], np.int64),
             y=flat(REC["y"], np.float32), yshapes=np.array([x.shape for x in REC["y"]], np.int64).reshape(-1, 2),
             loss=np.array(REC["loss"], np.float64), lgsum=np.array(REC["lgsum"], np.float64),
             calls=np.array(["|".join(str(v) for v in c) for c in REC["calls"]]), t=np.array(REC["t"]),
             ll_fidx=flat(REC["ll_fidx"], np.int64), ll_off=np.concatenate([[0], np.cumsum([len(x) for x in REC["ll_fidx"]])]).astype(np.int64),
             ll_split=np.array(REC["ll_split"]),
             ok=np.array(ok), minutes=np.array((time.time() - t0) / 60), argv=np.array(" ".join(sys.argv)),
             cap=np.array(os.environ.get("FD_EVAL_CAP", "")))
    print(f"[cpu_run] ok={ok} {(time.time() - t0) / 60:.1f} min; record -> {OUT / 'record.npz'}", flush=True)
