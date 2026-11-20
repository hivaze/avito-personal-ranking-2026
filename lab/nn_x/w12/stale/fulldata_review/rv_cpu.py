"""Reviewer's own CPU runner (independent of fulldata/cpu_run.py): executes a train file unchanged as __main__ on the CPU.
usage: CUDA_VISIBLE_DEVICES="" python rv_cpu.py <code_dir> <train_file> <out_dir> <args...>
Differences to the study's runner: eval cap = a hash-selected subset of feeds (not the first n), own recorder
(feed ids per training gather, listwise target, loss, pre-clip grad norm)."""
import os, sys, runpy, time
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "start with CUDA_VISIBLE_DEVICES=''"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    assert int(os.environ.get(k, "999")) <= 12, k
sys.dont_write_bytecode = True
code_dir, train_file, out_dir, args = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]), sys.argv[4:]
MY = os.path.dirname(os.path.abspath(__file__))
assert out_dir.startswith(MY + os.sep)
import warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(12)
assert not torch.cuda.is_available()
def _boom(*a, **k):
    raise RuntimeError("rv_cpu: CUDA attempted")
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = _boom
torch.use_deterministic_algorithms(True, warn_only=True)
_RealDev = torch.device
class _Meta(type):
    def __instancecheck__(cls, o):
        return isinstance(o, _RealDev)
    def __call__(cls, *a, **k):
        d = _RealDev(*a, **k)
        return _RealDev("cpu") if d.type == "cuda" else d
class FakeDevice(metaclass=_Meta):
    pass
torch.device = FakeDevice
sys.path.insert(0, code_dir)
sys.path.insert(0, os.path.dirname(train_file))   # production order: script dir first
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
try:
    import common
finally:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(12)
import data as datamod, model as modelmod
assert os.path.dirname(os.path.abspath(common.__file__)) == os.path.abspath(code_dir)
from pathlib import Path
OUT = Path(out_dir)
common.CKPT, common.LOGS, common.PREDS = OUT / "ckpt", OUT / "logs", OUT / "preds"
for p in (common.CKPT, common.LOGS, common.PREDS):
    p.mkdir(parents=True, exist_ok=True)
MOD = {"val": (14, 3), "train": (20, 11), "test": (19, 7)}
def sub(data, ids):
    ids = np.arange(data.n) if ids is None else np.asarray(ids)
    m, r = MOD[data.split]
    return ids[(ids.astype(np.int64) * 2654435761 % 4294967296) % m == r]
REC = dict(f=[], y=[], loss=[], gn=[], ev=[])
_g, _ll, _pr = datamod.SplitData.gather, modelmod.listwise_loss, common.predict
_clip = torch.nn.utils.clip_grad_norm_
def gather(self, fidx_np, with_pos=False):
    if with_pos:
        REC["f"].append((self.split, bool(getattr(self, "hzcd", False)), np.asarray(fidx_np, dtype=np.int64).copy()))
    return _g(self, fidx_np, with_pos)
def listwise_loss(lg, y, m, *a, **k):
    out = _ll(lg, y, m, *a, **k)
    REC["y"].append((y.detach().float() * m).numpy().astype(np.float32))
    REC["loss"].append(float(out.detach().double()))
    return out
def clip(params, mx, *a, **k):
    g = _clip(params, mx, *a, **k)
    REC["gn"].append(float(g))
    return g
def predict(model, data, feed_ids=None, *a, **k):
    ids = sub(data, feed_ids)
    REC["ev"].append(f"predict|{data.split}|{len(ids)}|{bool(getattr(data, 'hzcd', False))}")
    return _pr(model, data, ids, *a, **k)
def eval_subset(model, data, feed_ids, rows=None):
    feed_ids = np.sort(feed_ids)
    M, off = data.M_np[feed_ids], data.cand_off_np[feed_ids]
    if rows is None:
        rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off, M)])
    sc = predict(model, data, feed_ids)[rows]
    return common.ndcg(sc, data.target_np[rows], np.concatenate([[0], np.cumsum(M)[:-1]]), M)
datamod.SplitData.gather, modelmod.listwise_loss = gather, listwise_loss
torch.nn.utils.clip_grad_norm_ = clip
common.predict, common.eval_subset = predict, eval_subset
sys.argv = [train_file] + args
t0 = time.time(); ok = False
try:
    runpy.run_path(train_file, run_name="__main__")
    ok = True
finally:
    assert not torch.cuda.is_initialized()
    fl = REC["f"]
    np.savez(OUT / "rec.npz", fidx=np.concatenate([x[2] for x in fl]) if fl else np.zeros(0, np.int64),
             flen=np.array([len(x[2]) for x in fl], np.int64), fsplit=np.array([x[0] for x in fl]),
             fhz=np.array([x[1] for x in fl]), y=np.concatenate([v.ravel() for v in REC["y"]]) if REC["y"] else np.zeros(0, np.float32),
             yshape=np.array([v.shape for v in REC["y"]], np.int64).reshape(-1, 2), loss=np.array(REC["loss"]),
             gn=np.array(REC["gn"]), ev=np.array(REC["ev"]), ok=np.array(ok))
    print(f"[rv_cpu] ok={ok} {(time.time() - t0) / 60:.1f} min", flush=True)
