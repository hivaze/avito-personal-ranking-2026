"""CPU end-to-end wrapper of tests/test_moe_cpu.py (w12/moe, 2026-11-19; = lab/nn_x/w12/pseudokd/tests/e2e_cpu.py reduced to what the
MoE identity / arm tests compare). Runs <train file> as __main__ on the CPU with a MINIATURE split (tests/mini_split.MiniSplit: a few
dozen real train / val feeds, memory-mapped). sys.path = [dirname(train file), lab/nn_x/w5, tests/] = what `python <train file>` with
PYTHONPATH=lab/nn_x/w5 sees (+ the test helpers, none of which shadows a w5 module). Every output of the run (common.CKPT / LOGS / PREDS)
goes to <outdir>/{ckpt,logs,preds}: nothing is written into lab/nn_x/w5. Patches (none changes a result):
  torch.Tensor.to         -> any "cuda*" target becomes the CPU; torch.cuda.is_available -> False (autocast "cuda" disables itself);
  np.load                 -> the FULL train p_random file -> the mini split's slice; a --kd_file teacher (one logit per flat train
                             candidate) -> its rows at the mini split's candidates;
  data.SplitData          -> MiniSplitData(split, device, max_hist): MiniSplit over a FIXED feed list per split + attach_xt;
  SplitData.gather        -> records every TRAINING gather (with_pos=True): feed ids;
  Ranker.forward          -> (w5 model.py AND model_moe.py) records the logits of every TRAIN-mode forward;
  torch.Tensor.backward   -> records the scalar every .backward() is called on (the total loss of each training step).
  usage: CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python tests/e2e_moe_cpu.py <train file> <outdir> <record.npz> --variant <v> <args>
"""
import os
import runpy
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
train_file, outdir, rec_path, args = os.path.abspath(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
HERE = os.path.dirname(os.path.abspath(__file__))
MOE = os.path.dirname(HERE)
W5 = "<repo>/lab/nn_x/w5"
for p_ in (HERE, W5, os.path.dirname(train_file)):  # final order: [train dir, W5, tests]
    if p_ in sys.path:
        sys.path.remove(p_)
    sys.path.insert(0, p_)
if MOE not in sys.path:  # model_moe importable for the forward patch even when the train file is the w5 one (no name clash with w5)
    sys.path.append(MOE)
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU wrapper: run with CUDA_VISIBLE_DEVICES=''"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
torch.cuda.is_available = lambda: False  # noqa: E731
_tensor_to = torch.Tensor.to


def _is_cuda(x):
    return isinstance(x, (str, torch.device)) and str(x).startswith("cuda")


def _to_cpu(self, *a, **k):
    a = tuple("cpu" if _is_cuda(x) else x for x in a)
    if _is_cuda(k.get("device")):
        k["device"] = "cpu"
    return _tensor_to(self, *a, **k)


torch.Tensor.to = _to_cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # common.py's import-time guard only (CUDA is never touched)
import common as Cm  # noqa: E402
import data as D  # noqa: E402
import model as Mo  # noqa: E402
import model_moe as Mm  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""
assert os.path.dirname(Cm.__file__) == W5 and os.path.dirname(D.__file__) == W5 and os.path.dirname(Mo.__file__) == W5, \
    (Cm.__file__, D.__file__, Mo.__file__)
assert os.path.dirname(Mm.__file__) == MOE, Mm.__file__
from mini_split import MiniSplit, _xt_col, pick_train_feeds  # noqa: E402

for sub in ("ckpt", "logs", "preds", "preds/holdout"):
    (outdir / sub).mkdir(parents=True, exist_ok=True)
Cm.CKPT, Cm.LOGS, Cm.PREDS = outdir / "ckpt", outdir / "logs", outdir / "preds"

ROOT = "<repo>"
PRANDOM = f"{ROOT}/eda/labels_position/out/train_prandom.npz"
_np_load = np.load
ST = {"mini_prand": None, "crow_train": None}
REC = dict(fidx=[], lg_model=[], total=[], kd_file=[])


def val_feeds(n_each=10, seed=1):
    f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet", columns=["M", "H"])
    M, H = f["M"].to_numpy(), f["H"].to_numpy()
    rng = np.random.default_rng(seed)
    groups = [np.where(M < 50)[0], np.where((M >= 50) & (M <= 200))[0], np.where(M > 200)[0], np.where(H == 0)[0]]
    return np.unique(np.concatenate([rng.choice(g_, min(n_each, len(g_)), replace=False) for g_ in groups]))


def train_feeds(n_each=16, seed=0):
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])
    M = f["M"].to_numpy()
    pr = np.nan_to_num(_np_load(PRANDOM)["p_random"].astype(np.float32), nan=0.0)
    pool = np.where((pr > 0.5) & (pr <= 0.9) & (M >= 50))[0]
    rng = np.random.default_rng(seed)
    return np.unique(np.concatenate([pick_train_feeds(n_each, seed=seed), rng.choice(pool, n_each, replace=False)]))


FEEDS = {"train": train_feeds(), "val": val_feeds()}


class MiniSplitData(MiniSplit):
    def __init__(self, split, device, max_hist=256, verbose=True, **hx):
        assert not hx, f"e2e: hist_ext options not supported ({hx})"
        super().__init__(split, FEEDS[split], xt_names=None, max_hist=max_hist, with_domain=False)
        f = pl.read_parquet(f"{ROOT}/data/flat/{split}/feeds.parquet", columns=["M", "cand_off", "user_id"])
        Mf, coff = f["M"].to_numpy().astype(np.int64)[self.src_feeds], f["cand_off"].to_numpy().astype(np.int64)[self.src_feeds]
        self._crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, Mf)])
        self.user_id = f["user_id"].to_numpy()[self.src_feeds]
        if split == "train":
            pr = _np_load(PRANDOM)
            ST["mini_prand"] = {"fidx": np.arange(self.n), "p_random": pr["p_random"][self.src_feeds]}
            ST["crow_train"] = self._crow
        if verbose:
            print(f"[mini] {split}: {self.n} feeds, {self.n_cand} cands, {self.n_hist} hist (CPU miniature)", flush=True)

    def attach_xt(self, names):
        self.xt = torch.stack([_xt_col(self.split, n, self._crow) for n in names], 1)


def np_load(path, *a, **k):
    p = str(path)
    if p.endswith("train_prandom.npz") and ST["mini_prand"] is not None:
        return ST["mini_prand"]
    if "/kd/out/teacher" in p and p.endswith(".npy") and ST["crow_train"] is not None:
        REC["kd_file"].append(p)
        return np.asarray(_np_load(p, mmap_mode="r")[ST["crow_train"]])
    return _np_load(path, *a, **k)


np.load = np_load
_gather, _bw = MiniSplit.gather, torch.Tensor.backward


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    if with_pos:
        REC["fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
    return b


def _wrap_fwd(fwd):
    def forward(self, b, *a, **k):
        out = fwd(self, b, *a, **k)
        if self.training:
            REC["lg_model"].append(out.detach().float().cpu().numpy())
        return out
    return forward


def backward(self, *a, **k):
    if self.dim() == 0:
        REC["total"].append(float(self.detach()))
    return _bw(self, *a, **k)


MiniSplitData.gather = gather
D.SplitData = MiniSplitData
Mo.Ranker.forward = _wrap_fwd(Mo.Ranker.forward)
Mm.Ranker.forward = _wrap_fwd(Mm.Ranker.forward)
torch.Tensor.backward = backward
sys.argv = [train_file] + args
try:
    runpy.run_path(train_file, run_name="__main__")
finally:
    f = REC["fidx"]
    flat = lambda xs: np.concatenate([x.ravel() for x in xs]) if xs else np.zeros(0)  # noqa: E731
    offs = lambda xs: np.concatenate([[0], np.cumsum([len(x.ravel()) for x in xs])]).astype(np.int64)  # noqa: E731
    np.savez(rec_path, fidx=np.concatenate(f) if f else np.zeros(0, np.int64), off=offs(f),
             lg_model=flat(REC["lg_model"]).astype(np.float32), lg_model_off=offs(REC["lg_model"]),
             total=np.array(REC["total"], np.float64), kd_file=np.array(REC["kd_file"]))
