"""CPU end-to-end wrapper for tests/test_w9_catgrad.py (wave 9 CATGRAD): runs a train.py FILE of a given CODE DIRECTORY as
__main__ on the CPU with a MINIATURE split (tests/mini_split.MiniSplit: a few hundred real train / val feeds, memory-mapped),
recording what the identity tests compare. In the style of tests/e2e_w5.py; patches (none changes a result):
  torch.Tensor.to         -> any "cuda*" target becomes the CPU (train.py hard-codes dev = torch.device("cuda:0") and moves the
                             model / p_random there; nn.Module.to goes through Tensor.to); torch.cuda.is_available -> False
                             (autocast "cuda" then disables itself with a warning; nothing initialises CUDA);
  np.load                 -> the FULL train p_random file (eda/labels_position/out/train_prandom.npz) is answered with the mini
                             split's slice (fidx = arange(n_mini), p_random[src feeds]) once the train MiniSplit exists, so
                             train.py's holdout / domain / pool logic runs unchanged on the mini feeds;
  data.SplitData          -> MiniSplitData(split, device, max_hist): MiniSplit over a FIXED feed list per split (train:
                             mini_split.pick_train_feeds(16, seed 0); val: 40 stratified val feeds, seed 1) + attach_xt;
  data.SplitData.gather   -> records every TRAINING gather (with_pos=True): feed ids, candidate rows ci (-1 on padding);
  model.Ranker.forward    -> records the logits of every TRAIN-mode forward and, for a CATGRAD model, its cg_g / cg_h / ccat /
                             cmask (composite identity check);
  model.listwise_loss     -> records the logits that reach the main listwise loss;
  common.predict / eval_subset -> counts.
  usage: CUDA_VISIBLE_DEVICES="" python tests/e2e_cpu.py <code_dir> <record.npz> --variant <v> <train.py args>
"""
import os
import runpy
import sys
import warnings

code_dir, rec_path, args = sys.argv[1], sys.argv[2], sys.argv[3:]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, code_dir)
sys.path.insert(1, HERE)
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU wrapper: run with CUDA_VISIBLE_DEVICES=''"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
torch.cuda.is_available = lambda: False  # noqa: E731  (never initialise CUDA; autocast("cuda") disables itself)
_tensor_to = torch.Tensor.to


def _is_cuda(x):
    return isinstance(x, (str, torch.device)) and str(x).startswith("cuda")


def _to_cpu(self, *a, **k):
    """Tensor.to / Module.to with a cuda device -> the CPU (train.py hard-codes dev = torch.device('cuda:0'))."""
    a = tuple("cpu" if _is_cuda(x) else x for x in a)
    if _is_cuda(k.get("device")):
        k["device"] = "cpu"
    return _tensor_to(self, *a, **k)


torch.Tensor.to = _to_cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # common.py's import-time guard only (CUDA is never touched: see above)
import common as Cm  # noqa: E402
import data as D  # noqa: E402
import model as Mo  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from mini_split import MiniSplit, _xt_col, pick_train_feeds  # noqa: E402

ROOT = "<repo>"
PRANDOM = f"{ROOT}/eda/labels_position/out/train_prandom.npz"
_np_load = np.load
ST = {"mini_prand": None}
REC = dict(fidx=[], ci=[], lg_model=[], lg_loss=[], cg_g=[], cg_h=[], ccat=[], cmask=[], n_eval=0, n_predict=0)


def val_feeds(n_each=10, seed=1):
    f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet", columns=["M", "H"])
    M, H = f["M"].to_numpy(), f["H"].to_numpy()
    rng = np.random.default_rng(seed)
    groups = [np.where(M < 50)[0], np.where((M >= 50) & (M <= 200))[0], np.where(M > 200)[0], np.where(H == 0)[0]]
    return np.unique(np.concatenate([rng.choice(g_, min(n_each, len(g_)), replace=False) for g_ in groups]))


def train_feeds(n_each=16, seed=0):
    """pick_train_feeds (random-like p > 0.9 M >= 50 -> they become the --holdout 30000 = all-of-them mini holdout, production,
    M < 50, M > 200, H == 0, user0) + n_each feeds with 0.5 < p_random <= 0.9 and M >= 50 (the cooldown exploration pool)."""
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])
    M = f["M"].to_numpy()
    pr = np.nan_to_num(_np_load(PRANDOM)["p_random"].astype(np.float32), nan=0.0)
    pool = np.where((pr > 0.5) & (pr <= 0.9) & (M >= 50))[0]
    rng = np.random.default_rng(seed)
    return np.unique(np.concatenate([pick_train_feeds(n_each, seed=seed), rng.choice(pool, n_each, replace=False)]))


FEEDS = {"train": train_feeds(), "val": val_feeds()}


class MiniSplitData(MiniSplit):
    """data.SplitData stand-in: the MiniSplit of a fixed feed list, same constructor signature, + attach_xt."""

    def __init__(self, split, device, max_hist=256, verbose=True):
        super().__init__(split, FEEDS[split], xt_names=None, max_hist=max_hist, with_domain=False)
        f = pl.read_parquet(f"{ROOT}/data/flat/{split}/feeds.parquet", columns=["M", "cand_off"])
        Mf, coff = f["M"].to_numpy().astype(np.int64)[self.src_feeds], f["cand_off"].to_numpy().astype(np.int64)[self.src_feeds]
        self._crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, Mf)])
        if split == "train":
            pr = _np_load(PRANDOM)
            ST["mini_prand"] = {"fidx": np.arange(self.n), "p_random": pr["p_random"][self.src_feeds]}
        if verbose:
            print(f"[mini] {split}: {self.n} feeds, {self.n_cand} cands, {self.n_hist} hist (CPU miniature)", flush=True)

    def attach_xt(self, names):
        self.xt = torch.stack([_xt_col(self.split, n, self._crow) for n in names], 1)


def np_load(path, *a, **k):
    if str(path).endswith("train_prandom.npz") and ST["mini_prand"] is not None:
        return ST["mini_prand"]
    return _np_load(path, *a, **k)


np.load = np_load
_gather, _fwd, _ll, _eval, _predict = MiniSplit.gather, Mo.Ranker.forward, Mo.listwise_loss, Cm.eval_subset, Cm.predict


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    if with_pos:
        REC["fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
        REC["ci"].append(torch.where(b["cmask"], b["ci"], -1).cpu().numpy().astype(np.int64))
    return b


def forward(self, b, *a, **k):
    out = _fwd(self, b, *a, **k)
    if self.training:
        REC["lg_model"].append(out.detach().float().cpu().numpy())
        if getattr(self, "catw", None):
            REC["cg_g"].append(self.cg_g.detach().float().cpu().numpy())
            REC["cg_h"].append(self.cg_h.detach().float().cpu().numpy())
            REC["ccat"].append(b["ccat"].cpu().numpy().astype(np.int64))
            REC["cmask"].append(b["cmask"].cpu().numpy())
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


MiniSplitData.gather = gather
D.SplitData = MiniSplitData
Mo.Ranker.forward, Mo.listwise_loss = forward, listwise_loss
Cm.eval_subset, Cm.predict = eval_subset, predict
sys.argv = ["train.py"] + args
try:
    runpy.run_path(os.path.join(code_dir, "train.py"), run_name="__main__")
finally:
    f = REC["fidx"]
    flat = lambda xs: np.concatenate([x.ravel() for x in xs]) if xs else np.zeros(0)  # noqa: E731
    np.savez(rec_path, fidx=np.concatenate(f) if f else np.zeros(0, np.int64),
             off=np.concatenate([[0], np.cumsum([len(x) for x in f])]).astype(np.int64),
             shapes=np.array([x.shape for x in REC["ci"]]).reshape(-1, 2),
             ci=flat(REC["ci"]).astype(np.int64), lg_model=flat(REC["lg_model"]).astype(np.float32),
             lg_loss=flat(REC["lg_loss"]).astype(np.float32),
             g_shapes=np.array([x.shape for x in REC["cg_g"]]).reshape(-1, 2), cg_g=flat(REC["cg_g"]).astype(np.float32),
             cg_h=flat(REC["cg_h"]).astype(np.float32), ccat=flat(REC["ccat"]).astype(np.int64), cmask=flat(REC["cmask"]).astype(bool),
             n_model=np.array(len(REC["lg_model"])), n_loss=np.array(len(REC["lg_loss"])),
             n_eval=np.array(REC["n_eval"]), n_predict=np.array(REC["n_predict"]))
