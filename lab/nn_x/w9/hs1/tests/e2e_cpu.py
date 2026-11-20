"""CPU end-to-end wrapper for tests/test_hs1.py C3: runs a train.py FILE as __main__ on a MINIATURE split (tests/mini_split.py,
a few hundred real feeds) on the CPU, so that nn_x/w9/hs1/train.py (flag off) can be compared bit-for-bit with nn_x/w5/train.py
and the flag-on path can be smoke-tested without a GPU. Modelled on nn_x/w5/tests/e2e_w5.py (records every TRAINING batch, the
train-mode forward logits, the logits reaching the main listwise loss and the eval / predict call counts). Patches, applied
BEFORE the train file is imported and identical for every train file under test:
  * CUDA_VISIBLE_DEVICES is "" (pure CPU); common.py's one-GPU guard is passed by setting the variable to "0" only while that
    module is imported (no CUDA call is made before it is reset to "", so torch never sees a GPU);
  * torch.device("cuda:0") -> the CPU device until the first SplitData() call (then restored);
  * data.SplitData -> a MiniSplit factory (fixed feed lists per split: train = pick_train_feeds(48) + 24 feeds with
    0.5 < p_random <= 0.9, val = 160 feeds incl. the empty-s1-slice ones, test = 96 feeds) with attach_xt;
  * np.load(train_prandom.npz) -> the mini train feeds' entries (fidx = arange(n_mini)) outside the MiniSplit constructor;
  * common.CKPT / LOGS / PREDS -> <out_dir>/{ckpt,logs,preds} (nothing is written into nn_x/w5 or nn_x/w9/hs1).
  usage: python tests/e2e_cpu.py <train file> <record.npz> <out_dir> --variant w5_... <train.py args>
"""
import os
import runpy
import sys
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "e2e_cpu.py: run with CUDA_VISIBLE_DEVICES=''"
train_file, rec_path, out_dir, args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
CODE = os.path.dirname(os.path.abspath(train_file))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(1, HERE)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # common.py guard only; reset below before any CUDA call
import common as Cm  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""
assert not torch.cuda.is_available()
import data as D  # noqa: E402
import model as Mo  # noqa: E402
from mini_split import MiniSplit, _xt_col, pick_train_feeds  # noqa: E402

sys.path.insert(0, CODE)  # mini_split put its own parent first; the train file's directory must win for every later import

ROOT = "<repo>"
out = Path(out_dir)
for k in ("ckpt", "logs", "preds"):
    (out / k).mkdir(parents=True, exist_ok=True)
Cm.CKPT, Cm.LOGS, Cm.PREDS = out / "ckpt", out / "logs", out / "preds"


def mini_feeds():
    f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M"])
    M = f["M"].to_numpy()
    pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[M < 50] = 0.0
    rng = np.random.default_rng(7)
    mid = np.where((pr > 0.5) & (pr <= 0.9) & (M >= 50))[0]
    tr = np.unique(np.concatenate([pick_train_feeds(48, 0), rng.choice(mid, 24, replace=False)]))
    fv = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet", columns=["H", "hist_off"])
    H, ho = fv["H"].to_numpy().astype(np.int64), fv["hist_off"].to_numpy().astype(np.int64)
    es = np.load(f"{ROOT}/data/flat/val/hist_event_source.npy")
    s1 = np.bincount(np.repeat(np.arange(len(H)), H)[es == 1], minlength=len(H))
    emp = np.where((s1 == 0) & (H > 0))[0]
    va = np.unique(np.concatenate([rng.choice(len(H), 128, replace=False), rng.choice(emp, 24, replace=False),
                                   rng.choice(np.where(H == 0)[0], 8, replace=False)]))
    nt = len(pl.read_parquet(f"{ROOT}/data/flat/test/feeds.parquet", columns=["M"]))
    te = np.sort(rng.choice(nt, 96, replace=False))
    return {"train": tr, "val": va, "test": te}


FEEDS = mini_feeds()
XT_LAYOUT = {}


class MS(MiniSplit):
    """MiniSplit + the SplitData methods train.py calls on a fresh split (attach_xt; item ids unsupported)."""

    def __init__(self, split, feeds, max_hist=256):
        super().__init__(split, feeds, xt_names=None, max_hist=max_hist, with_domain=False)
        self.dom = None
        f = pl.read_parquet(f"{ROOT}/data/flat/{split}/feeds.parquet", columns=["M", "cand_off"])
        M = f["M"].to_numpy().astype(np.int64)[self.src_feeds]
        coff = f["cand_off"].to_numpy().astype(np.int64)[self.src_feeds]
        self._crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, M)])

    def attach_xt(self, names):
        self.xt = torch.stack([_xt_col(self.split, n, self._crow) for n in names], 1)

    def load_item_ids(self):
        raise NotImplementedError("e2e_cpu: item-ID tables are not part of the recipe")


_real_device, _real_load = torch.device, np.load
torch.device = lambda *a, **k: _real_device("cpu")  # train.py's dev = torch.device("cuda:0") -> cpu


def fake_split(split, device, max_hist=256, verbose=True):
    torch.device = _real_device  # restore at the first split (every later torch.device call is the real one)
    np.load = _real_load
    ms = MS(split, FEEDS[split], max_hist=max_hist)
    np.load = patched_load
    print(f"[e2e_cpu] {split}: {ms.n} feeds, {ms.n_cand} cands, {ms.n_hist} hist (mini split, CPU)", flush=True)
    return ms


def patched_load(p, *a, **k):
    if str(p).endswith("train_prandom.npz"):
        z = _real_load(p, *a, **k)
        return {"fidx": np.arange(len(FEEDS["train"])), "p_random": np.asarray(z["p_random"])[FEEDS["train"]]}
    return _real_load(p, *a, **k)


np.load = patched_load
D.SplitData = fake_split

REC = dict(fidx=[], ci=[], H=[], hes=[], lg_model=[], lg_loss=[], n_eval=0, n_predict=0)
_gather, _fwd, _ll, _eval, _predict = MS.gather, Mo.Ranker.forward, Mo.listwise_loss, Cm.eval_subset, Cm.predict


def gather(self, fidx_np, with_pos=False):
    b = _gather(self, fidx_np, with_pos)
    if with_pos:
        REC["fidx"].append(np.asarray(fidx_np, dtype=np.int64).copy())
        REC["ci"].append(torch.where(b["cmask"], b["ci"], -1).cpu().numpy().astype(np.int64))
        REC["H"].append(b["H"].cpu().numpy().astype(np.int64))
        REC["hes"].append(torch.where(b["hmask"], b["hes"], -1).cpu().numpy().astype(np.int64).ravel())
    return b


def forward(self, *a, **k):
    o = _fwd(self, *a, **k)
    if self.training:
        REC["lg_model"].append(o.detach().float().cpu().numpy())
    return o


def listwise_loss(logit, *a, **k):
    REC["lg_loss"].append(logit.detach().float().cpu().numpy())
    return _ll(logit, *a, **k)


def eval_subset(*a, **k):
    REC["n_eval"] += 1
    return _eval(*a, **k)


def predict(*a, **k):
    REC["n_predict"] += 1
    return _predict(*a, **k)


MS.gather, Mo.Ranker.forward, Mo.listwise_loss = gather, forward, listwise_loss
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
             ci=flat(REC["ci"]).astype(np.int64), H=flat(REC["H"]).astype(np.int64), hes=flat(REC["hes"]).astype(np.int64),
             lg_model=flat(REC["lg_model"]).astype(np.float32), lg_loss=flat(REC["lg_loss"]).astype(np.float32),
             n_model=np.array(len(REC["lg_model"])), n_loss=np.array(len(REC["lg_loss"])),
             n_eval=np.array(REC["n_eval"]), n_predict=np.array(REC["n_predict"]),
             mini_train=FEEDS["train"], mini_val=FEEDS["val"], mini_test=FEEDS["test"])
