"""CPU end-to-end wrapper of tests/test_pseudokd.py (pseudokd port, 2026-11-17; = lab/nn_x/w11/pseudo/tests/e2e_cpu.py adapted to a
train.py FILE that imports the unchanged lab/nn_x/w5 modules). Runs <train file> as __main__ on the CPU with a MINIATURE split
(tests/mini_split.MiniSplit: a few hundred real train / val feeds, memory-mapped) and records what the identity tests compare.
sys.path = [dirname(train file), lab/nn_x/w5, tests/] = what `python <train file>` with PYTHONPATH=lab/nn_x/w5 sees (+ the test helpers,
none of which shadows a w5 module). Every output of the run (common.CKPT / LOGS / PREDS) goes to <outdir>/{ckpt,logs,preds}: nothing is
written into lab/nn_x/w5. Patches (none changes a result):
  torch.Tensor.to         -> any "cuda*" target becomes the CPU; torch.cuda.is_available -> False (autocast "cuda" disables itself);
  np.load                 -> the FULL train p_random file -> the mini split's slice; a --kd_file teacher (one logit per flat train
                             candidate) -> its rows at the mini split's candidates (both once the train MiniSplit exists);
  data.SplitData          -> MiniSplitData(split, device, max_hist): MiniSplit over a FIXED feed list per split + attach_xt;
  data.SplitData.gather   -> records every TRAINING gather (with_pos=True): feed ids, candidate rows ci (-1 on padding);
  model.Ranker.forward    -> records the logits of every TRAIN-mode forward, tagged real (0) / pseudo (1: the batch carries pseudo_t);
  model.listwise_loss     -> records the logits reaching the listwise loss, w, norm, denom, value, tag, and whether its target is a
                             one-hot click vector (pseudo) or a soft mixture (KD real);
  cd_pseudokd.PseudoTrainer.build (pseudokd code only) -> the training step index of every call and the number of pseudo-feeds built;
                             VERIFIES every pseudo-feed against its source batch with the brute-force checkers (tests/pseudo_check.py:
                             the verbatim cd_arms11 checker + the port's locality / padding / key-set checks); violations are counted;
  torch.Tensor.backward   -> records the scalar every .backward() is called on (the total loss of each training step);
  common.predict / eval_subset -> counts.
  usage: CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python tests/e2e_cpu.py <train file> <outdir> <record.npz> --variant <v> <args>
"""
import os
import runpy
import sys
import warnings
from pathlib import Path

train_file, outdir, rec_path, args = os.path.abspath(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
HERE = os.path.dirname(os.path.abspath(__file__))
W5 = "<repo>/lab/nn_x/w5"
for p_ in (HERE, W5, os.path.dirname(train_file)):  # final order: [train dir, W5, tests]
    if p_ in sys.path:
        sys.path.remove(p_)
    sys.path.insert(0, p_)
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""
assert os.path.dirname(Cm.__file__) == W5 and os.path.dirname(D.__file__) == W5 and os.path.dirname(Mo.__file__) == W5, \
    (Cm.__file__, D.__file__, Mo.__file__)
from mini_split import MiniSplit, _xt_col, pick_train_feeds  # noqa: E402

for sub in ("ckpt", "logs", "preds", "preds/holdout"):
    (outdir / sub).mkdir(parents=True, exist_ok=True)
Cm.CKPT, Cm.LOGS, Cm.PREDS = outdir / "ckpt", outdir / "logs", outdir / "preds"

ROOT = "<repo>"
PRANDOM = f"{ROOT}/eda/labels_position/out/train_prandom.npz"
_np_load = np.load
ST = {"mini_prand": None, "crow_train": None, "step": 0, "ps_pending": False}
REC = dict(fidx=[], ci=[], lg_model=[], lg_tag=[], lg_loss=[], lw_w=[], lw_has_w=[], lw_norm=[], lw_denom=[], lw_val=[], lw_tag=[],
           lw_onehot=[], total=[], ps_step=[], ps_P=[], ps_viol=[], ps_viol2=[], ps_M=[], ps_H=[], ps_t=[], ps_g=[], ps_srcfidx=[],
           n_eval=0, n_predict=0, kd_file=[])


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
        self.user_id = f["user_id"].to_numpy()[self.src_feeds]  # holdout_user_mask / xfold read tr.user_id
        if split == "train":
            pr = _np_load(PRANDOM)
            ST["mini_prand"] = {"fidx": np.arange(self.n), "p_random": pr["p_random"][self.src_feeds]}
            ST["crow_train"] = self._crow
            ST["train_user_id"], ST["train_src_feeds"] = self.user_id.astype(np.int64), self.src_feeds.astype(np.int64)
        if verbose:
            print(f"[mini] {split}: {self.n} feeds, {self.n_cand} cands, {self.n_hist} hist (CPU miniature)", flush=True)

    def attach_xt(self, names):
        self.xt = torch.stack([_xt_col(self.split, n, self._crow) for n in names], 1)


def np_load(path, *a, **k):
    p = str(path)
    if p.endswith("train_prandom.npz") and ST["mini_prand"] is not None:
        return ST["mini_prand"]
    if "/kd/out/teacher" in p and p.endswith(".npy") and ST["crow_train"] is not None:  # --kd_file: one logit per flat train candidate
        REC["kd_file"].append(p)
        return np.asarray(_np_load(p, mmap_mode="r")[ST["crow_train"]])
    return _np_load(path, *a, **k)


np.load = np_load
_gather, _fwd, _ll, _eval, _predict, _bw = MiniSplit.gather, Mo.Ranker.forward, Mo.listwise_loss, Cm.eval_subset, Cm.predict, torch.Tensor.backward


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
        REC["lg_tag"].append(1 if "pseudo_t" in b else 0)
    return out


def listwise_loss(logit, y, mask, w=None, norm="batch", denom=None):
    REC["lg_loss"].append(logit.detach().float().cpu().numpy())
    REC["lw_w"].append(np.zeros(0, np.float32) if w is None else w.detach().float().cpu().numpy().ravel())
    REC["lw_has_w"].append(0 if w is None else 1)
    REC["lw_norm"].append(norm)
    REC["lw_denom"].append(-1 if denom is None else int(denom))
    REC["lw_tag"].append(1 if ST.get("ps_pending") else 0)  # the pseudo listwise call is the one right after a successful build()
    yy = y.detach().float()
    REC["lw_onehot"].append(int(bool(((yy == 0) | (yy == 1)).all()) and bool((yy.sum(1) <= 1).all())))
    ST["ps_pending"] = False
    out = _ll(logit, y, mask, w, norm=norm, denom=denom)
    REC["lw_val"].append(float(out.detach()))
    return out


def eval_subset(*a, **k):
    REC["n_eval"] += 1
    return _eval(*a, **k)


def predict(*a, **k):
    REC["n_predict"] += 1
    return _predict(*a, **k)


def backward(self, *a, **k):
    if self.dim() == 0:
        REC["total"].append(float(self.detach()))
        ST["step"] += 1
    return _bw(self, *a, **k)


MiniSplitData.gather = gather
D.SplitData = MiniSplitData
Mo.Ranker.forward, Mo.listwise_loss = forward, listwise_loss
Cm.eval_subset, Cm.predict = eval_subset, predict
torch.Tensor.backward = backward
C11 = None
if os.path.exists(os.path.join(os.path.dirname(train_file), "cd_pseudokd.py")):  # pseudokd code only: record + verify every build()
    import cd_pseudokd as C11  # noqa: E402
    from pseudo_check import check_pseudo_batch, check_pseudo_extra  # noqa: E402
    _build = C11.PseudoTrainer.build

    def build(self, b):
        bp, info = _build(self, b)
        ST["ps_pending"] = bp is not None
        REC["ps_step"].append(ST["step"])
        if bp is None:
            REC["ps_P"].append(0)
            REC["ps_viol"].append(0)
            REC["ps_viol2"].append(0)
            return bp, info
        cemb = self.tr.c_emb[b["ci"]].long()
        viol = check_pseudo_batch(b, bp, cemb, self.src_ok[b["fidx"]], self.gmin, self.gmax, self.tmax)
        viol2 = check_pseudo_extra(b, bp, self.max_hist)
        vm = bp["pseudo_valid"].cpu().numpy()
        REC["ps_P"].append(int(vm.sum()))
        REC["ps_viol"].append(int(len(viol)))
        REC["ps_viol2"].append(int(len(viol2)))
        REC["ps_M"].append(bp["M"].cpu().numpy().astype(np.int64)[vm])
        REC["ps_H"].append(bp["H"].cpu().numpy().astype(np.int64)[vm])
        REC["ps_t"].append(bp["pseudo_t"].cpu().numpy().astype(np.int64)[vm])
        REC["ps_g"].append(bp["pseudo_g"].cpu().numpy().astype(np.int64)[vm])
        REC["ps_srcfidx"].append(bp["fidx"].cpu().numpy().astype(np.int64)[vm])
        if viol or viol2:
            print(f"[e2e] PSEUDO VIOLATIONS at step {ST['step']}: {(viol + viol2)[:5]}", flush=True)
        return bp, info

    C11.PseudoTrainer.build = build
sys.argv = [train_file] + args
try:
    runpy.run_path(train_file, run_name="__main__")
finally:
    f = REC["fidx"]
    flat = lambda xs: np.concatenate([x.ravel() for x in xs]) if xs else np.zeros(0)  # noqa: E731
    offs = lambda xs: np.concatenate([[0], np.cumsum([len(x) for x in xs])]).astype(np.int64)  # noqa: E731
    np.savez(rec_path, fidx=np.concatenate(f) if f else np.zeros(0, np.int64), off=offs(f),
             ci=flat(REC["ci"]).astype(np.int64), lg_model=flat(REC["lg_model"]).astype(np.float32), lg_model_off=offs(REC["lg_model"]),
             lg_tag=np.array(REC["lg_tag"], np.int64), lg_loss=flat(REC["lg_loss"]).astype(np.float32), lg_loss_off=offs(REC["lg_loss"]),
             lw_w=flat(REC["lw_w"]).astype(np.float32), lw_w_off=offs(REC["lw_w"]), lw_has_w=np.array(REC["lw_has_w"], np.int64),
             lw_norm=np.array(REC["lw_norm"]), lw_denom=np.array(REC["lw_denom"], np.int64), lw_val=np.array(REC["lw_val"], np.float64),
             lw_tag=np.array(REC["lw_tag"], np.int64), lw_onehot=np.array(REC["lw_onehot"], np.int64), total=np.array(REC["total"], np.float64),
             ps_step=np.array(REC["ps_step"], np.int64), ps_P=np.array(REC["ps_P"], np.int64), ps_viol=np.array(REC["ps_viol"], np.int64),
             ps_viol2=np.array(REC["ps_viol2"], np.int64), ps_M=flat(REC["ps_M"]).astype(np.int64), ps_H=flat(REC["ps_H"]).astype(np.int64),
             ps_t=flat(REC["ps_t"]).astype(np.int64), ps_g=flat(REC["ps_g"]).astype(np.int64), ps_srcfidx=flat(REC["ps_srcfidx"]).astype(np.int64),
             n_model=np.array(len(REC["lg_model"])), n_loss=np.array(len(REC["lg_loss"])), n_eval=np.array(REC["n_eval"]),
             n_predict=np.array(REC["n_predict"]), kd_file=np.array(REC["kd_file"]),
             train_user_id=ST.get("train_user_id", np.zeros(0, np.int64)), train_src_feeds=ST.get("train_src_feeds", np.zeros(0, np.int64)))
