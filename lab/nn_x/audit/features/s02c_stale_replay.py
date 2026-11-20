"""CPU replay of a G3 member on val under the project's private-like coverage scenarios, separated by input family
(read-only; writes out/replay/<member>_st_<scenario>.npy):
  st_pop  : it_hist_{all,et1..et4}_luo + seq_cv_{invd20,norm,pop} <- nn_x/r6/stale_cf/both_*.npy (review_shift 'both')
  st_pre2 : pre2_fe_z / pre2_fe_s (and has:pre2_fe = non-NaN of pre2both_pre2_fe) <- solvers/out/stale/cf/pre2both_*
  st_both : both (= the project's 'both_pre2both' stale-val scenario of nn_x/r6/stale_pred.py)
The same code without overrides is the s02 base pass (corr 0.999999 with the stored train-only preds).
"""
import os, sys, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/nn_x/r6"); sys.path.insert(0, "<repo>/src")
import numpy as np, torch, polars as pl
torch.set_num_threads(12)
import data as D
from model import Ranker
from metrics import ndcg
from pathlib import Path

ROOT = Path("<repo>"); HERE = Path(__file__).resolve().parent; OUT = HERE / "out/replay"
CF_POP = ROOT / "nn_x/r6/stale_cf"; CF_PRE = ROOT / "solvers/out/stale/cf"
POP = ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo", "seq_cv_invd20", "seq_cv_norm", "seq_cv_pop"]
PRE = {"pre2_fe": CF_PRE / "pre2both_pre2_fe.npy", "pre2_fe_z": CF_PRE / "pre2both_pre2_fe_z.npy", "pre2_fe_s": CF_PRE / "pre2both_pre2_fe_s.npy"}
SCEN = {"st_pop": {n: CF_POP / f"both_{n}.npy" for n in POP}, "st_pre2": dict(PRE)}
SCEN["st_both"] = {**SCEN["st_pop"], **SCEN["st_pre2"]}
for s in SCEN.values():
    for p in s.values():
        assert p.exists(), p
member = sys.argv[1]
ckp = ROOT / "nn_x/r6/ckpt" / f"{member}.pt"
if not ckp.exists():
    ckp = ROOT / "nn/ckpt" / f"{member}.pt"
LOG = open(OUT / f"{member}.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


_fp = D.feat_path
ACTIVE = {}


def feat_path(split, name):
    if split == "val" and name in ACTIVE:
        return ACTIVE[name]
    return _fp(split, name)


D.feat_path = feat_path
f = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(ROOT / "data/flat/val/cand_target.npy")
ck = torch.load(ckp, map_location="cpu", weights_only=False); cfg = ck["cfg"]; names = cfg["xt_names"]
d = D.SplitData("val", torch.device("cpu"), max_hist=ck["args"]["max_hist"], verbose=False)
mdl = Ranker(**cfg); mdl.load_state_dict(ck["state_dict"]); mdl.eval()


@torch.no_grad()
def predict():
    out = torch.zeros(d.n_cand, dtype=torch.float32)
    for fb in d.batches(np.arange(d.n), 32768, 512, shuffle=False):
        b = d.gather(fb); d.scatter(out, b, mdl(b, use_pos=False))
    return out.numpy()


base = np.load(OUT / f"{member}_base.npy") if (OUT / f"{member}_base.npy").exists() else None
for tag, ovr in SCEN.items():
    p = OUT / f"{member}_{tag}.npy"
    if p.exists():
        continue
    ACTIVE.clear(); ACTIVE.update(ovr)
    t0 = time.time(); d.attach_xt(names); pr = predict(); np.save(p, pr.astype(np.float32))
    msg = f"{member} {tag:10s} ndcg {ndcg(pr, y, off, M):.5f} {time.time()-t0:.0f}s"
    if base is not None:
        msg += f" delta vs base {ndcg(pr, y, off, M) - ndcg(base, y, off, M):+.5f}"
    log(msg)
ACTIVE.clear()
log("DONE stale scenarios")
