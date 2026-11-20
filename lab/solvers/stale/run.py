"""CPU inference of a train-only NN member with optional counterfactual feature overrides (read-only on project).

Copied from solvers/errors/nncpu/run.py. Changes: checkpoints are looked up (read-only) in a list of directories
(default: nn/ckpt then nn_x/solvers/ckpt; override with ckdirs=... or STALE_CKDIRS=dir1:dir2); the __main__ output
goes to solvers/out/stale/replay/ instead of the scratchpad.
"""
import os, sys, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(16)
import data as D
from model import Ranker

CKDIRS = [p for p in os.environ.get("STALE_CKDIRS", "").split(":") if p] or [
    "<repo>/nn/ckpt", "<repo>/nn_x/solvers/ckpt"]
OUT = "<repo>/solvers/out/stale/replay"


@torch.no_grad()
def predict(model, d, cand_budget=32768, max_feeds=512):
    model.eval()
    out = torch.zeros(d.n_cand, dtype=torch.float32)
    for fb in d.batches(np.arange(d.n), cand_budget, max_feeds, shuffle=False):
        b = d.gather(fb)
        lg = model(b, use_pos=False)
        d.scatter(out, b, lg)
    return out.numpy()


def ckpt_path(variant, ckdirs=None):
    """First existing <dir>/<variant>.pt over ckdirs (read-only). Raises if absent or ambiguous across dirs."""
    hits = [os.path.join(c, f"{variant}.pt") for c in (ckdirs or CKDIRS) if os.path.exists(os.path.join(c, f"{variant}.pt"))]
    if not hits:
        raise FileNotFoundError(f"checkpoint {variant}.pt not found in {ckdirs or CKDIRS}")
    if len(hits) > 1:
        raise RuntimeError(f"checkpoint {variant}.pt is ambiguous: {hits}")
    return hits[0]


def load(variant, ckdirs=None):
    ck = torch.load(ckpt_path(variant, ckdirs), map_location="cpu", weights_only=False)
    m = Ranker(**ck["cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def run(variant, split="val", overrides=None, data=None, ckdirs=None):
    D.OVERRIDE.clear()
    if overrides:
        D.OVERRIDE.update({(split, k): v for k, v in overrides.items()})
    m, ck = load(variant, ckdirs)
    d = data or D.SplitData(split, torch.device("cpu"), max_hist=ck["args"]["max_hist"])
    assert d.max_hist == ck["args"]["max_hist"], (d.max_hist, ck["args"]["max_hist"])
    d.xt = None
    if ck["cfg"].get("xt_names"):
        d.attach_xt(ck["cfg"]["xt_names"])
    return predict(m, d), d


if __name__ == "__main__":
    sys.path.insert(0, "<repo>/src")
    from metrics import ndcg
    import polars as pl
    v = sys.argv[1]
    t0 = time.time()
    pr, d = run(v)
    f = pl.read_parquet("<repo>/data/flat/val/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy()
    y = np.load("<repo>/data/flat/val/cand_target.npy")
    print(v, "cpu val ndcg", ndcg(pr, y, off, M), f"{time.time()-t0:.0f}s")
    ref = f"<repo>/preds/nn_{v}_trainonly_val.npy"
    if os.path.exists(ref):
        r = np.load(ref)
        print("ref ndcg", ndcg(r, y, off, M), "corr", np.corrcoef(r, pr)[0, 1], "max abs diff", np.abs(r - pr).max())
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/cpu_{v}_val.npy", pr)
