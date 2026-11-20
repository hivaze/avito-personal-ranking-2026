"""CPU inference of a train-only NN member with optional counterfactual feature overrides (read-only on project)."""
import os, sys, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(16)
import data as D
from model import Ranker

CK = "<repo>/nn/ckpt"


@torch.no_grad()
def predict(model, d, cand_budget=32768, max_feeds=512):
    model.eval()
    out = torch.zeros(d.n_cand, dtype=torch.float32)
    for fb in d.batches(np.arange(d.n), cand_budget, max_feeds, shuffle=False):
        b = d.gather(fb)
        lg = model(b, use_pos=False)
        d.scatter(out, b, lg)
    return out.numpy()


def load(variant):
    ck = torch.load(f"{CK}/{variant}.pt", map_location="cpu", weights_only=False)
    m = Ranker(**ck["cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def run(variant, split="val", overrides=None, data=None):
    D.OVERRIDE.clear()
    if overrides:
        D.OVERRIDE.update({(split, k): v for k, v in overrides.items()})
    m, ck = load(variant)
    d = data or D.SplitData(split, torch.device("cpu"), max_hist=ck["args"]["max_hist"])
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
    np.save(f"/tmp/scratchpad/solvers/errors/out/cpu_{v}_val.npy", pr)
