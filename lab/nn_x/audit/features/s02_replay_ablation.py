"""CPU replay of a G3 member on val with per-column ablations of the 34 cvnd_pre2s train-stage inputs (read-only on
checkpoints / feature files; writes only out/replay/<member>_<tag>.npy).

  python s02_replay_ablation.py <member>        # nn_x/r6/ckpt/<member>.pt, else nn/ckpt/<member>.pt

Tags: base; z<j> = column j set to 0 (the NaN->0 'missing' state, i.e. cold / absent); p<j> = column j permuted within
each feed (keeps the feed-level distribution, destroys the within-feed ordering); g<name>_{z,p} = the same for a
column group (pop = cols 0-7, nd = 8-17, hn = 18-23, hs = 24-27, pre2 = 28-33). Inference as nn_x/r6/data.py +
model.py on the CPU (no autocast); the base pass is checked against the stored train-only val predictions.
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

ROOT = "<repo>"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "replay"); os.makedirs(OUT, exist_ok=True)
member = sys.argv[1]
only_groups = len(sys.argv) > 2 and sys.argv[2] == "groups"
ckp = f"{ROOT}/nn_x/r6/ckpt/{member}.pt"
if not os.path.exists(ckp):
    ckp = f"{ROOT}/nn/ckpt/{member}.pt"
LOG = open(os.path.join(OUT, f"{member}.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy")
ck = torch.load(ckp, map_location="cpu", weights_only=False)
cfg = ck["cfg"]; names = cfg["xt_names"]
log(f"==== {time.ctime()} {member} from {ckp}; arch {cfg['arch']} xt {len(names)} cols")
d = D.SplitData("val", torch.device("cpu"), max_hist=ck["args"]["max_hist"], verbose=False)
d.attach_xt(names)
mdl = Ranker(**cfg); mdl.load_state_dict(ck["state_dict"]); mdl.eval()
xt0 = d.xt.clone()
rng = np.random.default_rng(99)
perm = np.concatenate([o + rng.permutation(m) for o, m in zip(off, M)])
perm_t = torch.from_numpy(perm)


@torch.no_grad()
def predict():
    out = torch.zeros(d.n_cand, dtype=torch.float32)
    for fb in d.batches(np.arange(d.n), 32768, 512, shuffle=False):
        b = d.gather(fb); d.scatter(out, b, mdl(b, use_pos=False))
    return out.numpy()


def run(tag):
    p = os.path.join(OUT, f"{member}_{tag}.npy")
    if os.path.exists(p):
        return np.load(p)
    t0 = time.time(); pr = predict(); np.save(p, pr.astype(np.float32))
    log(f"{member} {tag:10s} ndcg {ndcg(pr, y, off, M):.5f} {time.time()-t0:.0f}s")
    return pr


base = run("base")
ref = f"{ROOT}/preds/nn_{member}_trainonly_val.npy"
if os.path.exists(ref):
    r = np.load(ref)
    log(f"stored ndcg {ndcg(r, y, off, M):.5f} corr(cpu, stored) {np.corrcoef(r, base)[0, 1]:.6f}")
GROUPS = {"pop": list(range(0, 8)), "nd": list(range(8, 18)), "hn": list(range(18, 24)), "hs": list(range(24, 28)),
          "pre2": list(range(28, 34))}
jobs = []
for g, cols in GROUPS.items():
    jobs += [(f"g{g}_z", cols, "z"), (f"g{g}_p", cols, "p")]
if not only_groups:
    for j in range(len(names)):
        jobs += [(f"z{j}", [j], "z"), (f"p{j}", [j], "p")]
for tag, cols, mode in jobs:
    d.xt = xt0.clone()
    for j in cols:
        if mode == "z":
            d.xt[:, j] = 0
        else:
            d.xt[:, j] = xt0[perm_t, j]
    run(tag)
d.xt = xt0
log("DONE")
