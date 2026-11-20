# paired.py: paired A/B on val (+ 30K random-like train holdout). {seed} in prefixes is substituted.
#   $PY paired.py --base <dir>/nn_base_s{seed} --treat <dir>/nn_<variant>_s{seed}
#   fine-tune OOF: --val_suffix _val_oof.npy (no holdout files needed)
import argparse, os, sys
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
FL = "<repo>/data/flat"
p = argparse.ArgumentParser()
p.add_argument("--base"); p.add_argument("--treat"); p.add_argument("--seeds", default="23,24")
p.add_argument("--val_suffix", default="_trainonly_val.npy")
p.add_argument("--treat_val_suffix", default=None)
a = p.parse_args()
f = pl.read_parquet(f"{FL}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{FL}/val/cand_target.npy").astype(np.float64)
def pf(s, y, off, M): return ndcg_per_feed(s, y, off, M)
D = {"val": [], "hold": []}
for s in a.seeds.split(","):
    b, t = a.base.format(seed=s), a.treat.format(seed=s)
    vb, vt = pf(np.load(b + a.val_suffix), y, off, M), pf(np.load(t + (a.treat_val_suffix or a.val_suffix)), y, off, M)
    D["val"].append(vt - vb)
    msg = f"seed {s}: val {vb.mean():.5f} -> {vt.mean():.5f} d={vt.mean()-vb.mean():+.5f} se={(vt-vb).std(ddof=1)/np.sqrt(len(vb)):.5f}"
    if os.path.exists(b + "_holdout.npz") and os.path.exists(t + "_holdout.npz"):
        hb, ht = np.load(b + "_holdout.npz"), np.load(t + "_holdout.npz")
        assert (hb["M"] == ht["M"]).all()
        Mh = hb["M"].astype(np.int64); oh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
        xb, xt = pf(hb["scores"], hb["y"], oh, Mh), pf(ht["scores"], ht["y"], oh, Mh)
        D["hold"].append(xt - xb)
        msg += f" | hold {xb.mean():.5f} -> {xt.mean():.5f} d={xt.mean()-xb.mean():+.5f} se={(xt-xb).std(ddof=1)/np.sqrt(len(xb)):.5f}"
    print(msg)
for k, v in D.items():
    if v:
        d = np.mean(v, 0); se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"POOLED {k}: d={d.mean():+.5f} se={se:.5f} z={d.mean()/se:+.2f} (n_seeds={len(v)})")
