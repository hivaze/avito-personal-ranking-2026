"""Stale-val check of the cross-recipe pool (TO halves): CPU replay of the 15 train-only members on val with the
review_shift counterfactual inputs (notest / trunc17 / both = private-like popularity coverage). Read-only on nn/ckpt, feat_cache."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import *  # noqa
import polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

OUT = "/tmp/scratchpad/solvers/errors/out/cf"
os.makedirs(OUT, exist_ok=True)
CF = "/tmp/scratchpad/review/shift/out/cf"
v12 = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]
v15 = ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"]
v13 = ["v13_lr1_mix0", "v13_lr1_mix1", "v13_lr2_mix1"]
scen = {"base": {}}
for s in ["notest", "trunc17", "both"]:
    scen[s] = {n.replace(f"{s}_", "").replace(".npy", ""): f"{CF}/{n}" for n in os.listdir(CF) if n.startswith(s + "_")}
prewb = {n.replace("prewboth_", "").replace(".npy", ""): f"{CF}/{n}" for n in os.listdir(CF) if n.startswith("prewboth_")}
f = pl.read_parquet("<repo>/data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy()
y = np.load("<repo>/data/flat/val/cand_target.npy")
d = D.SplitData("val", torch.device("cpu"), max_hist=256)
t0 = time.time()
for m in v12 + v15 + v13:
    for s in ("base", "both"):
        p = f"{OUT}/cf_{m}_{s}.npy"
        if os.path.exists(p):
            continue
        ov = dict(scen[s])
        if s == "both" and "prew" in m:
            ov.update(prewb)
        pr, _ = run(m, "val", ov, data=d)
        np.save(p, pr)
        if s == "base":
            ref = np.load(f"<repo>/preds/nn_{m}_trainonly_val.npy")
            print(m, "replay corr with stored", round(float(np.corrcoef(ref, pr)[0, 1]), 6), f"{time.time() - t0:.0f}s", flush=True)
