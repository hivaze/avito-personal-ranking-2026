"""T5b: confirmation of T5 (raw / global-z member mean vs per-feed z-mean) on member sets NOT used to find it:
 (1) NEW = 12 from-scratch m02 members (decide5 gate-(a) passers) as their own pool;
 (2) Pall = 6-family 38-member pool (families with different logit scales: the hard case for a raw mean);
 (3) Cd = F5cd+F6+F7 (older generation; public-verified);
 (4) holdout TO branch of G3 and of NEW (M>=50 only; expect ~0 there).
Paired per-feed deltas vs the per-feed z-mean, by M bucket."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
P = "<repo>/preds"
fam = json.load(open("<repo>/nn_x/stack/hfit/cache/families.json"))
NEW = ["v27_gru_m02_s511", "v27_gru_m02_s513", "v27_gru_m02_s515", "v28_lr1_m02_s411", "v28_lr1_m02_s413", "v28_lr1_m02_s414",
       "v28_lr1_m02_s415", "v28_lr1_m02_s417", "v29_d192_m02_s711", "v29_d192_m02_s712", "v30_f7_m02_s901", "v30_f7_m02_s902"]
pools = {"G3": fam["F10"] + fam["F7x"] + fam["F7x70"], "NEW": NEW, "Pall": fam["F5cd"] + fam["F6"] + fam["F7"] + fam["F9"] + fam["F10"] + fam["F12"],
         "Cd": fam["F5cd"] + fam["F6"] + fam["F7"]}
out = []
def say(s): print(s, flush=True); out.append(s)
buckets = [(5, 10), (10, 20), (20, 50), (50, 100), (100, 301)]

def run(split, M, off, y, members, suf_ft, suf_to, label):
    fid = np.repeat(np.arange(len(M)), M)
    def feed_z(s):
        s = s.astype(np.float64); mu = np.bincount(fid, s, len(M)) / M
        sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
        return (s - mu[fid]) / sd[fid]
    glob = lambda s: (s.astype(np.float64) - s.mean()) / s.std()
    branches = []
    for suf in [suf_ft, suf_to]:
        if suf is None: continue
        arrs = []
        for v in members:
            fp = f"{P}/{suf.format(v=v)}"
            if not os.path.exists(fp): say(f"  missing {fp}"); continue
            arrs.append(np.load(fp).astype(np.float64))
        branches.append(arrs)
    res = {}
    for name, fn in [("feedz", feed_z), ("global_z", glob), ("raw", lambda s: s.astype(np.float64))]:
        bs = [np.mean([fn(s) for s in arrs], 0) for arrs in branches]
        sc = np.mean([feed_z(b) for b in bs], 0)
        res[name] = ndcg_per_feed(sc, y, off, M)
    say(f"{label}: members {len(members)}; feedz {res['feedz'].mean():.5f}")
    for name in ["global_z", "raw"]:
        d = res[name] - res["feedz"]
        cells = [f"M[{a},{b}) {d[(M >= a) & (M < b)].mean():+.5f}+-{d[(M >= a) & (M < b)].std() / np.sqrt(max(((M >= a) & (M < b)).sum(), 1)):.5f}" for a, b in buckets if ((M >= a) & (M < b)).sum() > 0]
        say(f"  {name:9s} vs feedz: all {d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f} | " + " | ".join(cells))

f, M, off = feed_meta("val"); M = M.astype(np.int64); y = np.load(FLAT / "val/cand_target.npy")
for k, mem in pools.items():
    run("val", M, off, y, mem, "nn_{v}_n2cpv3_val_oof.npy", "nn_{v}_trainonly_val.npy", f"VAL {k} (0.5 ft + 0.5 to)")
    run("val", M, off, y, mem, None, "nn_{v}_trainonly_val.npy", f"VAL {k} TO-only")
hm = np.load(f"{P}/holdout/holdout_meta.npy"); hy = np.load(f"{P}/holdout/holdout_y.npy")
HM, HO = hm[0].astype(np.int64), hm[1].astype(np.int64)
for k in ["G3", "NEW", "Pall"]:
    run("holdout", HM, HO, hy, pools[k], None, "holdout/nn_{v}_holdout.npy", f"HOLDOUT {k} TO-only")
open("<repo>/nn_x/audit/weakspots/t5b_pool_norm_confirm.txt", "w").write("\n".join(out) + "\n")
