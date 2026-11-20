"""Summarize saved OOFs: per-seed deltas, mean paired per-feed delta + SE (harness definition), seed-ensemble delta.
Usage: vsum.py tag1[,tag2...] [--seeds 0,1,2,3,4] [--log] ; several tags = pooled candidate ensemble"""
import json, sys
sys.path.insert(0, "<repo>/src")
from common import *
SCR = Path("/tmp/scratchpad")
CACHE = Path("<repo>/data/eval_cache")
tags = sys.argv[1].split(",")
seeds = [int(x) for x in (sys.argv[sys.argv.index("--seeds") + 1] if "--seeds" in sys.argv else "0,1,2,3,4").split(",")]
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
fidx = np.load(FLAT / "val/cand_fidx.npy")
def fz(x):
    x = x.astype(np.float64); mu = np.bincount(fidx, weights=x, minlength=len(M)) / M; d = x - mu[fidx]
    return d / (np.sqrt(np.bincount(fidx, weights=d * d, minlength=len(M)) / M)[fidx] + 1e-9)
def base(s):
    cp = CACHE / f"base_oof_s{s}.npy"
    return np.load(cp) if cp.exists() else np.load(SCR / "oof" / f"base_s{s}.npy")
for tag in tags:
    ds, rb, rc = [], [], []
    for s in seeds:
        pb = ndcg_per_feed(base(s), y, off, M); pc = ndcg_per_feed(np.load(SCR / "oof" / f"{tag}_s{s}.npy"), y, off, M)
        ds.append(pc - pb); rb.append(pb.mean()); rc.append(pc.mean())
    d = np.mean(ds, 0)
    eb = ndcg_per_feed(np.mean([fz(base(s)) for s in seeds], 0), y, off, M)
    ec = ndcg_per_feed(np.mean([fz(np.load(SCR / "oof" / f"{tag}_s{s}.npy")) for s in seeds], 0), y, off, M)
    res = dict(tag=tag, seeds=seeds, base_ndcg=float(np.mean(rb)), cand_ndcg=float(np.mean(rc)), delta=float(d.mean()),
               delta_se=float(d.std() / np.sqrt(len(d))), per_seed_delta=[round(float(c - b), 5) for b, c in zip(rb, rc)],
               ens_base=float(eb.mean()), ens_cand=float(ec.mean()), ens_delta=float((ec - eb).mean()),
               ens_se=float((ec - eb).std() / np.sqrt(len(eb))))
    print(json.dumps(res))
if len(tags) > 1:
    eb = ndcg_per_feed(np.mean([fz(base(s)) for s in seeds], 0), y, off, M)
    ec = ndcg_per_feed(np.mean([fz(np.load(SCR / "oof" / f"{t}_s{s}.npy")) for t in tags for s in seeds], 0), y, off, M)
    print("POOLED", len(tags) * len(seeds), "models vs base", len(seeds), ":", round((ec - eb).mean(), 5), "se", round((ec - eb).std() / np.sqrt(len(eb)), 5))
