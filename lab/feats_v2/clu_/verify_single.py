"""Verifier: single-feature NDCG (both directions, random tie-break, NaN bottom) for KEEP features on val (all/Tue/Thu)
and on a 100K-feed train sample (train labels, position-biased; also restricted to feeds' deep part is not possible for
full-list NDCG, so we report all positions). Plus max |Spearman| with base features on val."""
import sys
sys.path.insert(0, "<repo>/src")
from common import *
from metrics import ndcg_per_feed
K = ["clu_hcnt1024", "clu_hmin1024", "clu_hrw4096", "clu_het14096", "clu_pmi1024_mean", "clu_pmi11024_mean", "cat_cnt_all", "cat_min_hpos", "maxsim32", "f1"]
def prep(x, sign):
    v = np.where(np.isnan(x), -np.inf, sign * x); fin = np.isfinite(v); lo, hi = v[fin].min(), v[fin].max()
    return np.where(fin, (v - lo) / (hi - lo + 1e-12), -1.0)
def per_feed(x, y, off, M, sign, rng):
    v = prep(x, sign); return np.mean([ndcg_per_feed(v + 1e-7 * rng.random(len(y)), y, off, M) for _ in range(3)], axis=0)
# val
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
day = ((f["timestamp"].to_numpy().astype(np.int64) + 3 * 3600) // 86400 - 20702)
# train sample of 100K feeds (keep only feeds with >=1 positive irrelevant: NDCG=1 for no-positive feeds in both)
ft, Mt, offt = feed_meta("train"); rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(Mt), 100_000, replace=False))
rows = np.concatenate([np.arange(offt[i], offt[i] + Mt[i]) for i in sel])
yt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows].astype(np.float64)
Ms = Mt[sel]; offs = np.r_[0, np.cumsum(Ms)[:-1]]
print("| feature | val ndcg(+) | val ndcg(-) | val Tue (best dir) | val Thu (best dir) | train-100K ndcg(+) | train-100K ndcg(-) |")
print("|---|---|---|---|---|---|---|")
rr = np.random.default_rng(1)
for k in K:
    x = np.load(FEATS / "val" / f"{k}.npy").astype(np.float64)
    pp, pn = per_feed(x, y, off, M, 1, rr), per_feed(x, y, off, M, -1, rr)
    best = pp if pp.mean() > pn.mean() else pn
    xt = np.load(FEATS / "train" / f"{k}.npy", mmap_mode="r")[rows].astype(np.float64)
    tp, tn = per_feed(xt, yt, offs, Ms, 1, rr), per_feed(xt, yt, offs, Ms, -1, rr)
    print(f"| {k} | {pp.mean():.4f} | {pn.mean():.4f} | {best[day == 16].mean():.4f} | {best[day == 18].mean():.4f} | {tp.mean():.4f} | {tn.mean():.4f} |", flush=True)
x0 = np.zeros(len(y)); r0 = per_feed(x0 + 0.0, y, off, M, 1, rr)  # random
xr = np.zeros(len(yt)); print("random: val", round(r0.mean(), 4), "train", round(per_feed(xr, yt, offs, Ms, 1, rr).mean(), 4))
# redundancy: Spearman with base on val
import json
base = json.loads(Path("<repo>/data/eval_cache/base_features.json").read_text())
from scipy.stats import rankdata
sub = np.sort(np.random.default_rng(2).choice(len(y), 300_000, replace=False))
B = {b: rankdata(np.nan_to_num(np.load(FEATS / "val" / f"{b}.npy")[sub], nan=-1e9)) for b in base}
for k in K[:6]:
    r = rankdata(np.nan_to_num(np.load(FEATS / "val" / f"{k}.npy")[sub], nan=-1e9))
    cs = sorted(((abs(np.corrcoef(r, v)[0, 1]), b) for b, v in B.items() if np.std(v) > 0), reverse=True)[:3]
    print(f"  {k}: top |spearman| with base: " + ", ".join(f"{b}={c:.3f}" for c, b in cs))
