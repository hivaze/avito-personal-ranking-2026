"""T7: pre-registration numbers for a public test of the FT branch: G3 blend vs TO-only branch vs FT-only branch on val,
by slice (M<50, M>=50, Tue, Thu, has_later), with paired SE. The FT branch has never been isolated on the public LB
(every submitted pool was a 0.5/0.5 blend). Also the holdout-vs-val-vs-public ordering context for the 5 scored pools."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
import calib
P = "<repo>/preds"
f, M, off = feed_meta("val"); M = M.astype(np.int64); y = np.load(FLAT / "val/cand_target.npy")
fid = np.repeat(np.arange(len(M)), M)
def feed_z(s):
    s = s.astype(np.float64); mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]
ft, to = np.load(f"{P}/ens_G3_ft_val.npy"), np.load(f"{P}/ens_G3_to_val.npy")
blend = 0.5 * feed_z(ft) + 0.5 * feed_z(to)
tv = f["timestamp"].to_numpy(); uv = f["user_id"].to_numpy(); tue = (tv + 10800) // 86400 < 20720
ftt = pl.read_parquet(FLAT / "test/feeds.parquet")
allu = np.concatenate([uv, ftt["user_id"].to_numpy()]); allt = np.concatenate([tv, ftt["timestamp"].to_numpy()])
last = pl.DataFrame({"u": allu, "t": allt}).group_by("u").agg(pl.col("t").max()); lastd = dict(zip(last["u"].to_list(), last["t"].to_list()))
has_later = np.array([(u != 0) and (lastd[u] > t) for u, t in zip(uv, tv)])
slices = {"all": np.ones(len(M), bool), "M<50": M < 50, "M>=50": M >= 50, "Tue": tue, "Thu": ~tue, "user0": uv == 0, "has_later": has_later}
out = []
def say(s): print(s, flush=True); out.append(s)
nb, nf, nt = ndcg_per_feed(blend, y, off, M), ndcg_per_feed(ft, y, off, M), ndcg_per_feed(to, y, off, M)
say(f"G3 val: blend {nb.mean():.5f}  FT-only {nf.mean():.5f}  TO-only {nt.mean():.5f}")
for nm, a, b in [("blend - TO", nb, nt), ("blend - FT", nb, nf), ("FT - TO", nf, nt)]:
    d = a - b
    say(f"{nm}: " + " | ".join(f"{k} {d[s].mean():+.5f}+-{d[s].std() / np.sqrt(s.sum()):.5f}" for k, s in slices.items()))
for nm, per in [("blend", nb), ("TO-only", nt), ("FT-only", nf)]:
    c = calib.predict(per); say(f"calib {nm}: public {c['public']:.5f} private {c['private']:.5f}")
# weight sweep (descriptive; w_ft closed before at +0.0002 nested)
for w in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
    s = w * feed_z(ft) + (1 - w) * feed_z(to); n = ndcg_per_feed(s, y, off, M)
    say(f"  w_ft {w:.2f}: val {n.mean():.5f}  M<50 {n[M < 50].mean():.5f}  M>=50 {n[M >= 50].mean():.5f}  (delta vs 0.5: {(n - nb).mean():+.5f}+-{(n - nb).std() / np.sqrt(len(n)):.5f})")
say("public-scored pools (public | val | holdout) from CURRENT_STATUS 18:53: Cd 0.39208|0.39901|0.32319; PF7x 0.39272|0.39974|0.32360; G3 0.39282|0.40032|0.32379; M2 0.39243|0.40094|0.32361; S2 0.39247|0.40088|0.32373")
pub = np.array([0.39208, 0.39272, 0.39282, 0.39243, 0.39247]); val = np.array([0.39901, 0.39974, 0.40032, 0.40094, 0.40088]); hold = np.array([0.32319, 0.32360, 0.32379, 0.32361, 0.32373])
from scipy.stats import spearmanr, pearsonr
say(f"  Spearman(public, val) {spearmanr(pub, val).statistic:+.2f}; Spearman(public, holdout) {spearmanr(pub, hold).statistic:+.2f}; n=5 (context only, all within ~1 public SE of each other except Cd)")
open("<repo>/nn_x/audit/weakspots/t7_branches.txt", "w").write("\n".join(out) + "\n")
