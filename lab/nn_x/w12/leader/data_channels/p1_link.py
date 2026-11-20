"""P1: history-overlap linking of feeds across user ids (same person / device behind different or zero user ids).

Question: the same-user later-snapshot leak was neutralised on public for feeds linked by a non-zero user_id (G3 + overlay
= G3 to 6 decimals). Feeds of user_id 0 (pooled guests) and a guest feed followed by a logged-in feed of the same person
are not linked by user_id. If their histories overlap strongly, a rival could link them by content and read the earlier
feed's clicks from the later snapshot. This script measures (label-free) how many val / test feeds have such a link and
(on val, with labels) how often the earlier feed's candidates that appear in the later history are positives.
This is a LEAK-CLASS channel under the team's rule (a later snapshot of the same person); it is measured, never used.

Steps: distinct history items per feed (val + test, 43,018 feeds); items held by more than MAXF feeds dropped;
pair overlaps by a sparse product; link = shared >= KMIN and shared / min(distinct) >= RMIN.
Outputs: out/p1_pairs.parquet, logs on stdout.
"""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import *  # noqa
import scipy.sparse as sp

MAXF = 60

v = Split("val"); t = Split("test")
nv, nt = v.n, t.n
ts = np.concatenate([v.ts, t.ts]); uid = np.concatenate([v.user, t.user]); split = np.r_[np.zeros(nv, int), np.ones(nt, int)]
hf = np.concatenate([v.hfid, t.hfid + nv]); hi = np.concatenate([v.hitem, t.hitem]).astype(np.int64)
log("events", len(hf))
# distinct (feed, item)
key = hf.astype(np.int64) * 200_000_000 + hi
u = np.unique(key)
pf = (u // 200_000_000).astype(np.int64); pi = (u % 200_000_000).astype(np.int64)
nd = np.bincount(pf, minlength=nv + nt)                     # distinct history items per feed
ii, inv = np.unique(pi, return_inverse=True)
fr = np.bincount(inv)
keep = fr[inv] <= MAXF
log("distinct pairs", len(u), "items", len(ii), "kept pairs", keep.sum(), "items with freq>MAXF", (fr > MAXF).sum())
A = sp.csr_matrix((np.ones(keep.sum(), np.int32), (pf[keep], inv[keep])), shape=(nv + nt, len(ii)))
C = (A @ A.T).tocoo()
m = (C.row < C.col) & (C.data >= 3)
a, b, c = C.row[m], C.col[m], C.data[m]
log("pairs with >=3 shared", len(a))
mn = np.minimum(nd[a], nd[b])
r = c / np.maximum(mn, 1)
P = pl.DataFrame({"a": a, "b": b, "shared": c, "nda": nd[a], "ndb": nd[b], "ratio": r,
                  "ua": uid[a], "ub": uid[b], "ta": ts[a], "tb": ts[b], "sa": split[a], "sb": split[b]})
P = P.with_columns(
    pl.when((pl.col("ua") == pl.col("ub")) & (pl.col("ua") != 0)).then(pl.lit("same_uid"))
      .when((pl.col("ua") == 0) & (pl.col("ub") == 0)).then(pl.lit("g0_g0"))
      .when((pl.col("ua") == 0) | (pl.col("ub") == 0)).then(pl.lit("g0_uid"))
      .otherwise(pl.lit("uid_uid")).alias("kind"))
P.write_parquet(f"{OUT}/out/p1_pairs.parquet")
# overlap distribution by kind
for kind in ["same_uid", "g0_g0", "g0_uid", "uid_uid"]:
    q = P.filter(pl.col("kind") == kind)
    if q.height == 0:
        print(kind, 0); continue
    print(f"{kind:9s} pairs {q.height:8d} | shared q50/q90/q99/max {q['shared'].quantile(0.5):.0f}/{q['shared'].quantile(0.9):.0f}/"
          f"{q['shared'].quantile(0.99):.0f}/{q['shared'].max()} | ratio q50/q90/q99 {q['ratio'].quantile(0.5):.3f}/"
          f"{q['ratio'].quantile(0.9):.3f}/{q['ratio'].quantile(0.99):.3f}")
    for k, rr in [(5, 0.2), (10, 0.3), (20, 0.5)]:
        print(f"      shared>={k:2d} & ratio>={rr}: {q.filter((pl.col('shared') >= k) & (pl.col('ratio') >= rr)).height}")
# ratio histogram for g0 kinds and uid_uid
for kind in ["same_uid", "g0_g0", "g0_uid", "uid_uid"]:
    q = P.filter((pl.col("kind") == kind) & (pl.col("shared") >= 5))
    h = np.histogram(q["ratio"].to_numpy(), bins=[0, .05, .1, .2, .3, .5, .7, .9, 1.01])[0]
    print(kind, "shared>=5 ratio hist [0,.05,.1,.2,.3,.5,.7,.9,1]:", h.tolist())
log("done")
