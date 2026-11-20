"""staleft_review: per-feed delta AVERAGED over all member subsets of size k (expected k-pool delta over member draws), with its per-feed SE.
Separates the common (feed-sampling) part of the noise from the member-combination part. CPU only."""
import os, sys, itertools
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft_review")
import numpy as np
import polars as pl
import rederive as R
V, P = R.V, R.P
R.lines.clear()
W8 = R.W8
cs = pl.read_parquet("<repo>/eda/errors/segments/feed_table.parquet")["cold_share"].to_numpy()
Z = {}
for i, m in enumerate(W8):
    Z[i] = {"p0f": V.z(R.ld(m, "p0", "val_oof")), "p0s": V.z(R.ld(m, "p0", "val_oof_stale")), "p1f": V.z(R.ld(m, "p10", "val_oof")), "p1s": V.z(R.ld(m, "p10", "val_oof_stale")),
            "p5f": V.z(R.ld(m, "p05", "val_oof")), "p5s": V.z(R.ld(m, "p05", "val_oof_stale")),
            "tof": V.z(np.load(R.W5 / f"nn_{m}_trainonly_val.npy")), "tos": V.z(R.ld(m, "p0", "val_e0_stale"))}
pool = lambda S, k: np.mean([Z[i][k] for i in S], 0)  # noqa: E731
P("expected k-pool final-file delta vs the P=0 control = per-feed delta averaged over ALL subsets of size k; +- = per-feed SE of that average (14,423 feeds)")
for k in (4, 6, 7):
    subs = list(itertools.combinations(range(8), k))
    acc = {(t, v): np.zeros(V.n) for t in ("p1", "p5") for v in ("f", "s")}
    sq = {(t, v): np.zeros(V.n) for t in ("p1", "p5") for v in ("f", "s")}
    for S in subs:
        tf, ts = pool(S, "tof"), pool(S, "tos")
        c = {"f": V.ndcg(R.fin(pool(S, "p0f"), tf)), "s": V.ndcg(R.fin(pool(S, "p0s"), ts))}
        for t in ("p1", "p5"):
            for v, to in (("f", tf), ("s", ts)):
                d = V.ndcg(R.fin(pool(S, t + v), to)) - c[v]
                acc[(t, v)] += d; sq[(t, v)] += d * d
    for t, lab in (("p1", "P=1.0"), ("p5", "P=0.5")):
        for v, vn in (("f", "fresh"), ("s", "stale-val")):
            d = acc[(t, v)] / len(subs)
            within = np.sqrt(np.maximum(sq[(t, v)] / len(subs) - d * d, 0))  # per-feed SD across subsets
            hi = cs >= 0.45
            P(f"k={k} {lab} final {vn:9s}: {R.row(d)} | cold>=45% {R.fm(d[hi])} | cold<45% {R.fm(d[~hi])} | share of per-feed variance that is member-combination noise "
              f"{(within ** 2).mean() / ((within ** 2).mean() + d.var()):.2f}")
        dm = 0.5 * acc[(t, "f")] / len(subs) + 0.5 * acc[(t, "s")] / len(subs)
        P(f"k={k} {lab} final mix read : {R.row(dm)}")
    d = (acc[("p1", "s")] - acc[("p5", "s")]) / len(subs) * 0.5 + (acc[("p1", "f")] - acc[("p5", "f")]) / len(subs) * 0.5
    P(f"k={k} P=1.0 - P=0.5, final mix read: {R.row(d)}")
(R.OUT / "poolsize2.txt").write_text("\n".join(R.lines) + "\n")
