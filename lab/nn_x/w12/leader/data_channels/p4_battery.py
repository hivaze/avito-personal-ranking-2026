"""P4: one lens for the remaining channels, on the best file (FD16P20@m0.4_0.2), plus the known leak as a yardstick.

Single-column nested reads (dc_nest; within-feed z of the column unless stated):
  stored order        position of the row in the stored val list, j / (M - 1)               (dead per readme: shuffled)
  emb_idx order       within-feed rank of item_id__embedding_idx
  item_id order       within-feed rank of item_id (the "item id as a clock" idea)
  f4 x item_id        item_id rank minus f4 rank (listing age vs id)
  LEAK yardstick      candidate appears in a LATER val/test history of the same non-zero user (later-snapshot leak,
                      neutralised on public: G3 + overlay = G3 to 6 decimals); reported to scale the other numbers.
"""
import sys, json
sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import *  # noqa
from dc_nest import Nested, fmt

v = Split("val"); t = Split("test", hist=True)
s = v.best()
N = Nested(v, s)


def wrank(x):
    o = np.lexsort((x, v.fid)); r = np.empty(v.N); r[o] = np.arange(v.N) - np.repeat(v.off, v.M)
    return r / np.maximum(v.M[v.fid] - 1, 1)


j = (np.arange(v.N) - np.repeat(v.off, v.M)) / np.maximum(v.M[v.fid] - 1, 1)
eidx = np.load(f"{FLAT}/val/cand_emb_idx.npy").astype(np.float64)
f4 = np.load(f"{FLAT}/val/cand_f4.npy").astype(np.float64)
cols = {
    "stored order": [j],
    "emb_idx order": [wrank(eidx)],
    "item_id order": [wrank(v.item.astype(np.float64))],
    "item_id rank - f4 rank": [wrank(v.item.astype(np.float64)) - wrank(f4)],
}
# same-user later-snapshot flag (LEAK yardstick)
hist = pl.DataFrame({"g": np.concatenate([v.hfid, t.hfid + v.n]), "item": np.concatenate([v.hitem, t.hitem])}).unique()
feeds = pl.DataFrame({"g": np.arange(v.n + t.n), "u": np.concatenate([v.user, t.user]), "ts": np.concatenate([v.ts, t.ts])})
cand = pl.DataFrame({"row": np.arange(v.N), "g": v.fid, "item": v.item}).join(feeds, on="g")
later = hist.join(feeds, on="g").filter(pl.col("u") != 0).rename({"g": "g2", "ts": "ts2"})
lk = cand.filter(pl.col("u") != 0).join(later, on=["u", "item"]).filter(pl.col("ts2") > pl.col("ts"))["row"].unique().to_numpy()
leak = np.zeros(v.N); leak[lk] = 1
cols["LEAK: in later same-user history"] = [leak]
log("leak rows", int(leak.sum()), "positive rate", v.y[leak > 0].mean())
res = []
for name, c in cols.items():
    r = N.read(name, c); print(fmt(r), flush=True); res.append(r)
json.dump(res, open(f"{OUT}/out/p4_battery.json", "w"), indent=1, default=float)
log("done")
