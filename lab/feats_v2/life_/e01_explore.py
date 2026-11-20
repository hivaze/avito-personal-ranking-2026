"""Exploration: item lifecycle signals for val candidates (signal check before building the family).
Computes, for val/test candidate rows, simple first/last-seen statistics over all streams and prints positive rates
and single-feature NDCG on val."""
import sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb

nb.set_num_threads(16)
SCR = Path("/tmp/scratchpad")
NI = 111_400_000


def load(split, kind, col):
    return np.load(FLAT / split / f"{kind}_{col}.npy", mmap_mode="r")


t0 = time.time()
U = np.zeros(NI, np.bool_)
for s in ["val", "test"]:
    U[np.asarray(load(s, "cand", "item_id"))] = True
print("U items", U.sum(), f"{time.time()-t0:.0f}s", flush=True)

# gather appearances: item, t, src (0 train cand, 1 train hist, 2 val cand, 3 val hist, 4 test cand, 5 test hist), hpos, et, user, split-fidx
parts = []
for si, s in enumerate(["train", "val", "test"]):
    f = pl.read_parquet(FLAT / s / "feeds.parquet")
    ts = f["timestamp"].to_numpy(); uid = f["user_id"].to_numpy()
    for k, kind in enumerate(["cand", "hist"]):
        it = load(s, kind, "item_id")
        fi = load(s, kind, "fidx")
        CH = 50_000_000
        for a in range(0, len(it), CH):
            iti = np.asarray(it[a:a + CH]); m = U[iti]
            fii = np.asarray(fi[a:a + CH])[m]
            d = dict(item=iti[m], t=ts[fii], src=np.full(m.sum(), 2 * si + k, np.int8), user=uid[fii],
                     fidx=fii.astype(np.int32), split=np.full(m.sum(), si, np.int8))
            if kind == "hist":
                d["hpos"] = np.asarray(load(s, kind, "hpos")[a:a + CH])[m].astype(np.int16)
                d["et"] = np.asarray(load(s, kind, "event_type")[a:a + CH])[m].astype(np.int8)
            else:
                d["hpos"] = np.full(m.sum(), -1, np.int16); d["et"] = np.zeros(m.sum(), np.int8)
            parts.append(pl.DataFrame(d))
        print(s, kind, f"{time.time()-t0:.0f}s", flush=True)
A = pl.concat(parts)
del parts
print("appearances", A.shape, f"{time.time()-t0:.0f}s", flush=True)
A.write_parquet(SCR / "app_valtest_items.parquet")
print(A.group_by("src").len().sort("src"))
