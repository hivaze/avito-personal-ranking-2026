"""Z3 (DECLARATION block 5): label-free sizing of 'borrowed history' for sparse known users from strictly earlier snapshots."""
from rv_lib import *
def load(sp):
    f = pl.read_parquet(FLAT / sp / "feeds.parquet", columns=["user_id", "timestamp", "H", "hist_off"])
    return f["user_id"].to_numpy().astype(np.int64), f["timestamp"].to_numpy().astype(np.int64), f["H"].to_numpy().astype(np.int64), f["hist_off"].to_numpy().astype(np.int64)
SRC = {sp: load(sp) + (np.load(FLAT / sp / "hist_item_id.npy", mmap_mode="r"),) for sp in ("train", "val", "test")}
IDX = {sp: (np.argsort(SRC[sp][0], kind="stable"),) for sp in ("train", "val")}
for sp in IDX:
    IDX[sp] += (SRC[sp][0][IDX[sp][0]],)
def old_only(sp, f, sources):
    u, ts, Hh, off, items = SRC[sp]
    cur = set(np.asarray(items[off[f]:off[f] + Hh[f]]).tolist())
    old = set(); n = 0; hmax = 0
    for s in sources:
        order, su = IDX[s]
        lo, hi = np.searchsorted(su, u[f]), np.searchsorted(su, u[f], side="right")
        for j in order[lo:hi]:
            if SRC[s][1][j] < ts[f]:
                n += 1; hmax = max(hmax, SRC[s][2][j])
                old.update(np.asarray(SRC[s][4][SRC[s][3][j]:SRC[s][3][j] + SRC[s][2][j]]).tolist())
    return n, len(old - cur), hmax
rows = []
for sp, sources in (("val", ("train",)), ("test", ("train", "val"))):
    u, ts, Hh, off, items = SRC[sp]
    nf = len(u)
    for K in (0, 9, 19, 49):
        ids = np.where((Hh <= K) & (u != 0))[0]
        res = np.array([old_only(sp, f, sources) for f in ids]) if len(ids) else np.zeros((0, 3))
        has = res[:, 0] > 0
        rows.append([sp, f"H <= {K}, known user id", len(ids), f"{100 * len(ids) / nf:.2f}", f"{100 * has.mean():.1f}", f"{100 * has.sum() / nf:.3f}", f"{100 * (res[:, 1] >= 10).sum() / nf:.3f}", f"{100 * (res[:, 1] >= 50).sum() / nf:.3f}",
                     f"{np.median(res[has, 1]) if has.any() else 0:.0f}", f"{np.median(res[has, 2]) if has.any() else 0:.0f}"])
    ids0 = np.where((Hh <= 19) & (u == 0))[0]
    rows.append([sp, "H <= 19, user_id 0 (pooled: no earlier snapshot can be attributed)", len(ids0), f"{100 * len(ids0) / nf:.2f}", "", "", "", "", "", ""])
print(tab(rows, ["split", "segment", "feeds", "% of feeds", "% of them with an earlier snapshot", "= % of ALL feeds", "with >= 10 old-only items, % of ALL", "with >= 50, % of ALL", "median old-only items", "median H of the earlier snapshot"]))
