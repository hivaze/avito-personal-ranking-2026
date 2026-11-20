"""P5: the user's LOST history (legal past): events of the same user's EARLIER train snapshots that are no longer in the
current (thinned, 256-capped) history of the val / test feed. The ranker reads only the current snapshot.

For val / test feeds of a non-zero user with >= 1 train feed: lost = distinct (item) of the user's train-snapshot
events whose item is not in the current history (union over the user's train feeds; all train feeds precede val/test).
Per candidate row:
  ex_lost       candidate item among the lost events (any type); ex_lost_ct the same for types 2-4 (contact / favourite)
  mx_lost       highest cosine of the candidate to a lost event's item (train-table vectors), mx_lost_ct for types 2-4
  cat_lost      log1p(lost events in the candidate's category), cat_cur = log1p(current-history events in it)
Nested read on the best file; coverage on test public / private (label-free).
"""
import sys, json
sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import *  # noqa
from dc_nest import Nested, fmt
import numba as nb

v = Split("val"); t = Split("test")
trf = pl.read_parquet(f"{FLAT}/train/feeds.parquet", columns=["fidx", "user_id"])
users = np.setdiff1d(np.union1d(v.user, t.user), [0])
tr_u = trf["user_id"].to_numpy()
sel_feed = np.isin(tr_u, users)
log("train feeds of val/test users", sel_feed.sum())
hf = np.load(f"{FLAT}/train/hist_fidx.npy", mmap_mode="r")
hm = sel_feed[np.asarray(hf)]
rows = np.flatnonzero(hm)
log("train history events of those feeds", len(rows))
H = pl.DataFrame({
    "u": tr_u[np.asarray(hf)[rows]],
    "item": np.asarray(np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r")[rows]),
    "et": np.asarray(np.load(f"{FLAT}/train/hist_event_type.npy", mmap_mode="r")[rows]),
    "cat": np.asarray(np.load(f"{FLAT}/train/hist_category_id.npy", mmap_mode="r")[rows]),
    "eidx": np.asarray(np.load(f"{FLAT}/train/hist_emb_idx.npy", mmap_mode="r")[rows]),
})
H = H.with_columns(((pl.col("et") >= 2) & (pl.col("et") <= 4)).alias("ct")).group_by(["u", "item"]).agg(
    pl.col("ct").any(), pl.col("cat").first(), pl.col("eidx").max())
log("distinct (user, item) in train snapshots", H.height)
tab_tr = np.memmap(f"{ROOT}/artifacts/data/competition/train/item_embeddings/data.memmap", dtype=np.int8, mode="r").reshape(-1, 128)


def build(sp):
    cur = pl.DataFrame({"g": sp.hfid, "item": sp.hitem, "cat": sp.hcat}).unique()
    fu = pl.DataFrame({"g": np.arange(sp.n), "u": sp.user})
    # lost events per feed = user's train-snapshot items not in the current history
    L = fu.filter(pl.col("u") != 0).join(H, on="u").join(cur.select("g", "item").with_columns(pl.lit(True).alias("incur")),
                                                          on=["g", "item"], how="left").filter(pl.col("incur").is_null())
    L = L.sort("g")
    lg = L["g"].to_numpy(); lit = L["item"].to_numpy(); lct = L["ct"].to_numpy(); lcat = L["cat"].to_numpy(); le = L["eidx"].to_numpy()
    ptr = np.searchsorted(lg, np.arange(sp.n + 1))
    E = np.asarray(tab_tr[np.maximum(le, 0)], np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9; E[le < 0] = 0
    tab = np.memmap(f"{ROOT}/artifacts/data/competition/{sp.split}/item_embeddings/data.memmap", dtype=np.int8, mode="r").reshape(-1, 128)
    ci = np.load(f"{FLAT}/{sp.split}/cand_emb_idx.npy")
    C = np.asarray(tab[np.maximum(ci, 0)], np.float32); C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9; C[ci < 0] = 0
    ccnt = cur.group_by(["g", "cat"]).len().rename({"len": "ncur"})
    out = np.zeros((sp.N, 6), np.float32)
    _kern(sp.fid, sp.item, sp.cat, C, ptr, lit, lct.astype(np.int8), lcat, E, out)
    cc = pl.DataFrame({"g": sp.fid, "cat": sp.cat}).join(ccnt, on=["g", "cat"], how="left")["ncur"].fill_null(0).to_numpy()
    out[:, 5] = np.log1p(cc)
    has_lost = np.diff(ptr) > 0
    return out, has_lost


@nb.njit(parallel=True, cache=True)
def _kern(fid, item, cat, C, ptr, lit, lct, lcat, E, out):
    for r in nb.prange(len(fid)):
        g = fid[r]; a, b = ptr[g], ptr[g + 1]
        exl = 0.0; exc = 0.0; mx = 0.0; mxc = 0.0; nc = 0.0
        for e in range(a, b):
            if lit[e] == item[r]:
                exl = 1.0
                if lct[e]:
                    exc = 1.0
            d = 0.0
            for k in range(128):
                d += C[r, k] * E[e, k]
            if d > mx:
                mx = d
            if lct[e] and d > mxc:
                mxc = d
            if lcat[e] == cat[r]:
                nc += 1.0
        out[r, 0] = exl; out[r, 1] = exc; out[r, 2] = mx; out[r, 3] = mxc; out[r, 4] = np.log1p(nc)


for sp in (v, t):
    sp.hcat = np.load(f"{FLAT}/{sp.split}/hist_category_id.npy")
Xv, hv = build(v); log("val built; feeds with lost events", hv.sum(), "rows ex_lost", int(Xv[:, 0].sum()))
Xt, ht = build(t); log("test built; feeds with lost events", ht.sum())
y = v.y
m = Xv[:, 0] > 0
print(f"val: candidate in lost history: rows {m.sum()} positive rate {y[m].mean() if m.any() else float('nan'):.4f} (base {y.mean():.4f}); "
      f"contact/fav: rows {(Xv[:, 1] > 0).sum()} pos rate {y[Xv[:, 1] > 0].mean() if (Xv[:, 1] > 0).any() else float('nan'):.4f}")
for lo, hi in ((0.95, 1.01), (0.98, 1.01)):
    mm = (Xv[:, 2] >= lo) & (Xv[:, 2] < hi)
    print(f"val rows with a lost-history item at cos >= {lo}: {mm.mean():.4f}, positive rate {y[mm].mean():.4f}")
s = v.best(); N = Nested(v, s)
fm = np.bincount(v.fid, weights=np.repeat(hv, v.M), minlength=v.n) > 0
cols = [Xv[:, j].astype(np.float64) for j in range(6)]
r = N.read("P5 lost history (6 cols)", cols, mask_feeds=fm); print(fmt(r))
r2 = N.read("P5 lost history, cos and exact only", cols[:4], mask_feeds=fm); print(fmt(r2))
rng = np.random.default_rng(3); perm = np.lexsort((rng.random(v.N), v.fid))
r3 = N.read("P5 lost history [placebo]", [c[perm] for c in cols], insample=False); print(fmt(r3))
pub = t.public
print(f"feeds with lost events: val {hv.mean():.4f} public {ht[pub].mean():.4f} private {ht[~pub].mean():.4f}")
json.dump([r, r2, r3], open(f"{OUT}/out/p5_read.json", "w"), indent=1, default=float)
log("done")
