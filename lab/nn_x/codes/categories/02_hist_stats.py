"""02: history-side per-category statistics.
Sources: (VT) val+test histories, one feed per non-zero user (latest) + every user-0 feed; (TR) train: 200K random non-zero users,
their latest train feed (seed 0). One snapshot per user avoids counting the same overlapping history many times.
Per category: event share, event-type mix (1..10), source mix of type-1 events, repeat rate, mean hpos, missing-embedding share,
per-(user,item) funnel (P(contact 2/3 | view), P(fav 4 | view), P(5/6/8 | view)), distinct items.
53x53 matrices: consecutive-event transitions (older -> newer, different items; all events and type-1 only) and user-level
co-occurrence (category sets of a history). Output: out/hist_stats.parquet, out/hist_mats.npz, log."""
from common_cat import *

rng = np.random.default_rng(0)


def pick_latest(m, users_subset=None):
    df = pl.DataFrame({"f": np.arange(m["n"]), "u": m["user"], "t": m["ts"]})
    nz = df.filter(pl.col("u") > 0).sort("t").group_by("u").last()
    f = nz["f"].to_numpy()
    if users_subset is not None:
        f = f[np.isin(nz["u"].to_numpy(), users_subset)]
    return f


def load_hist(split, feeds, m):
    idx = rows_of_feeds(m["hoff"], m["H"], np.sort(feeds))
    d = {k: np.asarray(hist(split, k)[idx]).astype(np.int64) for k in ["category_id", "event_type", "event_source", "item_id", "hpos", "fidx", "emb_idx"]}
    return d


parts = {}
vt = []
for s in ["val", "test"]:
    m = meta(s)
    f = np.concatenate([pick_latest(m), np.where(m["user"] == 0)[0]])
    d = load_hist(s, f, m)
    d["fidx"] = d["fidx"] + (0 if s == "val" else 10_000_000)
    vt.append(d)
    log(s, "feeds used", len(f), "events", len(d["fidx"]))
parts["VT"] = {k: np.concatenate([a[k] for a in vt]) for k in vt[0]}
del vt
mtr = meta("train")
u = np.unique(mtr["user"][mtr["user"] > 0])
sel_u = rng.choice(u, 200_000, replace=False)
ftr = pick_latest(mtr, sel_u)
parts["TR"] = load_hist("train", ftr, mtr)
log("train feeds used", len(ftr), "events", len(parts["TR"]["fidx"]))

res = {"code": np.arange(NC)}
mats = {}
for nm, d in parts.items():
    c = d["category_id"]; et = d["event_type"]; es = d["event_source"]; it = d["item_id"]; fi = d["fidx"]; hp = d["hpos"]
    n = np.bincount(c, minlength=NC).astype(float)
    res[f"hist_share_{nm}"] = n / n.sum()
    for e in range(1, 11):
        res[f"et{e}_{nm}"] = np.bincount(c[et == e], minlength=NC) / np.maximum(n, 1)
    m1 = et == 1
    n1 = np.bincount(c[m1], minlength=NC).astype(float)
    for sname, smask in [("s1", es == 1), ("s2", es == 2), ("s0", es == 0), ("s3p", es >= 3)]:
        res[f"t1_{sname}_{nm}"] = np.bincount(c[m1 & smask], minlength=NC) / np.maximum(n1, 1)
    res[f"mean_hpos_{nm}"] = np.bincount(c, weights=hp, minlength=NC) / np.maximum(n, 1)
    res[f"no_emb_{nm}"] = np.bincount(c, weights=(d["emb_idx"] < 0), minlength=NC) / np.maximum(n, 1)
    # per (feed,item) funnel
    df = pl.DataFrame({"f": fi, "i": it, "c": c, "e": et}).group_by(["f", "i"]).agg(
        pl.col("c").first(), pl.len().alias("n"), (pl.col("e") == 1).any().alias("v"), pl.col("e").is_in([2, 3]).any().alias("ct"),
        (pl.col("e") == 4).any().alias("fav"), pl.col("e").is_in([5, 6, 8]).any().alias("msg"), pl.col("e").is_in([7, 9]).any().alias("e79"),
        (pl.col("e") == 10).any().alias("e10"))
    cc = df["c"].to_numpy().astype(np.int64); npair = np.bincount(cc, minlength=NC).astype(float)
    res[f"pairs_{nm}"] = npair
    res[f"repeat_pair_{nm}"] = np.bincount(cc, weights=(df["n"].to_numpy() > 1), minlength=NC) / np.maximum(npair, 1)
    res[f"ev_per_pair_{nm}"] = np.bincount(cc, weights=df["n"].to_numpy(), minlength=NC) / np.maximum(npair, 1)
    v_ = df["v"].to_numpy()
    nv = np.bincount(cc[v_], minlength=NC).astype(float)
    for k in ["ct", "fav", "msg", "e79", "e10"]:
        x = df[k].to_numpy()
        res[f"p_{k}_given_view_{nm}"] = np.bincount(cc[v_ & x], minlength=NC) / np.maximum(nv, 1)
        res[f"p_{k}_pair_{nm}"] = np.bincount(cc[x], minlength=NC) / np.maximum(npair, 1)
    res[f"p_view_pair_{nm}"] = nv / np.maximum(npair, 1)
    res[f"items_{nm}"] = pl.DataFrame({"c": c, "i": it}).group_by("c").agg(pl.col("i").n_unique()).sort("c").pipe(
        lambda q: np.bincount(q["c"].to_numpy(), weights=q["i"].to_numpy(), minlength=NC))
    # users (histories) touching the category
    fc = np.unique(fi * NC + c)
    res[f"user_reach_{nm}"] = np.bincount(fc % NC, minlength=NC) / len(np.unique(fi))
    # ---- transitions older -> newer (consecutive positions in the same history), different items
    o = np.lexsort((hp, fi))
    c_o, f_o, i_o, e_o = c[o], fi[o], it[o], et[o]
    same = (f_o[1:] == f_o[:-1]) & (i_o[1:] != i_o[:-1])
    # position k (newer) is at index j, older at j+1
    T = np.zeros((NC, NC)); np.add.at(T, (c_o[1:][same], c_o[:-1][same]), 1)
    mats[f"trans_all_{nm}"] = T
    # type-1 only sequence
    m1o = e_o == 1
    c1, f1_, i1 = c_o[m1o], f_o[m1o], i_o[m1o]
    s1 = (f1_[1:] == f1_[:-1]) & (i1[1:] != i1[:-1])
    T1 = np.zeros((NC, NC)); np.add.at(T1, (c1[1:][s1], c1[:-1][s1]), 1)
    mats[f"trans_t1_{nm}"] = T1
    # co-occurrence of categories within a history (binary per history)
    fu, finv = np.unique(fc // NC, return_inverse=True)
    B = np.zeros((len(fu), NC), np.float32); B[finv, fc % NC] = 1
    mats[f"cooc_{nm}"] = (B.T @ B).astype(np.float64)
    log(nm, "done; self-transition share", np.trace(T) / T.sum(), "type-1 self", np.trace(T1) / T1.sum())

np.savez(OUT / "hist_mats.npz", **mats)
df = pl.DataFrame({k: np.asarray(v_, dtype=np.float64) for k, v_ in res.items()})
df.write_parquet(OUT / "hist_stats.parquet")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(300)
show = df.filter(pl.col("hist_share_VT") > 0).sort("hist_share_VT", descending=True)
print(show.select(["code", "hist_share_VT", "hist_share_TR"] + [f"et{e}_VT" for e in range(1, 11)]).with_columns(pl.all().round(3)))
print(show.select(["code", "t1_s1_VT", "t1_s2_VT", "repeat_pair_VT", "ev_per_pair_VT", "p_ct_given_view_VT", "p_fav_given_view_VT",
                   "p_msg_given_view_VT", "p_e79_given_view_VT", "p_e10_given_view_VT", "p_view_pair_VT", "mean_hpos_VT", "no_emb_VT",
                   "user_reach_VT", "items_TR"]).with_columns(pl.all().round(3)))
log("done")
