"""Leave-user-later-out (LUO) versions of the global item history counts it_hist_all / it_hist_et{e}.

it_hist_* count an item's occurrences in ALL history snapshots of all splits. A feed's positives reappear in the same user's
later snapshots (92% in train), so for train rows these counts carry the row's own label. LUO removes the occurrences that
come from feeds of the SAME user (user_id != 0) with a LATER timestamp, in any split. The same rule is applied to every split
so val/test semantics match train; the cross-feed signal itself is used only as an explicit post-hoc boost.
Writes it_hist_all_luo, it_hist_et{e}_luo and within-feed rank/z/gap of it_hist_all_luo.
"""
import sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import polars as pl

N_ET = 10


def main():
    t0 = time.time()
    splits = ["train", "val", "test"]
    feeds = {s: pl.read_parquet(FLAT / s / "feeds.parquet") for s in splits}
    # users with >= 2 feeds over all splits (others cannot have a later snapshot)
    allu = pl.concat([feeds[s].select("user_id") for s in splits])
    multi = allu.filter(pl.col("user_id") > 0).group_by("user_id").len().filter(pl.col("len") >= 2)["user_id"].to_numpy()
    print("multi-feed users", len(multi), f"{time.time()-t0:.0f}s", flush=True)
    hs = []
    for s in splits:
        f = feeds[s]
        hf = np.load(FLAT / s / "hist_fidx.npy")
        u = f["user_id"].to_numpy()[hf]
        keep = np.isin(u, multi)
        hs.append(pl.DataFrame({"user": u[keep].astype(np.int32),
                                "item": np.load(FLAT / s / "hist_item_id.npy")[keep],
                                "th": f["timestamp"].to_numpy()[hf][keep],
                                "et": np.load(FLAT / s / "hist_event_type.npy")[keep]}))
    h = pl.concat(hs)
    print("history rows of multi-feed users", len(h), f"{time.time()-t0:.0f}s", flush=True)
    for s in splits:
        f = feeds[s]
        cf = np.load(FLAT / s / "cand_fidx.npy")
        u = f["user_id"].to_numpy()[cf]
        rows = np.flatnonzero(np.isin(u, multi))
        c = pl.DataFrame({"row": rows, "user": u[rows].astype(np.int32),
                          "item": np.load(FLAT / s / "cand_item_id.npy")[rows],
                          "tc": f["timestamp"].to_numpy()[cf][rows]})
        j = (c.lazy().join(h.lazy(), on=["user", "item"], how="inner").filter(pl.col("th") > pl.col("tc"))
             .group_by("row").agg([pl.len().alias("n")] + [(pl.col("et") == e).sum().alias(f"n{e}") for e in range(1, N_ET + 1)])
             .collect())
        r = j["row"].to_numpy()
        base = np.load(FEATS / s / "it_hist_all.npy").astype(np.float32)
        sub = np.zeros(len(base), np.float32); sub[r] = j["n"].to_numpy()
        np.save(FEATS / s / "it_hist_all_luo.npy", base - sub)
        for e in range(1, N_ET + 1):
            b = np.load(FEATS / s / f"it_hist_et{e}.npy").astype(np.float32)
            se = np.zeros(len(b), np.float32); se[r] = j[f"n{e}"].to_numpy()
            np.save(FEATS / s / f"it_hist_et{e}_luo.npy", b - se)
        y = np.load(FLAT / s / "cand_target.npy", mmap_mode="r") if s != "test" else None
        msg = f"{s}: rows with later same-user occurrences {len(r)} ({len(r)/len(base):.4%})"
        if y is not None:
            yy = np.asarray(y[r]); msg += f", target rate among them {yy.mean():.3f}"
        print(msg, f"{time.time()-t0:.0f}s", flush=True)
        # within-feed transforms of the main count
        from stack_val import feed_rank_z
        fm, M, off = feed_meta(s)
        x = base - sub
        rz = feed_rank_z(off, M, x)
        mx = np.maximum.reduceat(x, off)
        np.save(FEATS / s / "it_hist_all_luo_rk.npy", rz[:, 0]); np.save(FEATS / s / "it_hist_all_luo_z.npy", rz[:, 1])
        np.save(FEATS / s / "it_hist_all_luo_gap.npy", x - np.repeat(mx, M))


if __name__ == "__main__":
    main()
