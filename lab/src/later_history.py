"""Cross-feed 'later history' signal.

For a feed of user u (u != 0) at time t, a candidate item that appears in the history of a LATER feed (t' > t) of the same user
was very likely acted on (history is snapshotted at feed time and positives reappear there). Returns per-candidate flags:
  lh_any      item in history of any later feed of the same user
  lh_et1      same, restricted to event_type 1
  lh_minpos   smallest hpos at which it appears (recency in the later history), large if absent
`sources` lists splits whose feeds may serve as "later" feeds (e.g. val uses val+test, test uses test only).
"""
import sys
sys.path.insert(0, "<repo>/src")
from common import *


def later_history_flags(split, sources):
    f, M, off = feed_meta(split)
    fidx = np.load(FLAT / split / "cand_fidx.npy")
    cu = f["user_id"].to_numpy().astype(np.int64)[fidx]
    ct = f["timestamp"].to_numpy().astype(np.int64)[fidx]
    ci = np.load(FLAT / split / "cand_item_id.npy").astype(np.int64)
    users = set(np.unique(cu[cu > 0]).tolist())
    # collect (user, item, time, event_type, hpos) of history events from feeds of these users in source splits
    rows = []
    for s in sources:
        fs = pl.read_parquet(FLAT / s / "feeds.parquet")
        hf = np.load(FLAT / s / "hist_fidx.npy")
        hu = fs["user_id"].to_numpy().astype(np.int64)[hf]
        keep = np.isin(hu, np.fromiter(users, np.int64))
        rows.append(pl.DataFrame({
            "user": hu[keep], "item": np.load(FLAT / s / "hist_item_id.npy")[keep].astype(np.int64),
            "t": fs["timestamp"].to_numpy().astype(np.int64)[hf][keep],
            "et": np.load(FLAT / s / "hist_event_type.npy")[keep], "hpos": np.load(FLAT / s / "hist_hpos.npy")[keep]}))
    h = pl.concat(rows)
    cand = pl.DataFrame({"row": np.arange(len(ci)), "user": cu, "item": ci, "tc": ct}).filter(pl.col("user") > 0)
    j = cand.join(h, on=["user", "item"], how="inner").filter(pl.col("t") > pl.col("tc"))
    agg = j.group_by("row").agg(pl.len().alias("n"), (pl.col("et") == 1).any().alias("et1"), pl.col("hpos").min().alias("minpos"))
    out = {"lh_any": np.zeros(len(ci), np.float32), "lh_et1": np.zeros(len(ci), np.float32),
           "lh_minpos": np.full(len(ci), 999, np.float32)}
    r = agg["row"].to_numpy()
    out["lh_any"][r] = 1
    out["lh_et1"][r] = agg["et1"].to_numpy().astype(np.float32)
    out["lh_minpos"][r] = agg["minpos"].to_numpy()
    return out


def apply_boost(scores, flags, off, M, mode="any"):
    """Move flagged candidates above all others in their feed, keeping the model order among each group."""
    from stack_val import feed_rank_z
    z = feed_rank_z(off, M, np.asarray(scores, np.float32))[:, 1].astype(np.float64)
    flag = flags["lh_et1"] if mode == "et1" else flags["lh_any"]
    return z + 100.0 * flag


if __name__ == "__main__":
    for split, src in [("val", ["val", "test"]), ("val", ["val"]), ("test", ["test"])]:
        fl = later_history_flags(split, src)
        n = fl["lh_any"].sum()
        msg = f"{split} sources={src}: flagged rows {int(n)}, et1 {int(fl['lh_et1'].sum())}"
        if split == "val":
            y = np.load(FLAT / "val/cand_target.npy")
            m = fl["lh_any"] > 0
            msg += f", target rate flagged {y[m].mean():.3f} (et1 {y[fl['lh_et1'] > 0].mean():.3f}), base {y.mean():.4f}"
        f, M, off = feed_meta(split)
        fidx = np.load(FLAT / split / "cand_fidx.npy")
        msg += f", feeds affected {len(np.unique(fidx[fl['lh_any'] > 0]))}"
        print(msg, flush=True)
        if split == "val":
            for nm in ["lh_any", "lh_et1", "lh_minpos"]:
                np.save(FEATS / "val" / f"x_{nm}_{'+'.join(src)}.npy", fl[nm])
        else:
            for nm in ["lh_any", "lh_et1", "lh_minpos"]:
                np.save(FEATS / "test" / f"x_{nm}_test.npy", fl[nm])
