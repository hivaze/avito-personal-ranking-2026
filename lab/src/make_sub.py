"""Write a submission parquet from flat-aligned test scores (data/flat/test cand order)."""
import sys
sys.path.insert(0, "<repo>/src")
from common import *


def make_sub(scores, out_path):
    f, M, off = feed_meta("test")
    fid = f["feed_id"].to_numpy()[np.load(FLAT / "test/cand_fidx.npy")]
    item = np.load(FLAT / "test/cand_item_id.npy")
    scores = np.asarray(scores, dtype=np.float64)
    assert len(scores) == len(item) and np.isfinite(scores).all()
    # only the within-feed order matters: emit the within-feed rank (0 = worst), which is exact in float32
    # and tie-free (exact float64 ties broken by a fixed random key) -> immune to the pessimistic tie rule
    fidx = np.load(FLAT / "test/cand_fidx.npy")
    key = np.random.default_rng(0).random(len(scores))
    order = np.lexsort((key, scores, fidx))
    rank = np.empty(len(scores), np.float64)
    rank[order] = np.arange(len(scores)) - np.repeat(off, M)
    sub = pl.DataFrame({"feed_id": fid.astype(np.int32), "item_id": item.astype(np.int32),
                        "score": rank.astype(np.float32)})
    assert not sub.select("feed_id", "score").is_duplicated().any()
    ss = pl.read_parquet("<repo>/data/raw/sample_submission.parquet")
    chk = ss.select("feed_id", "item_id").join(sub, on=["feed_id", "item_id"], how="left")
    assert chk["score"].null_count() == 0 and len(sub) == len(ss) and sub.select("feed_id", "item_id").is_unique().all()
    sub.write_parquet(out_path)
    return out_path


if __name__ == "__main__":
    make_sub(np.load(sys.argv[1]), sys.argv[2])
    print("wrote", sys.argv[2])
