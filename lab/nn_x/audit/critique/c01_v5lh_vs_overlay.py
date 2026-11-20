"""Critique check 1: was the day-1 public null of the later-history boost (sub_v5 vs sub_v5_lh, both 0.3802 public)
a valid test of the same leak that the overlay now exploits?  Compares the two day-1 files row by row, verifies the
boost direction (score = within-feed rank, higher = better), ties, public membership (ts < 1790497920), and the
overlap with nn_x/audit/testinfo/out_test_leak_pos.npy and with src/later_overlay.py applied to G3.
CPU only."""
import sys, os
os.environ.setdefault("POLARS_MAX_THREADS", "12")
sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from common import feed_meta, FLAT

ROOT = "<repo>"
BOUND = 1790497920

f, M, off = feed_meta("test")
fid_of_feed = f["feed_id"].to_numpy()
ts = f["timestamp"].to_numpy()
uid = f["user_id"].to_numpy()
cand_fidx = np.load(FLAT / "test/cand_fidx.npy")
cand_item = np.load(FLAT / "test/cand_item_id.npy")
row_fid = fid_of_feed[cand_fidx]
pub_feed = ts < BOUND
print(f"test feeds {len(M)}, public {pub_feed.sum()}, private {(~pub_feed).sum()}, rows {len(cand_item)}")

base = pl.read_parquet(f"{ROOT}/subs/sub_v5_nnfinal2_res2.parquet")
lh = pl.read_parquet(f"{ROOT}/subs/sub_v5_nnfinal2_res2_lh.parquet")
print("columns", base.columns, lh.columns, len(base), len(lh))
j = base.join(lh, on=["feed_id", "item_id"], how="inner", suffix="_lh")
assert len(j) == len(base) == len(lh)
diff = j.filter(pl.col("score") != pl.col("score_lh"))
print(f"rows with changed score: {len(diff)} in {diff['feed_id'].n_unique()} feeds")

# within-feed rank-from-top of every row in both files
def rank_from_top(df, col):
    return (df.with_columns(pl.col(col).rank(method="min", descending=True).over("feed_id").alias("r"))["r"].to_numpy())
j = j.with_columns(pl.col("score").rank(method="min", descending=True).over("feed_id").alias("r_base"),
                   pl.col("score_lh").rank(method="min", descending=True).over("feed_id").alias("r_lh"),
                   pl.col("score_lh").count().over("feed_id").alias("Mf"))
# boosted rows = rows whose rank from the top improved by more than the mechanical shift (score went up)
up = j.filter(pl.col("score_lh") > pl.col("score"))
down = j.filter(pl.col("score_lh") < pl.col("score"))
print(f"rows with score UP {len(up)} (feeds {up['feed_id'].n_unique()}), DOWN {len(down)} (feeds {down['feed_id'].n_unique()})")
# the boosted set: rows that moved to the top (r_lh <= number of up-rows in that feed)
up = up.with_columns(pl.len().over("feed_id").alias("n_up"))
at_top = up.filter(pl.col("r_lh") <= pl.col("n_up"))
print(f"boosted rows now at the top of their feed: {len(at_top)}/{len(up)}; mean rank-from-top before {up['r_base'].mean():.1f}, after {up['r_lh'].mean():.2f}; mean M of their feeds {up['Mf'].mean():.1f}")
# ties inside lh file within feed
ties = lh.group_by("feed_id").agg((pl.col("score").n_unique() != pl.len()).alias("tie"))["tie"].sum()
print(f"feeds with tied scores in the lh file: {ties}")
# which boosted feeds are public?
fid2ts = dict(zip(fid_of_feed.tolist(), ts.tolist()))
up_feeds = np.array(sorted(set(up["feed_id"].to_list())))
up_pub = np.array([fid2ts[x] < BOUND for x in up_feeds])
print(f"boosted feeds: {len(up_feeds)}; public {up_pub.sum()}, private {(~up_pub).sum()}")

# compare with the testinfo flags (flat test row order)
pos = np.load(f"{ROOT}/nn_x/audit/testinfo/out_test_leak_pos.npy")
neg = np.load(f"{ROOT}/nn_x/audit/testinfo/out_test_leak_neg.npy")
flag_keys = set(zip(row_fid[pos].tolist(), cand_item[pos].tolist()))
up_keys = set(zip(up["feed_id"].to_list(), up["item_id"].to_list()))
print(f"testinfo later_hist flags: rows {pos.sum()}, feeds {len(set(row_fid[pos].tolist()))}; day-1 boosted rows {len(up_keys)}")
print(f"overlap of (feed,item) keys: {len(flag_keys & up_keys)}; only day-1 {len(up_keys - flag_keys)}; only testinfo {len(flag_keys - up_keys)}")
pf = np.array(sorted(set(row_fid[pos].tolist())))
pf_pub = np.array([fid2ts[x] < BOUND for x in pf])
print(f"testinfo flagged feeds public {pf_pub.sum()} private {(~pf_pub).sum()}")
# per-feed NDCG impact simulation: what would the public delta have been if boosted rows were ALL positive /
# ALL negative, using the base file's ranks (binary, 1 positive assumed at the base rank of each boosted row)?
# -> gives the two-sided expectation that the 0.3802 == 0.3802 reading must be compared against.
import math
def ndcg_of_positions(pos_list, Mf):
    pos_list = sorted(pos_list)
    dcg = sum(1 / math.log2(p + 2) for p in pos_list)
    idcg = sum(1 / math.log2(i + 2) for i in range(len(pos_list)))
    return dcg / idcg
gain_if_pos = 0.0; loss_if_neg = 0.0
for fidv, g in up.filter(pl.col("feed_id").is_in(up_feeds[up_pub].tolist())).group_by("feed_id"):
    Mf = int(g["Mf"][0]); rb = (g["r_base"].to_numpy() - 1).tolist(); n = len(rb)
    # if all boosted rows were the feed's only positives: before = ndcg at base ranks, after = 1.0
    gain_if_pos += 1.0 - ndcg_of_positions(rb, Mf)
    # if they were negatives and the feed had one positive at base rank 0..: after, that positive moves down by n
    # (approximate: 1 positive at median base rank of G3 positives ~ 5 -> discount shift)
    loss_if_neg += (1 / math.log2(5 + 2 + n)) / (1 / math.log2(5 + 2)) - 1.0
npub = int(pub_feed.sum())
print(f"if boosted rows were all positive (and the only positives): public delta ~ +{gain_if_pos/npub:.5f}; if all negative: ~ {loss_if_neg/npub:+.5f}")
print("done c01")
