"""Part 4 (extra): does the m02 gain differ on weekend vs weekday feeds? Public = Sat + Sun until 11:28 MSK; val = Tue/Thu.
Uses the train-period holdout (30K random-like M>=50 feeds, train-only pools), which spans all weekdays.
Selection reproduced from nn/holdout_eval.py (asserted against holdout_meta)."""
import sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/nn_x/pubdiag/noise")
from common import *  # noqa
from metrics import ndcg_per_feed
from noise import POOLS, zfeed, P
f, M, o = feed_meta("train")
pr = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[M < 50] = 0
cand = np.where((pr > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
Mh, oh = np.load(P / "holdout/holdout_meta.npy").astype(np.int64); yh = np.load(P / "holdout/holdout_y.npy")
assert (M[hold] == Mh).all()
ts = f["timestamp"].to_numpy()[hold].astype(np.int64)
day = (ts + 10800) // 86400; dow = (day + 3) % 7; hour = ((ts + 10800) % 86400) // 3600
wkend = dow >= 5
publike = (dow == 5) | ((dow == 6) & (hour < 12))
valike = (dow == 1) | (dow == 3)


def hold_pool(ms):
    u = sorted(set(ms)); cnt = {m: ms.count(m) for m in u}
    e = sum(cnt[m] * zfeed(oh, Mh, np.load(P / f"holdout/nn_{m}_holdout.npy")) for m in u) / len(ms)
    return ndcg_per_feed(e.astype(np.float32), yh, oh, Mh)


H = {k: hold_pool(POOLS[k]) for k in ["Cd", "PF7x", "G3", "M2", "Pall", "G3c", "M2c"]}
print(f"holdout feeds: weekend {wkend.sum()}  public-like (Sat + Sun<12h) {publike.sum()}  val-like (Tue/Thu) {valike.sum()}")
print(f"{'contrast':11s} {'all':>18s} {'weekday':>18s} {'weekend':>18s} {'wkend-wkday':>18s} {'publike':>18s} {'valike':>18s}")
fmt = lambda d: f"{d.mean():+.5f}±{d.std() / np.sqrt(len(d)):.5f}"
for a, b in [("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"), ("M2", "PF7x"), ("PF7x", "Pall"), ("G3", "G3c"), ("M2", "M2c")]:
    d = H[a] - H[b]
    dd = d[wkend].mean() - d[~wkend].mean(); sd = np.sqrt(d[wkend].var() / wkend.sum() + d[~wkend].var() / (~wkend).sum())
    print(f"{a + '-' + b:11s} {fmt(d):>18s} {fmt(d[~wkend]):>18s} {fmt(d[wkend]):>18s} {dd:+.5f}±{sd:.5f} {fmt(d[publike]):>18s} {fmt(d[valike]):>18s}")
