"""Stage-E information columns for the R4 pair (rule amendment 16:10 (b)): the late (Sep 17-19) / early (Sep 6-8) holdout deltas of
w5_e10_<arm>_s<seed> vs w5_e10_base_s<seed> (selected ckpt and @last), from the preds copied into nn_x/w5. CPU only, read-only.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r4/e_holdcols.py [arm=r4] [seeds=951,952]"""
import os
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
sys.path.insert(0, "<repo>/src")
from common import *  # noqa: E402,F401,F403
from metrics import ndcg_per_feed  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "r4"
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "951,952").split(",")]
W = Path("<repo>/nn_x/w5/preds")
Mh, oh = np.load(W / "holdout/holdout_meta.npy")
Mh, oh = Mh.astype(np.int64), oh.astype(np.int64)
yh = np.load(W / "holdout/holdout_y.npy")
ftr, Mtr, _ = feed_meta("train")
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
prand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
prand[Mtr < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((prand > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
assert np.array_equal(Mtr[hold], Mh)
hday = (ftr["timestamp"].to_numpy().astype(np.int64)[hold] + 10800) // 86400
late, early = hday >= hday.min() + 11, hday <= hday.min() + 2
assert late.sum() == 8279
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
for ck, label in (("", "selected"), ("@last", "@last")):
    dh = np.mean([eh(np.load(W / f"holdout/nn_w5_e10_{ARM}_s{s}{ck}_holdout.npy")) - eh(np.load(W / f"holdout/nn_w5_e10_base_s{s}{ck}_holdout.npy"))
                  for s in SEEDS], 0)
    print(f"[{label}] pooled holdout {dh.mean():+.5f}+-{se(dh):.5f} | late(Sep17-19, n={late.sum()}) {dh[late].mean():+.5f}+-{se(dh[late]):.5f} "
          f"(z {dh[late].mean() / se(dh[late]):+.1f}) | early(Sep6-8, n={early.sum()}) {dh[early].mean():+.5f}+-{se(dh[early]):.5f} "
          f"(z {dh[early].mean() / se(dh[early]):+.1f}) | mid(Sep9-14) {dh[~late & ~early].mean():+.5f}")
