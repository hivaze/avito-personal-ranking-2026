"""Stage-E information columns for the WAVE-10 R3 pair (nn_x/w5/logs/E_r3_predeclared.txt): late (Sep 17-19) / early (Sep 6-8)
holdout deltas of w5_e10_<arm>_s<seed> vs w5_e10_base_s<seed> (selected ckpt primary, @last reported), the holdout-day profile, the
member / pool correlations on the holdout and the mean logit level (calibration) of the arm vs the base. CPU only, read-only.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w10/r3/e_extra.py [arm=r3] [seeds=951,952]
      -> nn_x/w5/logs/e_extra_<arm>.txt
"""
import os
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.path.insert(0, "<repo>/src")
from common import *  # noqa: E402,F401,F403
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402
import polars as pl  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "r3"
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "951,952").split(",")]
W = Path("<repo>/nn_x/w5/preds")
OUT = Path("<repo>/nn_x/w5/logs") / f"e_extra_{ARM}.txt"
LINES = []


def P(s):
    print(s, flush=True)
    LINES.append(s)


Mh, oh = np.load(W / "holdout/holdout_meta.npy")
Mh, oh = Mh.astype(np.int64), oh.astype(np.int64)
yh = np.load(W / "holdout/holdout_y.npy")
_f = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "timestamp"])
_M = _f["M"].to_numpy().astype(np.int64)
_p = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
_p[_M < 50] = 0.0
_hold = np.sort(np.random.default_rng(12345).choice(np.where((_p > 0.9) & (_M >= 50))[0], 30000, replace=False))
assert np.array_equal(_M[_hold], Mh)
_day = (_f["timestamp"].to_numpy()[_hold].astype(np.int64) + 10800) // 86400
_day = _day - _day.min()
LATE, EARLY = _day >= 11, _day <= 2
fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
A = lambda s: f"w5_e10_{ARM}_s{s}"  # noqa: E731
B = lambda s: f"w5_e10_base_s{s}"  # noqa: E731
P(f"== STAGE E extra columns {ARM} s{SEEDS} vs base ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): holdout late (Sep 17-19, n {LATE.sum()}) / early (Sep 6-8, n {EARLY.sum()})")
for tag, suf in (("selected", ""), ("@last", "@last")):
    try:
        dh = np.mean([eh(np.load(W / f"holdout/nn_{A(s)}{suf}_holdout.npy")) - eh(np.load(W / f"holdout/nn_{B(s)}{suf}_holdout.npy")) for s in SEEDS], 0)
    except FileNotFoundError as e:
        P(f"[{tag}] missing: {e}")
        continue
    P(f"[{tag}] pooled member holdout {dh.mean():+.5f}+-{se(dh):.5f} | late {dh[LATE].mean():+.5f}+-{se(dh[LATE]):.5f} (z {dh[LATE].mean() / se(dh[LATE]):+.2f}) | "
      f"early {dh[EARLY].mean():+.5f}+-{se(dh[EARLY]):.5f} (z {dh[EARLY].mean() / se(dh[EARLY]):+.2f}) | by day " + " ".join(f"d{d}:{dh[_day == d].mean():+.4f}" for d in np.unique(_day)))
    if tag == "selected":
        ZA = {s: zh(np.load(W / f"holdout/nn_{A(s)}_holdout.npy")) for s in SEEDS}
        ZB = {s: zh(np.load(W / f"holdout/nn_{B(s)}_holdout.npy")) for s in SEEDS}
        ph = eh(zh(np.mean([ZA[s] for s in SEEDS], 0))) - eh(zh(np.mean([ZB[s] for s in SEEDS], 0)))
        P(f"[{tag}] 2-member pool holdout {ph.mean():+.5f}+-{se(ph):.5f} | late {ph[LATE].mean():+.5f} early {ph[EARLY].mean():+.5f} | holdout member corr: "
          f"same-seed arm-base {np.mean([corr(ZA[s], ZB[s]) for s in SEEDS]):.5f}, cross-seed {np.mean([corr(ZA[SEEDS[0]], ZB[SEEDS[1]]), corr(ZA[SEEDS[1]], ZB[SEEDS[0]])]):.5f}, "
          f"base-base {corr(ZB[SEEDS[0]], ZB[SEEDS[1]]):.5f}, arm-arm {corr(ZA[SEEDS[0]], ZA[SEEDS[1]]):.5f}")
        for s in SEEDS:
            a_, b_ = np.load(W / f"nn_{A(s)}_trainonly_val.npy"), np.load(W / f"nn_{B(s)}_trainonly_val.npy")
            P(f"[{tag}] s{s} val logit level (calibration): {ARM} mean {a_.mean():+.2f} sd {a_.std():.2f} (sigmoid mean {1 / (1 + np.exp(-np.clip(a_, -50, 50))).mean():.4f}) | "
              f"base mean {b_.mean():+.2f} sd {b_.std():.2f} | positive rate {yv.mean():.4f}")
open(OUT, "w").write("\n".join(LINES) + "\n")
print(f"wrote {OUT}")
