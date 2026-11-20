"""why/fit F2 (CPU, read-only): compact transfer table over every variant of the w192 KD recipe (@last checkpoints, paired vs the same-seed
reference; from <OUT>/f12_summary@last.npy written by f12_train.py @last) and across-variant regressions:
  holdout / val member NDCG delta on the in-sample (RIN) NDCG delta, and on the out-of-sample KL(teacher || student) delta.
Output: why/fit/f2_ladder.txt"""
import numpy as np

OUT = "/tmp/scratchpad/why/fit"
S = np.load(f"{OUT}/f12_summary@last.npy", allow_pickle=True).item()
lab = {"kd05w192moeffn": "A routed FFN", "kd05w192moevo": "B routed V/O", "kd05w192b3ff1": "C depth 7 ff1", "kd05": "d128",
       "kd05w224": "d224", "kd08w192": "kd_w 0.8 (more teacher)", "kwd2": "dropout 0.2", "kw256d2": "d256 + dropout 0.2", "kwh6": "6 heads",
       "kw_ff4": "FFN x4 (2 seeds)", "kw_hl2": "2 hist layers (2 seeds)", "kw_drop0": "dropout 0 (2 seeds)", "kw_e20": "20 epochs (2 seeds)",
       "base": "click-only d128 (teacher member)"}
lines = []
P = lambda s="": (lines.append(s), print(s))
P("member deltas vs same-seed kd05w192, @last checkpoints (all at the end of the schedule), paired per feed (feed-level SE)")
P(f"{'variant':34s} {'RIN train-obj':>14s} {'RIN click CE':>13s} {'RIN KL(q|p)':>12s} {'RIN NDCG':>16s} {'HOLD NDCG':>16s} {'VAL NDCG':>16s} "
  f"{'HOLD KL':>9s} {'VAL KL':>9s}  hold/in-sample  val/in-sample")
X = []
for k, v in S.items():
    g = lambda a, b: v[(a, b)]
    rin, hold, val = g("rin", "nd"), g("hold_st", "nd"), g("val_st", "nd")
    P(f"{lab.get(k, k):34s} {g('rin', 'main')[0]:+14.5f} {g('rin', 'cc')[0]:+13.5f} {g('rin', 'kl')[0]:+12.5f} "
      f"{rin[0]:+.5f}+-{rin[1]:.5f} {hold[0]:+.5f}+-{hold[1]:.5f} {val[0]:+.5f}+-{val[1]:.5f} {g('hold_st', 'kl')[0]:+9.5f} "
      f"{g('val_st', 'kl')[0]:+9.5f}  {hold[0] / rin[0]:+8.2f}       {val[0] / rin[0]:+8.2f}")
    if k not in ("base", "kd08w192"):  # same objective (kd_w 0.5, teacher_mean11) only
        X.append((rin[0], hold[0], val[0], g("hold_st", "kl")[0], g("val_st", "kl")[0], g("rin", "cc")[0]))
X = np.array(X)
P("")
P(f"across the {len(X)} same-objective variants (kd_w 0.5; excludes kd_w 0.8 and click-only):")
for j, nm in ((1, "HOLD NDCG"), (2, "VAL NDCG")):
    b = np.polyfit(X[:, 0], X[:, j], 1)[0]
    r = np.corrcoef(X[:, 0], X[:, j])[0, 1]
    P(f"   {nm} delta on RIN (in-sample) NDCG delta: slope {b:+.3f}, r {r:+.2f}")
for j, nm, kj in ((1, "HOLD NDCG", 3), (2, "VAL NDCG", 4)):
    b = np.polyfit(X[:, kj], X[:, j], 1)[0]
    r = np.corrcoef(X[:, kj], X[:, j])[0, 1]
    P(f"   {nm} delta on its out-of-sample KL(teacher||student) delta: slope {b:+.4f} NDCG per 0.01 nats, r {r:+.2f}")
r = np.corrcoef(X[:, 5], X[:, 3])[0, 1]
P(f"   in-sample click-CE delta vs out-of-sample KL delta: r {r:+.2f} (more in-sample click fit <-> further from the teacher out of sample)")
open("<repo>/lab/nn_x/w12/moe/why/fit/f2_ladder.txt", "w").write("\n".join(lines) + "\n")
