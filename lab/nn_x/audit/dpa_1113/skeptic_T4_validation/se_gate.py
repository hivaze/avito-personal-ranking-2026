"""T4 SE clause (stale-holdout paired SE <= 0.7x S's) under the as-of d-7 spec: SE with all 30K feeds vs with only the
10,925 feeds that keep >= 1 day of as-of history (asof_cover.log). Per-feed delta sd = live holdout (s4_power.log), an
approximation (hypothesis: staling does not change the per-feed delta sd much); S SE = stale_both column of s5_stale_transfer.log."""
import numpy as np
C = ["Cd", "Pall", "PF7x", "M2", "Q1a", "Q2a"]
sd_h = np.array([0.0473, 0.0433, 0.0387, 0.0321, 0.0367, 0.0364])
se_S = np.array([0.00052, 0.00047, 0.00043, 0.00034, 0.00034, 0.00036])
for n, lab in ((30000, "all 30K (63.6% zero-history = popularity-zeroed)"), (10925, "10,925 feeds with >= 1 as-of history day"),
               (8279, "days 20713-15 only (4-6 history days)")):
    se = sd_h / np.sqrt(n)
    print(f"{lab:50s} SE " + " ".join(f"{c} {s:.5f}" for c, s in zip(C, se)) + " | ratio to S " + " ".join(f"{r:.2f}" for r in se / se_S)
          + f" | gate(<=0.7) {'PASS' if (se / se_S <= 0.7).all() else 'FAIL'}")
