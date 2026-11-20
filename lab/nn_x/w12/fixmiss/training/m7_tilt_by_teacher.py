"""M7 (CPU): popularity tilt and distance to the full-view pool, by how the popularity-light students were taught.
Within-feed popularity tercile (it_hist_all_luo rank) O/E on M>=50 and M<50, TO branch; corr to the W16 full-view TO pool;
admixture worth on W16 at w 0.2 (from M1).  PC = click-label d128 (no teacher), PD4d = d128 distilled (teacher_mean25, full view),
PD4w = d192 distilled (teacher_mean11, full view)."""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/training")
import numpy as np
from m3_branch_views import pool, Z, ev, corr_within
from m5_large_feed_pop import oe_t
S4, S8 = range(951, 955), range(951, 959)
base = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
bto = pool(base, "to")
t, s = oe_t(Z(bto)); print(f"W16 full TO (t {t:.3f}): {s}")
for nm, ms in (("PC4a", [f"abl_e10_pop1_s{s}" for s in S4]), ("PC8", [f"abl_e10_pop1_s{s}" for s in S8]),
               ("PD4d", [f"abl_e10_pop1kd05t25_s{s}" for s in S4]), ("PD4w", [f"abl_e10_pop1kd05w192_s{s}" for s in S4])):
    p = pool(ms, "to"); t, s = oe_t(Z(p))
    print(f"{nm:5s} TO (t {t:.3f}): {s} | corr to W16 TO {corr_within(p, bto):.4f} | pool NDCG {ev(p).mean():.5f}")
