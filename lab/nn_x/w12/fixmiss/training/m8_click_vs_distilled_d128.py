"""M8 (CPU): equal-width (d128) paired contrast of the popularity-light admixture on the W16 base: click-label pop-light (PC, no
teacher) minus distilled pop-light (PD4d, full-view teacher_mean25), w0.2 and m0.4_0.2."""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/training")
import numpy as np
from m3_branch_views import pool, Z, ev, line, Mv
MROW = np.repeat(Mv, Mv); S4, S8 = range(951, 955), range(951, 959)
base = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]
Fb, Tb = pool(base, "ft"), pool(base, "to")
def fin(pm, w):
    return ev(0.5 * Z((1 - w) * Z(Fb) + w * Z(pool(pm, "ft"))) + 0.5 * Z((1 - w) * Z(Tb) + w * Z(pool(pm, "to"))))
PD4d = [f"abl_e10_pop1kd05t25_s{s}" for s in S4]
for rule, w in (("w0.2", np.float32(0.2)), ("m0.4_0.2", np.where(MROW >= 50, 0.2, 0.4).astype(np.float32))):
    r = fin(PD4d, w)
    for nm, pm in (("PC4a", [f"abl_e10_pop1_s{s}" for s in S4]), ("PC4b", [f"abl_e10_pop1_s{s}" for s in range(955, 959)]), ("PC8", [f"abl_e10_pop1_s{s}" for s in S8])):
        print(f"[{rule}] {nm} minus PD4d: {line(fin(pm, w) - r)}")
