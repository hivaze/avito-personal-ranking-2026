"""M4 (CPU): the extreme of 'keep the popularity-light view pure in the fine-tune': pop-light part of the FT branch taken from the
pop-light students' TRAIN-ONLY predictions (no fine-tune at all), full-view part fine-tuned as stored. Paired vs the stored form.
Also the reverse control (full-view part not fine-tuned, pop-light part fine-tuned). Rules w0.2 and m0.4_0.2 (0.4 on M<50)."""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/training")
import numpy as np
from m3_branch_views import pool, Z, ev, line, Mv, SETS
MROW = np.repeat(Mv, Mv)
for nm, (base, pop) in SETS.items():
    Fb, Fp, Tb, Tp = pool(base, "ft"), pool(pop, "ft"), pool(base, "to"), pool(pop, "to")
    for rule, w in (("w0.2", np.float32(0.2)), ("m0.4_0.2", np.where(MROW >= 50, 0.2, 0.4).astype(np.float32))):
        to = (1 - w) * Z(Tb) + w * Z(Tp)
        cur = ev(0.5 * Z((1 - w) * Z(Fb) + w * Z(Fp)) + 0.5 * Z(to))
        v1 = ev(0.5 * Z((1 - w) * Z(Fb) + w * Z(Tp)) + 0.5 * Z(to))
        v2 = ev(0.5 * Z((1 - w) * Z(Tb) + w * Z(Fp)) + 0.5 * Z(to))
        print(f"{nm} {rule}: stored {cur.mean():.5f}")
        print(f"   pop-light part of FT branch NOT fine-tuned minus stored: {line(v1 - cur)}")
        print(f"   (control) full part of FT branch NOT fine-tuned minus stored: {line(v2 - cur)}")
