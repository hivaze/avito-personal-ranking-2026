"""Critic check 4 (CPU): item 3 inputs. (a) strength of the proposed teacher's members/pool (PC8 = abl_e10_pop1_s951-958,
the members summed in teacher_p1 + teacher_p2) vs the FDP4 students it would replace; (b) re-check of m8: click-label d128
pop-light minus full-teacher-distilled d128 pop-light (PD4d) as the admixture on W16, w0.2 and m0.4_0.2."""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/critic_training")
from c1_lib import *  # noqa
PC8 = [f"abl_e10_pop1_s{s}" for s in S8]
PD4d = [f"abl_e10_pop1kd05t25_s{s}" for s in S4]
PC4a, PC4b = PC8[:4], PC8[4:]
print(f"PC8 (proposed teacher members) TO pool val {ndcg(pool(PC8, TO)).mean():.5f}; members "
      + " ".join(f"{ndcg(mem(m, TO)).mean():.5f}" for m in PC8))
print(f"FDP4 (students to be replaced) TO members " + " ".join(f"{ndcg(mem(m, TO)).mean():.5f}" for m in FDP4)
      + f" | pool {ndcg(pool(FDP4, TO)).mean():.5f}")
print(f"PD4w TO members " + " ".join(f"{ndcg(mem(m, TO)).mean():.5f}" for m in PD4w))
Gb, Hb = pool(W16, FT), pool(W16, TO)
for rn, w in (("w0.2", 0.2), ("m0.4_0.2", wM)):
    ref = ndcg(final(Gb, pool(PD4d, FT), Hb, pool(PD4d, TO), w))
    for nm, ms in (("PC4a", PC4a), ("PC4b", PC4b), ("PC8", PC8)):
        e = ndcg(final(Gb, pool(ms, FT), Hb, pool(ms, TO), w))
        print(f"[{rn}] {nm} minus PD4d: {line(e - ref)}")
    refw = ndcg(final(Gb, pool(PD4w, FT), Hb, pool(PD4w, TO), w))
    e = ndcg(final(Gb, pool(PC8, FT), Hb, pool(PC8, TO), w))
    print(f"[{rn}] PC8 minus PD4w (wide distilled): {line(e - refw)}")
