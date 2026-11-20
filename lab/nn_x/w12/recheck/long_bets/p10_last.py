"""P10: is holdout-based checkpoint selection still needed for distilled students? selected vs @last (val is independent of the selection)."""
from lb_common import *
for nm, arms in (('W8 (d192 students)', ('kd05w192', 'kd05t25w192')), ('kd05 d128 students', ('kd05',)), ('base hard labels', ('base',))):
    sel = [ld(f'w5_e10_{a}_s{s}') for a in arms for s in S4]; last = [ld(f'w5_e10_{a}_s{s}', '@last') for a in arms for s in S4]
    dv = np.mean([ev(l[0]) - ev(s[0]) for l, s in zip(last, sel)], 0); dh = np.mean([eh(l[1]) - eh(s[1]) for l, s in zip(last, sel)], 0)
    ps, pl_ = pool(sel), pool(last)
    print(f'{nm:20s} member @last - selected: val {f(dv)} hold {f(dh)} | pool @last {ev(pl_[0]).mean():.5f}/{eh(pl_[1]).mean():.5f} vs selected {ev(ps[0]).mean():.5f}/{eh(ps[1]).mean():.5f}: val {f(ev(pl_[0]) - ev(ps[0]))} hold {f(eh(pl_[1]) - eh(ps[1]))}')
