"""N. Does the FT branch need the new members? file = 0.5 z(FT pool of a SUBSET) + 0.5 z(TO pool of all), vs the file with the FT of all members."""
from lib import *
S16 = W8 + N8
for nm, allm, subs in (('S16 (TO of 16 students)', S16, (('FT of W8 (8)', W8), ('FT of N8 (8)', N8), ('FT of kd05w192 (4)', GROUPS['S:kd05w192']), ('FT of kd05t25w192 (4)', GROUPS['S:kd05t25w192']))),
                       ('W8 (TO of 8 wide)', W8, (('FT of kd05w192 (4)', GROUPS['S:kd05w192']), ('FT of kd05t25w192 (4)', GROUPS['S:kd05t25w192']), ('FT of 2 (s951 of each arm)', [W8[0], W8[4]])))):
    to = zv(pool(allm, 'to', 'val')); ref = ev(0.5 * zv(pool(allm, 'ft', 'val')) + 0.5 * to)
    print(f'{nm}: file with the FT of all members {ref.mean():.5f}; TO only {ev(to).mean():.5f}')
    for sn, sm in subs:
        e = ev(0.5 * zv(pool(sm, 'ft', 'val')) + 0.5 * to); print(f'   {sn:28s} {e.mean():.5f}  vs all-member FT {f(e - ref)}')
    # test-side: correlation of the two files
    tt_ = zt(pool(allm, 'to', 'test')); a = zt(0.5 * zt(pool(allm, 'ft', 'test')) + 0.5 * tt_)
    for sn, sm in subs[:1]:
        b = zt(0.5 * zt(pool(sm, 'ft', 'test')) + 0.5 * tt_); print(f'   test corr of the file with [{sn}] to the all-member file: {np.corrcoef(a, b)[0, 1]:.5f}')
