"""e5b: cross-check of T11. The same members added to W16 are read (i) on the bare base pool (the "add" column of T4) and (ii) inside the branch 0.8 z(base) + 0.2 z(P1d + P1w) (T11).
The holdout reading (ii) minus (i) is a common offset: every row shares the baseline term [pop effect on W16]. Output out/addcheck.json."""
from ens_common import *
m = Meta(); st = Store(m); res = {}
pop = P1d + P1w; pt = m.z('val', st.pool('to_val', pop)); ph = m.z('hold', st.pool('hold', pop))
sets = {'kd05w224 x4': e10('kd05w224'), 'kd05w192gru x4': e10('kd05w192gru'), 'kdmixw192 x4': e10('kdmixw192'), 'gru + w224 + kdmix x12': e10('kd05w192gru') + e10('kd05w224') + e10('kdmixw192'), 'kwh6 x4': e10('kwh6'), 'kwsb2 x4': e10('kwsb2'),
        'kwr32 x4': e10('kwr32'), 'kwd2 x4': e10('kwd2'), 'RC ref@last x16': rc('ref'), 'RC ref2@last x16': rc('ref2')}
b0v = st.pool('to_val', W16); b0h = st.pool('hold', W16)
pe0 = m.eh(0.8 * m.z('hold', b0h) + 0.2 * ph) - m.eh(b0h); print(f'popularity-light admixture on W16, holdout: {fmt(pe0)}')
for nm, X in sets.items():
    bv = st.pool('to_val', W16 + X); bh = st.pool('hold', W16 + X)
    a = m.ev(bv) - m.ev(b0v); b = m.eh(bh) - m.eh(b0h); c = m.ev(0.8 * m.z('val', bv) + 0.2 * pt) - m.ev(0.8 * m.z('val', b0v) + 0.2 * pt); d = m.eh(0.8 * m.z('hold', bh) + 0.2 * ph) - m.eh(0.8 * m.z('hold', b0h) + 0.2 * ph); dd = d - b
    res[nm] = dict(bare_val=(float(a.mean()), se(a)), bare_hold=(float(b.mean()), se(b)), branch_val=(float(c.mean()), se(c)), branch_hold=(float(d.mean()), se(d)), hold_offset=(float(dd.mean()), se(dd)))
    print(f'{nm:24s} bare base pool: val {fmt(a)} hold {fmt(b)} | inside the branch with the admixture: val {fmt(c)} hold {fmt(d)} | holdout offset {fmt(dd)}')
res['_pop_effect_on_W16_hold'] = (float(pe0.mean()), se(pe0)); json.dump(res, open(OUT / 'addcheck.json', 'w'), indent=0)
