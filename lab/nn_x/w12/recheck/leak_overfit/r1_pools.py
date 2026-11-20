"""R1: headline pools, selected vs @last; R2: inherited selection bound (teacher pool selected vs @last); W8 vs G3/C3 on holdout."""
from lo_common import *
print('== R1 headline TO pools (val / holdout), selected and @last')
res = {}
for name, ms in (('base x4', E('base')), ('kd05 x4', E('kd05')), ('kd05w192 x4', E('kd05w192')), ('kd05t25w192 x4', E('kd05t25w192')), ('W8', W8), ('N8', N8)):
    out = []
    for suf in ('', '@last'):
        pv, ph = poolv(ms, suf), poolh(ms, suf); res[(name, suf)] = (ev(pv), eh(ph))
        out.append(f'{suf or "sel":5s} val {res[(name,suf)][0].mean():.5f} hold {res[(name,suf)][1].mean():.5f}')
    dv = res[(name, '')][0]-res[(name, '@last')][0]; dh = res[(name, '')][1]-res[(name, '@last')][1]
    print(f' {name:16s} {out[0]} | {out[1]} | sel-last val {f(dv)} hold {f(dh)}')
for a in ('kd05 x4', 'kd05w192 x4', 'kd05t25w192 x4'):
    for suf in ('', '@last'):
        print(f'   {a} - base x4 {suf or "sel":5s}: val {f(res[(a,suf)][0]-res[("base x4",suf)][0])} hold {f(res[(a,suf)][1]-res[("base x4",suf)][1])}')
print(f'   W8 - N8 sel: val {f(res[("W8","")][0]-res[("N8","")][0])} hold {f(res[("W8","")][1]-res[("N8","")][1])} | @last: val {f(res[("W8","@last")][0]-res[("N8","@last")][0])} hold {f(res[("W8","@last")][1]-res[("N8","@last")][1])}')
print('== final files (0.5 z(FT)+0.5 z(TO)) on val')
fin = {}
for k, t in (('W8', 'W8kd'), ('S16', 'S16kd'), ('C3', 'Q2a8s3'), ('G3', 'G3')):
    ft_, to_ = zv(np.load(P/f'ens_{t}_ft_val.npy')), zv(np.load(P/f'ens_{t}_to_val.npy'))
    fin[k] = (ev(0.5*ft_+0.5*to_), ev(ft_), ev(to_))
    print(f' {k:4s} final {fin[k][0].mean():.5f} FT {fin[k][1].mean():.5f} TO {fin[k][2].mean():.5f} | final-TO {f(fin[k][0]-fin[k][2])} | FT-TO {f(fin[k][1]-fin[k][2])}')
for a, b in (('W8', 'C3'), ('W8', 'G3'), ('W8', 'S16'), ('C3', 'G3')):
    d = fin[a][0]-fin[b][0]; print(f'   {a}-{b} final val {f(d)} (per-feed sd {d.std():.4f}; SE of a public delta ~ {d.std()/np.sqrt(8562):.5f}, private ~ {d.std()/np.sqrt(20033):.5f})')
print('== R2 inherited selection: teacher tf11 members = r6_cd_m02_r101-107 + base s951-954')
T11 = F7x + E('base')
hs = poolh(T11); e_sel = eh(hs)
print(f' teacher tf11 pool holdout (selected members) {e_sel.mean():.5f}')
# base members have @last; r6_cd_m02 @last may not exist -> use base4 only as the measurable part
try:
    hl = np.mean([LH(m, '@last') for m in T11], 0); print(f' teacher tf11 pool holdout (@last members) {eh(hl).mean():.5f}  sel-last {f(e_sel-eh(hl))}')
except FileNotFoundError as ex:
    print('  no @last for', ex, '-> partial: base4 part only')
    b_sel, b_last = eh(poolh(E('base'))), eh(poolh(E('base'), '@last'))
    print(f'  base4 pool sel-last {f(b_sel-b_last)}')
    mix = np.mean([LH(m) for m in F7x] + [LH(m, '@last') for m in E('base')], 0)
    print(f'  tf11 with base @last (r6 selected): {eh(mix).mean():.5f}  sel-thismix {f(e_sel-eh(mix))}')
print('== W8 (sel / @last) vs G3 TO pool and C3 TO pool on the holdout')
g3 = eh(poolh(G3M)); c3 = eh(poolh(G3M+POP1))
for suf in ('', '@last'):
    w = res[('W8', suf)][1]
    print(f' W8{suf or "    "} hold {w.mean():.5f} | -G3 ({g3.mean():.5f}) {f(w-g3)} z {((w-g3).mean()/se(w-g3)):+.2f} | -C3 ({c3.mean():.5f}) {f(w-c3)}')
np.savez(R/'nn_x/w12/recheck/leak_overfit/out/r1.npz', w8v=res[('W8','')][0], w8h=res[('W8','')][1], w8h_last=res[('W8','@last')][1], g3h=g3, c3h=c3)
