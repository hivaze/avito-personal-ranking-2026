"""R6c: W16 read (all 16 wide students finished ~21:20Z): TO pool val/holdout vs W8, by arm, @last, N-curve over random subsets of the 16, and the new-8 pool alone."""
from lo_common import *
rng = np.random.default_rng(21)
NEW = E('kd05w192', (955, 956, 957, 958))+E('kd05t25w192', (955, 956, 957, 958)); ALL = W8+NEW
ZV = {m: LV(m) for m in ALL}; ZH = {m: LH(m) for m in ALL}
pv = lambda ms: ev(np.mean([ZV[m] for m in ms], 0)); ph = lambda ms: eh(np.mean([ZH[m] for m in ms], 0))
w8v, w8h, n8v, n8h, av, ah = pv(W8), ph(W8), pv(NEW), ph(NEW), pv(ALL), ph(ALL)
print(f' W8 (orig)  val {w8v.mean():.5f} hold {w8h.mean():.5f}')
print(f' NEW8       val {n8v.mean():.5f} hold {n8h.mean():.5f} | NEW8 - W8: val {f(n8v-w8v)} hold {f(n8h-w8h)}')
print(f' W16        val {av.mean():.5f} hold {ah.mean():.5f} | W16 - W8: val {f(av-w8v)} hold {f(ah-w8h)} | M<50 {f((av-w8v)[Mv<50])} M>=50 {f((av-w8v)[Mv>=50])}')
print(' member val/hold: ' + ' '.join(f'{m.split("_")[2]}_{m[-4:]}:{ev(ZV[m]).mean():.4f}/{eh(ZH[m]).mean():.4f}' for m in ALL))
for k in (1, 2, 4, 8, 12, 16):
    subs = [rng.choice(16, k, replace=False) for _ in range(1 if k == 16 else 80)]
    v = [pv([ALL[i] for i in s]).mean() for s in subs]; h = [ph([ALL[i] for i in s]).mean() for s in subs]
    print(f'  N={k:2d}: val {np.mean(v):.5f} (sd {np.std(v):.5f}) hold {np.mean(h):.5f} (sd {np.std(h):.5f})')
try:
    lv = ev(np.mean([LV(m, '@last') for m in ALL], 0)); lh = eh(np.mean([LH(m, '@last') for m in ALL], 0))
    print(f' W16 @last  val {lv.mean():.5f} hold {lh.mean():.5f}')
except FileNotFoundError as ex: print(' @last missing', ex)
