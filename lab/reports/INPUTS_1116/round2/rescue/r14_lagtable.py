"""RESCUE r14 (label-free): the lag of the view D5 = (slot of the row) - (newest source slot of its window), and the calendar span of the window, per query set.
Output out/r14_lagtable.txt (exclusive create)."""
from rs_common import *
P = Log(OUT / 'r14_lagtable.txt')
V = Views(None)
def tab(name, slot, cut, w=None):
    ok = cut >= 0; lag = np.array([int(s - window(int(c)).max()) for s, c in zip(slot[ok], cut[ok])]); span = np.array([int(window(int(c)).max() - window(int(c)).min() + 1) for c in cut[ok]])
    u, n = np.unique(lag, return_counts=True); us, ns = np.unique(span, return_counts=True)
    P(f'{name}: feeds {int(ok.sum())}; lag to the newest source day: ' + ', '.join(f'{a} d {b / ok.sum():.3f}' for a, b in zip(u, n)) + f'; mean {lag.mean():.2f} d | calendar span of the 5 source days: ' + ', '.join(f'{a} d {b / ok.sum():.3f}' for a, b in zip(us, ns)))
a = V.A['train']; e = V.A['test']; v = V.A['val']
for nm, fm in (('pool', V.poolf), ('holdout', V.hold)):
    sl = a['slot'][fm]; ct = np.where(V.has[fm], V.tcut[fm], -1)
    tab(f'train {nm} feeds, slots 6-8 (val-like)', sl[(sl >= 6) & (sl <= 8)], ct[(sl >= 6) & (sl <= 8)]); tab(f'train {nm} feeds, slots 11-13 (test-like)', sl[sl >= 11], ct[sl >= 11])
tab('val natural', v['slot'], np.full(v['n'], NAT_CUT)); tab('val stale-val twin', v['slot'], V.vcut)
pub = e['ts'] < hc.PUB_TS
tab('test public', e['slot'][pub], np.full(int(pub.sum()), NAT_CUT)); tab('test private', e['slot'][~pub], np.full(int((~pub).sum()), NAT_CUT))
P('DONE')
