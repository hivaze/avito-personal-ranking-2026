"""Step 5 robustness (DECLARATION block 8): nested gains by day / M bucket / pmix weights; label-free test coverage of the gate cells; test ranking change against the PLAIN file of the same members."""
import sys, pickle; sys.path.insert(0, '<repo>/src'); sys.path.insert(0, '<repo>/eda/errors/segments')
from lib import *
import pmix
from segdefs import cut
f, M, off = meta('val'); pr = props('val'); gt = gates(pr); W = pmix.weights(); n = len(M); fams = families(); names = [x[0] for x in fams]; parts = partitions(gt)
SHOW = ['C(M50)', 'C(Mq)', 'G(M3q)', 'G(M3r)', 'K(Mq,Hq)', 'F(Hq)', 'A']
Mb = cut(M, [10, 25, 50, 100, 200], ['M5-10', 'M11-25', 'M26-50', 'M51-100', 'M101-200', 'M201-300']); day = pr['DAY']
print('pmix weights: mean cold share (segment definition) val / public / private', round(W['cs_val'], 3), round(W['cs_pub'], 3), round(W['cs_priv'], 3))
for c in COMPS:
    R = pickle.load(open(HERE / f'res_{c}.pkl', 'rb')); plain = R['plain_vec']; mr = R['mrule_vec']
    print(f'\n=== {c} {COMPS[c]}  (nested per-feed vectors, mean over 5 fold seeds; minus PLAIN unless stated)')
    rows = [('fixed MRULE', mr)] + [(k, R['vec'][names.index(k)]) for k in SHOW] + [('VAR B-3', R['var']['B-3']['vec']), ('VAR B-all', R['var']['B-all']['vec'])]
    print(f'{"rule":12s} | val                 | pmix-public         | pmix-private        | day 1 (n {int((~day).sum())})      | day 2 (n {int(day.sum())})      | ' + ' | '.join(f'{b} (n {int((Mb == b).sum())})' for b in ['M5-10', 'M11-25', 'M26-50', 'M51-100', 'M101-200', 'M201-300']))
    for tag, v in rows:
        d = v - plain
        print(f'{tag:12s} | {fmt(d)} | {fmt(d, W["public"])} | {fmt(d, W["private"])} | {fmt(d[~day])} | {fmt(d[day])} | ' + ' | '.join(f'{d[Mb == b].mean():+.5f}+-{d[Mb == b].std() / np.sqrt((Mb == b).sum()):.5f}' for b in ['M5-10', 'M11-25', 'M26-50', 'M51-100', 'M101-200', 'M201-300']))
    print('  minus fixed MRULE at the private mix: ' + ' | '.join(f'{tag} {fmt(v - mr, W["private"])}' for tag, v in rows[1:]))
# ---- label-free: gate cells on val / public-like / private-like test
pt = props('test'); g_t = gates(pt); pub = pt['ts'] < BOUND
print('\n=== LABEL-FREE gate coverage: share of feeds per cell, val | public-like test | private-like test   (cell 0 / 1 / 2)')
for g in G2 + G3:
    nc = 2 if g in G2 else 3
    print(f'{g:5s} ' + ' | '.join('/'.join(f'{(x == k).mean():.3f}' for k in range(nc)) for x in (gt[g], g_t[g][pub], g_t[g][~pub])))
print('ranges: ' + ' ; '.join(f'{k} val [{pr[k].min():.3g}, {pr[k].max():.3g}] test [{pt[k].min():.3g}, {pt[k].max():.3g}]' for k in ('M', 'H', 'CL', 'CS')))
for a, b in (('M50', 'Hq'), ('Mq', 'Hq')):
    print(f'joint {a} x {b} (cells 00/01/10/11): ' + ' | '.join('/'.join(f'{((2 * x[a] + x[b])[m] == k).mean():.3f}' for k in range(4)) for x, m in ((gt, slice(None)), (g_t, pub), (g_t, ~pub))))
# ---- label-free: test ranking change vs PLAIN for fixed rules on the FD16 + FDP members (and FD8)
ft_, Mt, ot = meta('test'); nt = len(Mt); Mrow = np.repeat(Mt, Mt); Hrow = np.repeat(pt['H'], Mt)
@nb.njit(parallel=True)
def cmp_rank(off, M, a, b):
    """per feed: same top-1 (0/1), Spearman rho of the two score vectors."""
    nf = len(M); top = np.zeros(nf); rho = np.zeros(nf)
    for f in nb.prange(nf):
        o = off[f]; m = M[f]; x = a[o:o + m]; y = b[o:o + m]
        top[f] = 1.0 if np.argmax(x) == np.argmax(y) else 0.0
        rx = np.argsort(np.argsort(x)).astype(np.float64); ry = np.argsort(np.argsort(y)).astype(np.float64)
        rho[f] = np.corrcoef(rx, ry)[0, 1] if m > 1 else 1.0
    return top, rho
R1 = pickle.load(open(HERE / 'res_C1.pkl', 'rb'))
for c in ('C1', 'C2'):
    b, p = COMPS[c]; cache = {}
    bF, bT = pool(MEMBERS[b], 'test'); pF, pT = pool(MEMBERS[p], 'test')
    plain_t = ref_final('test', bF, bT, pF, pT, 0.2, 0.2, 0.5)
    rules = {'MRULE (a .2 | .4 by M>=50)': (np.where(Mrow >= 50, 0.4, 0.2), None, 0.5),
             'M-rule .2 | .5': (np.where(Mrow >= 50, 0.5, 0.2), None, 0.5),
             'h3-TO (M-rule in the train-only branch only)': (0.2, np.where(Mrow >= 50, 0.4, 0.2), 0.5),
             'h4a (.2,.3,.4 by M 30/100)': (np.select([Mrow < 30, Mrow < 100], [0.2, 0.3], 0.4), None, 0.5),
             'F(Hq) best on C1 (a .4; c .7 if H<190 else .5)': (0.4, None, np.where(Hrow >= 190, 0.5, 0.7)),
             'CONST a .4 c .5': (0.4, None, 0.5), 'CONST a .4 c .7 (A best on C1)': (0.4, None, 0.7)}
    print(f'\n=== LABEL-FREE test ranking change vs the PLAIN file of the same members, {c} {COMPS[c]}: share of feeds with the same top-1 | mean per-feed Spearman   (all / public-like / private-like)')
    for tag, (a_ft, a_to, cc) in rules.items():
        s = ref_final('test', bF, bT, pF, pT, a_ft, a_ft if a_to is None else a_to, cc); top, rho = cmp_rank(ot, Mt, s.astype(np.float64), plain_t.astype(np.float64))
        print(f' {tag:48s}: same top-1 {top.mean():.4f} / {top[pub].mean():.4f} / {top[~pub].mean():.4f} | Spearman {rho.mean():.4f} / {rho[pub].mean():.4f} / {rho[~pub].mean():.4f} | feeds with any order change {1 - (rho > 1 - 1e-12).mean():.3f}')
