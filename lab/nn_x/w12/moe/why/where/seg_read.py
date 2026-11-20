"""why/where W1 + W3: where do the arms' val loss and holdout gain sit? CPU, read-only. Output out/seg_read.txt
Per-feed contrasts (paired per feed; +- = 1 SE over feeds of the segment):
  arm = mean over A, B, C of NDCG(arm 4-pool) - NDCG(reference)   [the three arms share the reference; per-arm columns too]
  reference = refN (per-feed mean over the 70 4-subsets of kd05w192 s951-958) [main], ref4 (s951-954, the screen's) [secondary]
  null = ref4b (s955-958) - ref4 : a same-recipe seed swap; its profile shows where reference seed luck sits.
Segments from out/seg_{val,hold}.npz (seg_build.py)."""
from common_where import *
m = Meta(); P = perfeed(m)
sv = dict(np.load(HERE / 'out/seg_val.npz')); sh = dict(np.load(HERE / 'out/seg_hold.npz'))
assert len(sv['M']) == len(m.Mv) and (sv['M'] == m.Mv).all() and (sh['M'] == m.Mh).all()
D = {}
for sp in ('v', 'h'):
    for ref in ('refN', 'ref4'):
        for k in ARMS: D[(sp, ref, k)] = P[f'{k}_{sp}'] - P[f'{ref}_{sp}']
        D[(sp, ref, 'arm')] = np.mean([D[(sp, ref, k)] for k in ARMS], 0)
    D[(sp, 'null')] = P[f'ref4b_{sp}'] - P[f'ref4_{sp}']
L = []
def row(lbl, selv, selh):
    def c(sp, sel):
        if sel is None or sel.sum() < 30: return f'{"-":>40s}'
        a = D[(sp, 'refN', 'arm')][sel]; r = D[(sp, 'ref4', 'arm')][sel]; nl = D[(sp, 'null')][sel]
        return f'n {int(sel.sum()):6d} arm-refN {fmt(a):>18s} arm-ref4 {fmt(r):>18s} null {fmt(nl):>18s}'
    L.append(f'  {lbl:22s}| VAL {c("v", selv)} || HOLD {c("h", selh)}')
def per_arm(lbl, selv, selh):
    s = '  '.join(f'{k}: val {fmt(D[("v", "refN", k)][selv])} hold {fmt(D[("h", "refN", k)][selh]) if selh is not None and selh.sum() >= 30 else "-"}' for k in ARMS)
    L.append(f'      per arm vs refN [{lbl}] {s}')
L.append('=== W1 segment profile (arm = mean of A,B,C 4-pool deltas; refN main reference; null = ref4b - ref4) ===')
allv = np.ones(len(m.Mv), bool); allh = np.ones(len(m.Mh), bool)
row('ALL', allv, allh); per_arm('ALL', allv, allh)
def seg(title, fv, fh, bins):
    L.append(f'-- {title}')
    for lbl, f_ in bins:
        selv = f_(sv) if fv else None; selh = f_(sh) if fh else None
        row(lbl, selv, selh)
seg('feed size M', 1, 1, [('M<20', lambda s: s['M'] < 20), ('20<=M<50', lambda s: (s['M'] >= 20) & (s['M'] < 50)),
                          ('50<=M<100', lambda s: (s['M'] >= 50) & (s['M'] < 100)), ('M>=100', lambda s: s['M'] >= 100),
                          ('M>=50 (all)', lambda s: s['M'] >= 50)])
seg('history length H', 1, 1, [('H<64', lambda s: s['H'] < 64), ('64<=H<256', lambda s: (s['H'] >= 64) & (s['H'] < 256)), ('H>=256 (full)', lambda s: s['H'] >= 256)])
seg('user', 1, 1, [('user 0 (anon)', lambda s: s['user0']), ('user, 0 train feeds', lambda s: s['user_ntr'] == 0),
                   ('user, 1-2 train feeds', lambda s: (s['user_ntr'] >= 1) & (s['user_ntr'] <= 2)), ('user, >=3 train feeds', lambda s: s['user_ntr'] >= 3)])
seg('positive items seen in training candidates (any day; model view)', 1, 1,
    [('all positives seen', lambda s: s['pos_seen'] >= 1), ('some seen', lambda s: (s['pos_seen'] > 0) & (s['pos_seen'] < 1)), ('no positive seen', lambda s: s['pos_seen'] <= 0)])
seg('positive items seen in train before the feed day (causal)', 1, 1,
    [('all seen before', lambda s: s['pos_seen_c'] >= 1), ('some', lambda s: (s['pos_seen_c'] > 0) & (s['pos_seen_c'] < 1)), ('none before', lambda s: s['pos_seen_c'] <= 0)])
seg('popularity input of the positives (it_hist_all_luo == 0 share, as fed)', 1, 1,
    [('all positives pop>0', lambda s: s['pos_pop0'] <= 0), ('some pop=0', lambda s: (s['pos_pop0'] > 0) & (s['pos_pop0'] < 1)), ('all positives pop=0', lambda s: s['pos_pop0'] >= 1)])
seg('candidate warm share (seen in training)', 1, 1, [('<0.6', lambda s: s['cand_seen'] < 0.6), ('0.6-0.85', lambda s: (s['cand_seen'] >= 0.6) & (s['cand_seen'] < 0.85)), ('>=0.85', lambda s: s['cand_seen'] >= 0.85)])
seg('listing age of the positives (mean, days; capped f4 = old)', 1, 1,
    [('<1d', lambda s: s['pos_age_d'] < 1), ('1-7d', lambda s: (s['pos_age_d'] >= 1) & (s['pos_age_d'] < 7)), ('7-30d', lambda s: (s['pos_age_d'] >= 7) & (s['pos_age_d'] < 30)), ('>=30d', lambda s: s['pos_age_d'] >= 30)])
seg('median candidate age (days)', 1, 1, [('<7d', lambda s: s['cand_age_med'] < 7), ('7-30d', lambda s: (s['cand_age_med'] >= 7) & (s['cand_age_med'] < 30)), ('>=30d', lambda s: s['cand_age_med'] >= 30)])
seg('number of positives', 1, 1, [('1', lambda s: s['npos'] == 1), ('2', lambda s: s['npos'] == 2), ('>=3', lambda s: s['npos'] >= 3)])
top = [c for c, _ in sorted(zip(*np.unique(sv['cat_mode'], return_counts=True)), key=lambda t: -t[1])[:7]]
seg('dominant candidate category (top 7 in val)', 1, 1, [(f'cat {c}', (lambda c_: lambda s: s['cat_mode'] == c_)(c)) for c in top] +
    [('other', lambda s: ~np.isin(s['cat_mode'], top))])
seg('day', 1, 1, [('val Sep 22', lambda s: s['day'] <= 20718), ('val Sep 24', lambda s: s['day'] >= 20720)] if False else [])
L.append('-- day (val: Sep 22 / Sep 24; holdout: Sep 6-10 / Sep 11-14 / Sep 17-19)')
for lbl, sel in (('val Sep 22', sv['day'] <= 20718), ('val Sep 24', sv['day'] >= 20720)): row(lbl, sel, None)
for lbl, sel in (('hold Sep 6-10', sh['day'] <= 20706), ('hold Sep 11-14', (sh['day'] > 20706) & (sh['day'] <= 20710)), ('hold Sep 17-19', sh['day'] >= 20713)): row(lbl, None, sel)
# heterogeneity: split z for key binary splits (val), arm vs refN
L.append('-- split tests (arm - refN, VAL): difference between two bins, z = d / sqrt(se1^2 + se2^2)')
def ztest(lbl, a, b, sp='v', ref='refN'):
    x, y = D[(sp, ref, 'arm')][a], D[(sp, ref, 'arm')][b]; d = x.mean() - y.mean(); s = np.sqrt(se(x) ** 2 + se(y) ** 2)
    nl = D[(sp, 'null')]; dn = nl[a].mean() - nl[b].mean(); sn = np.sqrt(se(nl[a]) ** 2 + se(nl[b]) ** 2)
    L.append(f'  {lbl:46s} arm {d:+.5f} +- {s:.5f} (z {d / s:+.2f}) | null {dn:+.5f} +- {sn:.5f} (z {dn / sn:+.2f})')
ztest('val: M<50 minus M>=50', sv['M'] < 50, sv['M'] >= 50)
ztest('val: no positive seen minus all seen', sv['pos_seen'] <= 0, sv['pos_seen'] >= 1)
ztest('val: any positive pop=0 minus all pop>0', sv['pos_pop0'] > 0, sv['pos_pop0'] <= 0)
ztest('val: Sep 24 minus Sep 22', sv['day'] >= 20720, sv['day'] <= 20718)
ztest('hold: no positive seen minus all seen', sh['pos_seen'] <= 0, sh['pos_seen'] >= 1, 'h')
ztest('hold: any positive pop=0 minus all pop>0', sh['pos_pop0'] > 0, sh['pos_pop0'] <= 0, 'h')
ztest('hold: Sep 17-19 minus Sep 6-10', sh['day'] >= 20713, sh['day'] <= 20706, 'h')
# covariate-shift reweighting: holdout reweighted to the val M>=50 cell mix (M bin x pos_seen class x pos_pop0>0)
L.append('-- covariate reweighting: holdout feeds reweighted to the cell mix of val M>=50 feeds; cells = M bin (50-100, >=100) x positives-seen class (all/some/none) x any positive pop=0')
def cells(s):
    a = (s['M'] >= 100).astype(int); b = np.where(s['pos_seen'] >= 1, 0, np.where(s['pos_seen'] > 0, 1, 2)); c = (s['pos_pop0'] > 0).astype(int)
    return a * 6 + b * 2 + c
cv, ch = cells(sv), cells(sh); big = sv['M'] >= 50
pv = np.bincount(cv[big], minlength=12) / big.sum(); ph = np.bincount(ch, minlength=12) / len(ch)
w = (pv / np.maximum(ph, 1e-12))[ch]; w = w / w.mean()
for key in (('refN', 'arm'), ('ref4', 'arm'), ('refN', 'A'), ('refN', 'B'), ('refN', 'C')):
    d = D[('h', *key)]; dw = d * w
    L.append(f'  {key[1]}-{key[0]}: holdout raw {fmt(d)} | reweighted {dw.mean():+.5f} +- {se(dw):.5f} | val M>=50 {fmt(D[("v", *key)][big])}')
dn = D[('h', 'null')] * w
L.append(f'  null ref4b-ref4: holdout raw {fmt(D[("h", "null")])} | reweighted {dn.mean():+.5f} +- {se(dn):.5f} | val M>=50 {fmt(D[("v", "null")][big])}')
L.append(f'  cell shares val(M>=50) / holdout: ' + ' '.join(f'{i}:{pv[i]:.3f}/{ph[i]:.3f}' for i in range(12)))
# ---------------------------------------------------------------- W3: concentration
L.append('')
L.append('=== W3 distribution of per-feed deltas (VAL, all 14,423 feeds unless stated) ===')
def dist(lbl, d):
    n = len(d); nz = d != 0; o = np.sort(d); k = max(1, n // 100)
    trim = o[k:-k]; neg1 = o[:k].sum() / n; pos1 = o[-k:].sum() / n
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    sgn_z = (wins - losses) / np.sqrt(wins + losses)
    L.append(f'  {lbl:22s} mean {d.mean():+.5f} +- {se(d):.5f} | nonzero {nz.mean():.3f} | win/loss {wins}/{losses} (sign z {sgn_z:+.2f}) | 1%-trimmed mean {trim.mean():+.5f} +- {trim.std(ddof=1) / np.sqrt(len(trim)):.5f} '
             f'| bottom-1% feeds contribute {neg1:+.5f}, top-1% {pos1:+.5f} to the mean | sd {d.std():.4f} | q01 {np.quantile(d, .01):+.4f} q99 {np.quantile(d, .99):+.4f}')
for k in ARMS: dist(f'{k} - ref4', D[('v', 'ref4', k)])
for k in ARMS: dist(f'{k} - refN', D[('v', 'refN', k)])
dist('mean(A,B,C) - refN', D[('v', 'refN', 'arm')])
dist('null ref4b - ref4', D[('v', 'null')])
L.append('  holdout, for comparison:')
for k in ARMS: dist(f'{k} - refN (hold)', D[('h', 'refN', k)])
dist('null ref4b-ref4 (hold)', D[('h', 'null')])
# where does the mean come from: deltas binned by |delta|
L.append('-- contribution to the val mean by size of the per-feed delta (arm mean of A,B,C vs refN): bins of |d|')
d = D[('v', 'refN', 'arm')]; ad = np.abs(d)
for lo, hi in ((0, 1e-9), (1e-9, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 9)):
    s = (ad >= lo) & (ad < hi); L.append(f'  |d| in [{lo:g},{hi:g}): n {int(s.sum()):6d} share {s.mean():.3f} contributes {d[s].sum() / len(d):+.5f} to the mean')
# correlation of the three arms' per-feed val deltas (do they lose on the same feeds?)
L.append('-- per-feed correlation of arm deltas vs ref4 (shared reference inflates it) and vs refN; and with the null (ref4b - ref4):')
for ref in ('ref4', 'refN'):
    X = np.array([D[('v', ref, k)] for k in ARMS]); C = np.corrcoef(X)
    L.append(f'  val vs {ref}: corr AB {C[0, 1]:+.3f} AC {C[0, 2]:+.3f} BC {C[1, 2]:+.3f}')
X = np.array([D[('v', 'ref4', k)] for k in ARMS] + [D[('v', 'null')]]); C = np.corrcoef(X)
L.append(f'  val: corr(arm - ref4, ref4b - ref4): A {C[0, 3]:+.3f} B {C[1, 3]:+.3f} C {C[2, 3]:+.3f}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/seg_read.txt', 'w').write(txt + '\n')
