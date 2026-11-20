"""why/where W1: formal sign-flip table. For every segment bin defined on both splits, z_flip = (val delta - holdout delta) / sqrt(se_v^2 + se_h^2),
arm = mean of A,B,C 4-pools minus refN. Also: val cross-tab M<50 x positive coldness, and the val delta predicted from holdout per-class
deltas (class = positives seen all / some / none) at the val class mix. CPU, read-only. Output out/flip_table.txt"""
from common_where import *
m = Meta(); P = perfeed(m)
sv = dict(np.load(HERE / 'out/seg_val.npz')); sh = dict(np.load(HERE / 'out/seg_hold.npz'))
dv = np.mean([P[f'{k}_v'] for k in ARMS], 0) - P['refN_v']; dh = np.mean([P[f'{k}_h'] for k in ARMS], 0) - P['refN_h']
top = [c for c, _ in sorted(zip(*np.unique(sv['cat_mode'], return_counts=True)), key=lambda t: -t[1])[:7]]
B = [('M 50-100', lambda s: (s['M'] >= 50) & (s['M'] < 100)), ('M>=100', lambda s: s['M'] >= 100),
     ('H<64', lambda s: s['H'] < 64), ('H 64-255', lambda s: (s['H'] >= 64) & (s['H'] < 256)), ('H 256', lambda s: s['H'] >= 256),
     ('user0', lambda s: s['user0']), ('user 0 trainfeeds', lambda s: s['user_ntr'] == 0), ('user 1-2', lambda s: (s['user_ntr'] >= 1) & (s['user_ntr'] <= 2)), ('user >=3', lambda s: s['user_ntr'] >= 3),
     ('pos all seen', lambda s: s['pos_seen'] >= 1), ('pos some seen', lambda s: (s['pos_seen'] > 0) & (s['pos_seen'] < 1)), ('pos none seen', lambda s: s['pos_seen'] <= 0),
     ('pos all pop>0', lambda s: s['pos_pop0'] <= 0), ('pos some pop=0', lambda s: (s['pos_pop0'] > 0) & (s['pos_pop0'] < 1)), ('pos all pop=0', lambda s: s['pos_pop0'] >= 1),
     ('pos age<1d', lambda s: s['pos_age_d'] < 1), ('pos age 1-7d', lambda s: (s['pos_age_d'] >= 1) & (s['pos_age_d'] < 7)), ('pos age 7-30d', lambda s: (s['pos_age_d'] >= 7) & (s['pos_age_d'] < 30)), ('pos age>=30d', lambda s: s['pos_age_d'] >= 30),
     ('cand age med<7d', lambda s: s['cand_age_med'] < 7), ('cand age 7-30d', lambda s: (s['cand_age_med'] >= 7) & (s['cand_age_med'] < 30)), ('cand age>=30d', lambda s: s['cand_age_med'] >= 30),
     ('npos 1', lambda s: s['npos'] == 1), ('npos 2', lambda s: s['npos'] == 2), ('npos>=3', lambda s: s['npos'] >= 3)] + \
    [(f'cat {c}', (lambda c_: lambda s: s['cat_mode'] == c_)(c)) for c in top] + [('cat other', lambda s: ~np.isin(s['cat_mode'], top))]
L = ['=== flip table: arm (mean A,B,C) - refN; val restricted to M>=50 where the bin is not about M (holdout has M>=50 only) ===']
L.append(f'{"segment":20s} {"val n":>6s} {"val delta":>20s} {"hold n":>7s} {"hold delta":>20s}  z_flip  sign-flip')
zs = []
big = sv['M'] >= 50
for lbl, f in B:
    a = f(sv) & (big if not lbl.startswith('M') else True); b = f(sh)
    if a.sum() < 50 or b.sum() < 50: continue
    x, y = dv[a], dh[b]; z = (x.mean() - y.mean()) / np.sqrt(se(x) ** 2 + se(y) ** 2); zs.append(z)
    L.append(f'{lbl:20s} {int(a.sum()):6d} {fmt(x):>20s} {int(b.sum()):7d} {fmt(y):>20s}  {z:+5.2f}   {"yes" if np.sign(x.mean()) != np.sign(y.mean()) else ""}')
zs = np.array(zs)
L.append(f'{len(zs)} bins: |z_flip| >= 2 in {int((np.abs(zs) >= 2).sum())} (expected by chance ~{0.046 * len(zs):.1f}); mean z_flip {zs.mean():+.2f} (bins overlap; the common shift is the val-vs-holdout level difference)')
L.append('-- val cross-tab: M<50 / M>=50 x positives seen class (arm - refN)')
for ml, ms in (('M<50', sv['M'] < 50), ('M>=50', sv['M'] >= 50)):
    for cl, cs in (('all seen', sv['pos_seen'] >= 1), ('some', (sv['pos_seen'] > 0) & (sv['pos_seen'] < 1)), ('none seen', sv['pos_seen'] <= 0)):
        s = ms & cs; L.append(f'  {ml:6s} {cl:9s} n {int(s.sum()):5d} share {s.mean():.3f}  {fmt(dv[s])}')
L.append('-- val delta predicted from holdout per-class deltas at the val class mix (class = positives seen all / some / none):')
cls = lambda s: np.where(s['pos_seen'] >= 1, 0, np.where(s['pos_seen'] > 0, 1, 2))
cv, ch = cls(sv), cls(sh)
for sub, sel in (('all val', np.ones(len(dv), bool)), ('val M>=50', big)):
    pv = np.bincount(cv[sel], minlength=3) / sel.sum(); hd = np.array([dh[ch == i].mean() for i in range(3)]); hs = np.array([se(dh[ch == i]) for i in range(3)])
    pred = (pv * hd).sum(); pse = np.sqrt(((pv * hs) ** 2).sum())
    L.append(f'  {sub}: class mix val {np.round(pv, 3)} vs holdout {np.round(np.bincount(ch, minlength=3) / len(ch), 3)} | predicted {pred:+.5f} +- {pse:.5f} | observed {fmt(dv[sel])} | holdout raw {fmt(dh)}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/flip_table.txt', 'w').write(txt + '\n')
