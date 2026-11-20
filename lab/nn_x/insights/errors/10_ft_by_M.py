"""Per-member FT effect (final 0.5z(ft)+0.5z(to) minus to-only, and ft-only minus to-only) by M<50 / M>=50 and cold/warm positives."""
from lib import *
V = Val('val'); M = V.M
T = pl.read_parquet(OUT / 'out/feed_table.parquet'); keep = ~T['leak'].to_numpy()
cp, wp = T['coldpos'].to_numpy(), T['warmpos'].to_numpy()
D = np.load(OUT / 'out/member_ndcg.npz'); mem = list(D['members'])
rows = []
for m in mem:
    ft = load_member(V, m, 'ft'); to = load_member(V, m, 'to')
    a = V.ndcg(0.5 * ft + 0.5 * to) - V.ndcg(to); b = V.ndcg(ft) - V.ndcg(to)
    r = [a[keep].mean(), a[keep & (M < 50)].mean(), a[keep & (M >= 50)].mean(), a[keep & cp].mean(), a[keep & wp].mean(), b[keep & (M < 50)].mean(), b[keep & (M >= 50)].mean()]
    rows.append(r); print(f'{m:24s} final-to all {r[0]:+.5f} M<50 {r[1]:+.5f} M>=50 {r[2]:+.5f} coldpos {r[3]:+.5f} warmpos {r[4]:+.5f} | ft-to M<50 {r[5]:+.5f} M>=50 {r[6]:+.5f}')
R = np.array(rows)
for nm, ix in [('G3 members', slice(0, int(D['G3n']))), ('NEW members', slice(int(D['G3n']), None)), ('all', slice(None))]:
    X = R[ix]
    print(f'{nm:12s} mean over {len(X)}: ' + ' '.join(f'{c} {X[:, j].mean():+.5f} (members > 0: {np.mean(X[:, j] > 0):.2f})' for j, c in enumerate(['all', 'M<50', 'M>=50', 'coldpos', 'warmpos', 'ftonly-to M<50', 'ftonly-to M>=50'])))
