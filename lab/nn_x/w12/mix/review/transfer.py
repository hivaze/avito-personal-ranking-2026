"""CHECK 4: transfer (label-free on test). T1 gate cells, T2 top-1 changes, T3 concentration of the val gain against the public / private mix, T4 pool geometry."""
import sys, time
sys.path.insert(0, '<repo>/nn_x/w12/mix/review')
from rlib import *
from scipy.stats import ks_2samp
t0 = time.time(); mv = meta('val'); mt = meta('test'); g = groups(); n = mv['n']
pub = mt['ts'] < BOUND; prv = ~pub
print(f'test feeds {mt["n"]}: public-like {int(pub.sum())}, private-like {int(prv.sum())}; val time {mv["ts"].min()}..{mv["ts"].max()}, test {mt["ts"].min()}..{mt["ts"].max()}')
PART = {'val': (mv, np.ones(n, bool)), 'public': (mt, pub), 'private': (mt, prv)}
print('\n=== T1: gate cells (share of feeds; KS distance of the variable inside the cell, val vs test part) ===')
for var, cut, lab in (('M', 50, 'M >= 50'), ('H', 190, 'H >= 190')):
    xv = mv[var]
    for cell, cn in ((0, f'NOT {lab}'), (1, lab)):
        line = f'  {cn}: share val {((xv >= cut) == cell).mean():.4f}'
        for pn in ('public', 'private'):
            m_, mk = PART[pn]; xt = m_[var][mk]; inc = (xt >= cut) == cell; ks = ks_2samp(xv[(xv >= cut) == cell], xt[inc]).statistic
            line += f' | {pn} {inc.mean():.4f} (KS {ks:.3f}; median {np.median(xt[inc]):.0f} vs val {np.median(xv[(xv >= cut) == cell]):.0f}; range {xt[inc].min()}..{xt[inc].max()} vs val {xv[(xv >= cut) == cell].min()}..{xv[(xv >= cut) == cell].max()})'
        print(line)
qs = [.1, .25, .5, .75, .9]
print('  M quantiles (10/25/50/75/90): val', np.quantile(mv['M'], qs), '| public', np.quantile(mt['M'][pub], qs), '| private', np.quantile(mt['M'][prv], qs))
print('  H quantiles: val', np.quantile(mv['H'], qs), '| public', np.quantile(mt['H'][pub], qs), '| private', np.quantile(mt['H'][prv], qs))
print('  user_id 0 share: val %.4f public %.4f private %.4f' % ((mv['uid'] == 0).mean(), (mt['uid'][pub] == 0).mean(), (mt['uid'][prv] == 0).mean()))
edges = [5, 11, 26, 50, 51, 101, 201, 301]
print('  M bins (5-10, 11-25, 26-49, 50, 51-100, 101-200, 201-300): val', np.round(np.histogram(mv['M'], edges)[0] / n, 4), '| public', np.round(np.histogram(mt['M'][pub], edges)[0] / pub.sum(), 4), '| private', np.round(np.histogram(mt['M'][prv], edges)[0] / prv.sum(), 4))
for lo, hi in ((40, 49), (50, 59), (45, 54)):
    print(f'  share of feeds with M in {lo}..{hi}: val {((mv["M"] >= lo) & (mv["M"] <= hi)).mean():.4f} public {((mt["M"][pub] >= lo) & (mt["M"][pub] <= hi)).mean():.4f} private {((mt["M"][prv] >= lo) & (mt["M"][prv] <= hi)).mean():.4f}')

print('\n=== T2 / T4: own-path scores of C1 (FD16 + 4 wide) and C2 (FD8 + 4), plain vs M-rule; NOT a submission, nothing written ===')


@nb.njit(parallel=True, cache=False)
def feed_stats(a, b, pB, pP, fB, fP, off, M):
    """per feed: top-1 differs (a vs b), Spearman-like Pearson of a and b ranks skipped -> Pearson(a, b); Pearson(base TO, pop TO); Pearson(base FT, pop FT)."""
    nf = len(M); out = np.zeros((nf, 5))
    for f in nb.prange(nf):
        o = off[f]; m = M[f]; ia = 0; ib = 0
        for k in range(m):
            if a[o + k] > a[o + ia]: ia = k
            if b[o + k] > b[o + ib]: ib = k
        out[f, 0] = 1.0 if ia != ib else 0.0
        # top-3 set differs
        ta = np.argsort(-a[o:o + m])[:3]; tb = np.argsort(-b[o:o + m])[:3]; same = 0
        for i in range(min(3, m)):
            for j in range(min(3, m)):
                if ta[i] == tb[j]: same += 1
        out[f, 1] = 1.0 if same < min(3, m) else 0.0
        for c, (x, yv) in enumerate(((a, b), (pB, pP), (fB, fP))):
            sx = 0.0; sy = 0.0; sxx = 0.0; syy = 0.0; sxy = 0.0
            for k in range(m):
                u = x[o + k]; v = yv[o + k]; sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v
            den = np.sqrt(max(sxx - sx * sx / m, 1e-18) * max(syy - sy * sy / m, 1e-18)); out[f, 2 + c] = (sxy - sx * sy / m) / den
    return out


for key in ('C1', 'C2'):
    ST = {}
    for sp in ('val', 'test'):
        m_ = meta(sp); q = pools4(key, sp); bigs = m_['M'] >= 50
        sa = final(sp, q, 0.2); sb = final(sp, q, np.where(bigs, 0.4, 0.2)); ST[sp] = (feed_stats(sa, sb, q[1], q[3], q[0], q[2], m_['off'], m_['M']), bigs)
        if sp == 'test' and key == 'C1':
            # sanity against the built file's stored branches (read-only): correlation of my own-path test score with the stored FD16P20 branches
            try:
                bf = np.load(P / 'ens_FD16P20kd_ft_test.npy'); bt = np.load(P / 'ens_FD16P20kd_to_test.npy'); ref = 0.5 * Z('test', bf) + 0.5 * Z('test', bt)
                print(f'  check: own-path C1 plain test score vs the stored FD16P20 branches: max abs diff {np.abs(ref - sa).max():.2e}, corr {np.corrcoef(ref, sa)[0, 1]:.8f}')
                bf = np.load(P / 'ens_FD16P20_m02_04kd_ft_test.npy'); bt = np.load(P / 'ens_FD16P20_m02_04kd_to_test.npy'); ref = 0.5 * Z('test', bf) + 0.5 * Z('test', bt)
                print(f'  check: own-path C1 M-rule test score vs the stored FD16P20_m02_04 branches: max abs diff {np.abs(ref - sb).max():.2e}')
            except Exception as e: print('  stored-branch check not possible:', e)
    sv, bv = ST['val']; st, bt_ = ST['test']
    print(f'  {key}: share of feeds whose top-1 differs between plain and M-rule: val all {sv[:, 0].mean():.4f} (long {sv[bv, 0].mean():.4f}, short {sv[~bv, 0].mean():.4f}) | public all {st[pub, 0].mean():.4f} (long {st[pub & bt_, 0].mean():.4f})'
          f' | private all {st[prv, 0].mean():.4f} (long {st[prv & bt_, 0].mean():.4f}) | test feeds changed top-1: {int(st[:, 0].sum())} of {mt["n"]} (public {int(st[pub, 0].sum())}, private {int(st[prv, 0].sum())})')
    print(f'      top-3 set differs on long feeds: val {sv[bv, 1].mean():.4f} public {st[pub & bt_, 1].mean():.4f} private {st[prv & bt_, 1].mean():.4f} | per-feed Pearson(plain, M-rule) on long feeds: val {sv[bv, 2].mean():.5f} public {st[pub & bt_, 2].mean():.5f} private {st[prv & bt_, 2].mean():.5f}')
    for c, nm in ((3, 'train-only branch'), (4, 'fine-tuned branch (val = out-of-fold, test = fold mean)')):
        print(f'      T4 per-feed Pearson(base pool, popularity-light pool), {nm}: short feeds val {sv[~bv, c].mean():.4f} public {st[pub & ~bt_, c].mean():.4f} private {st[prv & ~bt_, c].mean():.4f} | long feeds val {sv[bv, c].mean():.4f} public {st[pub & bt_, c].mean():.4f} private {st[prv & bt_, c].mean():.4f}')
    if key == 'C1': STC1 = ST

print('\n=== T3: where the fixed rule\'s val gain sits (long feeds only), against the val / public / private mix of the pmix cells ===')
sys.path.insert(0, '<repo>/src'); sys.path.insert(0, '<repo>/eda/errors/segments')
import pmix
W = pmix.weights(); csv = cold_pmix('val'); cst = cold_pmix('test'); cbin = lambda c: np.searchsorted([0.15, 0.30, 0.45], c, side='left'); mbin = lambda M_: np.searchsorted([10, 25, 50, 100, 200], M_, side='left')
CN = ['cold<15%', '15-30%', '30-45%', '>=45%']; MN = ['M5-10', 'M11-25', 'M26-50', 'M51-100', 'M101-200', 'M201-300']
DM = {k: np.where(mv['M'] >= 50, tensor(k)[:, C04] - tensor(k)[:, PLAIN], 0.0) for k in ('C1', 'C2', 'C3', 'C4')}
big = mv['M'] >= 50
print('  cold-share bins of LONG feeds (M >= 50): share of all feeds val / public / private | per-feed gain of the rule (C1, C2, C3, C4), +- clustered SE for C1')
for c in range(4):
    mk = big & (cbin(csv) == c); sh = (mk.mean(), ((mt['M'] >= 50) & (cbin(cst) == c))[pub].mean(), ((mt['M'] >= 50) & (cbin(cst) == c))[prv].mean())
    print(f'    {CN[c]:9s}: {sh[0]:.3f} / {sh[1]:.3f} / {sh[2]:.3f} | C1 {fm(DM["C1"][mk], g[mk])} | C2 {DM["C2"][mk].mean():+.5f} | C3 {DM["C3"][mk].mean():+.5f} | C4 {DM["C4"][mk].mean():+.5f} | contribution to the C1 total {DM["C1"][mk].sum() / n:+.5f}')
print('  M bins: share val / public / private | per-feed gain C1, C2, C3, C4')
for b in range(6):
    mk = mbin(mv['M']) == b; sh = (mk.mean(), (mbin(mt['M']) == b)[pub].mean(), (mbin(mt['M']) == b)[prv].mean())
    print(f'    {MN[b]:9s}: {sh[0]:.3f} / {sh[1]:.3f} / {sh[2]:.3f} | C1 {fm(DM["C1"][mk], g[mk])} | C2 {DM["C2"][mk].mean():+.5f} | C3 {DM["C3"][mk].mean():+.5f} | C4 {DM["C4"][mk].mean():+.5f}')
kv = mbin(mv['M']) * 4 + cbin(csv); kt = mbin(mt['M']) * 4 + cbin(cst)
for key in ('C1', 'C2', 'C3', 'C4'):
    d = DM[key]; tot_pos = 0.0; rare_pos = 0.0
    for c in np.unique(kv):
        mk = kv == c; contrib = d[mk].sum() / n; ratio = (kt[prv] == c).mean() / mk.mean()
        if contrib > 0:
            tot_pos += contrib
            if ratio < 0.5: rare_pos += contrib
    print(f'  {key}: val gain {d.mean():+.5f} | pmix public {wse(d, W["public"])[0]:+.5f}+-{wse(d, W["public"])[1]:.5f} | pmix private {wse(d, W["private"])[0]:+.5f}+-{wse(d, W["private"])[1]:.5f} | share of the positive cell contributions in cells with private/val share < 0.5: {rare_pos / tot_pos:.2f}'
          f' -> FLAG {"YES" if (rare_pos / tot_pos > 0.5 and wse(d, W["private"])[0] < 0.5 * d.mean()) else "no"}')
day = ((mv['ts'] - mv['ts'].min()) // 86400)
print('  by val day (days since the first val time stamp): ' + ' | '.join(f'day {int(dd)}: n {int((day == dd).sum())} C1 {fm(DM["C1"][day == dd], g[day == dd])} C2 {DM["C2"][day == dd].mean():+.5f} C3 {DM["C3"][day == dd].mean():+.5f} C4 {DM["C4"][day == dd].mean():+.5f}' for dd in np.unique(day)))
u0 = mv['uid'] == 0
print(f'  user 0 feeds (n {int(u0.sum())}): C1 {fm(DM["C1"][u0], g[u0])} C3 {DM["C3"][u0].mean():+.5f} | others C1 {fm(DM["C1"][~u0], g[~u0])} C3 {DM["C3"][~u0].mean():+.5f}')
# does the gain live in feeds where the rule changes the top of the list?
sv, bv = STC1['val']; ch = sv[:, 0] > 0
print(f'  C1: val feeds whose top-1 changes: {int(ch.sum())} ({ch.mean():.4f}); gain inside them {fm(DM["C1"][ch], g[ch])}, contribution {DM["C1"][ch].sum() / n:+.5f} of {DM["C1"].mean():+.5f}')
print(f'done {time.time() - t0:.0f}s')
