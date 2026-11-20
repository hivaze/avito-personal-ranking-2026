"""CHECK 1: reproduction with the reviewer's own path. Prints levels, kernel checks, the searchers' in-sample numbers. Builds and caches pools + tensors for all compositions."""
import sys, time
sys.path.insert(0, '<repo>/nn_x/w12/mix/review')
from rlib import *
t0 = time.time()
mv = meta('val'); y = labels(); g = groups(); M = mv['M']; big = M >= 50
print(f'val feeds {mv["n"]} rows {mv["rows"]} users(groups) {g.max() + 1} share M>=50 {big.mean():.4f} positives {int(y.sum())} feeds without positives {(np.bincount(np.repeat(np.arange(mv["n"]), M), y) == 0).sum()}')
# --- kernel check against the project functions (reference only)
sys.path.insert(0, '<repo>/src')
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
q = pools4('C1'); s_plain = final('val', q, 0.2); e_own = ndcg_feed(s_plain, y, mv['off'], M); e_ref = ndcg_per_feed(s_plain, y, mv['off'], M)
print(f'R0a kernel: own NDCG vs src/metrics on the same scores: max abs per-feed diff {np.abs(e_own - e_ref).max():.2e}; mean {e_own.mean():.6f}')
rng = np.random.default_rng(1); s_tie = np.round(rng.normal(size=mv['rows']), 1)
print(f'R0a kernel on heavily tied random scores: max abs diff {np.abs(ndcg_feed(s_tie, y, mv["off"], M) - ndcg_per_feed(s_tie, y, mv["off"], M)).max():.2e}')
# project path (float32, feed_rank_z) for C1 plain and M-rule
Zp = lambda x: feed_rank_z(mv['off'], M, np.asarray(x, np.float32))[:, 1]
def proj_pool(names):
    return (np.mean([Zp(np.load(P / f'nn_{m}_{SUF["val"][0]}.npy')) for m in names], 0), np.mean([Zp(np.load(P / f'nn_{m}_{SUF["val"][1]}.npy')) for m in names], 0))
b = proj_pool(MEM['FD16']); p = proj_pool(MEM['FDP'])
for nm, w in (('plain', np.float32(0.2)), ('M-rule', np.where(np.repeat(M, M) >= 50, 0.4, 0.2).astype(np.float32))):
    br = ((1 - w) * Zp(b[0]) + w * Zp(p[0]), (1 - w) * Zp(b[1]) + w * Zp(p[1])); sp_ = 0.5 * Zp(br[0]) + 0.5 * Zp(br[1])
    so = final('val', q, 0.2 if nm == 'plain' else np.where(big, 0.4, 0.2))
    print(f'R0a path C1 {nm}: project path {ndcg_per_feed(sp_, y, mv["off"], M).mean():.6f} | own path {ndcg_feed(so, y, mv["off"], M).mean():.6f}')
# --- levels
print('R0b levels (own path): key | plain | M-rule | rule minus plain (clustered SE) [per-feed SE] | M<50 | M>=50')
EV = {}
for key in COMPS:
    q = pools4(key); ep = ndcg_feed(final('val', q, 0.2), y, mv['off'], M); em = ndcg_feed(final('val', q, np.where(big, 0.4, 0.2)), y, mv['off'], M); d = em - ep; EV[key] = (ep, em)
    print(f'  {key} {COMPS[key]}: {ep.mean():.6f} | {em.mean():.6f} | {fm(d, g)} [{pse(d)[1]:.5f}] | {fm(d[~big], g[~big])} | {fm(d[big], g[big])}')
# --- tensors
for key in COMPS:
    t1 = time.time(); E = tensor(key, y)
    print(f'  tensor {key}: {E.shape} {time.time() - t1:.0f}s | E(0.2,0.5) {E[:, PLAIN].mean():.6f} (direct {EV[key][0].mean():.6f}, max abs per-feed diff {np.abs(E[:, PLAIN] - EV[key][0]).max():.1e})'
          f' | M-rule from tensor {np.where(big, E[:, C04], E[:, PLAIN]).mean():.6f} (direct {EV[key][1].mean():.6f})')
# --- searchers' in-sample numbers
print('R0c in-sample numbers of the searchers (own tensor)')
pr = dict(M=M, H=mv['H']); fams = fam_defs(pr)
for key in ('C1', 'C2', 'C3', 'C4'):
    E = tensor(key); D = E - E[:, [PLAIN]]; res = nested(D, fams, [], want_vec=False)
    line = f'  {key}: ' + ' | '.join(f'{nm} {res[nm]["ins"]:+.5f} {[dec(c) for c in res[nm]["ins_rule"][1]]}' + (f' cut#{res[nm]["ins_rule"][0]}' if len(dict(fams)[nm]) > 1 else '') for nm in res)
    print(line)
    dd = D[:, C04]; print(f'      dose contrast D = E(.4,.5) - E(.2,.5): all {fm(dd, g)} | M<50 {fm(dd[~big], g[~big])} | M>=50 {fm(dd[big], g[big])} | const (0.4, 0.7) {D[:, comb(0.4, 0.7)].mean():+.5f} const (0.4, 0.4) {D[:, comb(0.4, 0.4)].mean():+.5f}')
E1 = tensor('C1'); E3 = tensor('C3'); Dp = 0.5 * (E1 - E1[:, [PLAIN]]) + 0.5 * (E3 - E3[:, [PLAIN]]); dm = np.where(big, Dp[:, C04], 0.0)
print(f'  pooled (mean of C1 and C3) fixed M-rule minus plain {fm(dm, g)} (B: +0.00048 +- 0.00017)')
print(f'done {time.time() - t0:.0f}s')
