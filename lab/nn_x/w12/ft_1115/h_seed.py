"""ft_1115 step H: (1) fold assignment of the fine-tune; (2) the seed-averaging part: 1 fine-tune seed against 3 per fold, pool level, on the members that have both
(stored branches ens_G3 / ens_G3s3 = the 20 G3 members, ens_Q2a8pop1 / ens_Q2a8s3 = G3 + 8 popularity-light hard-label members); member level from the member files.
Output: out/h_seed.txt"""
from lib import *
from sklearn.model_selection import GroupKFold
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
g = user_groups(); gk = np.zeros(len(Mv), np.int64)
for k, (_, ho) in enumerate(GroupKFold(5).split(np.arange(len(Mv)), groups=g)): gk[ho] = k
w_('## Fold assignment (sklearn GroupKFold(5), no shuffle, groups = user_id with every user-0 feed as its own group; identical for every member; seeds 1000 k + {0, 1, 2})')
w_(f'val feeds {len(Mv)}, users {len(np.unique(uv))}, user-0 feeds {int((uv == 0).sum())}')
for k in range(5):
    m = gk == k; w_(f'fold {k}: feeds {int(m.sum())} | users {len(np.unique(g[m]))} | candidates {int(Mv[m].sum())} | feeds with M >= 50 {int((Mv[m] >= 50).sum())} | pmix-public weight share {WPUB[m].sum() / WPUB.sum():.3f}')
ut = ft_['user_id'].to_numpy(); w_(f'test feeds whose user has a val feed (user 0 excluded): {int((np.isin(ut, np.unique(uv[uv != 0])) & (ut != 0)).sum())} of {len(ut)}')
w_()
w_('## Seed averaging: 3 fine-tune seeds per fold against 1 (same members, same folds)')
for one, three, lab in (('G3', 'G3s3', 'G3 (20 hard-label members)'), ('Q2a8pop1', 'Q2a8s3', 'G3 + 8 popularity-light hard-label members (28)')):
    f1, t1 = stored(one, 'val'); f3, t3 = stored(three, 'val'); z1, z3, zo1, zo3 = zv(f1), zv(f3), zv(t1), zv(t3)
    dft = ev(z3) - ev(z1); dfi = ev(0.5 * z3 + 0.5 * zo3) - ev(0.5 * z1 + 0.5 * zo1)
    tf1, _ = stored(one, 'test'); tf3, _ = stored(three, 'test')
    w_(f'{lab}: FT pool alone 3 seeds - 1 seed {f(dft)} | file {f(dfi)} | pmix-public {f(dfi, WPUB)} | pmix-private {f(dfi, WPRIV)} | corr of the two FT pools: val {np.corrcoef(z1, z3)[0, 1]:.5f} test {np.corrcoef(zt(tf1), zt(tf3))[0, 1]:.5f}'
       f' | variance of the difference of the two FT pools (z units): val {((z1 - z3) ** 2).mean():.6f} test {((zt(tf1) - zt(tf3)) ** 2).mean():.6f}')
dm = []
for m in MEM['G3']:
    a = zv(np.load(P / f'nn_{m}_n2cpv3_val_oof.npy')); b = zv(np.load(P / f'nn_{m}_n2cpv3s3_val_oof.npy')); dm.append((ev(b).mean() - ev(a).mean(), ((a - b) ** 2).mean()))
dm = np.array(dm); w_(f'member level, 20 G3 members: 3 seeds - 1 seed {dm[:, 0].mean():+.5f}+-{dm[:, 0].std(ddof=1) / np.sqrt(len(dm)):.5f} (SE across members) | variance of z(1 seed) - z(3 seeds): {dm[:, 1].mean():.5f} -> seed-noise variance of one fine-tune about {dm[:, 1].mean() * 1.5:.5f} (z units; x 3/2 because the 3-seed mean contains the first seed)')
open(OUT / 'h_seed.txt', 'w').write('\n'.join(out) + '\n')
