"""G. Read-out of the CPU fine-tune pilot (ft_cpu.py). Feeds = held-out folds finished by ALL arms of a member.
 rows: arm x {epoch 1..4 raw weights, final EMA (= what the recipe saves)}; columns: member NDCG - TO (paired), arm - baseline arm (paired),
 pool proxy = z(TO pool W8) + (z(arm) - z(member TO)) minus TO pool, and the file proxy 0.5 z(proxy FT pool) + 0.5 z(TO pool) minus TO pool."""
from lib import *
import glob, re
members = ['w5_e10_kd05w192_s951', 'w5_e10_kd05t25w192_s951']
def sub(mask):
    """feed mask -> (cand rows, M, offsets) for metric on a feed subset"""
    rows = np.repeat(mask, Mv); Ms = Mv[mask]; return rows, Ms, np.concatenate([[0], np.cumsum(Ms)[:-1]])
def evs(s, rows, Ms, offs): return ndcg_per_feed(np.asarray(s, np.float64)[rows], yv[rows], offs, Ms)
def evsub(s, rows, Ms, offs): return ndcg_per_feed(np.asarray(s, np.float64), yv[rows], offs, Ms)
def zs(x, rows, Ms, offs): return feed_rank_z(offs, Ms, np.ascontiguousarray(np.asarray(x, np.float32)[rows]))[:, 1].astype(np.float64)
ptv = pool(W8, 'to', 'val'); ACC = {}
for m in members:
    arms = {}
    for fn in sorted(glob.glob(str(OUT / f'ftcpu_{m}_*_oof.npy'))):
        tag = re.search(rf'ftcpu_{m}_(.*)_oof.npy', fn).group(1); arms[tag] = (np.load(fn), np.load(fn.replace('_oof.npy', '_done.npy')))
    if 'A_base' not in arms: continue
    done = np.all([d for _, d in arms.values()], 0); rows, Ms, offs = sub(done)
    print(f'\n=== {m}: arms {list(arms)}; feeds done by all arms {done.sum()} of {len(done)}')
    A = arms['A_base'][0]; E = A.shape[0] - 2
    e_to = evs(A[0], rows, Ms, offs); g = raw(m, 'ft', 'val'); t = raw(m, 'to', 'val')
    print(f' sanity: CPU epoch-0 vs saved TO preds corr {np.corrcoef(A[0][rows], t[rows])[0,1]:.6f} (NDCG {e_to.mean():.5f} vs {evs(t, rows, Ms, offs).mean():.5f}); CPU baseline EMA vs GPU n2cpv3s3 OOF: per-feed-z corr {np.corrcoef(zs(A[E+1], rows, Ms, offs), zs(g, rows, Ms, offs))[0,1]:.5f}, NDCG {evs(A[E+1], rows, Ms, offs).mean():.5f} vs GPU {evs(g, rows, Ms, offs).mean():.5f} (paired {f(evs(A[E+1], rows, Ms, offs) - evs(g, rows, Ms, offs))})')
    zpt = zs(ptv, rows, Ms, offs); e_pt = evsub(zpt, rows, Ms, offs); zto = zs(A[0], rows, Ms, offs)
    for tag, (O, _) in arms.items():
        for e in list(range(1, E + 1)) + [E + 1]:
            ee = evs(O[e], rows, Ms, offs); za = zs(O[e], rows, Ms, offs); prox = zpt + (za - zto)
            ep = evsub(prox, rows, Ms, offs); zp = feed_rank_z(offs, Ms, prox.astype(np.float32))[:, 1]; efile = evsub(0.5 * zp + 0.5 * zpt, rows, Ms, offs)
            eb = evs(arms['A_base'][0][E + 1], rows, Ms, offs)
            # de-projected proxy: remove the member->pool component alpha * (z_pool - z_member) that a soft-target arm adds by construction
            dvec = zpt - zto; sh = za - zto; alpha = float(sh @ dvec / (dvec @ dvec)); rest = sh - alpha * dvec
            zp2 = feed_rank_z(offs, Ms, (zpt + rest).astype(np.float32))[:, 1]; efile2 = evsub(0.5 * zp2 + 0.5 * zpt, rows, Ms, offs); ep2 = evsub(zpt + rest, rows, Ms, offs)
            ACC[(m, tag, e)] = (ee - e_to, ep - e_pt, efile - e_pt, done, np.sqrt(((za - zto) ** 2).mean()), ep2 - e_pt, efile2 - e_pt, alpha)
            print(f'  {tag:10s} {"ep" + str(e) if e <= E else "EMA":4s} member-TO {f(ee - e_to)} | vs base-EMA {f(ee - eb)} | RMS shift {np.sqrt(((za - zto) ** 2).mean()):.4f} | pool proxy-TOpool {f(ep - e_pt)} | file proxy-TOpool {f(efile - e_pt)}')

print('\n=== mean over the two members, feeds done by all arms of BOTH members (final EMA; paired per feed)')
ms2 = [m for m in members if (m, 'A_base', 5) in ACC and (m, 'B_kd05', 5) in ACC]
if len(ms2) == 2:
    dn = ACC[(ms2[0], 'A_base', 5)][3] & ACC[(ms2[1], 'A_base', 5)][3]
    def full(m, tag, e, j):
        x = np.full(len(Mv), np.nan); x[ACC[(m, tag, e)][3]] = ACC[(m, tag, e)][j]; return x[dn]
    for e in (1, 2, 3, 4, 5):
        lab = 'EMA' if e == 5 else f'ep{e}'
        print(f'  {lab:4s} alpha (member->pool share of the shift): A {np.mean([ACC[(m, "A_base", e)][7] for m in ms2]):+.3f} B {np.mean([ACC[(m, "B_kd05", e)][7] for m in ms2]):+.3f}')
        for j, nm in ((0, 'member - TO'), (1, 'pool proxy - TO pool'), (2, 'file proxy - TO pool'), (5, 'deproj pool proxy - TO'), (6, 'deproj file proxy - TO')):
            a = np.mean([full(m, 'A_base', e, j) for m in ms2], 0); b = np.mean([full(m, 'B_kd05', e, j) for m in ms2], 0)
            print(f'  {lab:4s} {nm:22s} A_base {f(a)} | B_kd05 {f(b)} | B - A {f(b - a)}   (feeds {dn.sum()})')

m = members[0]
if (m, 'C_lrhalf', 5) in ACC:
    print(f'\n=== {m}: C_lrhalf vs A_base (paired, feeds done by all arms of this member)')
    for e in (1, 2, 3, 4, 5):
        lab = 'EMA' if e == 5 else f'ep{e}'
        for j, nm in ((0, 'member - TO'), (2, 'file proxy - TO pool'), (6, 'deproj file proxy - TO')):
            a, c, b = ACC[(m, 'A_base', e)][j], ACC[(m, 'C_lrhalf', e)][j], ACC[(m, 'B_kd05', e)][j]
            print(f'  {lab:4s} {nm:22s} A {f(a)} | C {f(c)} | C - A {f(c - a)} | B - C {f(b - c)}')
for m in members:
    if (m, 'B_kd05', 5) in ACC:
        for j, nm in ((0, 'member - TO'), (2, 'file proxy - TO pool'), (6, 'deproj file proxy')):
            print(f'  {m} EMA {nm}: B - A {f(ACC[(m, "B_kd05", 5)][j] - ACC[(m, "A_base", 5)][j])}')
