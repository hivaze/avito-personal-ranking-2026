"""ft_1115 step E: is an earlier-epoch (or epoch-averaged) fine-tune a different object from a shrunk final fine-tune?
The only per-epoch fine-tune PREDICTIONS of students on disk are the CPU pilot of 2026-11-14 (nn_x/w12/recheck/ft_redesign/out/ftcpu_<m>_A_base_oof.npy:
rows 0 = train-only, 1..4 = raw weights after each epoch, 5 = final EMA = what the recipe saves; 2 W8 members, folds 0-2 = 8,655 feeds, 1 fine-tune seed, fp32).
For each epoch: size and direction of the shift against the saved (EMA) shift, its cold-minus-warm offset, member NDCG, and the FILE PROXY
0.5 z(z(TO pool W8) + member shift) + 0.5 z(TO pool W8) minus the TO pool, under plain val / pmix-public / pmix-private weights; also the saved shift shrunk to
the epoch's size (same RMS), which separates 'earlier epoch' from 'smaller step'.
Output: out/e_epoch.txt"""
from lib import *
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
PO = R / 'nn_x/w12/recheck/ft_redesign/out'
members = ['w5_e10_kd05w192_s951', 'w5_e10_kd05t25w192_s951']
cold = (np.load(FEATS / 'val/it_hist_all_luo.npy') == 0)
ptv = mpool(MEM['W8'], 'trainonly', 'val').astype(np.float64)
done = np.all([np.load(PO / f'ftcpu_{m}_A_base_done.npy') for m in members], 0)
rows = np.repeat(done, Mv); Ms = Mv[done]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]); ys = yv[rows]; cs = cold[rows]
zs = lambda x: feed_rank_z(offs, Ms, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1].astype(np.float64)
evs = lambda s: ndcg_per_feed(np.asarray(s, np.float32), ys, offs, Ms)
wp, wq = WPUB[done], WPRIV[done]
def cw_offset(s):
    """mean over feeds (with both kinds) of mean shift on cold candidates minus mean shift on warm candidates"""
    nc = np.add.reduceat(cs.astype(np.float64), offs); sc = np.add.reduceat(s * cs, offs); sw = np.add.reduceat(s * ~cs, offs)
    ok = (nc > 0) & (nc < Ms); return float((sc[ok] / nc[ok] - sw[ok] / (Ms[ok] - nc[ok])).mean())
zpt = zs(ptv[rows]); e_pt = evs(zpt)
w_(f'feeds finished by both members: {int(done.sum())} of {len(done)}; TO pool W8 on them {e_pt.mean():.5f}; cold candidates {cs.mean():.3f}')
LAB = ['ep1', 'ep2', 'ep3', 'ep4', 'EMA (saved)', 'mean of ep1-4 (z)']
ACC = {l: [] for l in LAB}; ACCS = {l: [] for l in LAB}; ST = {l: [] for l in LAB}
for m in members:
    A = np.load(PO / f'ftcpu_{m}_A_base_oof.npy')[:, rows]
    zto = zs(A[0]); e_to = evs(A[0]); Z = [zs(A[e]) for e in range(1, 6)]; Z.append(np.mean(Z[:4], 0)); s_ema = Z[4] - zto
    for l, z in zip(LAB, Z):
        s = z - zto; k = float(s @ s_ema / (s_ema @ s_ema)); rms = np.sqrt((s ** 2).mean())
        prox = zs(zpt + s); efile = evs(0.5 * prox + 0.5 * zpt) - e_pt; epool = evs(prox) - e_pt
        lam = rms / np.sqrt((s_ema ** 2).mean()); proxs = zs(zpt + lam * s_ema); efs = evs(0.5 * proxs + 0.5 * zpt) - e_pt; eps = evs(proxs) - e_pt
        ACC[l].append((evs(z) - e_to, epool, efile)); ACCS[l].append((evs(zto + lam * s_ema) - e_to, eps, efs))
        ST[l].append((rms, k, float(np.corrcoef(s, s_ema)[0, 1]), cw_offset(s), cw_offset(s) / rms))
w_('epoch | RMS shift (z) | regression coefficient on the saved shift | corr with the saved shift | cold-minus-warm offset (z) | offset / RMS   (mean of the 2 members)')
for l in LAB:
    a = np.mean(ST[l], 0); w_(f'{l} | {a[0]:.4f} | {a[1]:.3f} | {a[2]:.3f} | {a[3]:+.4f} | {a[4]:+.3f}')
w_()
w_('mean of the 2 members, paired per feed (8,655 feeds; +- SE). "same-size shrink" = the saved (EMA) shift scaled to the RMS of that row.')
w_('epoch | member - TO | pool proxy - TO pool | FILE proxy - TO pool: plain val | pmix-public | pmix-private || same-size shrink of the saved shift: member - TO | FILE proxy plain | pmix-public | pmix-private || FILE proxy: epoch minus same-size shrink (plain / pmix-public / pmix-private)')
for l in LAB:
    a = [np.mean([x[j] for x in ACC[l]], 0) for j in range(3)]; b = [np.mean([x[j] for x in ACCS[l]], 0) for j in range(3)]; d = a[2] - b[2]
    w_(f'{l} | {f(a[0])} | {f(a[1])} | {f(a[2])} | {f(a[2], wp)} | {f(a[2], wq)} || {f(b[0])} | {f(b[2])} | {f(b[2], wp)} | {f(b[2], wq)} || {f(d)} / {f(d, wp)} / {f(d, wq)}')
ref = np.mean([x[2] for x in ACC['EMA (saved)']], 0)
w_('FILE proxy against the saved recipe (epoch minus EMA), paired:')
for l in LAB[:4] + LAB[5:]:
    d = np.mean([x[2] for x in ACC[l]], 0) - ref; dm = np.mean([x[0] for x in ACC[l]], 0) - np.mean([x[0] for x in ACC['EMA (saved)']], 0)
    w_(f'  {l}: member {f(dm)} | file proxy plain {f(d)} | pmix-public {f(d, wp)} | pmix-private {f(d, wq)}')
open(OUT / 'e_epoch.txt', 'w').write('\n'.join(out) + '\n')
