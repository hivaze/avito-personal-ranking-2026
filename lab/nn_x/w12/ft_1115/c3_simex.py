"""ft_1115 step C3 (replaces the control-variate block of c_foldavg.py, which is confounded: z(single) - z(bag) contains a shrink of the common shift,
corr 0.45 with the student shift, so 'F - beta N' is mostly a lower w_ft, not a denoising).
REAL-NOISE SIMEX: the six fields n = z(one fine-tune on 80%) - z(bag of 5 inner fold models) of the 3 nested members x 2 seeds are real fine-tune instability fields on
val rows (label-free for the scored feed). Each is made orthogonal to its member's bag shift and to the pool's own FT-minus-TO shift, scaled so that the added variance
is lambda x sigma_e^2 (sigma_e^2 = fold-noise variance of one fold-branch on test rows, c_foldavg C1) and added with both signs (the first-order term cancels).
Loss against lambda -> slope -> what removing 4/5 of the fold noise (test: mean of 5 folds) is worth, for the FT branch alone and for the file at each w_ft.
Output: out/c3_simex.txt, out/c3_simex.npz"""
from lib import *
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
NM = ['v15_cd_all_e10', 'v14_s93', 'v17_pre2s_s102']; NP = R / 'nn_x/solvers/preds/refit'
fields = []
for m in NM:
    z = np.load(NP / f'nested_{m}.npz', allow_pickle=True)
    zb = zv(z['P_i']).astype(np.float64); zto = zv(np.load(P / f'nn_{m}_trainonly_val.npy')).astype(np.float64); sb = zb - zto
    for nm, arr in (('refit', z['P_ii']), ('oof', np.load(P / f'nn_{m}_n2cpv3_val_oof.npy'))):
        n = zv(arr).astype(np.float64) - zb; g = float(n @ sb / (sb @ sb)); n = n - g * sb
        fields.append((f'{m}:{nm}', n, g))
w_('noise fields (z(single 80%) - z(bag), minus its projection on the bag shift): ' + ' | '.join(f'{k} RMS {np.sqrt((n ** 2).mean()):.4f} (shrink coefficient removed {g:+.3f})' for k, n, g in fields))
C = np.corrcoef(np.array([n for _, n, _ in fields])); w_('pairwise correlation of the six fields: same member, two seeds ' + ' '.join(f'{C[2 * i, 2 * i + 1]:.2f}' for i in range(3)) + f' | different members (mean) {np.mean([C[i, j] for i in range(6) for j in range(i) if i // 2 != j // 2]):.2f}')
SIG = {'W8P20': 0.00247, 'FD16P20': 0.00257, 'W8': 0.00268}       # c_foldavg C1 (out/c_foldavg.txt)
GRID = [0.25, 0.5, 0.65, 0.75, 1.0]; LAM = [1.0, 2.0, 4.0]; RES = {}
for name, tag in (('W8P20', 'W8P20kd'), ('FD16P20', 'FD16P20kd'), ('W8', 'W8kd')):
    b_ft, b_to = stored(tag, 'val'); F, T = zv(b_ft).astype(np.float64), zv(b_to).astype(np.float64); S = F - T
    base = np.array([ev(w * F + (1 - w) * T) for w in GRID])
    L = np.zeros((len(LAM), len(fields) * 2, len(GRID), 3))                                   # loss by lambda, draw, w, view (val, pmix-public, pmix-private)
    for fi, (k, n, _) in enumerate(fields):
        n2 = n - float(n @ S / (S @ S)) * S
        for li, lam in enumerate(LAM):
            c = np.sqrt(lam * SIG[name] / n2.var())
            for si, sgn in enumerate((1.0, -1.0)):
                Fn = zv(F + sgn * c * n2).astype(np.float64)
                for wi, w in enumerate(GRID):
                    d = ev(w * Fn + (1 - w) * T) - base[wi]
                    L[li, 2 * fi + si, wi] = (d.mean(), np.average(d, weights=WPUB), np.average(d, weights=WPRIV))
    RES[name] = L
    w_(f'## {name}: sigma_e^2 = {SIG[name]:.5f}; loss of the file when lambda x sigma_e^2 of real fine-tune noise is added to the FT branch (mean over 12 draws = 6 fields x 2 signs; +- SE over draws)')
    w_('w_ft | lambda 1 | lambda 2 | lambda 4 | slope per unit lambda (least squares through 0) | gain of removing 4/5 of the fold noise: val | pmix-public | pmix-private | w^2 law from the w_ft 1.0 row')
    lam = np.array(LAM); g1 = None
    for wi, w in enumerate(GRID):
        mu = L[:, :, wi, 0].mean(1); sem = L[:, :, wi, 0].std(1, ddof=1) / np.sqrt(L.shape[1])
        sl = [float((lam * L[:, :, wi, v].mean(1)).sum() / (lam ** 2).sum()) for v in range(3)]
        # SE of the slope from the draws: slope per draw
        sld = (lam[:, None] * L[:, :, wi, 0]).sum(0) / (lam ** 2).sum(); sse = sld.std(ddof=1) / np.sqrt(len(sld))
        if w == 1.0: g1 = -0.8 * sl[0]
        RES[(name, w)] = (-0.8 * sl[0], 0.8 * sse, -0.8 * sl[1], -0.8 * sl[2])
        w_(f'{w:.2f} | ' + ' | '.join(f'{mu[i]:+.5f}+-{sem[i]:.5f}' for i in range(3)) + f' | {sl[0]:+.6f}+-{sse:.6f} | {-0.8 * sl[0]:+.5f}+-{0.8 * sse:.5f} | {-0.8 * sl[1]:+.5f} | {-0.8 * sl[2]:+.5f} |')
    for wi, w in enumerate(GRID):
        w_(f'   w^2 law check, w_ft {w:.2f}: measured gain {RES[(name, w)][0]:+.5f} against G1 x w^2 = {g1 * w * w:+.5f}')
    w_()
np.savez(OUT / 'c3_simex.npz', **{k: v for k, v in RES.items() if isinstance(k, str)}, lam=np.array(LAM), grid=np.array(GRID))
import json; json.dump({f'{k[0]}@{k[1]}': v for k, v in RES.items() if isinstance(k, tuple)}, open(OUT / 'c3_simex.json', 'w'))
open(OUT / 'c3_simex.txt', 'w').write('\n'.join(out) + '\n')
