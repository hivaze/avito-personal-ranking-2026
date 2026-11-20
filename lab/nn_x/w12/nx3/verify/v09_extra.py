"""nx3 verifier v09: follow-ups to v02 (label-free)
 1 leaky variants with a clear margin: rows where a variant's forms differ from V0 by > 1e-3 (z) or > 1e-4 (rank), and how many of them the installed value sides with
 2 the single zero-pattern mismatches of v02 traced
 3 audit cross-check: my admitted-pair count minus the pairs of query feeds with an empty history against the builder's C2 counts (build.log)
 4 the one val row that is non-zero in the stored c09_val array and zero in the build of record (builder: "not traced")
 5 share of rows that the model-side standardisation clamps (|x - f_mean| / f_std > 6 with the smoke checkpoint's statistics)"""
from v_common import *
P = Log(f'{OUT}/v09_extra.txt')
FC = ROOT + '/nn/feat_cache'; TW = ROOT + '/nn_x/w12/nx3/staleval'
V = np.load(f'{OUT}/v01_natural_val.npz'); K = 1.0 / np.median(V['rp3'][0][V['rp3'][0] > 0])
def offs(M): return np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
SETS = (('val', 'v01_natural_val', FC + '/val/'), ('test', 'v01_natural_test', FC + '/test/'), ('twin', 'v01_twin_twin', TW + '/'), ('train sample', 'v01_train_train', FC + '/train/'))
P('=== 1. leaky variants, clear margin (|dz| > 1e-3 or |drank| > 1e-4 against V0)')
for tag, fn, d_ in SETS:
    d = np.load(f'{OUT}/{fn}.npz'); rows = d['rows']; M = d['M']; off = offs(M); q = d['q']
    zi = np.load(d_ + 'nx_rp3_z.npy', mmap_mode='r'); ri = np.load(d_ + 'nx_rp3_rk.npy', mmap_mode='r'); o = np.argsort(rows, kind='stable')
    zin = np.empty(len(rows)); rin = np.empty(len(rows)); zin[o] = np.asarray(zi[rows[o]]); rin[o] = np.asarray(ri[rows[o]])
    z0, r0 = my_forms(d['rp3'][0], off, M, K)
    for v, nm in ((1, 'own user admitted'), (2, 'sparse source at/after the own time admitted'), (3, 'user 0 linked')):
        zv, rv = my_forms(d['rp3'][v], off, M, K); clear = (np.abs(zv - z0) > 1e-3) | (np.abs(rv - r0) > 1e-4)
        side = clear & ((np.abs(zin - zv) + 10 * np.abs(rin - rv)) < (np.abs(zin - z0) + 10 * np.abs(rin - r0)))
        P(f'  {tag} | {nm}: clearly different rows {int(clear.sum())} in {int((np.bincount(q[clear], minlength=len(M)) > 0).sum())} feeds; of them the installed value is closer to the leaky variant on {int(side.sum())} rows')
    # 2
    zmin = np.minimum.reduceat(zin, off)[q]; rmin = np.minimum.reduceat(rin, off)[q]; nz = d['p3a'][0] > 0
    mm = np.nonzero((zin > zmin) != nz)[0]
    for i in mm:
        P(f'  {tag} | zero-pattern mismatch at flat row {int(rows[i])}: my raw rp3b05 {d["rp3"][0][i]:.3e} (log1p(K x) = {np.log1p(K * d["rp3"][0][i]):.3e}), installed z - feed minimum {zin[i] - zmin[i]:.2e}, installed rank - feed minimum {rin[i] - rmin[i]:.4f} (the rank file separates it)')
    if len(mm) == 0: P(f'  {tag} | zero pattern from the z file: no mismatch')
    P(f'  {tag} | zero pattern from the RANK file (rank above the feed minimum) against my walk > 0: mismatches {int(((rin > rmin) != nz).sum())}; feeds without any zero row: {int((np.minimum.reduceat(d["p3a"][0], off) > 0).sum())}')
P('=== 3. my admitted (row, sequence) pairs against the builder\'s C2 audit (which skips query feeds with an empty history)')
VA = Split('val', hist=False); TE = Split('test', hist=False); TRf = Split('train', hist=False)
for tag, fn, S, bc in (('val natural', 'v01_natural_val', VA, 17218069), ('test natural', 'v01_natural_test', TE, 32741802), ('val twin', 'v01_twin_twin', VA, 12808523)):
    d = np.load(f'{OUT}/{fn}.npz'); h0 = (S.H[d['qf']] == 0)[d['q']]
    P(f'  {tag}: my pairs {int(d["cnt"][0])}, of them on feeds with H = 0: {int(d["nseq"][h0].sum())}; difference {int(d["cnt"][0]) - int(d["nseq"][h0].sum())}; builder {bc}; equal: {int(d["cnt"][0]) - int(d["nseq"][h0].sum()) == bc}')
tot = 0
for k in range(3):
    tot += int(np.load(f'{OUT}/v03_trainfull_{k}of3.npz')['cnt'][0])
P(f'  train lag view: my pairs over all has = 1 feeds {tot}; builder 1388410566 (difference {tot - 1388410566} = pairs of the {int(((TRf.slot >= 6) & (TRf.H == 0)).sum())} has = 1 feeds with an empty history; not stored per row in v03)')
P('=== 4. the val row that is non-zero in the stored c09_val (day admission) and zero in the build of record')
c9v = np.load(ROOT + '/nn_x/insights/ccb_1113/scratch/tester3/cache/c09_val.npz'); d = np.load(f'{OUT}/v01_natural_val.npz')
ii = np.nonzero((c9v['p3a'] > 0) & (d['p3a'][0] == 0))[0]
for i in ii:
    P(f'  val row {int(i)}: stored c09_val p3a {c9v["p3a"][i]:.3e}; mine 0; my admitted occurrences of the candidate {int(d["nocc"][i])} (cap {CAP}); contributing sequences under the cap {int(d["nseq"][i])} -> the 500-occurrence cap moved the window: {bool(d["nocc"][i] > CAP)}')
P('=== 5. model-side clamp (inputs are standardised with f_mean / f_std and clamped to +-6): smoke nx3 checkpoint statistics')
import torch
ck = torch.load(ROOT + '/nn_x/abl/ckpt/abl_smoke_nx3_s23.pt', map_location='cpu', weights_only=False); fm = ck['state_dict']['f_mean'].numpy(); fs = ck['state_dict']['f_std'].numpy()
P(f'  f_mean / f_std at dense indices 94, 95, 96 (nx_rp3_z, nx_rp3_rk, has): {fm[94]:.4f} / {fs[94]:.4f}; {fm[95]:.4f} / {fs[95]:.4f}; {fm[96]:.4f} / {fs[96]:.4f}')
thr = fm[94] + 6 * fs[94]
for sp in ('train', 'val', 'test'):
    z = np.load(f'{FC}/{sp}/nx_rp3_z.npy'); ok = ~np.isnan(z)
    P(f'  {sp}: rows with nx_rp3_z above the clamp ({thr:.3f}): {np.mean(z[ok] > thr):.5f} of the non-NaN rows; standardised sd on non-NaN rows {z[ok].std() / fs[94]:.3f}; standardised value of a has = 0 row {(0 - fm[94]) / fs[94]:+.3f}')
P('DONE')
