"""DESIGN p05: the candidates of inventory D (f1/f2 equality forms, same-user earlier-feed forms, item click / source statistics) read with the
CALIBRATED lens of inventory B (score test given pool + 84 columns of the ranker's own inputs; placebo sd 0.95-0.99), which D did not use.
Witnesses: val (train-only pool of the 16 held-out wide students; FD16P20 file) and the 30,000 holdout feeds. CPU only, read-only.
Output: out/p05_cond_DE.txt. numba is imported first so that its cache stays in this directory (lens.py would point it elsewhere)."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
os.environ['NUMBA_CACHE_DIR'] = '<repo>/reports/INPUTS_1116/design/out/numba_cache'
sys.dont_write_bytecode = True
import numba  # noqa: fixes the cache dir before lens.py sets its own
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
sys.argv = [sys.argv[0], 'val', 'TO16']
from lens import *
import condlib
RF = ROOT / 'reports/INPUTS_1116/raw_fields'; C9 = ROOT / 'nn_x/insights/ccb_1113/scratch/tester3/cache'
out = open(ROOT / 'reports/INPUTS_1116/design/out/p05_cond_DE.txt', 'w')
def Pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.write(s + '\n'); out.flush()
@nb.njit(parallel=True, cache=True)
def grp(off, M, x):
    o_ = np.zeros(len(x))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        for k in range(m):
            c = 0
            for q in range(m):
                if x[o + q] == x[o + k]: c += 1
            o_[o + k] = np.log(c)
    return o_
names = ['E:f1_floor', 'E:f2_home', 'E:f1_tie', 'U:uc_ctr', 'U:uc_coec', 'U:click_sim', 'U:skip_sim', 'U:reshown', 'U:uc_ctr_stale(val only)',
         'I:it_coec(val only)', 'I:it_coec_stale(val only)', 'I:s1share(val only)', 'I:s1share_stale(val only)', 'ctl:rp3b05', 'ctl:p3a']
def cols(sp, D):
    t = 'val' if sp == 'val' else 'hold'
    if sp == 'val': f1 = np.load(FLAT / 'val/cand_f1.npy')
    else: f1 = np.asarray(np.load(FLAT / 'train/cand_f1.npy', mmap_mode='r')[D['rows']])
    L = lambda n: np.load(RF / f'x_{n}_{t}.npy') if (RF / f'x_{n}_{t}.npy').exists() else None
    tie = L('V8b_f1_tie'); chk = float(np.abs(grp(D['off'], D['M'], f1.astype(np.float32)) - tie).max())
    Pr(f'[{sp}] alignment check: log f1 tie-group size recomputed on this lens\'s rows vs inventory D\'s array: max |diff| {chk:.2e}')
    c9 = np.load(C9 / f'c09_{t}.npz')
    return {'E:f1_floor': (f1 < 0.018).astype(np.float64), 'E:f2_home': L('V8a_f2_home'), 'E:f1_tie': tie, 'U:uc_ctr': L('V2b_uc_ctr'), 'U:uc_coec': L('C3_uc_coec'),
            'U:click_sim': L('C2_click_sim'), 'U:skip_sim': L('C1_skip_sim'), 'U:reshown': L('V2a_ui_reshown'), 'U:uc_ctr_stale(val only)': L('V2bs_uc_ctr_stale'),
            'I:it_coec(val only)': L('V3_it_coec'), 'I:it_coec_stale(val only)': L('V3s_it_coec_stale'), 'I:s1share(val only)': L('V5a_it_s1share'),
            'I:s1share_stale(val only)': L('V5as_it_s1share_stale'), 'ctl:rp3b05': c9['rp3b05'], 'ctl:p3a': c9['p3a']}
res = {n: {} for n in names}; blocks = {}
for sp, base in (('val', 'TO16'), ('val', 'FILE'), ('hold', 'TO16')):
    D = load_split(sp, base); condlib.sp = sp; nm, cl = condlib.control_block(D); N = condlib.Null(D, nm, cl)
    X = cols(sp, D); tag = f'{sp}-{base}'
    for n in names:
        x = X.get(n)
        if x is None: continue
        F = forms(np.asarray(x, np.float64), D, log=False)
        if F['varshare'] < 0.01: continue
        a = N.score(F['z']); b = N.score(F['rk'])
        res[n][tag] = (a['cz'], b['cz'], a['r2'], a['dnats'])
    # placebo: within-feed permutations of the two binary flags and of uc_coec (20 each) under the same null model
    pz = []
    for n in ('E:f1_floor', 'E:f2_home', 'U:uc_coec'):
        F = forms(np.asarray(X[n], np.float64), D, log=False)
        for s in range(20): pz.append(N.score(condlib.perm_within(F['z'], D['off'], D['M'], 1000 + s))['cz'])
    pz = np.array(pz); Pr(f'[{tag}] placebo (60 within-feed permutations): sd {pz.std():.2f}, max |cz| {np.abs(pz).max():.2f}; control block {len(nm)} columns')
Pr('\ncolumn | val TO16: cz z-form / rank-form (R2, dnats) | val FILE | holdout')
for n in names:
    cells = []
    for tag in ('val-TO16', 'val-FILE', 'hold-TO16'):
        r = res[n].get(tag)
        cells.append('-' if r is None else f'{r[0]:+5.2f} / {r[1]:+5.2f} (R2 {r[2]:.2f}, {r[3]:.5f})')
    Pr(f'{n:28s} | ' + ' | '.join(cells))
out.close()
