"""RESCUE r11: robustness of the column-forms that passed G-info in r10 (INFORMATION ONLY: written after r10 was read; nothing here can promote
a column; it can only qualify a pass).
  (a) 100 within-feed permutations of each form on each witness (seeds 7000-7099): placebo sd, maximum, calibrated z = cz / sd, share of placebos beyond |cz|
  (b) slices: H slots 6-8 / 11-13 (with their own 100 placebos for slots 11-13 = the test-like half); T val slot 16 / 18, pmix-public / pmix-private weights
      (score_f / cz of reports/INPUTS_1116/residual_rows/s09_slices.py, weights of src/pmix.py)
  (c) overlap: cz of each form given the null model (pool + own inputs + nx3) plus ONE other form; and given ALL the other forms
  (d) size: log-likelihood per feed of the null model, + nx3, + all forms (fit_big), on H and on T
usage: python r11_robust.py <tag> <col:transform:form,...>     e.g. bt5::rk,ns2_5:log1p:z,never5:flag:z
CPU only, read-only. Output out/r11_robust_<tag>.txt (exclusive create)."""
from rs_common import *
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows'); sys.path.insert(0, '<repo>/src')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
import lens
import condlib
import pmix
sys.argv = _argv
nb.set_num_threads(NT)
TAG = sys.argv[1]; FORMS = [tuple(x.split(':')) for x in sys.argv[2].split(',')]
P = Log(OUT / f'r11_robust_{TAG}.txt')
V = Views(None)
NPL = 100
R = {q: V.rows(q) for q in ('hold', 'val_twin')}


def col(q, c, t, fm):
    x = get(q, c); x[~np.isfinite(x)] = np.nan
    if t == 'log1p':
        x = np.log1p(np.maximum(x, 0))
    return (lens.feed_z(x, R[q][2], R[q][3], 5.0) if fm == 'z' else lens.feed_rk(x, R[q][2], R[q][3])).astype(np.float32)


def score_f(N, x):
    D = N.D; x = np.asarray(x, np.float32); p = N.r['p']
    mx = np.bincount(N.fid, p * x, D['n']); g = np.bincount(N.fid, D['y'] * x, D['n']) - N.npos * mx
    Ixc = (N.rw * x) @ N.X - mx @ N.nMU
    return g - N.GA @ Ixc


def czw(D, Uf, w=None):
    w = np.ones(len(Uf)) if w is None else w
    Uc = np.bincount(D['clus'], Uf * w); s = np.sqrt((Uc ** 2).sum())
    return float((Uf * w).sum() / s) if s > 0 else 0.0


names = [f'{c}:{fm}' for c, t, fm in FORMS]
for wit, q in (('H', 'hold'), ('T', 'val_twin')):
    sp = 'hold' if wit == 'H' else 'val'
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; nm_, cl = condlib.control_block(D)
    rows = R[q][0]
    if wit == 'H':
        assert (D['rows'] == rows).all()
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        slot_f = V.A['train']['slot'][V.hold]; slot_r = np.repeat(slot_f, R[q][3])
        SLF = [('all', None), ('slots 6-8', ((slot_f >= 6) & (slot_f <= 8)) * 1.0), ('slots 11-13', (slot_f >= 11) * 1.0)]
    else:
        nz = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nr = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
        slot_f = V.A['val']['slot']; slot_r = np.repeat(slot_f, R[q][3]); W = pmix.weights()
        SLF = [('all', None), ('slot 16', (slot_f == 16) * 1.0), ('slot 18', (slot_f == 18) * 1.0), ('M < 50', (D['M'] < 50) * 1.0), ('M >= 50', (D['M'] >= 50) * 1.0), ('pmix-public w', W['public']), ('pmix-private w', W['private'])]
    X = {n: col(q, c, t, fm) for n, (c, t, fm) in zip(names, FORMS)}
    N = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + [nz, nr])
    P(f'=== witness {wit} ({"holdout rows, lag view" if wit == "H" else "val rows, stale-val twin, base TO16"}); null model = pool + {len(nm_)} own inputs + nx_rp3_z + nx_rp3_rk ===')
    P('(a) + (b): form | ' + ' | '.join(n for n, _ in SLF) + f' | {NPL} placebos: sd, max |cz|, cz / sd, share of placebos with |cz| >= the read' + (' | placebos on slots 11-13 only: sd, max, cz / sd' if wit == 'H' else ''))
    for n in names:
        Uf = score_f(N, X[n]); cells = [czw(D, Uf, w) for _, w in SLF]
        pl_ = np.array([czw(D, score_f(N, condlib.perm_within(X[n], D['off'], D['M'], 7000 + s))) for s in range(NPL)])
        s_ = f'{n:14s} | ' + ' | '.join(f'{c_:+.2f}' for c_ in cells) + f' | sd {pl_.std():.2f}, max {np.abs(pl_).max():.2f}, cz / sd {cells[0] / pl_.std():+.2f}, beyond {np.mean(np.abs(pl_) >= abs(cells[0])):.2f}'
        if wit == 'H':
            xr = np.where(slot_r >= 11, X[n], 0.0).astype(np.float32)
            pr_ = np.array([czw(D, score_f(N, condlib.perm_within(xr, D['off'], D['M'], 7000 + s))) for s in range(NPL)])
            s_ += f' | sd {pr_.std():.2f}, max {np.abs(pr_).max():.2f}, cz / sd {cells[2] / pr_.std():+.2f}'
        P(s_)
    # (c) overlap
    P('(c) overlap: cz of the row form given the null model + the column form (diagonal = given the null model alone); last column = given ALL other forms')
    M_ = np.full((len(names), len(names) + 1), np.nan)
    for i, n in enumerate(names):
        M_[i, i] = N.score(X[n])['cz']
    Ls = {'null + nx3': N.r['L']}
    del N
    for j, nj in enumerate(names):
        Nj = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk', nj], cl + [nz, nr, X[nj]])
        for i, n in enumerate(names):
            if i != j:
                M_[i, j] = Nj.score(X[n])['cz']
        del Nj
    for i, n in enumerate(names):
        oth = [m for m in names if m != n]
        Na = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'] + oth, cl + [nz, nr] + [X[m] for m in oth])
        M_[i, len(names)] = Na.score(X[n])['cz']
        del Na
    P(f'{"":14s} | ' + ' | '.join(f'{n:>13s}' for n in names) + ' | given all others')
    for i, n in enumerate(names):
        P(f'{n:14s} | ' + ' | '.join(f'{M_[i, j]:+13.2f}' for j in range(len(names))) + f' | {M_[i, len(names)]:+.2f}')
    # (d) size
    N0 = condlib.Null(D, nm_, cl); Ls['null'] = N0.r['L']; Lp = N0.L_pool; del N0
    Nf = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'] + names, cl + [nz, nr] + [X[m] for m in names]); Ls['null + nx3 + all forms'] = Nf.r['L']
    th = Nf.r['theta'][-len(names):]; del Nf
    n_ = D['n']
    P(f'(d) log-likelihood per feed ({n_} feeds): pool alone {Lp / n_:.5f}; + own inputs {Ls["null"] / n_:.5f} (gain {(Ls["null"] - Lp) / n_:.5f}); + nx3 {Ls["null + nx3"] / n_:.5f} (gain {(Ls["null + nx3"] - Ls["null"]) / n_:.5f}); '
      f'+ the {len(names)} forms {Ls["null + nx3 + all forms"] / n_:.5f} (gain {(Ls["null + nx3 + all forms"] - Ls["null + nx3"]) / n_:.5f} nats per feed, LR {2 * (Ls["null + nx3 + all forms"] - Ls["null + nx3"]):.1f} on {len(names)} df)')
    P('    joint coefficients of the forms: ' + ', '.join(f'{n} {t_:+.4f}' for n, t_ in zip(names, th)))
    del D
P('DONE')
