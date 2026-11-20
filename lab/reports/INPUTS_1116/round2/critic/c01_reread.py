"""CRITIC2 c01: my own re-read of the rescue survivor and near-misses from the arrays (labels are opened here).
Null model = pool (16 held-out wide students) + control block of the ranker's own inputs + nx_rp3_z + nx_rp3_rk.
  H = holdout rows, lag view (0 on the 14,775 feeds without it);  T = val rows, stale-val twin of the column;  N = val rows, natural column.
For each form: cz (my numpy score path), model-variance z, nats per feed, R2 with the null model;
  placebo = NPL within-feed permutations (one row order per permutation, applied to all main forms): sd, max, share beyond;
  slices from the per-feed scores: H by slot group and by lag to the newest source day; T / N by val day, M, pmix-public / pmix-private weights;
  'region' re-score with the column zeroed outside holdout slots 11-13 (the test-like-only view);
  redundancy: cz of each main form given the null model + each other main form, + the two others; joint fit of the three.
Yardstick: nx_rp3_z under the null model WITHOUT nx3, with the same slices.
usage: c01_reread.py     output out/c01_reread.txt"""
from cc_common import *
import pmix
P = Log(OUT / 'c01_reread.txt')
V = rs.Views(P)
NPL = 200
MAIN = [('kn1_5', None, 'z'), ('ns2_5', 'log1p', 'z'), ('bt5', None, 'rk')]
INFO = [('kn1_5', None, 'rk'), ('kn20_5', None, 'z'), ('ns2_5', 'log1p', 'rk'), ('bt5', None, 'z'), ('bt5_seen', None, 'z'), ('coec5', None, 'rk'), ('never5', None, 'z'), ('X3_5', None, 'z')]
nm = lambda c: f'{c[0]}:{c[2]}'


def lag_newest(sp, q):
    """days between the row's slot and the newest source slot of its window (feed level); -1 without a view"""
    _, sel, cut = V.Q[q]; sl = V.A[sp]['slot'][sel]
    return np.array([int(s - rs.window(int(c)).max()) if c >= 0 else -1 for s, c in zip(sl, cut)])


def run(wit):
    sp = 'hold' if wit == 'H' else 'val'; q = {'H': 'hold', 'T': 'val_twin', 'N': 'val_nat'}[wit]
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D)
    off, M = D['off'], D['M']; rows, fl, off_q, M_q = V.rows(q)
    assert (off_q == off).all() and (M_q == M).all()
    if wit == 'H':
        assert (D['rows'] == rows).all()
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        slot_f = V.A['train']['slot'][V.hold]; lag_f = lag_newest('train', 'hold')
        SL = [('slots 6-8', ((slot_f >= 6) & (slot_f <= 8)) * 1.0), ('slots 11-13', (slot_f >= 11) * 1.0)] + [(f'lag {d} d', (lag_f == d) * 1.0) for d in (2, 3, 4, 7, 8, 9)]
    else:
        if wit == 'T':    # the stale-val twin of nx3 next to the twin of the column (staleness-matched); the natural nx3 is read as a second null below
            nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
        else:
            nz = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nr = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
        slot_f = V.A['val']['slot']; W = pmix.weights(); lag_f = lag_newest('val', q)
        SL = [('slot 16', (slot_f == 16) * 1.0), ('slot 18', (slot_f == 18) * 1.0), ('M < 50', (M < 50) * 1.0), ('M >= 50', (M >= 50) * 1.0), ('pmix-public w', W['public']), ('pmix-private w', W['private'])]
        if wit == 'T':
            SL += [(f'lag {d} d', (lag_f == d) * 1.0) for d in (7, 8, 9, 10)]
    X = {nm(c): form(rget(q, c[0]), off, M, c[1], c[2]) for c in MAIN + INFO}
    P(f'\n=== witness {wit}: {q}; feeds {D["n"]}, rows {len(D["y"])}; control block {len(cn)} columns ===')
    # ---------- yardstick: nx3 itself under the null model without it
    N0 = MyNull(D, cl)
    P(f'null model WITHOUT nx3: loglik {N0.L:.3f} (my own recomputation {N0.Lmine:.3f}); max |gradient| over its columns {np.abs(N0.grad).max():.2e}')
    for name, x in (('nx_rp3_z', nz), ('nx_rp3_rk', nr)):
        c, Vm, Ixx, Uf = N0.cz(x)
        P(f'  yardstick {name}: cz {c:+.2f}, nats/feed {0.5 * Uf.sum() ** 2 / Vm / D["n"]:.5f}, R2 {1 - Vm / Ixx:.2f} | ' + ' | '.join(f'{n_} {czw(D, Uf, w):+.2f}' for n_, w in SL))
    c0 = {k: N0.cz(X[k])[0] for k in X}
    th0 = N0.theta; del N0
    # ---------- the null model with nx3
    N = MyNull(D, cl + [nz, nr], theta0=list(th0) + [0.0, 0.0])
    P(f'null model WITH nx3 ({"twin" if wit == "T" else ("lag view" if wit == "H" else "natural")}): loglik {N.L:.3f} (mine {N.Lmine:.3f}); max |gradient| {np.abs(N.grad).max():.2e}; nx3 gain {(N.L - 0) :.1f}')
    chk = N.cz(nz)[0]; P(f'  check: nx_rp3_z under the model that holds it: cz {chk:+.4f}')
    P('form | cz | cz without nx3 in the null | z by the model variance | nats/feed | R2 with the null model | ' + ' | '.join(n_ for n_, _ in SL))
    UF = {}
    for c in MAIN + INFO:
        k = nm(c); cz_, Vm, Ixx, Uf = N.cz(X[k]); UF[k] = Uf
        P(f'  {k:11s} | {cz_:+.2f} | {c0[k]:+.2f} | {Uf.sum() / np.sqrt(Vm):+.2f} | {0.5 * Uf.sum() ** 2 / Vm / D["n"]:.5f} | {1 - Vm / Ixx:.2f} | ' + ' | '.join(f'{czw(D, Uf, w):+.2f}' for _, w in SL))
    # ---------- placebo
    rng = np.random.default_rng({'H': 9101, 'T': 9102, 'N': 9103}[wit]); PL = {nm(c): [] for c in MAIN}; PLs = {nm(c): [] for c in MAIN}
    if wit == 'H':
        test_like = np.repeat(slot_f >= 11, M)
    for s in range(NPL):
        o = perm_order(N.fid, rng)
        for c in MAIN:
            k = nm(c); xp = X[k][o]; PL[k].append(N.cz(xp)[0])
            if wit == 'H':
                PLs[k].append(N.cz(np.where(test_like, xp, 0.0))[0])
    P(f'placebo, {NPL} within-feed permutations:')
    for c in MAIN:
        k = nm(c); pl_ = np.array(PL[k]); r = czw(D, UF[k])
        s_ = f'  {k:11s}: placebo mean {pl_.mean():+.2f}, sd {pl_.std():.2f}, max |cz| {np.abs(pl_).max():.2f}, q99 |cz| {np.quantile(np.abs(pl_), 0.99):.2f}; read {r:+.2f} -> cz / sd {r / pl_.std():+.2f}, placebos beyond {np.mean(np.abs(pl_) >= abs(r)):.3f}'
        if wit == 'H':
            ps = np.array(PLs[k]); rr = N.cz(np.where(test_like, X[k], 0.0))[0]
            s_ += f' || region slots 11-13 only: read {rr:+.2f}, placebo sd {ps.std():.2f}, max {np.abs(ps).max():.2f}, cz / sd {rr / ps.std():+.2f}, beyond {np.mean(np.abs(ps) >= abs(rr)):.3f}'
        P(s_)
    # ---------- second null on T: the natural nx3 (the explorer's choice)
    if wit == 'T':
        nzv = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nrv = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
        N2 = MyNull(D, cl + [nzv, nrv], theta0=list(th0) + [0.0, 0.0])
        P('T with the NATURAL nx3 in the null model (the explorer\'s deciding read): ' + ', '.join(f'{nm(c)} {N2.cz(X[nm(c)])[0]:+.2f}' for c in MAIN + INFO)); del N2
    # ---------- redundancy among the three main forms
    names = [nm(c) for c in MAIN]; thN = list(N.theta); LN = N.L; del N
    P('redundancy: cz of the row form given the null model + the listed forms')
    for j, kj in enumerate(names):
        Nj = MyNull(D, cl + [nz, nr, X[kj]], theta0=thN + [0.0])
        P(f'  given + {kj}: ' + ', '.join(f'{k} {Nj.cz(X[k])[0]:+.2f}' for k in names if k != kj) + f' | coefficient of {kj} {Nj.theta[-1]:+.4f}, LR {2 * (Nj.L - LN):.1f}')
        del Nj
    for i, ki in enumerate(names):
        oth = [k for k in names if k != ki]
        Na = MyNull(D, cl + [nz, nr] + [X[k] for k in oth], theta0=thN + [0.0] * len(oth))
        P(f'  {ki} given the other two: {Na.cz(X[ki])[0]:+.2f}'); del Na
    Nf = MyNull(D, cl + [nz, nr] + [X[k] for k in names], theta0=thN + [0.0] * 3)
    P(f'joint fit of the three: LR {2 * (Nf.L - LN):.1f} on 3 df, nats/feed {(Nf.L - LN) / D["n"]:.5f}; coefficients ' + ', '.join(f'{k} {t:+.4f}' for k, t in zip(names, Nf.theta[-3:])))
    del Nf, D


for w in ('H', 'T', 'N'):
    run(w)
P('DONE')
