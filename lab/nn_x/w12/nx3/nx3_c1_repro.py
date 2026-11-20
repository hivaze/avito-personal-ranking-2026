"""nx3 STEP 0, check C1 (G-code, reproduction): the kernel of record driven by THIS builder's driver (nx3_common.run_kernel / Src) in the c09 conventions
reproduces the stored arrays of record:
  (R-hold) cut = own day, no sparse members, deg = posting count of the day's source: cache/c09_hold.npz (rp3b05, p3a) on ALL 4,630,966 holdout rows (14 day indexes)
  (R-val)  all-train source + val feeds of earlier DAYS, deg = train + val feeds of earlier days (c09_build.py main()): cache/c09_val.npz on all 1,389,971 rows
Bar (rule, STEP 0 G-code): 1e-5 relative. CPU only, 12 threads. Output nn_x/w12/nx3/out/c1_repro.txt (exclusive create)."""
from nx3_common import *

P = Log(OUT / f'c1_repro{TAG}.txt')
P(f'kernel of record: {T3}/c09_build.py md5 {md5(T3 + "/c09_build.py")}; numba {nb.__version__}, numpy {np.__version__}')
D = Data(P)
ft = D.F['train']; At, Av = D.A['train'], D.A['val']


def cmp_(tag, mine, ref):
    res = []
    for j, k in enumerate(('rp3b05', 'p3a')):
        a_, b_ = mine[:, j], ref[k]; nz = b_ > 0
        rel = np.abs(a_ - b_)[nz] / b_[nz]
        zm = int(((a_ > 0) != (b_ > 0)).sum())
        P(f'{tag} {k}: rows {len(a_)}, max abs diff {np.abs(a_ - b_).max():.3e}, max rel diff on non-zero {rel.max() if nz.any() else 0.0:.3e}, zero-pattern mismatches {zm}, non-zero share mine {np.mean(a_ > 0):.6f} stored {np.mean(b_ > 0):.6f}')
        res.append((float(rel.max()) if nz.any() else 0.0, zm))
    return res

# ---------------- (R-val) c09 convention on val
src_all = c9.source_fidx(ft, None)
S = Src(D, [('train', src_all, np.full(len(src_all), -1, np.int64), True), ('val', np.arange(Av['n']), Av['day'], False)])
dq = np.unique(Av['day'])
nde = int(S.idx.lens[:S.n_dense].sum()); sde = np.repeat(S.idx.days[S.n_dense:], S.idx.lens[S.n_dense:]); evs = S.idx.ev[nde:]
deg2 = np.stack([S.deg_dense + np.bincount(evs[sde < d_], minlength=D.G) for d_ in dq])
o, rows, cd, offq, Mq = run_kernel(D, 'val', S.idx, deg2, np.arange(Av['n']), np.searchsorted(dq, Av['day']), Av['day'])
assert (rows == np.arange(NROWS['val'])).all()
rv = cmp_('(R-val) c09 convention (days; deg with the earlier val feeds)', o, np.load(C9 / 'c09_val.npz'))
del S, deg2, o

# ---------------- (R-hold) all holdout days, cut = own day, no sparse members
hold, _p = hc.holdout_feeds(ft); Mh = At['M'][hold]; hoff = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64); nrow_h = int(Mh.sum())
c9h = np.load(C9 / 'c09_hold.npz'); assert nrow_h == len(c9h['p3a']) == 4630966
mine = np.zeros((nrow_h, 2)); done = np.zeros(nrow_h, bool); hd = At['day'][hold]
for d_ in np.unique(hd):
    qi = np.nonzero(hd == d_)[0]; src = c9.source_fidx(ft, int(d_))
    lrow = np.concatenate([np.arange(hoff[q], hoff[q] + Mh[q]) for q in qi])
    if len(src) == 0:
        done[lrow] = True; P(f'(R-hold) day {d_} (slot {d_ - hc.DAY0}): empty source -> zeros ({len(qi)} feeds)'); continue
    S = Src(D, [('train', src, np.full(len(src), -1, np.int64), True)])
    assert (S.deg_dense == S.idx.cnt).all()
    o, rows, cd, offq, Mq = run_kernel(D, 'train', S.idx, S.idx.cnt.astype(np.float64)[None, :], hold[qi], np.zeros(len(qi), np.int64), np.full(len(qi), 1 << 40, np.int64))
    mine[lrow] = o[:, :2]; done[lrow] = True
    P(f'(R-hold) day {d_} (slot {d_ - hc.DAY0}): {len(qi)} feeds, {len(lrow)} rows, source seqs {len(src)}, non-zero share {np.mean(o[:, 1] > 0):.4f}; '
      f'max abs diff rp3b05 {np.abs(o[:, 0] - c9h["rp3b05"][lrow]).max():.3e} p3a {np.abs(o[:, 1] - c9h["p3a"][lrow]).max():.3e}')
    del S, o
assert done.all()
rh = cmp_('(R-hold) ALL holdout rows, cut = own day, no sparse members', mine, c9h)
ok = all(r[0] <= 1e-5 and r[1] == 0 for r in rv + rh)
P(f'C1 VERDICT (bar 1e-5 relative, zero pattern equal): {"PASS" if ok else "FAIL"}')
P('DONE')
