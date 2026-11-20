# Builder of the one new input that reached training: a two-step co-visitation walk (RP3beta) from a candidate
# through other users' whole histories to the items of the feed's own history, code name nx3.
# Train rows are built in a lag view: each row reads only sources older than a cut 1 to 8 days back, drawn to
# match the staleness of val and test rows. Rows of Sep 6-11 have no admissible source and are flagged missing.
# CPU only, 12 threads, no labels opened. It wrote staging files under nn_x/w12/nx3/out/ and a stale twin of
# val; log: lab/nn_x/w12/nx3/logs/build.stdout. Helpers are in lab/nn_x/w12/nx3/nx3_common.py.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/nx3/nx3_build.py
"""nx3 STEP 0, BUILD OF RECORD of family W (whole-sequence RP3beta walk): raw (rp3b05, p3a) and the two entered value columns nx_rp3_z, nx_rp3_rk for
  train  all 3,151,208 feeds in the LAG VIEW of BUILD_SPEC.txt 1-2 (cut' = max(slot - L, 5), L = nn_x/w12/stale/hz2/feats_L/train/hz2_lag.npy; has = slot >= 6;
         NaN on every row of a has = 0 feed; the 30,000 holdout feeds are train feeds and get the same view)
  val / test  natural: all train sequences (slots 0-14) + val and test feeds with timestamp < own timestamp
  val    STALE-VAL TWIN: dense = train slots < slot - L (L = hz2_common.assign_lags), sparse = the 1/37 train sample on slots >= cut + val feeds with timestamp < own
with, for every index, the causality audit C2 (counts over all admitted (row, sequence) pairs) and an independent numpy re-derivation on a feed sample.
Label-free: no target array is opened. CPU only, 12 threads. Writes NEW files under nn_x/w12/nx3/ only (staging: out/files/<split>/, out/raw/, staleval/);
nothing is installed into data/feats or nn/feat_cache by this script (that is nx3_install.py, after STEP 0 reads PASS)."""
from nx3_common import *
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
from lens import feed_z, feed_rk          # the within-feed z / mid-rank kernels of the lens (the forms the critic's advance read used)
sys.argv = _argv

for d_ in (RAW_DIR, FILES_DIR / 'train', FILES_DIR / 'val', FILES_DIR / 'test', TWIN_DIR):
    os.makedirs(d_, exist_ok=True)
P = Log(OUT / f'build{TAG}.log')
P(f'kernel of record: {T3}/c09_build.py md5 {md5(T3 + "/c09_build.py")}; lag draw md5 {md5(str(hc.HERE / "feats_L/train/hz2_lag.npy"))}; numba {nb.__version__}, numpy {np.__version__}')
D = Data(P)
ft = D.F['train']; At, Av, Ae = D.A['train'], D.A['val'], D.A['test']
rng = np.random.default_rng(20261116)
AUD = {}       # tag -> 8 counts
BR = {}        # tag -> (feeds, rows, max rel diff rp3b05, max rel diff p3a, zero-pattern mismatches)


def check_sample(tag, split, S, feeds_q, cut_of, out, offq, Mq, sel):
    """independent re-derivation on the sampled feeds that fall into this query set"""
    pos = {int(f): i for i, f in enumerate(sel.tolist())}
    n = r = zm = 0; m0 = m1 = 0.0
    for f in feeds_q:
        i = pos[int(f)]; b = brute(D, split, S, int(f), int(cut_of(int(f)))); k = out[offq[i]:offq[i] + Mq[i], :2]
        den = np.maximum(np.abs(b), 1e-300); rel = np.abs(k - b) / den; rel[(b == 0) & (k == 0)] = 0.0
        m0 = max(m0, float(rel[:, 0].max())); m1 = max(m1, float(rel[:, 1].max())); zm += int(((k > 0) != (b > 0)).sum()); n += 1; r += len(b)
    o = BR.get(tag, (0, 0, 0.0, 0.0, 0)); BR[tag] = (o[0] + n, o[1] + r, max(o[2], m0), max(o[3], m1), o[4] + zm)


def add_aud(tag, c):
    AUD[tag] = AUD.get(tag, np.zeros(8, np.int64)) + c

# =========================================================== 1. natural val / test
src_all = c9.source_fidx(ft, None)
assert At['slot'][src_all].max() <= 14
S = Src(D, [('train', src_all, np.full(len(src_all), -1, np.int64), True), ('val', np.arange(Av['n']), Av['ts'], False), ('test', np.arange(Ae['n']), Ae['ts'], False)])
P(f'[natural] index: dense train seqs {S.n_dense} (slots 0-{int(S.s_slot[:S.n_dense].max())}), sparse seqs {S.n_sparse} (val {Av["n"]} + test {Ae["n"]}); items with dense deg 0: {int((S.deg_dense == 0).sum())} of {D.G}')
deg2 = S.deg_floor1(); NAT = {}
for sp in ('val', 'test'):
    a = D.A[sp]; sel = np.arange(a['n'], dtype=np.int64)
    o, rows, cd, offq, Mq = run_kernel(D, sp, S.idx, deg2, sel, np.zeros(a['n'], np.int64), a['ts'])
    assert (rows == np.arange(NROWS[sp])).all()
    add_aud(f'{sp} natural', run_audit(D, sp, S, sel, a['ts'], np.full(a['n'], 15, np.int64), offq, Mq, cd))
    check_sample(f'{sp} natural', sp, S, rng.choice(sel, 60, replace=False), lambda f: 15, o, offq, Mq, sel)
    NAT[sp] = o[:, :2].copy()
    P(f'[natural] {sp}: rows {len(o)}, non-zero share {np.mean(o[:, 1] > 0):.4f}; rows with p3a > 0 and rp3b05 = 0: {int(((o[:, 1] > 0) & (o[:, 0] == 0)).sum())}; candidates in the id map {np.mean(cd >= 0):.4f}; '
      f'non-zero rows whose candidate has dense deg 0 (reached through sparse members only): {int(((o[:, 1] > 0) & (S.deg_dense[np.maximum(cd, 0)] == 0)).sum())}')
del S, deg2
c9v = np.load(C9 / 'c09_val.npz')
for j, k in enumerate(('rp3b05', 'p3a')):
    a_, b_ = NAT['val'][:, j], c9v[k]
    P(f'[natural] val {k} against the stored cache/c09_val.npz (days, deg with sparse; BUILD_SPEC C1 "correlation printed"): corr {np.corrcoef(a_, b_)[0, 1]:.6f}, corr of log1p(1e6 x) {np.corrcoef(np.log1p(1e6 * a_), np.log1p(1e6 * b_))[0, 1]:.6f}; '
      f'non-zero share mine {np.mean(a_ > 0):.6f} stored {np.mean(b_ > 0):.6f}; non-zero in mine only {int(((a_ > 0) & (b_ == 0)).sum())}, in stored only {int(((a_ == 0) & (b_ > 0)).sum())}; rows equal to 1e-9 relative {np.mean(np.abs(a_ - b_) <= 1e-9 * np.maximum(np.abs(b_), 1e-300)):.4f}')
Kr = 1.0 / np.median(NAT['val'][:, 0][NAT['val'][:, 0] > 0]); Kp = 1.0 / np.median(NAT['val'][:, 1][NAT['val'][:, 1] > 0])
P(f'K (rp3b05) = 1 / median non-zero rp3b05 of val rows = {Kr:.6e}; K (p3a) = {Kp:.6e} (p3a is not an input of nx3; printed as BUILD_SPEC 2.3 asks)')

# =========================================================== 2. train rows, LAG VIEW
L = np.load(hc.HERE / 'feats_L/train/hz2_lag.npy').astype(np.int64); assert len(L) == At['n'] and L.min() >= 6 and L.max() <= 9
tslot = At['slot']; cut = np.maximum(tslot - L, 5); has = tslot >= 6
hold, _p = hc.holdout_feeds(ft); ish = np.zeros(At['n'], bool); ish[hold] = True
P(f'[lag view] train feeds {At["n"]}; has = 1 feeds {int(has.sum())} (rows {int(At["M"][has].sum())}); holdout feeds with has = 1: {int((has & ish).sum())} of {len(hold)}; '
  f'cut counts over has = 1 feeds {dict(zip(*[x.tolist() for x in np.unique(cut[has], return_counts=True)]))}')
assert int(has.sum()) == 1470065 and int((has & ish).sum()) == 15225
assert (cut[has] < tslot[has]).all() and (cut[has] >= 5).all()
spf = hc.sparse_flag(At['fid'])
P(f'[lag view] sparse 1/37 sample: {int(spf.sum())} train feeds ({spf.mean():.5f})')
RAW = np.full((NROWS['train'], 2), np.nan)
tr_off = At['co']
for c in np.unique(cut[has]):
    dense = c9.source_fidx(ft, int(hc.DAY0 + c)); sp_ = np.nonzero(spf & (tslot >= c))[0]
    assert tslot[dense].max() < c
    S = Src(D, [('train', dense, np.full(len(dense), -1, np.int64), True), ('train', sp_, At['ts'][sp_], False)])
    deg2 = S.deg_floor1()
    qf = np.nonzero(has & (cut == c))[0]
    P(f'[lag view] cut {c}: query feeds {len(qf)} (slots {dict(zip(*[x.tolist() for x in np.unique(tslot[qf], return_counts=True)]))}), dense seqs {len(dense)} (max slot {int(tslot[dense].max())}), sparse seqs {len(sp_)} (slots {int(tslot[sp_].min())}-{int(tslot[sp_].max())})')
    samp = set(rng.choice(qf, min(60, len(qf)), replace=False).tolist()) | set(rng.choice(qf[ish[qf]], min(20, int(ish[qf].sum())), replace=False).tolist() if ish[qf].any() else [])
    nz = tot = 0
    for ch in np.array_split(qf, max(1, int(np.ceil(len(qf) / 150000)))):
        o, rows, cd, offq, Mq = run_kernel(D, 'train', S.idx, deg2, ch, np.zeros(len(ch), np.int64), At['ts'][ch])
        add_aud('train lag view', run_audit(D, 'train', S, ch, At['ts'][ch], np.full(len(ch), c, np.int64), offq, Mq, cd))
        ss = [f for f in ch.tolist() if f in samp]
        if ss: check_sample('train lag view', 'train', S, ss, lambda f: c, o, offq, Mq, ch)
        RAW[rows] = o[:, :2]; nz += int((o[:, 1] > 0).sum()); tot += len(o)
        del o, rows, cd
    P(f'[lag view] cut {c}: rows {tot}, non-zero share {nz / tot:.4f}')
    del S, deg2
hasrow = np.repeat(has, At['M'])
assert not np.isnan(RAW[hasrow]).any() and np.isnan(RAW[~hasrow]).all()

# =========================================================== 3. stale-val twin
vL = np.asarray(hc.assign_lags(Av['fid'])[1]).astype(np.int64); vcut = Av['slot'] - vL
P(f'[twin] val feeds by cut {dict(zip(*[x.tolist() for x in np.unique(vcut, return_counts=True)]))}; lags {dict(zip(*[x.tolist() for x in np.unique(vL, return_counts=True)]))}')
TW = np.full((NROWS['val'], 2), np.nan)
for c in np.unique(vcut):
    dense = c9.source_fidx(ft, int(hc.DAY0 + c)); sp_ = np.nonzero(spf & (tslot >= c))[0]
    assert tslot[dense].max() < c
    S = Src(D, [('train', dense, np.full(len(dense), -1, np.int64), True), ('train', sp_, At['ts'][sp_], False), ('val', np.arange(Av['n']), Av['ts'], False)])
    deg2 = S.deg_floor1(); qf = np.nonzero(vcut == c)[0]
    o, rows, cd, offq, Mq = run_kernel(D, 'val', S.idx, deg2, qf, np.zeros(len(qf), np.int64), Av['ts'][qf])
    add_aud('val stale-val twin', run_audit(D, 'val', S, qf, Av['ts'][qf], np.full(len(qf), c, np.int64), offq, Mq, cd))
    check_sample('val stale-val twin', 'val', S, rng.choice(qf, min(25, len(qf)), replace=False), lambda f: c, o, offq, Mq, qf)
    TW[rows] = o[:, :2]
    P(f'[twin] cut {c}: {len(qf)} val feeds, dense seqs {len(dense)}, sparse train seqs {len(sp_)}, non-zero share {np.mean(o[:, 1] > 0):.4f}')
    del S, deg2, o
assert not np.isnan(TW).any()
P(f'[twin] val twin non-zero share {np.mean(TW[:, 1] > 0):.4f} (natural {np.mean(NAT["val"][:, 1] > 0):.4f})')
hf = np.nonzero(has)[0]; Mh_ = At['M'][hf]; hrows = np.nonzero(hasrow)[0]
assert len(hrows) == int(Mh_.sum())
save_new(RAW_DIR / 'val_nat.npy', NAT['val']); save_new(RAW_DIR / 'test_nat.npy', NAT['test']); save_new(RAW_DIR / 'val_twin.npy', TW)
save_new(RAW_DIR / 'train_lag_hasrows.npy', RAW[hrows]); save_new(RAW_DIR / 'train_has_feed.npy', has); save_new(RAW_DIR / 'train_cut_feed.npy', np.where(has, cut, -1).astype(np.int8))
save_new(RAW_DIR / 'val_twin_cut_feed.npy', vcut.astype(np.int8)); save_new(RAW_DIR / 'K.npy', np.array([Kr, Kp]))
P(f'raw arrays saved in {RAW_DIR}')

# =========================================================== 4. C2 summary and the independent re-derivation
P('=== C2 causality audit: every admitted (row, sequence) pair of every query row, true timestamps / slots / users of the feeds tables ===')
c2ok = True
for tag, c in AUD.items():
    P(f'  {tag}: ' + '; '.join(f'{n} {int(v)}' for n, v in zip(ANAMES, c)))
    c2ok &= bool((c[3:] == 0).all())
P(f'  C2 VERDICT (every VIOLATION count 0): {"PASS" if c2ok else "FAIL"}')
P('=== independent numpy re-derivation of BUILD_SPEC 2 on sampled feeds (members by true slot / timestamp / user; kernel output compared) ===')
brok = True
for tag, (n, r, m0, m1, zm) in BR.items():
    P(f'  {tag}: {n} feeds, {r} rows; max relative difference rp3b05 {m0:.3e}, p3a {m1:.3e}; zero-pattern mismatches {zm}')
    brok &= (m0 <= 1e-9 and m1 <= 1e-9 and zm == 0)
P(f'  RE-DERIVATION VERDICT (1e-9 relative, zero pattern equal): {"PASS" if brok else "FAIL"}')

# =========================================================== 5. entered forms and files (staging)
def offs(M):
    return np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)

FILES = {}
for sp, raw, M in (('val', NAT['val'], Av['M']), ('test', NAT['test'], Ae['M']), ('twin', TW, Av['M'])):
    z, rk = forms_of(raw[:, 0], offs(M), M, Kr, feed_z, feed_rk); FILES[sp] = (z.astype(np.float32), rk.astype(np.float32))
zc, rkc = forms_of(RAW[hrows, 0], offs(Mh_), Mh_, Kr, feed_z, feed_rk)          # has = 1 feeds are contiguous in the compact row order
ztr = np.full(NROWS['train'], np.nan, np.float32); rtr = np.full(NROWS['train'], np.nan, np.float32); ztr[hrows] = zc; rtr[hrows] = rkc
FILES['train'] = (ztr, rtr)
for sp, (z, rk) in FILES.items():
    assert z.dtype == np.float32 and rk.dtype == np.float32
    fin = ~np.isnan(z)
    assert (np.isnan(z) == np.isnan(rk)).all() and np.abs(z[fin]).max() <= 5.0 and np.abs(rk[fin]).max() <= 0.5
    if sp != 'train': assert fin.all()
    else: assert (fin == hasrow).all()
for sp in ('train', 'val', 'test'):
    save_new(FILES_DIR / sp / 'nx_rp3_z.npy', FILES[sp][0]); save_new(FILES_DIR / sp / 'nx_rp3_rk.npy', FILES[sp][1])
save_new(TWIN_DIR / 'nx_rp3_z.npy', FILES['twin'][0]); save_new(TWIN_DIR / 'nx_rp3_rk.npy', FILES['twin'][1])
P('files written (exclusive create): ' + ', '.join(f'{p} {os.path.getsize(p)} bytes md5 {md5(str(p))}' for p in
                                                   [FILES_DIR / sp / f'{n}.npy' for sp in ('train', 'val', 'test') for n in ('nx_rp3_z', 'nx_rp3_rk')] + [TWIN_DIR / f'{n}.npy' for n in ('nx_rp3_z', 'nx_rp3_rk')]))

# =========================================================== 6. shapes, NaN / zero shares per split and per day
P('=== per split and per day (slot = MSK day - 20702; Sep 6 = slot 0): feeds | rows | NaN share of nx_rp3_z (= of nx_rp3_rk) | of the non-NaN rows: raw rp3b05 = 0 | nx_rp3_z = 0 exactly | nx_rp3_rk = 0 exactly | feeds without any non-zero value (of has = 1 feeds) | mean / sd of nx_rp3_z on non-NaN rows ===')
def day_table(tag, slots_f, M, z, rk, raw0, pub=None):
    srow = np.repeat(slots_f, M); o = offs(M)
    anyz = np.maximum.reduceat((raw0 > 0).astype(np.int8), o) if len(M) else np.zeros(0)
    keys = [(f'slot {s:2d} (Sep {6 + s:2d})', srow == s, slots_f == s) for s in np.unique(slots_f)] + [('ALL', np.ones(len(srow), bool), np.ones(len(M), bool))]
    if pub is not None:
        keys += [('PUBLIC', np.repeat(pub, M), pub), ('PRIVATE', np.repeat(~pub, M), ~pub)]
    for nm, mr, mf in keys:
        zz = z[mr]; fin = ~np.isnan(zz); r0 = raw0[mr]
        if fin.any():
            P(f'  {tag} {nm}: {int(mf.sum())} | {int(mr.sum())} | {1 - fin.mean():.4f} | {np.mean(r0[fin] == 0):.4f} | {np.mean(zz[fin] == 0):.4f} | {np.mean(rk[mr][fin] == 0):.4f} | {np.mean(anyz[mf][~np.isnan(z[o[mf]])] == 0):.4f} | {zz[fin].mean():+.4f} / {zz[fin].std():.4f}')
        else:
            P(f'  {tag} {nm}: {int(mf.sum())} | {int(mr.sum())} | {1 - fin.mean():.4f} | - | - | - | - | -')
raw0_tr = np.nan_to_num(RAW[:, 0], nan=0.0)
day_table('train', tslot, At['M'], ztr, rtr, raw0_tr)
day_table('train HOLDOUT feeds', tslot[hold], At['M'][hold], ztr[np.repeat(ish, At['M'])], rtr[np.repeat(ish, At['M'])], raw0_tr[np.repeat(ish, At['M'])])
day_table('val', Av['slot'], Av['M'], FILES['val'][0], FILES['val'][1], NAT['val'][:, 0])
day_table('val STALE-VAL TWIN', Av['slot'], Av['M'], FILES['twin'][0], FILES['twin'][1], TW[:, 0])
day_table('test', Ae['slot'], Ae['M'], FILES['test'][0], FILES['test'][1], NAT['test'][:, 0], pub=Ae['ts'] < hc.PUB_TS)
P(f'BUILD VERDICT (C2 and re-derivation): {"PASS" if (c2ok and brok) else "FAIL"}')
P('DONE')
