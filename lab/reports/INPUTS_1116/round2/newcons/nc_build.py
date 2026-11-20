"""newcons step 2: raw columns of the six constructions (PRELIST.txt) for
  hold   the 15,225 holdout feeds that have the LAG VIEW (cut' = max(slot - L, 5), L = hz2_lag.npy; dense = source_fidx(train, cut'), sparse = 1/37 sample on slots >= cut')
  val    natural (dense = all train, sparse = val + test feeds, admitted iff timestamp < own)  and the STALE-VAL TWIN (dense = train slots < slot - L', sparse = 1/37 + val)
  test   natural
exactly as nn_x/w12/nx3/nx3_build.py drives the kernel of record (same member lists, same kernel time, same degree rule), with
  - the causality audit of every admitted (row, sequence) pair, for the item postings and for the cell postings,
  - an independent numpy re-derivation on sampled feeds,
  - a driver check: column rp3_rec (kernel of record through my driver) against nx3's stored raw arrays (must be equal).
Label-free: no target array is opened. No full-train build. Writes NEW files under reports/INPUTS_1116/round2/newcons/out/ only.
NC_TAG=_smoke: one lag cut, 300 query feeds, 4 re-derived feeds, nothing of record written."""
from nc_common import *
TAGN = os.environ.get('NC_TAG', ''); SMOKE = TAGN == '_smoke'
os.makedirs(NRAW, exist_ok=True)
P = Log(NCO / f'build{TAGN}.log')
P(f'kernel of record md5 {md5(T3 + "/c09_build.py")}; nx3_common md5 {md5(str(NX3 / "nx3_common.py"))}; numba cache {nb.config.CACHE_DIR}')
D = Data(P); X = Ctx(D, P)
ft = D.F['train']; At, Av, Ae = D.A['train'], D.A['val'], D.A['test']
rng = np.random.default_rng(20261116)
AUD = {}; BR = {}


def add_aud(tag, c):
    AUD[tag] = AUD.get(tag, np.zeros(16, np.int64)) + c


def check_sample(tag, split, S, feeds_q, cut, out, offq, Mq, sel):
    pos = {int(f): i for i, f in enumerate(sel.tolist())}
    o = BR.get(tag, [0, 0, np.zeros(8), 0])
    for f in feeds_q:
        i = pos[int(f)]; b = brute2(D, X, split, S, int(f), int(cut)); k = out[offq[i]:offq[i] + Mq[i], :8]
        den = np.maximum(np.abs(b), 1e-300); den[:, [2, 6]] = np.maximum(np.abs(b[:, [2, 6]]), 1e-3)     # audience columns: float32 products, floor 1e-3
        rel = np.abs(k - b) / den; rel[(b == 0) & (k == 0)] = 0.0
        o[0] += 1; o[1] += len(b); o[2] = np.maximum(o[2], rel.max(0)); o[3] += int(((k != 0) != (b != 0)).sum())
    BR[tag] = o


def cov(tag, o):
    P(f'  [{tag}] rows {len(o)}: non-zero share ' + ' '.join(f'{n} {np.mean(o[:, j] != 0):.4f}' for j, n in enumerate(RAWCOLS)) + f' | cold rows (n_item = 0) {np.mean(o[:, 3] == 0):.4f}; no cell audience (n_cell = 0) {np.mean(o[:, 7] == 0):.4f}')


spf = hc.sparse_flag(At['fid']); tslot = At['slot']
# =========================================================== lag view of the holdout feeds
L = np.load(hc.HERE / 'feats_L/train/hz2_lag.npy').astype(np.int64); assert len(L) == At['n'] and L.min() >= 6 and L.max() <= 9
cut = np.maximum(tslot - L, 5); has = tslot >= 6
hold, _p = hc.holdout_feeds(ft); assert len(hold) == 30000
Mh = At['M'][hold]; offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64); NH = int(Mh.sum()); assert NH == 4630966
qall = hold[has[hold]]; assert len(qall) == 15225
P(f'[lag view] holdout feeds with the view {len(qall)}; cuts {dict(zip(*[x.tolist() for x in np.unique(cut[qall], return_counts=True)]))}')
HOLD = np.full((NH, 10), np.nan)
cuts = np.unique(cut[qall])
if SMOKE: cuts = cuts[:1]
for c in cuts:
    dense = c9.source_fidx(ft, int(hc.DAY0 + c)); sp_ = np.nonzero(spf & (tslot >= c))[0]; assert tslot[dense].max() < c
    S = Src2(D, [('train', dense, np.full(len(dense), -1, np.int64), True), ('train', sp_, At['ts'][sp_], False)], X.hist_cell, X.NCELL)
    qf = qall[cut[qall] == c]
    if SMOKE: qf = qf[:300]
    P(f'[lag view] cut {c}: query feeds {len(qf)}, dense seqs {len(dense)} (max slot {int(tslot[dense].max())}), sparse seqs {len(sp_)}; cells with a dense posting {int((S.degc_dense > 0).sum())} of {X.NCELL}')
    out, rows, cd, cc, offq, Mq, aud = run_all(D, X, 'train', S, qf, np.full(len(qf), c, np.int64))
    add_aud('hold lag view', aud)
    check_sample('hold lag view', 'train', S, rng.choice(qf, min(4, len(qf)), replace=False), c, out, offq, Mq, qf)
    pos = np.searchsorted(hold, qf); assert (hold[pos] == qf).all()
    HOLD[np.repeat(offh[pos] - offq, Mq) + np.arange(len(out))] = out
    cov(f'hold lag cut {c}', out)
    del S, out
hasrow_h = np.repeat(has[hold], Mh)
if not SMOKE:
    assert not np.isnan(HOLD[hasrow_h]).any() and np.isnan(HOLD[~hasrow_h]).all()
    # driver check against nx3's stored lag-view raw array
    Mt = At['M']; coff = np.cumsum(np.where(has, Mt, 0)) - np.where(has, Mt, 0)
    rec = np.load(NX3 / 'out/raw/train_lag_hasrows.npy', mmap_mode='r')
    pos = np.searchsorted(hold, qall); crow = np.repeat(coff[qall] - offh[pos], Mh[pos]) + (np.nonzero(hasrow_h)[0])
    ref = np.asarray(rec[crow, 0]); mine = HOLD[hasrow_h, 8]
    P(f'[driver check] hold lag view: rp3_rec against nn_x/w12/nx3/out/raw/train_lag_hasrows.npy on {len(ref)} rows: max abs diff {np.abs(ref - mine).max():.3e}, zero-pattern mismatches {int(((ref > 0) != (mine > 0)).sum())}')
    save_new(NRAW / 'hold_lag.npy', HOLD); save_new(NRAW / 'hold_has_row.npy', hasrow_h)
    cov('hold lag view, all has = 1 rows', HOLD[hasrow_h])

if not SMOKE:
    # =========================================================== natural val / test
    src_all = c9.source_fidx(ft, None)
    S = Src2(D, [('train', src_all, np.full(len(src_all), -1, np.int64), True), ('val', np.arange(Av['n']), Av['ts'], False), ('test', np.arange(Ae['n']), Ae['ts'], False)], X.hist_cell, X.NCELL)
    P(f'[natural] dense train seqs {S.n_dense}, sparse {S.n_sparse}; cells with a dense posting {int((S.degc_dense > 0).sum())} of {X.NCELL}; cell postings: median {np.median(S.ccnt[S.ccnt > 0]):.0f}, share of cells over the cap {np.mean(S.ccnt > CAP):.3f}')
    NAT = {}
    for sp in ('val', 'test'):
        a = D.A[sp]; sel = np.arange(a['n'], dtype=np.int64)
        out, rows, cd, cc, offq, Mq, aud = run_all(D, X, sp, S, sel, np.full(a['n'], 15, np.int64))
        assert (rows == np.arange(NROWS[sp])).all()
        add_aud(f'{sp} natural', aud)
        check_sample(f'{sp} natural', sp, S, rng.choice(sel, 5, replace=False), 15, out, offq, Mq, sel)
        ref = np.load(NX3 / f'out/raw/{sp}_nat.npy')[:, 0]
        P(f'[driver check] {sp} natural: rp3_rec against nx3 out/raw/{sp}_nat.npy: max abs diff {np.abs(ref - out[:, 8]).max():.3e}, zero-pattern mismatches {int(((ref > 0) != (out[:, 8] > 0)).sum())}')
        save_new(NRAW / f'{sp}_nat.npy', out); cov(f'{sp} natural', out); NAT[sp] = out
        if sp == 'test':
            pub = np.repeat(a['ts'] < hc.PUB_TS, a['M']); cov('test PUBLIC', out[pub]); cov('test PRIVATE', out[~pub])
    del S
    # =========================================================== stale-val twin
    vL = np.asarray(hc.assign_lags(Av['fid'])[1]).astype(np.int64); vcut = Av['slot'] - vL
    P(f'[twin] val feeds by cut {dict(zip(*[x.tolist() for x in np.unique(vcut, return_counts=True)]))}')
    TW = np.full((NROWS['val'], 10), np.nan)
    for c in np.unique(vcut):
        dense = c9.source_fidx(ft, int(hc.DAY0 + c)); sp_ = np.nonzero(spf & (tslot >= c))[0]; assert tslot[dense].max() < c
        S = Src2(D, [('train', dense, np.full(len(dense), -1, np.int64), True), ('train', sp_, At['ts'][sp_], False), ('val', np.arange(Av['n']), Av['ts'], False)], X.hist_cell, X.NCELL)
        qf = np.nonzero(vcut == c)[0]
        out, rows, cd, cc, offq, Mq, aud = run_all(D, X, 'val', S, qf, np.full(len(qf), c, np.int64))
        add_aud('val stale-val twin', aud)
        check_sample('val stale-val twin', 'val', S, rng.choice(qf, min(2, len(qf)), replace=False), c, out, offq, Mq, qf)
        TW[rows] = out
        P(f'[twin] cut {c}: {len(qf)} val feeds, dense seqs {len(dense)}, sparse train seqs {len(sp_)}')
        del S, out
    assert not np.isnan(TW).any()
    ref = np.load(NX3 / 'out/raw/val_twin.npy')[:, 0]
    P(f'[driver check] val twin: rp3_rec against nx3 out/raw/val_twin.npy: max abs diff {np.abs(ref - TW[:, 8]).max():.3e}, zero-pattern mismatches {int(((ref > 0) != (TW[:, 8] > 0)).sum())}')
    save_new(NRAW / 'val_twin.npy', TW); cov('val stale-val twin', TW)

P('=== causality audit, every admitted (row, sequence) pair; first block = item postings (N4, N6; N5 uses the kernel of record on the same members), second = cell postings (N1, N2, N3) ===')
ok = True
for tag, c in AUD.items():
    P(f'  {tag} | ITEM postings: ' + '; '.join(f'{n} {int(v)}' for n, v in zip(AUDN, c[:8])))
    P(f'  {tag} | CELL postings: ' + '; '.join(f'{n} {int(v)}' for n, v in zip(AUDN, c[8:])))
    ok &= bool((c[3:8] == 0).all() and (c[11:16] == 0).all())
P(f'  AUDIT VERDICT (every VIOLATION count 0): {"PASS" if ok else "FAIL"}')
P('=== independent numpy re-derivation on sampled feeds (max relative difference per kern column; zero-pattern mismatches) ===')
bok = True
for tag, (n, r, m, zm) in BR.items():
    P(f'  {tag}: {n} feeds, {r} rows; ' + ' '.join(f'{nm} {v:.1e}' for nm, v in zip(RAWCOLS[:8], m)) + f'; zero-pattern mismatches {zm}')
    bok &= bool(m[[0, 1, 3, 4, 5, 7]].max() <= 1e-9 and m[[2, 6]].max() <= 1e-4 and zm == 0)
P(f'  RE-DERIVATION VERDICT (1e-9 relative; 1e-4 with a floor of 1e-3 for the two audience columns, which are sums of float32 products): {"PASS" if bok else "FAIL"}')
P('DONE')
