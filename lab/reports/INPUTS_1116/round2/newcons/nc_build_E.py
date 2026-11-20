"""newcons follow-up (FOLLOWUP.txt, 2026-11-16T05:14:30Z): third witness E = the holdout feeds WITHOUT the lag view (slots 1-5), FRESH view:
dense = c9.source_fidx(train, own slot), sparse = the 1/37 sample on slots >= own slot admitted iff timestamp < own timestamp (the lag-view driver with L = 0).
Same kernels, audit and re-derivation as nc_build.py. Label-free. Writes out/raw/hold_early_fresh.npy (4,630,966 x 10, NaN outside the E feeds)."""
from nc_common import *
P = Log(NCO / 'build_E.log')
D = Data(P); X = Ctx(D, P)
ft = D.F['train']; At = D.A['train']; rng = np.random.default_rng(20261117)
spf = hc.sparse_flag(At['fid']); tslot = At['slot']
hold, _p = hc.holdout_feeds(ft); Mh = At['M'][hold]; offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64); NH = int(Mh.sum())
P('holdout feeds by slot: ' + str(dict(zip(*[x.tolist() for x in np.unique(tslot[hold], return_counts=True)]))))
E = np.full((NH, 10), np.nan); AUD = np.zeros(16, np.int64); BRm = np.zeros(8); nb_ = 0; zm = 0
for c in range(1, 6):
    dense = c9.source_fidx(ft, int(hc.DAY0 + c)); sp_ = np.nonzero(spf & (tslot >= c))[0]; assert tslot[dense].max() < c
    S = Src2(D, [('train', dense, np.full(len(dense), -1, np.int64), True), ('train', sp_, At['ts'][sp_], False)], X.hist_cell, X.NCELL)
    qf = hold[tslot[hold] == c]
    out, rows, cd, cc, offq, Mq, aud = run_all(D, X, 'train', S, qf, np.full(len(qf), c, np.int64)); AUD += aud
    pos = {int(f): i for i, f in enumerate(qf.tolist())}
    for f in rng.choice(qf, 3, replace=False):
        i = pos[int(f)]; b = brute2(D, X, 'train', S, int(f), c); k = out[offq[i]:offq[i] + Mq[i], :8]
        den = np.maximum(np.abs(b), 1e-300); den[:, [2, 6]] = np.maximum(np.abs(b[:, [2, 6]]), 1e-3); rel = np.abs(k - b) / den; rel[(b == 0) & (k == 0)] = 0.0
        BRm = np.maximum(BRm, rel.max(0)); nb_ += 1; zm += int(((k != 0) != (b != 0)).sum())
    p_ = np.searchsorted(hold, qf); E[np.repeat(offh[p_] - offq, Mq) + np.arange(len(out))] = out
    P(f'[fresh view] slot {c}: query feeds {len(qf)}, dense seqs {len(dense)}, sparse seqs {len(sp_)}; non-zero share ' + ' '.join(f'{n} {np.mean(out[:, j] != 0):.4f}' for j, n in enumerate(RAWCOLS)) + f' | cold rows {np.mean(out[:, 3] == 0):.4f}')
    del S, out
P('audit ITEM postings: ' + '; '.join(f'{n} {int(v)}' for n, v in zip(AUDN, AUD[:8])))
P('audit CELL postings: ' + '; '.join(f'{n} {int(v)}' for n, v in zip(AUDN, AUD[8:])))
P(f'AUDIT VERDICT: {"PASS" if (AUD[3:8] == 0).all() and (AUD[11:16] == 0).all() else "FAIL"}')
P(f're-derivation on {nb_} feeds: max relative difference ' + ' '.join(f'{nm} {v:.1e}' for nm, v in zip(RAWCOLS[:8], BRm)) + f'; zero-pattern mismatches {zm}; VERDICT {"PASS" if BRm[[0, 1, 3, 4, 5, 7]].max() <= 1e-9 and BRm[[2, 6]].max() <= 1e-4 and zm == 0 else "FAIL"}')
c9h = np.load(C9 / 'c09_hold.npz')['rp3b05']; er = ~np.isnan(E[:, 0])
P(f'information: my fresh rp3_rec against the stored cache/c09_hold.npz (cut = own day, NO sparse members, degree with all postings) on the E rows: corr {np.corrcoef(E[er, 8], c9h[er])[0, 1]:.4f}, non-zero share mine {np.mean(E[er, 8] > 0):.4f} stored {np.mean(c9h[er] > 0):.4f}')
save_new(NRAW / 'hold_early_fresh.npy', E)
P('DONE')
