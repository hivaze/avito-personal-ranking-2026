"""nx3 mechanics test before the build of record (label-free; disclosed in the report): one lag-view index (cut 7), 400 query feeds, the kernel of record,
the audit, the numpy re-derivation on 8 feeds, the forms. Also: the audit must COUNT violations when the driver is deliberately wrong (sparse members
handed to the kernel with time -1, i.e. always admitted) and the re-derivation must then disagree. Output out/minitest.txt."""
from nx3_common import *
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
from lens import feed_z, feed_rk
sys.argv = _argv
P = Log(OUT / f'minitest{TAG}.txt')
D = Data(P)
ft = D.F['train']; At = D.A['train']
L = np.load(hc.HERE / 'feats_L/train/hz2_lag.npy').astype(np.int64); tslot = At['slot']; cut = np.maximum(tslot - L, 5); has = tslot >= 6
spf = hc.sparse_flag(At['fid']); c = 7
dense = c9.source_fidx(ft, int(hc.DAY0 + c)); sp_ = np.nonzero(spf & (tslot >= c))[0]
qf = np.nonzero(has & (cut == c))[0]; rng = np.random.default_rng(1); qf = np.sort(rng.choice(qf, 400, replace=False))
for mode in ('correct', 'WRONG (sparse members always admitted)'):
    kt = At['ts'][sp_] if mode == 'correct' else np.full(len(sp_), -1, np.int64)
    S = Src(D, [('train', dense, np.full(len(dense), -1, np.int64), True), ('train', sp_, kt, False)])
    o, rows, cd, offq, Mq = run_kernel(D, 'train', S.idx, S.deg_floor1(), qf, np.zeros(len(qf), np.int64), At['ts'][qf])
    cnt = run_audit(D, 'train', S, qf, At['ts'][qf], np.full(len(qf), c, np.int64), offq, Mq, cd)
    P(f'[{mode}] dense {len(dense)} sparse {len(sp_)}; rows {len(o)}, non-zero share {np.mean(o[:, 1] > 0):.4f}; audit: ' + '; '.join(f'{n} {int(v)}' for n, v in zip(ANAMES, cnt)))
    m0 = m1 = 0.0; zm = 0; t1 = time.time()
    for i in range(8):
        b = brute(D, 'train', S, int(qf[i * 50]), c); k = o[offq[i * 50]:offq[i * 50] + Mq[i * 50], :2]
        rel = np.abs(k - b) / np.maximum(np.abs(b), 1e-300); rel[(b == 0) & (k == 0)] = 0.0
        m0 = max(m0, float(rel[:, 0].max())); m1 = max(m1, float(rel[:, 1].max())); zm += int(((k > 0) != (b > 0)).sum())
    P(f'[{mode}] re-derivation on 8 feeds ({time.time() - t1:.1f} s): max rel diff rp3b05 {m0:.3e}, p3a {m1:.3e}, zero-pattern mismatches {zm}')
    z, rk = forms_of(o[:, 0], offq, Mq, 2e5, feed_z, feed_rk)
    P(f'[{mode}] forms: z dtype {z.dtype} range {z.min():.3f}..{z.max():.3f} mean {z.mean():+.4f}; rk range {rk.min():.3f}..{rk.max():.3f}; feeds constant {int((np.maximum.reduceat(np.abs(z), offq) == 0).sum())}')
    del S
P('DONE')
