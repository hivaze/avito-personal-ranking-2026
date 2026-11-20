"""nx3 verifier v03: first-principles walk (v_common._scan, my own code) on ALL has = 1 train feeds (1,470,065 feeds, 115,150,397 rows), in the lag view
of BUILD_SPEC 1-2: cut' = max(slot - L, 5), L = hz2_lag.npy; dense = train slots < cut' (latest feed per non-zero user + every user-0 feed);
sparse = the 1/37 train sample on slots >= cut', admitted iff timestamp < own timestamp; own non-zero user skipped; deg = dense postings, floor 1.
usage: v03_trainfull.py <part k> <of n>   (query feeds with fidx % n == k)  -> out/v03_trainfull_<k>of<n>.npz (compact, V0 only), label-free."""
from v_common import *
k_, n_ = int(sys.argv[1]), int(sys.argv[2])
P = Log(f'{OUT}/v03_trainfull_{k_}of{n_}.log')
TR = Split('train')
maxid = 111319574
assert int(TR.hist.max()) <= maxid
L = np.load(ROOT + '/nn_x/w12/stale/hz2/feats_L/train/hz2_lag.npy').astype(np.int64); assert len(L) == TR.n
cut = np.maximum(TR.slot - L, 5); has = TR.slot >= 6
spf = sparse_flag(TR.fid)
part = has & (np.arange(TR.n) % n_ == k_)
P(f'part {k_} of {n_}: {int(part.sum())} of {int(has.sum())} has = 1 feeds, rows {int(TR.M[part].sum())}')
acc = {}
tot_cnt = np.zeros(8, np.int64)
for c in np.unique(cut[has]):
    qf = np.nonzero(part & (cut == c))[0]
    if len(qf) == 0:
        continue
    dn = dense_members(TR, int(c)); sp_ = np.nonzero(spf & (TR.slot >= c))[0]
    assert TR.slot[dn].max() < c
    U = Universe([(TR, dn, True), (TR, sp_, False)], maxid)
    Q = Queries(TR, qf, np.full(len(qf), c), maxid)
    assert int(Q.row_item.max()) <= maxid
    t = time.time(); r = run(U, Q)
    P(f'cut {c}: dense seqs {len(dn)}, sparse seqs {len(sp_)}, events {len(U.ev)}; queries {Q.nq} feeds, {Q.nrow} rows; scan {time.time() - t:.0f}s; non-zero share {np.mean(r["p3a"][0] > 0):.4f}; '
      f'V0 pairs {int(r["cnt"][0])} (dense {int(r["cnt"][1])}, sparse {int(r["cnt"][2])}); AUDIT dense slot >= cut {int(r["cnt"][3])}, sparse ts >= own {int(r["cnt"][4])}, own non-zero user {int(r["cnt"][5])}, own feed {int(r["cnt"][6])}, any ts >= own {int(r["cnt"][7])}; '
      f'rows where V1 != V0 {int((r["p3a"][1] != r["p3a"][0]).sum())}, V2 != V0 {int((r["p3a"][2] != r["p3a"][0]).sum())}, V3 != V0 {int((r["p3a"][3] != r["p3a"][0]).sum())}; rows over the cap {int((r["nocc"][0] > CAP).sum())}')
    tot_cnt += r['cnt']
    d = dict(qf=qf, M=Q.M, rows=Q.rows_flat, rp3=r['rp3'][0], p3a=r['p3a'][0], newest_d=r['newest_d'], newest_s=r['newest_s'],
             fr_sparse=(r['tot_sparse'] / np.maximum(r['p3a'][0], 1e-300)).astype(np.float32), mean_src_ts=(r['wts'] / np.maximum(r['p3a'][0], 1e-300)),
             nocc=np.minimum(r['nocc'][0], 2 ** 31 - 1).astype(np.int32),
             dv1=r['p3a'][1] != r['p3a'][0], dv2=r['p3a'][2] != r['p3a'][0], dv3=r['p3a'][3] != r['p3a'][0])
    for kk, v in d.items():
        acc.setdefault(kk, []).append(v)
    del U, Q, r
o = {kk: np.concatenate(v) for kk, v in acc.items()}
o['cnt'] = tot_cnt
np.savez(f'{OUT}/v03_trainfull_{k_}of{n_}.npz', **o)
P(f'saved: feeds {len(o["qf"])}, rows {len(o["rows"])}; V0 audit counts {tot_cnt.tolist()}')
P('DONE')
