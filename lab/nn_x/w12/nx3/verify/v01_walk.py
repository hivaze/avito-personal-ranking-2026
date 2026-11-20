"""nx3 verifier v01: first-principles walk (v_common._scan) on
  natural : ALL val feeds and ALL test feeds        (universe: train dense slots 0-14 + every val feed + every test feed as sparse members)
  train   : a random sample of lag-view train feeds spread over all has = 1 days + a sample of lag-view holdout feeds + user-0 feeds + all stragglers
            (universes per cut' = 5, 6, 7, 8: train dense slots < cut' + the 1/37 train sample on slots >= cut')
  twin    : ALL val feeds in the stale-val twin view (universes per cut = slot - L: train dense slots < cut + 1/37 train sample on slots >= cut + every val feed)
usage: v01_walk.py natural | train | twin        -> out/v01_<mode>.npz, out/v01_<mode>.log   (label-free: no target array is opened)"""
from v_common import *
mode = sys.argv[1]
TAG = os.environ.get('V_TAG', '')
P = Log(f'{OUT}/v01_{mode}{TAG}.log')
TR = Split('train'); VA = Split('val'); TE = Split('test')
maxid = int(max(TR.hist.max(), VA.hist.max(), TE.hist.max(), np.asarray(VA.cand).max(), np.asarray(TE.cand).max(), 111319574))
tmx = 0
for a in range(0, NROWS['train'], 50_000_000):
    tmx = max(tmx, int(np.asarray(TR.cand[a:a + 50_000_000]).max()))
maxid = max(maxid, tmx)
P(f'mode {mode}; feeds train {TR.n} val {VA.n} test {TE.n}; max item id {maxid}; distinct timestamps train {len(np.unique(TR.ts))} val {len(np.unique(VA.ts))} test {len(np.unique(TE.ts))}; '
  f'timestamp granularity (gcd of differences) {int(np.gcd.reduce(np.diff(np.unique(TR.ts))))} s')
P(f'slots: train {dict(zip(*[x.tolist() for x in np.unique(TR.slot, return_counts=True)]))}')
P(f'slots: val {dict(zip(*[x.tolist() for x in np.unique(VA.slot, return_counts=True)]))}; test {dict(zip(*[x.tolist() for x in np.unique(TE.slot, return_counts=True)]))}')
assert TR.ts.max() < VA.ts.min() and VA.ts.max() < TE.ts.min()
P(f'time order of the splits: train max ts {TR.ts.max()} < val min ts {VA.ts.min()}; val max ts {VA.ts.max()} < test min ts {TE.ts.min()}')
spf = sparse_flag(TR.fid)
sys.path.insert(0, ROOT + '/nn_x/w12/stale/hz2')
import hz2_common as hc           # constants / lag hash of record (read-only): used for the twin lags and to cross-check my copies
assert (spf == hc.sparse_flag(TR.fid)).all() and hc.DAY0 == DAY0 and hc.PUB_TS == PUB_TS and hc.MSK == MSK
P(f'1/37 sample: {int(spf.sum())} train feeds')
res = {}


def add(tag, U, Q, extra):
    t = time.time(); r = run(U, Q)
    P(f'  [{tag}] queries {Q.nq} feeds, {Q.nrow} rows; scan {time.time() - t:.0f}s; V0 non-zero share {np.mean(r["p3a"][0] > 0):.4f}; V0 admitted pairs {int(r["cnt"][0])} (dense {int(r["cnt"][1])}, sparse {int(r["cnt"][2])}); '
      f'AUDIT of V0 pairs: dense with slot >= cut {int(r["cnt"][3])}, sparse with ts >= own {int(r["cnt"][4])}, own non-zero user {int(r["cnt"][5])}, own feed {int(r["cnt"][6])}, any member with ts >= own {int(r["cnt"][7])}; '
      f'rows where V1 (own user admitted) != V0: {int((r["p3a"][1] != r["p3a"][0]).sum())}; V2 (sparse at/after own time admitted) != V0: {int((r["p3a"][2] != r["p3a"][0]).sum())}; V3 (user 0 linked) != V0: {int((r["p3a"][3] != r["p3a"][0]).sum())}; '
      f'rows with > {CAP} admitted occurrences: {int((r["nocc"][0] > CAP).sum())}')
    d = dict(rows=Q.rows_flat, q=Q.row_q, qf=Q.qf, M=Q.M, q_ts=Q.q_ts, q_user=Q.q_user, q_cut=Q.q_cut, rp3=r['rp3'], p3a=r['p3a'], nseq=r['nseq'][0], nocc=r['nocc'][0], newest_d=r['newest_d'], newest_s=r['newest_s'],
             tot_sparse=r['tot_sparse'], wts=r['wts'], cnt=r['cnt'], deg_c=r['deg_c'])
    d.update(extra)
    for k, v in d.items():
        res.setdefault(tag.split('|')[0], {}).setdefault(k, []).append(v)


if mode == 'natural':
    dn = dense_members(TR, 15)
    assert TR.slot.max() == 14
    P(f'natural universe: dense train seqs {len(dn)} (non-zero users {int((TR.user[dn] != 0).sum())}, user-0 feeds {int((TR.user[dn] == 0).sum())} of {int((TR.user == 0).sum())}); sparse val {VA.n} + test {TE.n}')
    U = Universe([(TR, dn, True), (VA, np.arange(VA.n), False), (TE, np.arange(TE.n), False)], maxid)
    P(f'  events {len(U.ev)} (dense {U.nde})')
    for S in (VA, TE):
        Q = Queries(S, np.arange(S.n), np.full(S.n, 15), maxid)
        add(S.name, U, Q, {})
elif mode == 'train':
    L = np.load(ROOT + '/nn_x/w12/stale/hz2/feats_L/train/hz2_lag.npy').astype(np.int64)
    assert len(L) == TR.n
    P(f'lag draw hz2_lag.npy: counts {dict(zip(*[x.tolist() for x in np.unique(L, return_counts=True)]))}')
    La = np.asarray(hc.assign_lags(TR.fid)[1]).astype(np.int64)
    P(f'lag draw against hz2_common.assign_lags(feed_id) (a hash of feed_id, label-free): equal on {int((L == La).sum())} of {TR.n} feeds')
    cut = np.maximum(TR.slot - L, 5); has = TR.slot >= 6
    Mt = TR.M
    pr = np.load(ROOT + '/eda/labels_position/out/train_prandom.npz'); assert (pr['fidx'] == np.arange(TR.n)).all()
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[Mt < 50] = 0.0
    hold = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (Mt >= 50))[0], 30000, replace=False))
    hm = np.load(ROOT + '/nn_x/w5/preds/holdout/holdout_meta.npy'); assert (hm[0] == Mt[hold]).all()
    ish = np.zeros(TR.n, bool); ish[hold] = True
    Lh = La[hold]; P(f'holdout feeds: lag draw equal to assign_lags on {int((L[hold] == Lh).sum())} of 30000; lag counts on holdout feeds {dict(zip(*[x.tolist() for x in np.unique(L[hold], return_counts=True)]))}')
    P(f'has = 1 feeds {int(has.sum())} (rows {int(Mt[has].sum())}); holdout with has = 1 {int((has & ish).sum())}; cut counts {dict(zip(*[x.tolist() for x in np.unique(cut[has], return_counts=True)]))}')
    rng = np.random.default_rng(777001)
    samp = np.zeros(TR.n, np.int8)        # 1 random, 2 holdout sample, 3 user-0 extra, 4 straggler
    for s in (6, 7, 8, 11, 12, 13):
        c_ = np.nonzero(has & (TR.slot == s) & ~ish)[0]; samp[rng.choice(c_, 1500, replace=False)] = 1
        c_ = np.nonzero(has & (TR.slot == s) & ish)[0]; samp[rng.choice(c_, 600, replace=False)] = 2
        c_ = np.nonzero(has & (TR.slot == s) & ~ish & (TR.user == 0) & (samp == 0))[0]; samp[rng.choice(c_, 100, replace=False)] = 3
    samp[np.nonzero(has & ((TR.slot == 9) | (TR.slot == 14)))[0]] = 4
    P(f'train sample: random {int((samp == 1).sum())}, holdout {int((samp == 2).sum())}, user-0 extra {int((samp == 3).sum())}, stragglers (slots 9, 14) {int((samp == 4).sum())}; '
      f'by slot {dict(zip(*[x.tolist() for x in np.unique(TR.slot[samp > 0], return_counts=True)]))}; by cut {dict(zip(*[x.tolist() for x in np.unique(cut[samp > 0], return_counts=True)]))}; user-0 query feeds {int((TR.user[samp > 0] == 0).sum())}')
    for c in np.unique(cut[has]):
        dn = dense_members(TR, int(c)); sp_ = np.nonzero(spf & (TR.slot >= c))[0]
        assert TR.slot[dn].max() < c and len(np.unique(TR.user[dn][TR.user[dn] != 0])) == int((TR.user[dn] != 0).sum())
        assert int((TR.user[dn] == 0).sum()) == int(((TR.user == 0) & (TR.slot < c)).sum())
        U = Universe([(TR, dn, True), (TR, sp_, False)], maxid)
        qf = np.nonzero((samp > 0) & (cut == c))[0]
        P(f'cut {c}: dense seqs {len(dn)} (max slot {int(TR.slot[dn].max())}; user-0 {int((TR.user[dn] == 0).sum())}), sparse seqs {len(sp_)} (slots {int(TR.slot[sp_].min())}-{int(TR.slot[sp_].max())}), events {len(U.ev)}')
        Q = Queries(TR, qf, np.full(len(qf), c), maxid)
        add('train', U, Q, dict(samp=samp[qf], slot=TR.slot[qf], hold=ish[qf], prand=p[qf]))
        del U, Q
elif mode == 'twin':
    vL = np.asarray(hc.assign_lags(VA.fid)[1]).astype(np.int64); vcut = VA.slot - vL
    P(f'twin: val feeds by cut {dict(zip(*[x.tolist() for x in np.unique(vcut, return_counts=True)]))}; lags {dict(zip(*[x.tolist() for x in np.unique(vL, return_counts=True)]))}')
    for c in np.unique(vcut):
        dn = dense_members(TR, int(c)); sp_ = np.nonzero(spf & (TR.slot >= c))[0]
        assert TR.slot[dn].max() < c
        U = Universe([(TR, dn, True), (TR, sp_, False), (VA, np.arange(VA.n), False)], maxid)
        qf = np.nonzero(vcut == c)[0]
        P(f'cut {c}: dense seqs {len(dn)} (max slot {int(TR.slot[dn].max())}), sparse train seqs {len(sp_)}, events {len(U.ev)}')
        Q = Queries(VA, qf, np.full(len(qf), c), maxid)
        add('twin', U, Q, dict(slot=VA.slot[qf]))
        del U, Q
for tag, d in res.items():
    o = {}
    nq0 = 0
    qs = []
    for i, qv in enumerate(d['q']):
        qs.append(qv + nq0); nq0 += len(d['qf'][i])
    for k, v in d.items():
        if k == 'q': o[k] = np.concatenate(qs)
        elif k == 'cnt': o[k] = np.sum(v, 0)
        elif k in ('rp3', 'p3a'): o[k] = np.concatenate(v, 1)
        else: o[k] = np.concatenate(v)
    np.savez(f'{OUT}/v01_{mode}_{tag}{TAG}.npz', **o)
    P(f'saved out/v01_{mode}_{tag}.npz: feeds {len(o["qf"])}, rows {len(o["rows"])}; total V0 audit counts {o["cnt"].tolist()}')
P('DONE')
