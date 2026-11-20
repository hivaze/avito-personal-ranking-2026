"""hz2_review: INDEPENDENT recount of the hz2 train view from raw data (data/flat) on a feed sample.
Reviewer's own code: imports nothing from nn_x/w12/stale/hz2 (no kernel, no cache, no mask file). The rule is implemented from the text of
nn_x/w12/stale/hz2/DECLARATION.txt (first block). Only inputs of the study that are read: the per-feed lag file (hz2_lag.npy) and the view columns
that are being checked. Label-free: no target file is opened.
Also validates the reading of the STORED rule on val / test feeds against nn/feat_cache/{val,test}.
Usage: CUDA_VISIBLE_DEVICES="" python recount.py <n_feeds_per_day> <seed>
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
import sys, time, json
sys.dont_write_bytecode = True
import numpy as np
import polars as pl

R = '<repo>'
FLAT = f'{R}/data/flat'
HZ2 = f'{R}/nn_x/w12/stale/hz2'
HERE = f'{R}/nn_x/w12/stale/hz2_review'
MSK, DAY = 3 * 3600, 86400
NPER = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


F = {s: pl.read_parquet(f'{FLAT}/{s}/feeds.parquet') for s in ('train', 'val', 'test')}
nF = {s: len(F[s]) for s in F}
ts = {s: F[s]['timestamp'].to_numpy().astype(np.int64) for s in F}
usr = {s: F[s]['user_id'].to_numpy().astype(np.int64) for s in F}
Mf = {s: F[s]['M'].to_numpy().astype(np.int64) for s in F}
Hf = {s: F[s]['H'].to_numpy().astype(np.int64) for s in F}
coff = {s: F[s]['cand_off'].to_numpy().astype(np.int64) for s in F}
hoff = {s: F[s]['hist_off'].to_numpy().astype(np.int64) for s in F}
day = {s: (ts[s] + MSK) // DAY for s in F}
DAY0 = int(day['train'].min())           # first train day = slot 0
slot = {s: (day[s] - DAY0).astype(np.int64) for s in F}
u, c = np.unique(slot['train'], return_counts=True)
lg('DAY0', DAY0, 'train feeds per slot', dict(zip(u.tolist(), c.tolist())))
DENSE = [int(a) for a, b in zip(u, c) if b > 1000]
lg('dense train slots', DENSE)
for s in ('val', 'test'):
    uu, cc = np.unique(slot[s], return_counts=True)
    lg(s, 'feeds per slot', dict(zip(uu.tolist(), cc.tolist())))
CS = 14                                   # Sep 20 = dense cutoff slot of val / test rows (first slot after the last dense train day 13)
DPAT = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -3, -2, -1]
PAT = [2, 4, 6, 7, 8, 9]
assert sorted(CS + o for o in DPAT) == DENSE, 'dense pattern relative to Sep 20 is not the set of dense train days'
fid_tr = F['train']['feed_id'].to_numpy()
spflag = (((fid_tr.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)) >> np.uint64(40)) % np.uint64(37)) == 0     # feat_new/hz_common.sparse_flag
lg('sparse-sample train feeds', int(spflag.sum()), 'per dense slot', {d: int(spflag[slot['train'] == d].sum()) for d in DENSE})

LAG = np.load(f'{HZ2}/feats/train/hz2_lag.npy').astype(np.int64)
PARTPUB = np.load(f'{HZ2}/feats/train/hz2_part_pub.npy')
assert len(LAG) == nF['train']
# ------------------------------------------------------------------ lag file against the test lag distribution (own computation)
PUB_TS = 1790497920
tl = slot['test'] - CS
pub = ts['test'] < PUB_TS
lg('test lag counts public', dict(zip(*[x.tolist() for x in np.unique(tl[pub], return_counts=True)])), 'private', dict(zip(*[x.tolist() for x in np.unique(tl[~pub], return_counts=True)])),
   'public share', round(float(pub.mean()), 4))
for nm, sel in (('public-like', PARTPUB), ('private-like', ~PARTPUB)):
    uu, cc = np.unique(LAG[sel], return_counts=True)
    lg(f'view lag shares {nm}: ' + ' '.join(f'{a}:{b / sel.sum():.4f}' for a, b in zip(uu, cc)), '| feeds', int(sel.sum()), 'share', round(float(sel.mean()), 4))
uu, cc = np.unique(LAG, return_counts=True)
lg('view lag counts', dict(zip(uu.tolist(), cc.tolist())))
# lag must not depend on the day or on M (hash of feed_id)
lg('mean lag by slot', {d: round(float(LAG[slot['train'] == d].mean()), 4) for d in DENSE})

# ------------------------------------------------------------------ sample
rng = np.random.default_rng(SEED)
samp_tr = np.sort(np.concatenate([rng.choice(np.flatnonzero(slot['train'] == d), NPER, replace=False) for d in DENSE]))
samp_va = np.sort(rng.choice(nF['val'], 250, replace=False))
samp_te = np.sort(rng.choice(nF['test'], 400, replace=False))
SAMP = {'train': samp_tr, 'val': samp_va, 'test': samp_te}


def rows_of(s, sel):
    return np.concatenate([np.arange(coff[s][f], coff[s][f] + Mf[s][f]) for f in sel])


ROWS = {s: rows_of(s, SAMP[s]) for s in SAMP}
CIT = {s: np.asarray(np.load(f'{FLAT}/{s}/cand_item_id.npy', mmap_mode='r')[ROWS[s]]).astype(np.int64) for s in SAMP}
F4 = {s: np.asarray(np.load(f'{FLAT}/{s}/cand_f4.npy', mmap_mode='r')[ROWS[s]]).astype(np.float64) for s in SAMP}
items = np.unique(np.concatenate(list(CIT.values())))
lg('sample feeds', {s: len(SAMP[s]) for s in SAMP}, 'rows', {s: len(ROWS[s]) for s in SAMP}, 'distinct items', len(items))
NI = int(max(items.max(), max(int(np.load(f'{FLAT}/{s}/hist_item_id.npy', mmap_mode='r').max()) for s in F))) + 1
look = np.full(NI, -1, np.int64); look[items] = np.arange(len(items))

# ------------------------------------------------------------------ events of the sampled items in every history (train, val, test)
gbase = {'train': 0, 'val': nF['train'], 'test': nF['train'] + nF['val']}
G_ts = np.concatenate([ts[s] for s in F]); G_user = np.concatenate([usr[s] for s in F]); G_slot = np.concatenate([slot[s] for s in F])
G_istr = np.zeros(len(G_ts), bool); G_istr[:nF['train']] = True
G_sp = np.ones(len(G_ts), bool); G_sp[:nF['train']] = spflag          # sparse-source snapshots: train 1/37 sample, every val / test feed
ev_k, ev_f, ev_et = [], [], []
for s in F:
    hi = np.load(f'{FLAT}/{s}/hist_item_id.npy')
    k = look[hi]
    m = np.flatnonzero(k >= 0)
    hf = np.load(f'{FLAT}/{s}/hist_fidx.npy', mmap_mode='r')
    ev_k.append(k[m]); ev_f.append(np.asarray(hf[m]).astype(np.int64) + gbase[s])
    ev_et.append(np.asarray(np.load(f'{FLAT}/{s}/hist_event_type.npy', mmap_mode='r')[m]).astype(np.int64))
    lg(s, 'history events', len(hi), 'matched', len(m))
    del hi, k
ev_k = np.concatenate(ev_k); ev_f = np.concatenate(ev_f); ev_et = np.concatenate(ev_et)
o = np.argsort(ev_k, kind='stable')
ev_k, ev_f, ev_et = ev_k[o], ev_f[o], ev_et[o]
ev_start = np.searchsorted(ev_k, np.arange(len(items) + 1))
E_slot = G_slot[ev_f]; E_tr = G_istr[ev_f]; E_sp = G_sp[ev_f]; E_ts = G_ts[ev_f]; E_user = G_user[ev_f]

# later-feed slot mask per TRAIN feed (same non-zero user, later in (ts, fidx) order), own computation
order = np.lexsort((np.arange(nF['train']), ts['train'], usr['train']))
uo = usr['train'][order]
run = np.cumsum(np.r_[True, uo[1:] != uo[:-1]]) - 1
pos = np.arange(nF['train'])
so = slot['train'][order]
later = np.zeros(nF['train'], np.int64)
for sl in range(15):
    mm = so == sl
    if not mm.any():
        continue
    last = np.full(run[-1] + 1, -1, np.int64)
    np.maximum.at(last, run[mm], pos[mm])
    has = (last[run] > pos) & (uo != 0)
    later[order[has]] |= (1 << sl)
E_later = np.zeros(len(ev_f), np.int64); E_later[E_tr] = later[ev_f[E_tr]]
lg('train feeds with a later train feed of the same non-zero user', int((later != 0).sum()), '| user-0 train feeds', int((usr['train'] == 0).sum()),
   'val', int((usr['val'] == 0).sum()), 'test', int((usr['test'] == 0).sum()))

# impressions of the sampled items per train slot
ci = np.load(f'{FLAT}/train/cand_item_id.npy', mmap_mode='r')
cf = np.load(f'{FLAT}/train/cand_fidx.npy', mmap_mode='r')
IM = np.zeros((len(items), 15), np.int64)
STEP = 40_000_000
for a in range(0, len(ci), STEP):
    kk = look[np.asarray(ci[a:a + STEP])]
    m = kk >= 0
    np.add.at(IM, (kk[m], slot['train'][np.asarray(cf[a:a + STEP])[m]]), 1)
lg('impressions counted', int(IM.sum()))

kn = np.load(f'{R}/data/feats/life_f4_age_knots.npy').astype(np.float64)


def creation_slot(f4, t):
    """slot of the listing's creation day from f4 (feat_new/hz_common.age_hours knots); capped f4 -> very old."""
    if f4 >= 63 / 64:
        return -10 ** 6
    uu_ = f4 * 64.0
    k = int(min(max(np.floor(uu_), 1), 62))
    w = uu_ - k
    a = kn[k] * (1 - w) + kn[k + 1] * w
    return int(np.floor((t - 3600.0 * a + MSK) / DAY)) - DAY0


def count_row(k, Wset, Sset, r, t, uq, own_gf, causal_ts_sparse=True):
    """Counts of one row from the event list of item k. Returns dict with it_hist all / et1, seq pop parts, and audit numbers."""
    a, b = ev_start[k], ev_start[k + 1]
    sl = E_slot[a:b]; tr = E_tr[a:b]; sp = E_sp[a:b]; et = ev_et[a:b]; tt = E_ts[a:b]; uu_ = E_user[a:b]; ff = ev_f[a:b]
    Wl = np.zeros(24, bool); Wl[list(Wset)] = True
    Sl = np.zeros(24, bool); Sl[list(Sset - Wset)] = True
    inW = tr & Wl[sl]
    inS = sp & Sl[sl] & ~inW
    own = (ff == own_gf) & ~(inW | inS)                       # own snapshot, if not yet counted
    cnt = inW | inS | own
    lat = cnt & (uq != 0) & (uu_ == uq) & (tt > t)            # counted snapshots of the same non-zero user with a later timestamp: removed
    wgt = np.where(inW, r, 1.0)
    keep = cnt & ~lat
    all_ = float(wgt[keep].sum()); e1 = float(wgt[keep & (et == 1)].sum())
    # seq_cv_pop: dense members = latest feed of its user among the W days (user 0 always); sparse members ts < t; own user skipped
    Wbits = sum(1 << int(x) for x in Wset if x < 15)
    dm = inW & ((E_later[a:b] & Wbits) == 0)
    sm = inS & (tt < t)
    ncol = int(dm.sum() + inS.sum())                          # collected before the user / ts filters (the 3000 cap applies here)
    if uq != 0:
        dm = dm & (uu_ != uq); sm = sm & (uu_ != uq)
    pop = min(r * float(dm.sum()) + float(sm.sum()), 3000.0)
    cs = sl[keep]
    return dict(all=all_, e1=e1, pop=pop, ncol=ncol, n_dense=int((inW & keep).sum()), n_sparse=int((inS & keep).sum()),
                max_slot=int(cs.max()) if len(cs) else -99, max_dense_slot=int(sl[inW & keep].max()) if (inW & keep).any() else -99,
                n_later_ts=int((keep & (tt > t)).sum()), n_own_user=int((keep & (uq != 0) & (uu_ == uq)).sum()),
                n_user0_later_same_day=int((keep & (uu_ == 0) & (tt > t)).sum()), n_before_creation=0)


# ------------------------------------------------------------------ 1. stored rule on val / test rows
STORED = {}
for s in ('val', 'test'):
    res = dict(all=[], e1=[], pop=[], ncol=[], xp=[])
    Wset = set(range(15)); Sset = set(range(15, 24))      # stored rule: every train snapshot (stragglers of slots 9 / 14 included) + every val / test snapshot
    fo = np.repeat(np.arange(len(SAMP[s])), Mf[s][SAMP[s]])
    for i in range(len(ROWS[s])):
        q = SAMP[s][fo[i]]; k = look[CIT[s][i]]
        d = count_row(k, Wset, Sset, 1.0, ts[s][q], usr[s][q], q + gbase[s])
        # stored it_hist: later val / test snapshots of OTHER users count (transductive); count_row removes only same-user later ones
        res['all'].append(d['all']); res['e1'].append(d['e1']); res['pop'].append(d['pop']); res['ncol'].append(d['ncol'])
        res["xp"].append(np.log1p(min(float(IM[k].sum()), 50.0)))
    STORED[s] = {kk: np.array(v) for kk, v in res.items()}
    fc = f'{R}/nn/feat_cache/{s}'
    for nme, key in (('it_hist_all_luo', 'all'), ('it_hist_et1_luo', 'e1'), ('xp_imp_c50', 'xp'), ('seq_cv_pop', 'pop')):
        ref = np.load(f'{fc}/{nme}.npy')[ROWS[s]].astype(np.float64)
        mine = STORED[s][key]
        ok = np.ones(len(ref), bool) if key != 'pop' else (STORED[s]['ncol'] < 3000)
        dd = np.abs(mine - ref)[ok]
        lg(f'STORED RULE {s} {nme}: rows compared {int(ok.sum())} of {len(ref)}; equal (|d| <= 1e-4) {float((dd <= 1e-4).mean()):.6f}; max |d| {dd.max():.4g}; mean stored {ref[ok].mean():.4f} mine {mine[ok].mean():.4f}')

# ------------------------------------------------------------------ 2. hz2 rule on train rows, both variants
OUT = {}
fo = np.repeat(np.arange(len(samp_tr)), Mf['train'][samp_tr])
for var, m, sub in (('primary', 12, 'feats'), ('lagfirst', 5, 'feats_L')):
    rec = []
    for i in range(len(ROWS['train'])):
        q = samp_tr[fo[i]]; k = look[CIT['train'][i]]
        d = int(slot['train'][q]); L = int(LAG[q]); c = d - L; t = int(ts['train'][q])
        p = creation_slot(F4['train'][i], t)
        T = [c + o_ for o_ in DPAT if c + o_ >= p]
        A = [x for x in T if x >= 0]
        xt = len(T)
        if len(A) == len(T):                                   # EXACT
            W_decl = set(A)                                    # declaration text: W = A
            W = set(c + o_ for o_ in DPAT if c + o_ >= 0)      # what kernels.row_masks does: every pattern day with data
            r = 1.0; trunc = False
        else:                                                  # TRUNCATED
            cr = min(d, max(c, m))
            cand = [x for x in DENSE if 0 <= x < cr]
            W = set(cand[:min(xt, len(cand))]); W_decl = W
            r = min(xt / len(W), 4.0) if len(W) else 1.0
            trunc = True
        S = set(c + o_ for o_ in PAT if 0 <= c + o_ <= d)
        dct = count_row(k, W, S, r, t, usr['train'][q], q)
        if not trunc and W_decl != W:
            d2 = count_row(k, W_decl, S, r, t, usr['train'][q], q)
            dct['all_decl'] = d2['all']
        else:
            dct['all_decl'] = dct['all']
        dct['xp'] = float(np.log1p(min(r * float(IM[k, sorted(W)].sum()) if W else 0.0, 50.0)))
        dct.update(d=d, L=L, trunc=trunc, r=r, xt=xt, nW=len(W), p=p, maxW=max(W) if W else -99, q=int(q))
        rec.append(dct)
    D = pl.DataFrame(rec)
    OUT[var] = D
    vf = f'{HZ2}/{sub}/train'
    lg(f'=== {var} (m = {m}) : {len(D)} rows of {len(samp_tr)} train feeds; truncated share {D["trunc"].mean():.4f}')
    for nme, key in (('it_hist_hz_all', 'all'), ('it_hist_hz_et1', 'e1'), ('xp_imp_hz_c50', 'xp'), ('seq_cv_hz_pop', 'pop')):
        ref = np.asarray(np.load(f'{vf}/{nme}.npy', mmap_mode='r')[ROWS['train']]).astype(np.float64)
        mine = D[key].to_numpy()
        ok = np.ones(len(ref), bool) if key != 'pop' else (D['ncol'].to_numpy() < 3000)
        dd = (np.abs(mine - ref) / (1 + np.abs(ref)))[ok]
        lg(f'  {nme}: rows compared {int(ok.sum())}; equal (rel <= 1e-5) {float((dd <= 1e-5).mean()):.6f}; max rel {dd.max():.4g}; zero share file {float((ref[ok] == 0).mean()):.4f} mine {float((mine[ok] == 0).mean()):.4f}')
        if key == 'all':
            dd2 = np.abs(D['all_decl'].to_numpy() - ref) / (1 + np.abs(ref))
            lg(f'  it_hist_hz_all under the DECLARATION wording for exact rows (W = A, days >= creation only): rows differing {int((dd2 > 1e-5).sum())} ({float((dd2 > 1e-5).mean()):.5f})')
    n_d = int(D['n_dense'].sum()); n_s = int(D['n_sparse'].sum())
    lg(f'  CAUSALITY: counted pairs dense {n_d} sparse {n_s}; rows with a counted pair on a slot AFTER the row day: {int((D["max_slot"] > D["d"]).sum())}; '
       f'rows with a DENSE pair on or after the row day: {int((D["max_dense_slot"] >= D["d"]).sum())}; rows whose window W reaches the own day or later: {int((D["maxW"] >= D["d"]).sum())}')
    lg(f'  pairs with ts later than the row (same-day sparse, other users; allowed by the rule): {int(D["n_later_ts"].sum())}; of which user-0 snapshots: {int(D["n_user0_later_same_day"].sum())}; '
       f'pairs from the row\'s own non-zero user (earlier or own snapshot): {int(D["n_own_user"].sum())}')
    D = D.with_columns((pl.col('d') - pl.col('maxW')).alias('gap'))
    warm = D.filter(pl.col('n_dense') > 0)
    lg(f'  STALENESS of the dense part (row day - latest dense day in W), rows with a dense count > 0 ({len(warm)}): mean {warm["gap"].mean():.2f} d; '
       + ' '.join(f'{g}d:{float((warm["gap"] == g).mean()):.3f}' for g in range(1, 11)))
    for dg, sel in (('Sep6-8', (0, 2)), ('Sep9-12', (3, 6)), ('Sep13-14', (7, 8)), ('Sep17-19', (11, 13))):
        w2 = warm.filter((pl.col('d') >= sel[0]) & (pl.col('d') <= sel[1]))
        if len(w2):
            lg(f'    {dg}: warm rows {len(w2)}; mean gap {w2["gap"].mean():.2f} d; share of warm rows with gap <= 2 d {float((w2["gap"] <= 2).mean()):.3f}; mean r {w2["r"].mean():.2f}; truncated share {w2["trunc"].mean():.3f}')
    D.write_parquet(f'{HERE}/recount_{var}_seed{SEED}.parquet')
lg('RECOUNT DONE')
