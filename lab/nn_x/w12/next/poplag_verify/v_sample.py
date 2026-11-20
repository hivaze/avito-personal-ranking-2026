"""poplag_verify step F (LABEL-FREE): my own rebuild of ALL NINE lag-dependent columns for a random sample of SUBSET holdout feeds (days >= Sep 13),
and my own CPU scoring of members on that sample with MY columns, compared with the study's staged private-like predictions.
  - seq_cv_{invd20,pop,norm}: event-level numpy implementation written from feats_v2/seq_/feat_covis.py (the lead's stored feature) and the membership text of
    nn_x/w12/next/poplag/DECLARATION.txt (no code of the study, no numba). Validated first on the stored lag-0 columns.
  - it_hist_{all,et1..4}_luo: event-level join (as v_ithist.py).
  - xp_imp_c50: my full rebuild of v_gate.py (cache/my_<view>_v3_xp_imp_c50.npy).
No target file is opened.
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_sample.py [n_feeds=400]"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *
import time, json

T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


NS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
G = 3; PAT = [2, 4, 6, 7, 8, 9]; BIG = 1 << 30; CAP = 3000
KN = np.load(ROOT / 'data/feats/life_f4_age_knots.npy')
ftr, hold, prand = my_holdout()
nF = len(ftr)
fts = ftr['timestamp'].to_numpy().astype(np.int64); fday = (fts + TZ) // DAY; fuser = ftr['user_id'].to_numpy().astype(np.int64)
fM = ftr['M'].to_numpy().astype(np.int64); fH = ftr['H'].to_numpy().astype(np.int64)
fco = ftr['cand_off'].to_numpy().astype(np.int64); fho = ftr['hist_off'].to_numpy().astype(np.int64); ffid = ftr['feed_id'].to_numpy()
assert (ftr['fidx'].to_numpy() == np.arange(nF)).all()
fsp = (((ffid.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)) >> np.uint64(40)) % np.uint64(37)) == 0      # shared 1/37 sample (feat_new/hz_common.py sparse_flag, re-typed)
# day of the NEXT train feed of the same non-zero user in (timestamp, fidx) order (BIG if none, and for user 0)
o = np.lexsort((np.arange(nF), fts, fuser)); uo = fuser[o]
same = (uo[1:] == uo[:-1]) & (uo[:-1] != 0)
fnd = np.full(nF, BIG, np.int64); fnd[o[:-1][same]] = fday[o[1:][same]]
hd_all = fday[hold]
late = np.flatnonzero(hd_all >= SEP6 + 7)
pick = np.sort(np.random.default_rng(20261116).choice(late, NS, replace=False))                # positions in the holdout list, SUBSET feeds only
fx = hold[pick]
off_h = np.concatenate([[0], np.cumsum(fM[hold])[:-1]])
hpos = np.concatenate([np.arange(off_h[i], off_h[i] + fM[hold[i]]) for i in pick])              # row positions inside the 4,630,966 holdout rows
jl = np.repeat(np.arange(NS), fM[fx])                                                            # local feed index of each sampled row
rows = np.concatenate([np.arange(fco[f], fco[f] + fM[f]) for f in fx])
item = np.asarray(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')[rows]).astype(np.int64)
f4 = np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[rows]).astype(np.float64)
u_ = f4 * 64.0; k_ = np.clip(np.floor(u_).astype(np.int64), 1, 62)
ah = np.where(f4 < 63 / 64, KN[k_] + (u_ - k_) * (KN[k_ + 1] - KN[k_]), np.nan)
nR = len(rows)
NOH = (fH[fx] == 0)[jl]                                                                           # rows of feeds with an empty history (run 1 of this script missed the rule: logs/v_sample_run1_popbug.log)
ht = fts[fx]; hd = fday[fx]; hu = fuser[fx]
lg(f'rows of feeds with an empty history: {NOH.mean():.4f} of rows, {np.mean(fH[fx] == 0):.4f} of feeds')
lg(f'sample: {NS} SUBSET holdout feeds (days >= Sep 13; {len(late)} in the subset), {nR} rows, {len(np.unique(item))} items; holdout days in the sample: {dict(zip(*[x.tolist() for x in np.unique(hd - SEP6, return_counts=True)]))}')

# ---- every train history event of the sampled items, with its global event index (-> position in the source history)
hi = np.load(FLAT / 'train/hist_item_id.npy')                                                     # 583.6M int32 in RAM (2.3 GB)
need = np.unique(item); NI = int(need.max()) + 1; look = np.zeros(NI, np.bool_); look[need] = True
ev_e = []
STEP = 32_000_000
for s in range(0, len(hi), STEP):
    x = hi[s:s + STEP]; ev_e.append(np.flatnonzero((x < NI) & look[np.minimum(x, NI - 1)]) + s)
ev_e = np.concatenate(ev_e); ev_i = hi[ev_e].astype(np.int64)
ev_f = np.asarray(np.load(FLAT / 'train/hist_fidx.npy', mmap_mode='r')[ev_e]).astype(np.int64)
ev_t = np.asarray(np.load(FLAT / 'train/hist_event_type.npy', mmap_mode='r')[ev_e]).astype(np.int64)
ev_p = ev_e - fho[ev_f]
assert (ev_p >= 0).all() and (ev_p < fH[ev_f]).all()
nE = len(ev_e)
lg(f'train history events of the sampled items: {nE} of {len(hi)}')
# +-20 window of every such event inside its source history (item ids; -1 outside the history)
OFFS = np.concatenate([np.arange(-20, 0), np.arange(1, 21)]); INVD = 1.0 / np.abs(OFFS)
WIN = np.full((nE, 40), -1, np.int64)
for c0 in range(0, nE, 500_000):
    sl = slice(c0, c0 + 500_000)
    q = ev_p[sl, None] + OFFS[None, :]; ok = (q >= 0) & (q < fH[ev_f[sl]][:, None])
    idx = np.clip(ev_e[sl, None] + OFFS[None, :], 0, len(hi) - 1)
    WIN[sl] = np.where(ok, hi[idx], -1)
# ---- query side: the feed's own history, de-duplicated by item with the largest 0.95^hpos
hq_rows = np.concatenate([np.arange(fho[f], fho[f] + fH[f]) for f in fx]); hq_j = np.repeat(np.arange(NS), fH[fx])
hq_item = hi[hq_rows].astype(np.int64); hq_w = 0.95 ** np.asarray(np.load(FLAT / 'train/hist_hpos.npy', mmap_mode='r')[hq_rows]).astype(np.float64)
del hi
KEYM = int(max(NI, hq_item.max() + 1, WIN.max() + 1)) + 1
qd = pl.DataFrame({'key': hq_j * KEYM + hq_item, 'w': hq_w}).group_by('key').agg(pl.col('w').max()).sort('key')
QK = qd['key'].to_numpy(); QW = qd['w'].to_numpy()
lg(f'query histories: {len(hq_rows)} events, {len(QK)} distinct (feed, item) pairs')

EVT = pl.DataFrame({'item': ev_i, 'k': np.arange(nE), 'e': ev_e, 's': ev_f, 'et': ev_t, 's_user': fuser[ev_f], 's_ts': fts[ev_f], 's_day': fday[ev_f], 's_nd': fnd[ev_f], 's_sp': fsp[ev_f]})
parts = [EVT.select('item', 's', 'et', 's_user', 's_ts', 's_day', 's_sp').with_columns(pl.lit(0, pl.Int8).alias('src'))]
for sp in ('val', 'test'):
    f = pl.read_parquet(FLAT / sp / 'feeds.parquet'); x = np.load(FLAT / sp / 'hist_item_id.npy'); xf = np.load(FLAT / sp / 'hist_fidx.npy').astype(np.int64)
    m = np.flatnonzero((x < NI) & look[np.minimum(x, NI - 1)]); ts_ = f['timestamp'].to_numpy().astype(np.int64)[xf[m]]
    parts.append(pl.DataFrame({'item': x[m].astype(np.int64), 's': np.full(len(m), -1, np.int64), 'et': np.load(FLAT / sp / 'hist_event_type.npy')[m].astype(np.int64),
                               's_user': f['user_id'].to_numpy().astype(np.int64)[xf[m]], 's_ts': ts_, 's_day': (ts_ + TZ) // DAY, 's_sp': np.ones(len(m), np.bool_)}).with_columns(pl.lit(1, pl.Int8).alias('src')))
EVALL = pl.concat(parts)
te = pl.read_parquet(FLAT / 'test/feeds.parquet'); tday = (te['timestamp'].to_numpy().astype(np.int64) + TZ) // DAY; tpub = te['timestamp'].to_numpy() < PUB_TS
stored = {n: np.asarray(np.load(FC / 'train' / f'{n}.npy', mmap_mode='r')[rows]) for n in
          ['it_hist_all_luo'] + [f'it_hist_et{k}_luo' for k in (1, 2, 3, 4)] + ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'xp_imp_c50']}


def covis(J, dense, sparse):
    """J: numpy dict of the (row, source event) pairs. dense / sparse: boolean membership of the source sequence for the row. Returns invd20, pop (float64)."""
    member = dense | sparse
    ix = np.flatnonzero(member)
    oo = np.lexsort((-J['e'][ix], J['row'][ix])); ix = ix[oo]
    r = J['row'][ix]; first = np.concatenate([[True], r[1:] != r[:-1]]); start = np.maximum.accumulate(np.where(first, np.arange(len(ix)), 0))
    ix = ix[(np.arange(len(ix)) - start) < CAP]                                                   # the CAP most recent member occurrences (stored cap), before the skips
    keep = ~((J['u'][ix] != 0) & (J['s_user'][ix] == J['u'][ix])) & ~(sparse[ix] & ~dense[ix] & (J['s_ts'][ix] >= J['t'][ix]))
    ix = ix[keep]
    pop = np.bincount(J['row'][ix], minlength=nR).astype(np.float64)
    pop[NOH] = 0.0                                                                                  # stored rule (feat_covis.py: "... or nu == 0: continue"): a feed with an EMPTY history gets pop = 0
    inv = np.zeros(nR)
    for c0 in range(0, len(ix), 400_000):
        a = ix[c0:c0 + 400_000]
        w = WIN[J['k'][a]]                                                                          # (n, 40) neighbour items
        key = J['j'][a][:, None] * KEYM + w
        pos = np.minimum(np.searchsorted(QK, key), len(QK) - 1)
        hit = (w >= 0) & (w != J['item'][a][:, None]) & (QK[pos] == key)
        inv += np.bincount(J['row'][a], weights=(np.where(hit, QW[pos], 0.0) * INVD[None, :]).sum(1), minlength=nR)
    return inv, pop, len(ix)


def luo_counts(Jp, keep):
    Jk = Jp.filter(keep); out = {}
    a = Jk.group_by('row').agg(pl.len().alias('n')); x = np.zeros(nR, np.int64); x[a['row'].to_numpy()] = a['n'].to_numpy(); out['it_hist_all_luo'] = x
    for k in (1, 2, 3, 4):
        a = Jk.filter(pl.col('et') == k).group_by('row').agg(pl.len().alias('n')); x = np.zeros(nR, np.int64); x[a['row'].to_numpy()] = a['n'].to_numpy(); out[f'it_hist_et{k}_luo'] = x
    return out


def cmp(nm, mine, ref, tol=1e-4):
    mine = np.asarray(mine, np.float64); ref = np.asarray(ref, np.float64); d = np.abs(mine - ref)
    okr = d <= tol * (1 + np.abs(ref))
    return f'{nm}: equal (rel 1e-4) on {okr.mean():.6f} of {len(ref)} rows, exact {np.mean(mine.astype(np.float32) == ref.astype(np.float32)):.6f}, max |diff| {d.max():.3g}, zero share mine {np.mean(mine == 0):.4f} / ref {np.mean(ref == 0):.4f}', float(okr.mean())


MY = {}; RES = {}
for nm in ('lag0', 'priv', 'pub'):
    if nm == 'lag0':
        L = np.zeros(NS, np.int64)
    else:
        sel = tpub if nm == 'pub' else ~tpub
        a, b = np.unique(tday[sel] - SEP20, return_counts=True); cdf = np.cumsum(b) / b.sum()
        un = ((ffid[fx].astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
        L = a[np.minimum(np.searchsorted(cdf, un, side='right'), len(a) - 1)].astype(np.int64)
    cd = hd - L; c_sec = cd * DAY - TZ
    E = c_sec[jl] - (ht[jl] - 3600.0 * ah)
    w = np.minimum(SEP20 * DAY - TZ, np.maximum(c_sec[jl], (SEP6 * DAY - TZ) + np.clip(np.where(np.isnan(E), 1e12, E), 0, 14 * DAY)))
    D = np.floor((w + TZ) / DAY + 0.5).astype(np.int64)
    Q = pl.DataFrame({'row': np.arange(nR), 'item': item, 'j': jl, 'u': hu[jl], 't': ht[jl], 'dt': hd[jl], 'cd': cd[jl], 'D': D, 'own': fx[jl].astype(np.int64)})
    # ---------------- seq_cv
    Jd = Q.join(EVT, on='item', how='inner')
    J = {c: Jd[c].to_numpy() for c in ('row', 'item', 'j', 'u', 't', 'dt', 'cd', 'D', 'k', 'e', 's_user', 's_ts', 's_day', 's_nd', 's_sp')}
    if nm == 'lag0':
        dense = (J['s_day'] < J['dt']) & (J['s_nd'] >= J['dt'])                                     # stored rule: latest feed of the user strictly before the row's day
        inv, pop, npair = covis(J, dense, np.zeros(len(dense), np.bool_))
        for cn, mine in (('seq_cv_invd20', inv), ('seq_cv_pop', pop), ('seq_cv_norm', inv / np.sqrt(pop + 1.0))):
            s_, r_ = cmp(f'CHECK my co-visitation code, lag-0 rule vs STORED {cn}', mine, stored[cn]); lg(s_); RES[f'lag0_{cn}'] = r_
    else:
        band_s = np.abs(J['s_day'] - J['dt']) < G
        dense = (J['s_day'] < J['D']) & ~band_s & ((J['s_nd'] >= J['D']) | (np.abs(J['s_nd'] - J['dt']) < G))
        onpat = J['s_sp'] & np.isin(J['s_day'] - J['cd'], PAT)
        sparse_a = onpat & ((J['s_day'] >= J['D']) | band_s)          # reading A: a sparse day is a day outside the counted dense window
        sparse_b = onpat & ~dense                                      # reading B: literal "not a counted dense snapshot" (also de-duplicated feeds inside the window)
        inv, pop, npair = covis(J, dense, sparse_a)
        inv_b, pop_b, _ = covis(J, dense, sparse_b)
        MY[nm] = {'seq_cv_invd20': inv.astype(np.float32), 'seq_cv_pop': pop.astype(np.float32), 'seq_cv_norm': (inv / np.sqrt(pop + 1.0)).astype(np.float32)}
        for cn in ('seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm'):
            st = np.load(STUDY / f'lag_{nm}_v3_{cn}.npy', mmap_mode='r')[hpos]
            s_, r_ = cmp(f'[{nm}] MY REBUILD vs the study\'s lag_{nm}_v3_{cn}', MY[nm][cn], st); lg(s_); RES[f'{nm}_{cn}'] = r_
        st = np.load(STUDY / f'lag_{nm}_v3_seq_cv_pop.npy', mmap_mode='r')[hpos]
        lg(f'[{nm}]   reading B of the sparse rule: seq_cv_pop equal to the study on {np.mean(pop_b == st):.6f} of rows; rows where reading B differs from reading A {np.mean(pop_b != pop):.6f}, mean pop A {pop.mean():.3f} B {pop_b.mean():.3f}')
        fut = dense & (J['s_day'] > J['dt'])
        lg(f'[{nm}]   member source sequences: dense {int(dense.sum())} (of which dated AFTER the row\'s day {fut.sum() / max(dense.sum(), 1):.3f}), sparse {int((sparse_a & ~dense).sum())}')
    # ---------------- it_hist (event-level join, all snapshots of train + val + test)
    Jp = Q.join(EVALL, on='item', how='inner')
    nlsu = ~((pl.col('u') != 0) & (pl.col('s_user') == pl.col('u')) & (pl.col('s_ts') > pl.col('t')))
    if nm == 'lag0':
        c = luo_counts(Jp, nlsu)
        for cn in c:
            s_, r_ = cmp(f'CHECK lag-0 rule vs STORED {cn}', c[cn], stored[cn]); lg(s_); RES[f'lag0_{cn}'] = r_
    else:
        dn = (pl.col('src') == 0) & (pl.col('s_day') < pl.col('D')) & ((pl.col('s_day') - pl.col('dt')).abs() >= G)
        own = (pl.col('src') == 0) & (pl.col('s') == pl.col('own'))
        spx = pl.col('s_sp') & (pl.col('s_day') - pl.col('cd')).is_in(PAT)
        c = luo_counts(Jp, nlsu & (dn | own | spx))
        for cn in c:
            st = np.load(STUDY / f'lag_{nm}_v3_{cn}.npy', mmap_mode='r')[hpos]
            s_, r_ = cmp(f'[{nm}] MY REBUILD vs the study\'s lag_{nm}_v3_{cn}', c[cn], st); lg(s_); RES[f'{nm}_{cn}'] = r_
            MY[nm][cn] = c[cn].astype(np.float32)
        xm = np.load(MYC / f'my_{nm}_v3_xp_imp_c50.npy', mmap_mode='r')[hpos]; st = np.load(STUDY / f'lag_{nm}_v3_xp_imp_c50.npy', mmap_mode='r')[hpos]
        s_, r_ = cmp(f'[{nm}] MY REBUILD (v_gate.py, full) vs the study\'s lag_{nm}_v3_xp_imp_c50', xm, st); lg(s_); RES[f'{nm}_xp_imp_c50'] = r_
        MY[nm]['xp_imp_c50'] = np.asarray(xm, np.float32)
        for cn in MY[nm]:
            np.save(MYC / f'sample_{nm}_{cn}.npy', MY[nm][cn])
np.save(MYC / 'sample_pick.npy', pick); np.save(MYC / 'sample_hpos.npy', hpos)
del WIN

# =============================================================== scoring of the sample with MY columns
import torch  # noqa: E402
torch.set_num_threads(int(NT))
sys.path.insert(0, str(ROOT / 'nn_x/w5'))
from data import SplitData  # noqa: E402
from model import Ranker  # noqa: E402

MEM = ['w5_e10_kd05w192_s952', 'w5_e10_kd05t25w192_s953', 'abl_e10_pop1kd05t25_s952']
crow = rows; hrow = hq_rows
mm = lambda n: np.load(FLAT / 'train' / f'{n}.npy', mmap_mode='r')
g = lambda a: torch.from_numpy(np.ascontiguousarray(a))
ce = np.asarray(mm('cand_emb_idx')[crow]); he = np.asarray(mm('hist_emb_idx')[hrow])
d = ROOT / 'data/competition/train/item_embeddings'; shape = tuple(json.loads((d / 'data.meta').read_text())['shape'])
tab = np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=shape)
used = np.unique(np.concatenate([ce[ce >= 0], he[he >= 0]])); pos = np.full(int(used.max()) + 1, -1, np.int64); pos[used] = np.arange(len(used))
s = SplitData.__new__(SplitData)
s.split = 'verify_sample'; s.device = torch.device('cpu'); s.max_hist = 256
s.M_np = fM[fx]; s.H_np = fH[fx]; s.n = NS
s.cand_off_np = np.concatenate([[0], np.cumsum(s.M_np)[:-1]]).astype(np.int64)
s.user_id = ftr['user_id'].to_numpy()[fx]
s.M = g(s.M_np); s.H = g(s.H_np); s.cand_off = g(s.cand_off_np); s.hist_off = g(np.concatenate([[0], np.cumsum(s.H_np)[:-1]]).astype(np.int64))
s.user_emb = g(np.asarray(mm('user_emb')[fx])); s.user0 = g(s.user_id == 0)
s.c_emb = g(np.where(ce >= 0, pos[np.maximum(ce, 0)], -1).astype(np.int32)); s.h_emb = g(np.where(he >= 0, pos[np.maximum(he, 0)], -1).astype(np.int32))
s.table = g(np.asarray(tab[used])); s.n_cand = len(crow); s.n_hist = len(hrow)
s.c_cat = g(np.asarray(mm('cand_category_id')[crow]).astype(np.int16)); s.c_f = g(np.stack([np.asarray(mm(f'cand_f{i}')[crow]) for i in range(1, 6)], 1))
s.h_cat = g(np.asarray(mm('hist_category_id')[hrow]).astype(np.int16)); s.h_et = g(np.asarray(mm('hist_event_type')[hrow])); s.h_es = g(np.asarray(mm('hist_event_source')[hrow]))
s.c_item = None; s.h_item = None; s.xt = None; s.has_target = False; s.target_np = None; s.c_tgt = None; s.has_pos = False; s.c_pos = None
pp = g(prand[fx].astype(np.float32)); s.dom = torch.stack([1 - pp, pp, torch.zeros_like(pp)], 1)       # nn_x/w5/train.py l.372-373
LAGC = list(MY['priv'].keys())


def xt_for(names, view):
    cols = []; rep = []
    for n in names:
        base = n[4:] if n.startswith('has:') else n
        if view != 'lag0' and base in LAGC:
            x = MY[view][base].astype(np.float32); rep.append(base)                                 # MY rebuilt column
        else:
            x = np.asarray(np.load(FC / 'train' / f'{base}.npy', mmap_mode='r')[crow]).astype(np.float32)
        if n.startswith('has:'):
            x = (~np.isnan(x)).astype(np.float32)
        elif n.endswith('_rk'):
            pass
        elif n.endswith('_z'):
            x = np.clip(x, -5, 5)
        elif n.startswith(('seq_cv', 'it_hist')):
            x = np.log1p(np.maximum(x, 0))
        else:
            x = np.clip(x, -10, 10)
        cols.append(torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16))
    return torch.stack(cols, 1), rep


Ms = s.M_np; os_ = s.cand_off_np; fid = np.repeat(np.arange(NS), Ms)
zf = lambda x: zfeed(x, os_, Ms).astype(np.float64)


def top1(x):
    return np.lexsort((-x, fid))[os_]


for v in MEM:
    ck = torch.load(ROOT / ('nn_x/w5/ckpt' if v.startswith('w5_') else 'nn_x/abl/ckpt') / f'{v}.pt', map_location='cpu', weights_only=False)
    m = Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
    names = ck['cfg']['xt_names']
    lg(f'{v}: d={ck["cfg"]["d"]} xt cols {len(names)}; lag-dependent inputs present: {[n for n in names if (n[4:] if n.startswith("has:") else n) in LAGC]}')
    out = {}
    for view in ('lag0', 'priv', 'pub'):
        s.xt, rep = xt_for(names, view)
        o_ = torch.zeros(s.n_cand, dtype=torch.float32)
        with torch.inference_mode():
            for fb in s.batches(np.arange(s.n), 65536, 1024, shuffle=False):
                b = s.gather(fb); s.scatter(o_, b, m(b, use_pos=False))
        out[view] = o_.numpy().astype(np.float64)
        np.save(MYC / f'sample_pred_{view}_{v}.npy', o_.numpy())
    c0 = np.load(STUDY / f'pred_lag0_{v}.npy', mmap_mode='r')[hpos].astype(np.float64); c1 = np.load(STUDY / f'pred_priv_v3_{v}.npy', mmap_mode='r')[hpos].astype(np.float64)
    for nm_, mine, ref in (('MY lag-0 vs the study\'s CPU lag-0 file', out['lag0'], c0), ('MY private-like v3 (ALL nine columns mine) vs the study\'s STAGED private-like file', out['priv'], c1),
                           ('(contrast) MY private-like v3 vs MY lag-0', out['priv'], out['lag0'])):
        fc = np.add.reduceat(zf(mine) * zf(ref), os_) / Ms
        lg(f'   {nm_}: max |diff| {np.abs(mine - ref).max():.2e}, mean |diff| {np.abs(mine - ref).mean():.2e}, mean per-feed corr {fc.mean():.6f}, min per-feed corr {fc.min():.6f}, top-1 changed in {np.mean(top1(mine) != top1(ref)):.4f} of {NS} feeds')
        RES[f'pred_{v}_{nm_[:20]}'] = [float(np.abs(mine - ref).max()), float(fc.mean()), float(np.mean(top1(mine) != top1(ref)))]
json.dump(RES, open(MYC / 'v_sample.json', 'w'), indent=1)
lg('done')
