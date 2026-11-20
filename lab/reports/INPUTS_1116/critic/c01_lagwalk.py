"""CRITIC c01: family W (RP3 whole-sequence walk) built by the KERNEL OF RECORD (imported read-only from
nn_x/insights/ccb_1113/scratch/tester3/c09_build.py: rp3, Index, source_fidx) in four forms, on small sets:
  (R)  reproduction: natural val (train + earlier val feeds) against cache/c09_val.npz; holdout rows of two days with cut = own day against c09_hold.npz
  (L)  LAG VIEW of DESIGN.md 1.2 / BUILD_SPEC.txt 1-2 on the 15,225 holdout feeds of slots >= 6: cut' = max(slot - L, 5), dense = train feeds of
       slots < cut', sparse = the shared 1/37 sample of train feeds on slots >= cut' with day < own day
  (T)  stale-val twin on all val feeds: cut = slot - L (hz2_common.assign_lags), dense slots < cut, sparse = 1/37 train sample on slots >= cut + val feeds of earlier days
  (N)  natural test rows: all train + val / test feeds of earlier days (label-free; coverage reference for the parity gate)
Deviation from BUILD_SPEC, stated: sparse members are admitted by DAY (day < own day), not by timestamp, and they enter deg as in c09_build (the
code that produced the arrays whose cz is the support on record). CPU only, 12 threads, read-only on the project; writes only critic/out/."""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
CR = '<repo>/reports/INPUTS_1116/critic'
os.environ['NUMBA_CACHE_DIR'] = CR + '/out/numba_cache'
sys.dont_write_bytecode = True
import numba as nb
import numpy as np
import polars as pl
sys.path.insert(0, '<repo>/nn_x/insights/ccb_1113/scratch/tester3')
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import c09_build as c9
import hz2_common as hc
from pathlib import Path
FLAT = Path('<repo>/data/flat'); OUT = Path(CR) / 'out'; C9 = Path('<repo>/nn_x/insights/ccb_1113/scratch/tester3/cache')
T0 = time.time(); logf = open(OUT / 'c01_lagwalk.log', 'w')
def P(*a):
    s = f'[{time.time() - T0:6.0f}s] ' + ' '.join(str(x) for x in a); print(s, flush=True); logf.write(s + '\n'); logf.flush()

nb.set_num_threads(12)
F = {s: pl.read_parquet(FLAT / s / 'feeds.parquet') for s in ('train', 'val', 'test')}
ft, fv, fe = F['train'], F['val'], F['test']
# dense item ids over all history items of the three splits (lookup table; same order as np.unique)
H_ = {s: np.load(FLAT / s / 'hist_item_id.npy') for s in F}
mx = max(int(h.max()) for h in H_.values()) + 1
pres = np.zeros(mx, np.bool_)
for h in H_.values(): pres[h] = True
lut = (np.cumsum(pres, dtype=np.int64) - 1).astype(np.int32); G = int(pres.sum())
hist_dense = {s: lut[H_[s]] for s in F}
del H_
P(f'dense ids: {G} items (max id {mx - 1})')
def cand_dense(items):
    items = np.asarray(items).astype(np.int64); ok = (items < mx)
    out = np.full(len(items), -1, np.int64); out[ok] = np.where(pres[items[ok]], lut[items[ok]], -1)
    return out
HP = {s: np.asarray(np.load(FLAT / s / 'hist_hpos.npy')).astype(np.int16) for s in F}
A = {s: dict(M=F[s]['M'].to_numpy().astype(np.int64), H=F[s]['H'].to_numpy().astype(np.int64), co=F[s]['cand_off'].to_numpy().astype(np.int64),
             ho=F[s]['hist_off'].to_numpy().astype(np.int64), u=F[s]['user_id'].to_numpy().astype(np.int64), day=c9.day_of(F[s]['timestamp'].to_numpy()),
             fid=F[s]['feed_id'].to_numpy()) for s in F}
tslot = A['train']['day'] - hc.DAY0

def run(split, idx, deg2, sel, fslot, fday, cand_d=None, coff=None):
    """query feeds `sel` of `split`; returns (rows x 3 array in the order of sel's concatenated candidates, degc per row)"""
    a = A[split]; Mq = a['M'][sel]; offq = np.concatenate([[0], np.cumsum(Mq)[:-1]]).astype(np.int64)
    rows = np.repeat(a['co'][sel] - offq, Mq) + np.arange(int(Mq.sum()), dtype=np.int64)
    cd = cand_dense(np.asarray(np.load(FLAT / split / 'cand_item_id.npy', mmap_mode='r')[rows]))
    out = np.zeros((len(rows), 3), np.float64); q = np.arange(len(sel), dtype=np.int64)
    c9.rp3(q, offq, Mq, a['ho'][sel], a['H'][sel], a['u'][sel], fslot.astype(np.int64), cd, hist_dense[split], HP[split], idx.ev, idx.starts, idx.lens,
           idx.users, idx.days, fday.astype(np.int64), idx.pst, idx.cnt, idx.p_seq, deg2, G, 12, 500, out)
    degc = np.where(cd >= 0, deg2[np.repeat(fslot, Mq), np.maximum(cd, 0)], 0.0).astype(np.float32)
    return out, degc, rows

def deg_rows(idx, n_dense_seq, days_q):
    """deg per query day: dense part + sparse sequences of earlier days (c09_build convention)"""
    nde = int(idx.lens[:n_dense_seq].sum()); degd = np.bincount(idx.ev[:nde], minlength=G).astype(np.float64)
    sde = np.repeat(idx.days[n_dense_seq:], idx.lens[n_dense_seq:]); evs = idx.ev[nde:]
    return np.stack([degd + np.bincount(evs[sde < D], minlength=G) for D in days_q])

res = {}
# ---------------- (R1, N) natural val and test: train all + val + test feeds of earlier days ----------------
src_all = c9.source_fidx(ft, None)
idx = c9.Index([('train', src_all, np.full(len(src_all), -1, np.int64)), ('val', np.arange(len(fv)), A['val']['day']), ('test', np.arange(len(fe)), A['test']['day'])],
               G, hist_dense, F)
dq = np.unique(np.concatenate([A['val']['day'], A['test']['day']])); deg2 = deg_rows(idx, len(src_all), dq)
for sp in ('val', 'test'):
    sel = np.arange(len(F[sp]), dtype=np.int64)
    o, dc, rows = run(sp, idx, deg2, sel, np.searchsorted(dq, A[sp]['day']), A[sp]['day'])
    assert (rows == np.arange(len(rows))).all()
    res[f'{sp}_nat'] = o[:, :2].astype(np.float64); res[f'{sp}_nat_degc'] = dc
    P(f'natural {sp}: non-zero share {np.mean(o[:, 1] > 0):.4f}; of non-zero rows deg(c) > 500: {np.mean(dc[o[:, 1] > 0] > 500):.4f}; index seqs {len(idx.lens)}')
c9v = np.load(C9 / 'c09_val.npz')
for j, k in enumerate(('rp3b05', 'p3a')):
    a_, b_ = res['val_nat'][:, j], c9v[k]; rel = np.abs(a_ - b_) / np.maximum(np.abs(b_), 1e-30)
    P(f'(R) val {k}: max abs diff {np.abs(a_ - b_).max():.3e}, max rel diff on non-zero {rel[b_ > 0].max():.3e}, zero-pattern mismatches {int(((a_ > 0) != (b_ > 0)).sum())}')
del idx, deg2

# ---------------- holdout feeds ----------------
hold, _p = hc.holdout_feeds(ft); Mh = A['train']['M'][hold]; hoff = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64)
L = np.load(hc.HERE / 'feats_L/train/hz2_lag.npy').astype(np.int64)
hslot = tslot[hold]; hL = L[hold]; hcut = np.maximum(hslot - hL, 5); has = hslot >= 6
P(f'holdout: {len(hold)} feeds, lag-view feeds {int(has.sum())}; cut counts {dict(zip(*[x.tolist() for x in np.unique(hcut[has], return_counts=True)]))}')
c9h = np.load(C9 / 'c09_hold.npz'); nrow_h = int(Mh.sum()); assert nrow_h == len(c9h['p3a'])
def put(dst, qi, o):
    Mq = Mh[qi]; offq = np.concatenate([[0], np.cumsum(Mq)[:-1]]).astype(np.int64)
    for j, qpos in enumerate(qi): dst[hoff[qpos]:hoff[qpos] + Mq[j]] = o[offq[j]:offq[j] + Mq[j]]
# (R2) reproduction of the stored holdout array on two days (cut = own day, no sparse members)
for s in (6, 13):
    qi = np.nonzero(hslot == s)[0]; src = c9.source_fidx(ft, int(hc.DAY0 + s))
    idx = c9.Index([('train', src, np.full(len(src), -1, np.int64))], G, hist_dense, F); deg2 = idx.cnt.astype(np.float64)[None, :]
    o, dc, rows = run('train', idx, deg2, hold[qi], np.zeros(len(qi), np.int64), np.full(len(qi), 1 << 40, np.int64))
    tmp = np.full((nrow_h, 3), np.nan); put(tmp, qi, o); m = ~np.isnan(tmp[:, 0])
    for j, k in enumerate(('rp3b05', 'p3a')):
        a_, b_ = tmp[m, j], c9h[k][m]
        P(f'(R) holdout slot {s} {k}: rows {int(m.sum())}, max abs diff {np.abs(a_ - b_).max():.3e}, zero-pattern mismatches {int(((a_ > 0) != (b_ > 0)).sum())}, non-zero share {np.mean(a_ > 0):.4f}')
    del idx, deg2
# (L) lag view
spf = hc.sparse_flag(A['train']['fid'])
lag = np.full((nrow_h, 2), np.nan); lag_degc = np.full(nrow_h, np.nan, np.float32); lag_nseq = np.full(nrow_h, np.nan, np.float32)
for c in (5, 6, 7):
    qi = np.nonzero(has & (hcut == c))[0]
    dense = c9.source_fidx(ft, int(hc.DAY0 + c)); sp_ = np.nonzero(spf & (tslot >= c))[0]
    idx = c9.Index([('train', dense, np.full(len(dense), -1, np.int64)), ('train', sp_, A['train']['day'][sp_])], G, hist_dense, F)
    dq = np.unique(A['train']['day'][hold[qi]]); deg2 = deg_rows(idx, len(dense), dq)
    o, dc, rows = run('train', idx, deg2, hold[qi], np.searchsorted(dq, A['train']['day'][hold[qi]]), A['train']['day'][hold[qi]])
    tmp = np.full((nrow_h, 3), np.nan); put(tmp, qi, o); m = ~np.isnan(tmp[:, 0]); lag[m] = tmp[m, :2]; lag_nseq[m] = tmp[m, 2]
    t2 = np.full((nrow_h, 1), np.nan); put(t2, qi, dc[:, None].astype(np.float64)); lag_degc[m] = t2[m, 0]
    P(f'(L) cut {c}: {len(qi)} holdout feeds, dense seqs {len(dense)}, sparse seqs {len(sp_)}, non-zero share {np.mean(o[:, 1] > 0):.4f}; of non-zero rows deg(c) > 500: {np.mean(dc[o[:, 1] > 0] > 500):.4f}')
    del idx, deg2
res['hold_lag'] = lag; res['hold_lag_degc'] = lag_degc; res['hold_lag_nseq'] = lag_nseq
res['hold_slot'] = np.repeat(hslot, Mh).astype(np.int8); res['hold_cut'] = np.repeat(hcut, Mh).astype(np.int8)
fresh = c9h['p3a']
for s in (6, 7, 8, 11, 12, 13):
    m = res['hold_slot'] == s
    P(f'(L) holdout slot {s}: feeds {int((hslot == s).sum())}; non-zero share fresh (cut = own day) {np.mean(fresh[m] > 0):.4f} -> lag view {np.mean(lag[m, 1] > 0):.4f}; '
      f'Spearman-like agreement: corr of log1p on rows non-zero in both {np.corrcoef(np.log1p(1e6 * fresh[m][(fresh[m] > 0) & (lag[m, 1] > 0)]), np.log1p(1e6 * lag[m, 1][(fresh[m] > 0) & (lag[m, 1] > 0)]))[0, 1]:.3f}')

# ---------------- (T) stale-val twin ----------------
vslot = A['val']['day'] - hc.DAY0; vL = np.asarray(hc.assign_lags(A['val']['fid'])[1]).astype(np.int64); vcut = vslot - vL
nv = int(A['val']['M'].sum()); twin = np.full((nv, 2), np.nan); twin_degc = np.full(nv, np.nan, np.float32)
for c in np.unique(vcut):
    qi = np.nonzero(vcut == c)[0]
    dense = c9.source_fidx(ft, int(hc.DAY0 + c)); sp_ = np.nonzero(spf & (tslot >= c))[0]
    idx = c9.Index([('train', dense, np.full(len(dense), -1, np.int64)), ('train', sp_, A['train']['day'][sp_]), ('val', np.arange(len(fv)), A['val']['day'])],
                   G, hist_dense, F)
    dq = np.unique(A['val']['day'][qi]); deg2 = deg_rows(idx, len(dense), dq)
    o, dc, rows = run('val', idx, deg2, qi, np.searchsorted(dq, A['val']['day'][qi]), A['val']['day'][qi])
    twin[rows] = o[:, :2]; twin_degc[rows] = dc
    P(f'(T) val cut {c}: {len(qi)} feeds, dense seqs {len(dense)}, sparse train seqs {len(sp_)}, non-zero share {np.mean(o[:, 1] > 0):.4f}')
    del idx, deg2
assert not np.isnan(twin).any()
res['val_twin'] = twin; res['val_twin_degc'] = twin_degc; res['val_cut'] = np.repeat(vcut, A['val']['M']).astype(np.int8)
P(f'(T) val twin: non-zero share {np.mean(twin[:, 1] > 0):.4f} (natural {np.mean(res["val_nat"][:, 1] > 0):.4f})')
np.savez(OUT / 'c01_walk.npz', **res)
P('DONE')
