"""Leak / mismatch audit of famN's hand-made columns (cache/hx/n30k): distribution and label association on fit (out-of-fold), selection,
holdout and val rows; user overlap between the witnesses and the 30K fit feeds; fold exclusion re-derived for the LDA column."""
import os, sys, json
sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review'); sys.path.insert(0, '<repo>/nn_x/w12/cb/famN/scripts')
from rv_common import *
import fn; from fn import cbx
import polars as pl
tr = cbx.load('xm'); ho = cbx.load('hold'); va = cbx.load('val')
fit, sel = fn.subset(30000, 5000)
rf = np.repeat(fit, tr.M); rs = np.repeat(sel, tr.M)
wv, wh = Wit('val'), Wit('hold')
assert (ho.M == wh.M).all() and (va.M == wv.M).all()
y_xm = np.asarray(tr.y, np.float64); t_xm = np.asarray(tr.teacher('oof', 'centred'), np.float64); t_ho = np.asarray(ho.teacher('oof', 'centred'), np.float64)
def cen(x, gid, n):
    mu = np.bincount(gid, x, n) / np.bincount(gid, minlength=n); return x - mu[gid]
def wcorr(a, b, gid, n):
    a = cen(a, gid, n); b = cen(b, gid, n); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
H = ROOT / 'nn_x/w12/cb/famN/cache/hx/n30k'
out = {}
gx = np.asarray(tr.gid); nfx = tr.n_feeds
print('column | set | mean | sd | within-feed sd | within-feed corr with click | within-feed corr with teacher_oof')
for c in ['hlda_pair', 'hlda_ehm', 'hlda_ue', 'hknn50_pair', 'hknn200_pair', 'hknn50_ehm', 'hknn200_ehm', 'hknn50_ue', 'hknn200_ue']:
    x = np.load(H / f'{c}_xm.npy').astype(np.float64); xh = np.load(H / f'{c}_hold.npy').astype(np.float64); xv = np.load(H / f'{c}_val.npy').astype(np.float64)
    assert np.isfinite(x[rf]).all() and np.isfinite(x[rs]).all() and np.isnan(x[~(rf | rs)]).all()
    o = {}
    for nm, v, yy, tt, gid, n in (('fit(oof)', x[rf], y_xm[rf], t_xm[rf], gx[rf], nfx), ('sel', x[rs], y_xm[rs], t_xm[rs], gx[rs], nfx),
                                   ('hold', xh, wh.y, t_ho, wh.fid, wh.nf), ('val', xv, wv.y, None, wv.fid, wv.nf)):
        o[nm] = dict(mean=float(v.mean()), sd=float(v.std()), wsd=float(cen(v, gid, n).std()), cy=wcorr(v, yy, gid, n), ct=None if tt is None else wcorr(v, tt, gid, n))
        print(f"{c} | {nm} | {o[nm]['mean']:+.4f} | {o[nm]['sd']:.4f} | {o[nm]['wsd']:.4f} | {o[nm]['cy']:+.4f} | {'' if tt is None else format(o[nm]['ct'], '+.4f')}", flush=True)
    out[c] = o
# user overlap
ft = pl.read_parquet(ROOT / 'data/flat/train/feeds.parquet', columns=['timestamp', 'user_id']); fv = pl.read_parquet(ROOT / 'data/flat/val/feeds.parquet', columns=['timestamp', 'user_id'])
ts = ft['timestamp'].to_numpy(); us = ft['user_id'].to_numpy()
assert (us[tr.fidx] == tr.uid_feed).all()
fit_u = us[tr.fidx[fit]]; fit_t = ts[tr.fidx[fit]]
hu = us[ho.fidx]; ht = ts[ho.fidx]
import collections
last = {}
first = {}
for u, t in zip(fit_u.tolist(), fit_t.tolist()):
    if u == 0: continue
    last[u] = max(last.get(u, -1), t); first[u] = min(first.get(u, 2**62), t)
inh = np.array([u in last for u in hu.tolist()]); later = np.array([(u in last) and last[u] > t for u, t in zip(hu.tolist(), ht.tolist())])
vu = fv['user_id'].to_numpy(); inv = np.array([u in last for u in vu.tolist()])
print(f'user overlap: holdout feeds whose user has a feed in the 30K fit subset: {int(inh.sum())} of {len(hu)} ({inh.mean():.4f}); of these with a LATER fit feed of the same user: {int(later.sum())} ({later.mean():.4f})')
print(f'user overlap: val feeds whose user has a feed in the 30K fit subset: {int(inv.sum())} of {len(vu)} ({inv.mean():.4f})')
print(f'time: xm fit feeds ts range {fit_t.min()}..{fit_t.max()}; holdout {ht.min()}..{ht.max()}; val {fv["timestamp"].min()}..{fv["timestamp"].max()}; xm feed timestamps non-decreasing in file order: {bool((np.diff(ts[tr.fidx]) >= 0).all())}')
out['overlap'] = dict(hold_in_fit=int(inh.sum()), hold_later=int(later.sum()), val_in_fit=int(inv.sum()))
# fold exclusion re-derived for hlda_pair on fold 3: direction from the other folds must reproduce the stored column; a direction that includes fold 3 must not
bf = fn.Block(tr, fit); fold_row = tr.xfold[bf.frow].astype(np.int64); V = bf.vec('pair'); yb = bf.label('y').astype(np.float64)
def direction(m):
    Vm = V[m].astype(np.float64); ym = yb[m]; n1 = ym.sum(); n0 = len(ym) - n1; mu1 = Vm[ym > 0].sum(0) / n1; mu0 = Vm[ym == 0].sum(0) / n0
    Sw = (Vm.T @ Vm - n1 * np.outer(mu1, mu1) - n0 * np.outer(mu0, mu0)) / (n1 + n0); return np.linalg.solve(Sw + 1e-4 * np.eye(V.shape[1]), mu1 - mu0)
stored = np.load(H / 'hlda_pair_xm.npy')[bf.idx]; k = 3; m = fold_row == k
w_out = direction(~m); w_all = direction(np.ones(len(m), bool))
print('hlda_pair fold 3: max abs diff stored vs own out-of-fold direction', float(np.abs(V[m] @ w_out - stored[m]).max()), '| vs own all-fold direction', float(np.abs(V[m] @ w_all - stored[m]).max()), '| sd of column', float(stored[m].std()))
# fold membership: every feed in one fold, every user in one fold
uf = tr.uid_feed[fit]; xf = tr.xfold[fit]; d = {}
bad = 0
for u, f_ in zip(uf.tolist(), xf.tolist()):
    if u == 0: continue
    if d.setdefault(u, f_) != f_: bad += 1
print('users of the fit subset split over two folds:', bad, '| fit feeds with user 0:', int((uf == 0).sum()))
json.dump(out, open(HERE / 'out/hx_audit.json', 'w'), indent=1)
