"""Embedding stats: value distributions, norms, missing rates, item_id<->emb_idx mapping,
cross-split identity of item vectors, user embedding constancy."""
import os
os.environ.setdefault('POLARS_MAX_THREADS', '40')
import numpy as np
import polars as pl
from common import feeds, arr, item_table, gather_emb

rng = np.random.default_rng(0)
out = {}


def vstats(name, X):
    X = np.asarray(X)
    Xf = X.astype(np.float32)
    nrm = np.linalg.norm(Xf, axis=1)
    vals, cnts = np.unique(X, return_counts=True)
    frac = dict(zip(vals.tolist(), (cnts / cnts.sum()).tolist()))
    print(f'== {name}: n={len(X)}')
    print(f'  value mean={Xf.mean():.3f} std={Xf.std():.3f} min={X.min()} max={X.max()}  '
          f'P(-128)={frac.get(-128,0):.5f} P(-127)={frac.get(-127,0):.5f} P(127)={frac.get(127,0):.5f} P(0)={frac.get(0,0):.5f}')
    q = np.quantile(np.abs(Xf), [0.5, 0.9, 0.99, 0.999])
    print(f'  |value| quantiles 50/90/99/99.9: {q}')
    print(f'  rows all-zero: {(nrm == 0).mean():.6f}; rows with any saturated (|v|>=127): {(np.abs(Xf) >= 127).any(1).mean():.4f}')
    qn = np.quantile(nrm, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1])
    print(f'  L2 norm quantiles 0/1/10/50/90/99/100: {np.round(qn, 1)}  mean={nrm.mean():.1f} std={nrm.std():.1f}')
    dm = Xf.mean(0)
    ds = Xf.std(0)
    print(f'  per-dim mean: range [{dm.min():.2f},{dm.max():.2f}], |mean| avg {np.abs(dm).mean():.2f}; per-dim std range [{ds.min():.2f},{ds.max():.2f}]')
    # norm of mean vector relative to mean norm (anisotropy)
    print(f'  ||mean vec||/mean||x|| = {np.linalg.norm(dm) / nrm.mean():.3f}')
    # random pair cosine
    i = rng.integers(0, len(X), 20000); j = rng.integers(0, len(X), 20000)
    a = Xf[i]; b = Xf[j]
    cs = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    print(f'  random-pair cosine mean={cs.mean():.3f} std={cs.std():.3f}')
    # effective dimensionality (PCA)
    C = np.cov((Xf - dm)[:min(len(X), 300000)].T)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    pr = ev.sum() ** 2 / (ev ** 2).sum()
    print(f'  PCA: top1 {cum[0]:.3f}, top10 {cum[9]:.3f}, top32 {cum[31]:.3f}, top64 {cum[63]:.3f}; dims for 90% var: {np.searchsorted(cum, 0.9) + 1}; participation ratio {pr:.1f}')
    return nrm


# ---------- item tables ----------
for s in ['val', 'test', 'train']:
    T = item_table(s)
    n = T.shape[0]
    if n > 3_000_000:
        idx = np.sort(rng.choice(n, 2_000_000, replace=False))
        X = np.asarray(T[idx])
    else:
        X = np.asarray(T[:])
    vstats(f'item table {s} (rows {n})', X)

# ---------- user embeddings ----------
for s in ['val', 'test', 'train']:
    U = arr(s, 'user_emb', mmap=False)
    if len(U) > 1_000_000:
        U = U[np.sort(rng.choice(len(U), 1_000_000, replace=False))]
    vstats(f'user_emb {s}', U)

# ---------- missing rates ----------
print('\n== missing emb_idx rates')
for s in ['val', 'test', 'train']:
    ce = arr(s, 'cand_emb_idx'); he = arr(s, 'hist_emb_idx')
    cm = float((ce < 0).mean()); hm = float((he < 0).mean())
    print(f'  {s}: cand missing {cm:.5f}  hist missing {hm:.4f}')
    if s == 'val':
        et = arr(s, 'hist_event_type', mmap=False)
        es = arr(s, 'hist_event_source', mmap=False)
        hem = np.asarray(he) < 0
        print('   hist missing by event_type:', {int(t): round(float(hem[et == t].mean()), 3) for t in np.unique(et)})
        hi = arr(s, 'hist_item_id', mmap=False)
        print('   hist missing items: item_id==0 frac', float((hi[hem] == 0).mean()), ' unique missing items', len(np.unique(hi[hem])))

# ---------- item_id <-> emb_idx mapping within split ----------
print('\n== item_id <-> emb_idx mapping within split')
maps = {}
for s in ['val', 'test', 'train']:
    ci = arr(s, 'cand_item_id'); ce = arr(s, 'cand_emb_idx')
    hi = arr(s, 'hist_item_id'); he = arr(s, 'hist_emb_idx')
    df = pl.concat([
        pl.DataFrame({'item': np.asarray(ci), 'idx': np.asarray(ce)}),
        pl.DataFrame({'item': np.asarray(hi), 'idx': np.asarray(he)}),
    ]).filter(pl.col('idx') >= 0).unique()
    n_items = df['item'].n_unique(); n_idx = df['idx'].n_unique()
    T = item_table(s)
    print(f'  {s}: distinct (item,idx) pairs={len(df)}, items={n_items}, idx={n_idx}, table rows={T.shape[0]} '
          f'-> items with >1 idx: {len(df) - n_items}, idx with >1 item: {len(df) - n_idx}')
    maps[s] = df.unique(subset='item', keep='first')
    del df

# ---------- cross-split identity ----------
print('\n== cross-split item vector identity (shared item_id)')
for a, b in [('val', 'train'), ('test', 'train'), ('val', 'test')]:
    j = maps[a].join(maps[b], on='item', suffix='_b')
    print(f'  {a} vs {b}: shared items {len(j)} ({len(j) / len(maps[a]):.3f} of {a} items)')
    if len(j) == 0:
        continue
    jj = j.sample(min(len(j), 200000), seed=0)
    A = gather_emb(item_table(a), jj['idx'].to_numpy())
    B = gather_emb(item_table(b), jj['idx_b'].to_numpy())
    eq = (A == B).all(1)
    cs = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
    print(f'    exact equal rows: {eq.mean():.4f}; cosine mean={cs.mean():.4f}, q01/q10/q50={np.quantile(cs, [0.01, 0.1, 0.5])}; '
          f'mean |diff|={np.abs(A - B).mean():.3f}')

# share of val/test candidate items seen anywhere in train (cand or hist)
print('\n== candidate item overlap')
for s in ['val', 'test']:
    ci = np.unique(np.asarray(arr(s, 'cand_item_id')))
    tr = maps['train']['item'].to_numpy()
    print(f'  {s}: unique cand items {len(ci)}, seen in train (cand or hist with emb) {np.isin(ci, tr).mean():.4f}')
    # seen in this split's history (any user)
    hi = np.unique(np.asarray(arr(s, 'hist_item_id')))
    print(f'  {s}: unique cand items seen in same-split histories (any user) {np.isin(ci, hi).mean():.4f}')

# ---------- user embedding constancy ----------
print('\n== user embedding constancy')
f = feeds('train').select('fidx', 'user_id', 'timestamp')
U = arr('train', 'user_emb', mmap=False)
f = f.with_columns(((pl.col('timestamp') - pl.col('timestamp').min()) // 86400).alias('day'))
g = f.group_by('user_id').agg(pl.len().alias('n'))
print('  train feeds/user quantiles:', np.quantile(g['n'].to_numpy(), [0.5, 0.9, 0.99]), 'users with >1 feed:', (g['n'] > 1).mean())
multi = g.filter(pl.col('n') > 1)['user_id']
sub = f.filter(pl.col('user_id').is_in(multi.sample(min(len(multi), 100000), seed=0))).sort(['user_id', 'timestamp'])
uid = sub['user_id'].to_numpy(); fx = sub['fidx'].to_numpy(); day = sub['day'].to_numpy(); ts = sub['timestamp'].to_numpy()
same_user_next = uid[1:] == uid[:-1]
A = U[fx[:-1]][same_user_next].astype(np.float32); B = U[fx[1:]][same_user_next].astype(np.float32)
eq = (A == B).all(1)
cs = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
dd = (day[1:] - day[:-1])[same_user_next]
dt = (ts[1:] - ts[:-1])[same_user_next]
print(f'  consecutive feed pairs of same user: {len(eq)}, identical vector: {eq.mean():.4f}, cosine mean {cs.mean():.4f}')
for lab, m in [('same day', dd == 0), ('next day', dd == 1), ('2+ days', dd >= 2), ('dt<1h', dt < 3600), ('dt<10min', dt < 600)]:
    if m.sum():
        print(f'    {lab}: n={m.sum()}, identical {eq[m].mean():.4f}, cosine mean {cs[m].mean():.4f}, q10 {np.quantile(cs[m], 0.1):.3f}')
# distinct vectors per user per day
sub2 = sub.with_columns(pl.Series('h', [hash(U[i].tobytes()) for i in fx]))
pd_ = sub2.group_by(['user_id', 'day']).agg(pl.len().alias('n'), pl.col('h').n_unique().alias('nu'))
pd_ = pd_.filter(pl.col('n') > 1)
print(f'  user-days with >1 feed: {len(pd_)}, frac with a single distinct vector: {(pd_["nu"] == 1).mean():.4f}, mean distinct/feeds {(pd_["nu"] / pd_["n"]).mean():.3f}')
pu = sub2.group_by('user_id').agg(pl.len().alias('n'), pl.col('h').n_unique().alias('nu'), pl.col('day').n_unique().alias('nd'))
print(f'  per user: mean feeds {pu["n"].mean():.2f}, mean distinct vectors {pu["nu"].mean():.2f}, mean distinct days {pu["nd"].mean():.2f}')
# random pair baseline
i = rng.integers(0, len(U), 20000); j = rng.integers(0, len(U), 20000)
a = U[i].astype(np.float32); b = U[j].astype(np.float32)
print(f'  random user pair cosine mean {((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)).mean():.4f}')

# across splits: val user vs last train vector of same user
fv = feeds('val').select('fidx', 'user_id', 'timestamp')
Uv = arr('val', 'user_emb', mmap=False)
last_tr = f.sort('timestamp').group_by('user_id').agg(pl.col('fidx').last().alias('tr_fidx'))
jv = fv.join(last_tr, on='user_id')
print(f'  val feeds whose user appears in train: {len(jv) / len(fv):.4f}')
A = Uv[jv['fidx'].to_numpy()].astype(np.float32); B = U[jv['tr_fidx'].to_numpy()].astype(np.float32)
cs = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
print(f'  val user vec vs last train vec of same user: identical {(A == B).all(1).mean():.4f}, cosine mean {cs.mean():.4f} q10/q50/q90 {np.quantile(cs, [0.1, 0.5, 0.9])}')
# val users with multiple val feeds
gv = fv.sort(['user_id', 'timestamp'])
uid = gv['user_id'].to_numpy(); fx = gv['fidx'].to_numpy()
m = uid[1:] == uid[:-1]
A = Uv[fx[:-1]][m].astype(np.float32); B = Uv[fx[1:]][m].astype(np.float32)
cs = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
print(f'  val consecutive same-user pairs: {m.sum()}, identical {(A == B).all(1).mean():.4f}, cosine mean {cs.mean():.4f}')
print(f'  val users: {fv["user_id"].n_unique()} for {len(fv)} feeds; test users: {feeds("test")["user_id"].n_unique()} for {len(feeds("test"))} feeds')
ft = feeds('test')
print(f'  test feeds whose user appears in train: {ft["user_id"].is_in(f["user_id"].unique()).mean():.4f}; in val: {ft["user_id"].is_in(fv["user_id"].unique()).mean():.4f}')
