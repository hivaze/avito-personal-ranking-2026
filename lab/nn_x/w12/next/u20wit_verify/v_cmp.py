"""u20wit_verify: (1) my own CPU logits on the sample vs the study's cached logits; (2) the transductive read re-derived with my own
NDCG / z / correlation code from the study's cached logits (used only after (1) validates them); (3) teacher check; (4) A/A null.
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit_verify/v_cmp.py > nn_x/w12/next/u20wit_verify/out_v_cmp.txt"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '4'
import numpy as np, polars as pl, numba as nb, itertools
from pathlib import Path

ROOT = Path('<repo>'); FLAT = ROOT / 'data/flat/train'; W = ROOT / 'nn_x/w5/preds'
HERE = ROOT / 'nn_x/w12/next/u20wit_verify'; C = HERE / 'cache'; ST = ROOT / 'nn_x/w12/next/u20wit/cache'
P = lambda *a: print(*a, flush=True)


@nb.njit(parallel=True, cache=False)
def my_ndcg(s, y, off, M):
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o = off[f]; m = M[f]; npos = 0
        for k in range(m):
            if y[o + k] > 0: npos += 1
        if npos == 0: continue
        dcg = 0.0
        for k in range(m):
            if y[o + k] > 0:
                sk = s[o + k]; r = 0
                for j in range(m):
                    sj = s[o + j]
                    if sj > sk: r += 1
                    elif sj == sk:
                        if y[o + j] <= 0: r += 1
                        elif j < k: r += 1
                dcg += 1.0 / np.log2(r + 2.0)
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


@nb.njit(parallel=True, cache=False)
def my_z(x, off, M):
    out = np.empty(len(x), np.float64)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; mu = 0.0
        for k in range(m): mu += x[o + k]
        mu /= m; v = 0.0
        for k in range(m): v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / m) + 1e-9
        for k in range(m): out[o + k] = (x[o + k] - mu) / sd
    return out


@nb.njit(parallel=True, cache=False)
def feed_corr(a, b, off, M):
    out = np.full(len(M), np.nan)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; ma = 0.0; mb = 0.0
        for k in range(m): ma += a[o + k]; mb += b[o + k]
        ma /= m; mb /= m; sab = 0.0; saa = 0.0; sbb = 0.0
        for k in range(m):
            da = a[o + k] - ma; db = b[o + k] - mb; sab += da * db; saa += da * da; sbb += db * db
        if saa > 0 and sbb > 0: out[f] = sab / np.sqrt(saa * sbb)
    return out


def ms(d):
    d = np.asarray(d, np.float64); return f'{d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(len(d)):.5f}'


def fold_of(uid, idx):
    u = uid.astype(np.uint64); h = ((u * np.uint64(2654435761)) % np.uint64(4294967296)) % np.uint64(5)
    return np.where(uid == 0, idx % 5, h.astype(np.int64))


f = pl.read_parquet(FLAT / 'feeds.parquet', columns=['user_id', 'M', 'cand_off'])
Mtr = f['M'].to_numpy().astype(np.int64); otr = f['cand_off'].to_numpy().astype(np.int64); uid = f['user_id'].to_numpy().astype(np.int64)
ytr = np.load(FLAT / 'cand_target.npy', mmap_mode='r')
V6 = [f'w5_e10_u20{a}_s{s}' for a in 'abc' for s in (951, 952)]

# ------------------------------------------------------------------ study caches: row maps (checked, not trusted)
sub_f = np.load(ST / 'sub/src_fidx.npy'); sub_rows = np.load(ST / 'sub/rows.npy'); sm = np.load(ST / 'sub_meta.npz')
xh_f = np.load(ST / 'subxh/src_fidx.npy'); xh_rows = np.load(ST / 'subxh/rows.npy')
for nme, ff, rr in (('sub', sub_f, sub_rows), ('subxh', xh_f, xh_rows)):
    exp = np.concatenate([np.arange(o, o + k) for o, k in zip(otr[ff], Mtr[ff])])
    assert len(exp) == len(rr) and (exp == rr).all(), nme
    P(f'{nme}: {len(ff)} feeds, {len(rr)} rows; rows.npy == my reconstruction of the train flat rows of src_fidx: True')
y_sub = np.asarray(ytr[sub_rows]).astype(np.float32); y_xh = np.asarray(ytr[xh_rows]).astype(np.float32)
assert (y_sub == np.load(ST / 'sub/cand_target.npy').astype(np.float32)).all() and (y_xh == np.load(ST / 'subxh/cand_target.npy').astype(np.float32)).all()
P('labels of the study caches == train cand_target at those rows: True')
Msub = Mtr[sub_f]; osub = np.concatenate([[0], np.cumsum(Msub)[:-1]]); Mxh = Mtr[xh_f]; oxh = np.concatenate([[0], np.cumsum(Mxh)[:-1]])
SETS = ('W0S', 'W0L', 'H0', 'W1S', 'W1L'); sid = sm['set_id']
mysets = np.load(C / 'sample.npz')

# ------------------------------------------------------------------ (1) my logits vs cached logits on the sample
P('\n=== (1) my independent CPU path vs the study cache, sample feeds (W0S 2000, W0L 1500, H0 600, XH 600) ===')
samp = {k: mysets[k] for k in ('W0S', 'W0L', 'H0', 'XH')}; fids = np.concatenate([samp[k] for k in samp]); Ms = Mtr[fids]; os_ = np.concatenate([[0], np.cumsum(Ms)[:-1]])
rows_s = np.concatenate([np.arange(o, o + k) for o, k in zip(otr[fids], Ms)]); y_s = np.asarray(ytr[rows_s]).astype(np.float32)
set_of_feed = np.concatenate([[k] * len(samp[k]) for k in samp]); is_xh_row = np.repeat(set_of_feed == 'XH', Ms)
pos_sub = np.searchsorted(sub_rows, rows_s[~is_xh_row]); assert (sub_rows[pos_sub] == rows_s[~is_xh_row]).all()
pos_xh = np.searchsorted(xh_rows, rows_s[is_xh_row]); assert (xh_rows[pos_xh] == rows_s[is_xh_row]).all()
mine = {}; theirs = {}
for v in V6 + ['w5_e10_xf0_s951']:
    p = C / f'my_{v}.npy'
    if not p.exists(): P(f'  {v}: my scores missing'); continue
    a = np.load(p); t = np.full(len(a), np.nan, np.float32)
    t[~is_xh_row] = np.load(ST / f'sc/{v}__tr.npy')[pos_sub]
    px = ST / f'scxh/{v}__tr.npy'
    if px.exists(): t[is_xh_row] = np.load(px)[pos_xh]
    ok = np.isfinite(t); d = np.abs(a - t)[ok]
    na = my_ndcg(a, y_s, os_, Ms); okf = np.array([ok[o:o + k].all() for o, k in zip(os_, Ms)]); nt = my_ndcg(np.nan_to_num(t), y_s, os_, Ms)
    mine[v] = na; theirs[v] = np.where(okf, nt, np.nan)
    P(f'  {v}: rows compared {int(ok.sum())} of {len(a)} | max |logit diff| {d.max():.2e} mean {d.mean():.2e} (logit sd {a.std():.2f}) | NDCG mine - cache on compared feeds {np.nanmean(na[okf] - nt[okf]):+.2e}')
if all(v in mine for v in V6):
    for k in samp:
        mk = set_of_feed == k
        dba = np.mean([mine[f'w5_e10_u20b_s{s}'][mk] - mine[f'w5_e10_u20a_s{s}'][mk] for s in (951, 952)], 0)
        dbt = np.mean([theirs[f'w5_e10_u20b_s{s}'][mk] - theirs[f'w5_e10_u20a_s{s}'][mk] for s in (951, 952)], 0)
        P(f'  sample {k} (n {int(mk.sum())}): B-A member from MY logits {ms(dba)} | from the cache, same feeds {ms(dbt)}')

# ------------------------------------------------------------------ (2) transductive read, my code on the cached logits
P('\n=== (2) transductive read re-derived (study cache logits, my NDCG / z; CPU fp32 vs CPU fp32) ===')
def load_sc(v, suf=''):
    p = ST / f'sc/{v}{suf}__tr.npy'; return np.load(p) if p.exists() else None
sc = {(v, suf): load_sc(v, suf) for v in V6 for suf in ('', '@last')}
nd = {}; zz = {}
for (v, suf), x in sc.items():
    if x is None: continue
    okf = np.array([np.isfinite(x[o:o + k]).all() for o, k in zip(osub, Msub)])
    xx = np.nan_to_num(x).astype(np.float32); e = my_ndcg(xx, y_sub, osub, Msub); e[~okf] = np.nan; nd[v, suf] = e; zz[v, suf] = my_z(xx, osub, Msub)
ndx = {}; zx = {}
for v in V6:
    x = np.load(ST / f'scxh/{v}__tr.npy'); assert np.isfinite(x).all(); ndx[v] = my_ndcg(x, y_xh, oxh, Mxh); zx[v] = my_z(x, oxh, Mxh)
def member(arm, ref, mk, suf=''):
    return np.mean([nd[f'w5_e10_u20{arm}_s{s}', suf][mk] - nd[f'w5_e10_u20{ref}_s{s}', suf][mk] for s in (951, 952)], 0)
def pool2(arm, mk, suf=''):
    z = np.mean([zz[f'w5_e10_u20{arm}_s{s}', suf] for s in (951, 952)], 0).astype(np.float32); return my_ndcg(z, y_sub, osub, Msub)[mk]
POOL = {(arm, suf): my_ndcg(np.mean([zz[f'w5_e10_u20{arm}_s{s}', suf] for s in (951, 952)], 0).astype(np.float32), y_sub, osub, Msub)
        for arm in 'abc' for suf in ('', '@last') if all((f'w5_e10_u20{arm}_s{s}', suf) in zz for s in (951, 952))}
R = {}
P('set | n | B-A member | per seed | B-A 2-pool | B-A member @last | C-A member (IN-SAMPLE on W0*) | A level | B level')
MASKS = {'W0S': sid == 0, 'W0L': sid == 1, 'H0': sid == 2, 'W0S+W0L': (sid == 0) | (sid == 1)}
for k, mk in MASKS.items():
    d = member('b', 'a', mk); R[k] = d
    ps = ', '.join(f'{np.mean(nd[f"w5_e10_u20b_s{s}", ""][mk] - nd[f"w5_e10_u20a_s{s}", ""][mk]):+.5f}' for s in (951, 952))
    dl = member('b', 'a', mk, '@last') if all((f'w5_e10_u20{a}_s{s}', '@last') in nd for a in 'ab' for s in (951, 952)) else None
    R[k, 'last'] = dl
    dc = member('c', 'a', mk); R[k, 'c'] = dc
    P(f'{k} | {int(mk.sum())} | {ms(d)} | ({ps}) | {ms(POOL["b", ""][mk] - POOL["a", ""][mk])} | {ms(dl) if dl is not None and np.isfinite(dl).all() else "n/a"} | {ms(dc)} | '
      f'{np.mean([nd[f"w5_e10_u20a_s{s}", ""][mk].mean() for s in (951, 952)]):.5f} | {np.mean([nd[f"w5_e10_u20b_s{s}", ""][mk].mean() for s in (951, 952)]):.5f}')
dxh = np.mean([ndx[f'w5_e10_u20b_s{s}'] - ndx[f'w5_e10_u20a_s{s}'] for s in (951, 952)], 0); dxc = np.mean([ndx[f'w5_e10_u20c_s{s}'] - ndx[f'w5_e10_u20a_s{s}'] for s in (951, 952)], 0)
pxa, pxb = (my_ndcg(np.mean([zx[f'w5_e10_u20{a}_s{s}'] for s in (951, 952)], 0).astype(np.float32), y_xh, oxh, Mxh) for a in 'ab')
P(f'XH | {len(Mxh)} | {ms(dxh)} | ({", ".join(f"{np.mean(ndx[f"w5_e10_u20b_s{s}"] - ndx[f"w5_e10_u20a_s{s}"]):+.5f}" for s in (951, 952))}) | {ms(pxb - pxa)} | n/a | {ms(dxc)} (out-of-sample)')
def diff(a, b):
    return f'{a.mean() - b.mean():+.5f} +- {np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)):.5f}'
P(f'excess of B-A: W0L - XH {diff(R["W0L"], dxh)} | W0S - XH {diff(R["W0S"], dxh)} | W0L - H0 {diff(R["W0L"], R["H0"])} | H0 - XH {diff(R["H0"], dxh)}')
if R['W0L', 'last'] is not None: P(f'   @last: W0L - H0 {diff(R["W0L", "last"], R["H0", "last"])}')
# short witness by M bin
Mb = Msub
for lo, hi in ((5, 10), (10, 20), (20, 35), (35, 50)):
    mk = (sid == 0) & (Mb >= lo) & (Mb < hi); P(f'   W0S M {lo}-{hi - 1}: n {int(mk.sum())} B-A member {ms(member("b", "a", mk))} | positives per feed {np.mean([y_sub[o:o + k].sum() for o, k in zip(osub[mk], Msub[mk])]):.3f}')
# feeds without a positive
npos = np.add.reduceat(y_sub, osub); P(f'   witness feeds without a positive: W0S {int((npos[sid == 0] == 0).sum())} W0L {int((npos[sid == 1] == 0).sum())} H0 {int((npos[sid == 2] == 0).sum())}')
# time split inside W0L (first half vs second half of the fold-0 witness by timestamp): stability of B-A
ts = sm['ts']
for k, s_ in (('W0S', 0), ('W0L', 1)):
    mk = sid == s_; med = np.median(ts[mk]); P(f'   {k} by time: early half {ms(member("b", "a", mk & (ts <= med)))} | late half {ms(member("b", "a", mk & (ts > med)))}')
# user-0 bucket inside the witness (those feeds' "user" is also in A's training set)
u0 = sm['uid'] == 0
for k, s_ in (('W0S', 0), ('W0L', 1)):
    mk = sid == s_; P(f'   {k} without the user-0 bucket: n {int((mk & ~u0).sum())} B-A member {ms(member("b", "a", mk & ~u0))}')

# ------------------------------------------------------------------ (3) teacher checks
P('\n=== (3) teacher: cached CPU xf0 3-mean vs teacher_xf0.npy / 3 (file is the SUM of n members), within-feed corr; B and A match to the teacher ===')
P('teacher_xf0.n =', open(ROOT / 'nn_x/w12/kd/out/teacher_xf0.n').read().strip())
T = np.load(ROOT / 'nn_x/w12/kd/out/teacher_xf0.npy', mmap_mode='r'); t_sub = np.asarray(T[sub_rows]).astype(np.float64) / 3
x3 = [load_sc(f'w5_e10_xf0_s{s}') for s in (951, 952, 953)]
if all(x is not None for x in x3):
    m3 = np.mean(x3, 0).astype(np.float64); cc = feed_corr(m3, t_sub, osub, Msub)
    for j, k in enumerate(SETS):
        mk = (sid == j) & np.isfinite(cc); P(f'  {k}: n {int(mk.sum())} mean within-feed corr {cc[mk].mean():.6f} median {np.median(cc[mk]):.6f} share >= 0.999 {np.mean(cc[mk] >= 0.999):.4f} min {cc[mk].min():.4f}')
    c1 = feed_corr(x3[0].astype(np.float64), t_sub, osub, Msub); P(f'  single member xf0_s951 vs teacher: W0L mean corr {np.nanmean(c1[sid == 1]):.4f}')
    tn = my_ndcg(t_sub.astype(np.float32), y_sub, osub, Msub)
    P('  teacher NDCG on W0S / W0L / H0: ' + ' / '.join(f'{tn[sid == j].mean():.5f}' for j in range(3)))
for arm in 'abc':
    cs = np.mean([feed_corr(np.nan_to_num(sc[f'w5_e10_u20{arm}_s{s}', '']).astype(np.float64), t_sub, osub, Msub) for s in (951, 952)], 0)
    R['corr', arm] = cs
    P(f'  u20{arm} within-feed corr to the teacher: W0S {np.nanmean(cs[sid == 0]):.5f} W0L {np.nanmean(cs[sid == 1]):.5f} H0 {np.nanmean(cs[sid == 2]):.5f}')
dcr = R['corr', 'b'] - R['corr', 'a']
P('  B minus A corr to the teacher: ' + ' | '.join(f'{k} {ms(dcr[(sid == j) & np.isfinite(dcr)])}' for j, k in enumerate(SETS[:3])))
a_, b_ = dcr[(sid == 1) & np.isfinite(dcr)], dcr[(sid == 2) & np.isfinite(dcr)]
P(f'  seen (W0L) minus unseen (H0): {diff(a_, b_)}')

# ------------------------------------------------------------------ (4) base - xf0 known answer on the witness (cached CPU logits)
P('\n=== (4) known answer: base - xf0 member (3 seeds), cached CPU logits ===')
bx = {}
for arm in ('base', 'xf0', 'xf1'):
    for s in (951, 952, 953):
        x = load_sc(f'w5_e10_{arm}_s{s}')
        if x is not None: bx[arm, s] = my_ndcg(np.nan_to_num(x).astype(np.float32), y_sub, osub, Msub)
for j, k in enumerate(SETS):
    mk = sid == j
    if all(np.isfinite(load_sc(f'w5_e10_{arm}_s951')[np.repeat(mk, Msub)]).all() for arm in ('base', 'xf0')):
        P(f'  {k}: n {int(mk.sum())} base - xf0 {ms(np.mean([bx["base", s][mk] - bx["xf0", s][mk] for s in (951, 952, 953)], 0))}')

# ------------------------------------------------------------------ (5) A/A null of the declared statistic (cross-user holdout, 2 seeds per arm)
P('\n=== (5) A/A null of "mean of 2 members minus mean of 2 other members" on the cross-user holdout (stored GPU files) ===')
pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz'); pold = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); pold[Mtr < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((pold > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W / 'holdout/holdout_y.npy').astype(np.float32)
xfh = fold_of(uid, np.arange(len(uid)))[hold]; cu = xfh != 0
for arm in ('kd05w192', 'kd05t25w192'):
    lv = []; lvl = []
    for s in range(951, 959):
        for suf, L in (('', lv), ('@last', lvl)):
            p = W / f'holdout/nn_w5_e10_{arm}_s{s}{suf}_holdout.npy'
            if p.exists(): L.append(my_ndcg(np.load(p).astype(np.float32), yh, oh, Mh)[cu].mean())
    for tag, L in (('selected', lv), ('@last', lvl)):
        if len(L) < 4: continue
        L = np.array(L); st = [L[list(a)].mean() - L[list(b)].mean() for a in itertools.combinations(range(len(L)), 2) for b in itertools.combinations([i for i in range(len(L)) if i not in a], 2)]
        st = np.array(st)
        P(f'  {arm} {tag}: {len(L)} members, member cross-user holdout NDCG sd {L.std(ddof=1):.5f} (range {L.min():.5f}..{L.max():.5f}); statistic sd {st.std():.5f} over {len(st)} arrangements; '
          f'share >= +0.00061: {np.mean(st >= 0.00061):.4f}; share >= +0.00040: {np.mean(st >= 0.00040):.4f}; share |x| >= 0.00061: {np.mean(np.abs(st) >= 0.00061):.4f}')
P('DONE')
