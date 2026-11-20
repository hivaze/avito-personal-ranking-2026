# Four cross-checks of the decoding (C1-C4): the share of delivery events per category, the home-region
# effect within feed and within (feed, category) as a Mantel-Haenszel odds ratio, the same for candidates
# near contacted items, and the guest marker e10 against user_id 0.
# Run without arguments on CPU with 12 threads (it imports lab/nn_x/codes/events/e05_intent.py).
# The command line is not on record.
# Produced logs/x01.log and logs/c1_delivery_share.csv (copies in ../results/).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/codes/synth/x01_skeptic_checks.py
#
"""Synthesiser skeptic checks (2026-11-13). CPU, 12 threads.
C1 delivery (e5/e6/e8) share per category: val + test + train 100K-feed sample (seed 0) -> reconcile the 3 analysts' lists.
C2 f2 'home level' (feed mode of non-zero f2) effect: MH OR within feed vs within (feed, category).
C3 'near e7/e9, not near e1' effect: MH OR within feed vs within (feed, category); same for e1 and e2/e3 as reference.
C4 e10 vs user_id 0 denominators.
Splits for C2/C3: val, and the events agent's 40K train-exploration sample (nn_x/codes/events/out/e05_train_explore40k_fidx.npy)."""
import os, sys, json
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[k] = '12'
import numpy as np, polars as pl
sys.path.insert(0, '<repo>/nn_x/codes/events')
from e05_intent import build
ROOT = '<repo>/data/flat'
VERT = json.load(open('<repo>/nn_x/codes/categories/out/code_vertical.json'))
rng = np.random.default_rng(0)


def ld(split, n, rows=None):
    x = np.load(f'{ROOT}/{split}/{n}.npy', mmap_mode='r')
    return np.asarray(x) if rows is None else np.asarray(x[rows])


def feeds(split):
    return pl.read_parquet(f'{ROOT}/{split}/feeds.parquet').sort('fidx')


def rows_of(F, off, n):
    o = F[off].to_numpy(); m = F[n].to_numpy()
    return np.concatenate([np.arange(a, a + b) for a, b in zip(o, m)])


def mh_strata(flag, tgt, strata, fid, B=200):
    """MH OR over strata; feed-cluster bootstrap CI of log OR."""
    s = strata
    a = np.bincount(s, weights=(flag & (tgt == 1)).astype(float)); b = np.bincount(s, weights=(flag & (tgt == 0)).astype(float))
    c = np.bincount(s, weights=(~flag & (tgt == 1)).astype(float)); d = np.bincount(s, weights=(~flag & (tgt == 0)).astype(float))
    n = np.maximum(a + b + c + d, 1)
    P = a * d / n; Q = b * c / n
    # stratum -> feed
    sf = np.zeros(len(a), np.int64); sf[s] = fid
    nf = fid.max() + 1
    Pf = np.bincount(sf, weights=P, minlength=nf); Qf = np.bincount(sf, weights=Q, minlength=nf)
    est = Pf.sum() / max(Qf.sum(), 1e-12)
    bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, nf, nf), minlength=nf)
        bs.append(np.log((w * Pf).sum() / max((w * Qf).sum(), 1e-12)))
    se = np.std(bs)
    inf = int(((a + b > 0) & (c + d > 0)).sum())
    return est, se, int(flag.sum()), inf


def C1():
    print('\n=== C1: delivery-type (e5/e6/e8) share of history events per category ===')
    out = {}
    for split, samp in [('val', None), ('test', None), ('train', 100_000)]:
        F = feeds(split)
        if samp:
            F = F.filter(pl.col('fidx').is_in(rng.choice(F.height, samp, replace=False))).sort('fidx')
            r = rows_of(F, 'hist_off', 'H')
        else:
            r = None
        cat = ld(split, 'hist_category_id', r).astype(np.int64); et = ld(split, 'hist_event_type', r).astype(np.int64)
        tot = np.bincount(cat, minlength=54); dl = np.bincount(cat, weights=np.isin(et, [5, 6, 8]), minlength=54)
        out[split] = (tot, dl)
    tot = sum(v[0] for v in out.values()); dl = sum(v[1] for v in out.values())
    rows = []
    for c in range(54):
        if tot[c] == 0: continue
        rows.append(dict(cat=c, vert=VERT.get(str(c), '?'), n_events=int(tot[c]), n_deliv=int(dl[c]), share=dl[c] / tot[c],
                         val=out['val'][1][c] / max(out['val'][0][c], 1), test=out['test'][1][c] / max(out['test'][0][c], 1),
                         train=out['train'][1][c] / max(out['train'][0][c], 1)))
    t = pl.DataFrame(rows).sort('share')
    with pl.Config(tbl_rows=60, float_precision=4, tbl_width_chars=200):
        print(t)
    t.write_csv('<repo>/nn_x/codes/synth/logs/c1_delivery_share.csv')


def C4():
    print('\n=== C4: e10 presence vs user_id 0 ===')
    for split in ['val', 'test']:
        F = feeds(split)
        fid = ld(split, 'hist_fidx').astype(np.int64); et = ld(split, 'hist_event_type')
        has10 = np.zeros(F.height, bool); has10[np.unique(fid[et == 10])] = True
        u0 = F['user_id'].to_numpy() == 0; H = F['H'].to_numpy()
        print(f'{split}: user0 feeds {u0.sum()} ({u0.mean():.3%}); P(e10|u0) all {has10[u0].mean():.3f}, H>0 {has10[u0 & (H > 0)].mean():.3f}; '
              f'P(e10|u>0) all {has10[~u0].mean():.4f}, H>0 {has10[~u0 & (H > 0)].mean():.4f}; share of u0 with H=0 {(H[u0] == 0).mean():.3f}')


def C23(name, split, fx):
    d = build(split, fx)
    F = feeds(split)
    if fx is not None:
        F = F.filter(pl.col('fidx').is_in(fx)).sort('fidx')
    r = rows_of(F, 'cand_off', 'M')
    cat = ld(split, 'cand_category_id', r).astype(np.int64)
    M = d['M']; tgt = d['tgt']; fs = d['fs']; mc = d['mc']; hasT = d['hasT']
    fid = np.repeat(np.arange(len(M)), M)
    s_feed = fid
    s_fc = fid * 54 + cat
    _, s_fc = np.unique(s_fc, return_inverse=True)
    vert = np.array([VERT.get(str(c), 'X') for c in range(54)])
    _, vcode = np.unique(vert, return_inverse=True)
    s_fv = np.unique(fid * 20 + vcode[cat], return_inverse=True)[1]
    print(f'\n=== {name}: {len(M)} feeds, {len(tgt):,} cands ===')
    # C2 f2 home level
    f2 = fs[:, 1]
    mode = np.zeros(len(M), np.float32)
    df = pl.DataFrame({'f': fid, 'v': f2}).filter(pl.col('v') > 0).group_by(['f', 'v']).len().sort(['f', 'len', 'v'], descending=[False, True, False]).group_by('f', maintain_order=True).first()
    mode[df['f'].to_numpy()] = df['v'].to_numpy()
    home = (f2 == mode[fid]) & (f2 > 0)
    print('C2 f2 home level (f2 == feed mode of non-zero f2):')
    for lab, sel in [('home vs all others', np.ones_like(home)), ('home vs other non-zero', f2 > 0)]:
        for sn, st in [('feed', s_feed), ('feed x vertical', s_fv), ('feed x category', s_fc)]:
            e, se, nf, ninf = mh_strata(home[sel], tgt[sel], st[sel], fid[sel])
            print(f'   {lab:24s} | strata {sn:16s}: MH OR {e:.3f}  (log se {se:.3f}, z {np.log(e)/se:+.1f})  n_flag {nf:,}  informative strata {ninf:,}')
    # composition of home rows by vertical (val)
    t = pl.DataFrame({'v': vert[cat], 'home': home, 'y': tgt}).group_by('v').agg(pl.col('home').mean().alias('home_share'), pl.col('y').mean().alias('ctr'), pl.len()).sort('ctr')
    with pl.Config(tbl_rows=20, float_precision=4):
        print(t)
    # C3 near-type flags
    print('C3 near-type flags (cos >= 0.8), feeds having the type(s):')
    near1 = mc[:, 0] >= 0.8
    defs = {
        'near e1': (near1, hasT[fid, 0]),
        'near e7|e9, not near e1': (((mc[:, 6] >= 0.8) | (mc[:, 8] >= 0.8)) & ~near1, hasT[fid, 6] | hasT[fid, 8]),
        'near e9, not near e1': ((mc[:, 8] >= 0.8) & ~near1, hasT[fid, 8]),
        'near e2|e3, not near e1': (((mc[:, 1] >= 0.8) | (mc[:, 2] >= 0.8)) & ~near1, hasT[fid, 1] | hasT[fid, 2]),
        'near e5|e6|e8, not near e1': (((mc[:, 4] >= 0.8) | (mc[:, 5] >= 0.8) | (mc[:, 7] >= 0.8)) & ~near1, hasT[fid, 4] | hasT[fid, 5] | hasT[fid, 7]),
        'near e4, not near e1': ((mc[:, 3] >= 0.8) & ~near1, hasT[fid, 3]),
    }
    for k, (fl, hf) in defs.items():
        fl = fl & hf
        for sn, st in [('feed', s_feed), ('feed x vertical', s_fv), ('feed x category', s_fc)]:
            e, se, nf, ninf = mh_strata(fl[hf], tgt[hf], st[hf], fid[hf])
            print(f'   {k:28s} | strata {sn:16s}: MH OR {e:.3f} (log se {se:.3f}, z {np.log(e)/se:+.1f}) n_flag {nf:,} inf {ninf:,}')
    # near e7/e9 composition by vertical
    fl = ((mc[:, 6] >= 0.8) | (mc[:, 8] >= 0.8)) & ~near1
    t = pl.DataFrame({'v': vert[cat], 'fl': fl}).group_by('v').agg(pl.col('fl').sum().alias('n_near79')).with_columns((pl.col('n_near79') / pl.col('n_near79').sum()).alias('share')).sort('share', descending=True)
    with pl.Config(tbl_rows=20, float_precision=3):
        print(t.head(6))


if __name__ == '__main__':
    C4()
    C1()
    C23('val', 'val', None)
    C23('train_explore40k', 'train', np.load('<repo>/nn_x/codes/events/out/e05_train_explore40k_fidx.npy'))
