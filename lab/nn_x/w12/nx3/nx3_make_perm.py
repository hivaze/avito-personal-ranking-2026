"""nx3 stage A, VALUE clause of amendment A3: the two NX value columns PERMUTED WITHIN THE FEED (one permutation per feed, the same for both
columns; the NaN pattern, hence has:nx_rp3_z, is unchanged). CPU only, label-free, write-new: nothing existing is read for writing.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
     POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/nx3/nx3_make_perm.py

Reads   nn/feat_cache/{val,train}/nx_rp3_z.npy, nx_rp3_rk.npy (the installed files of STEP 0), data/flat/{val,train}/feeds.parquet,
        eda/labels_position/out/train_prandom.npz (the fixed holdout draw, as train.py --holdout 30000). No label array, no prediction.
Writes  nn_x/w12/nx3/perm/val/nx_rp3_{z,rk}.npy     every val feed permuted (1,389,971 rows)
        nn_x/w12/nx3/perm/train/nx_rp3_{z,rk}.npy   copy of the train file with the rows of the 30,000 holdout feeds permuted within the feed
                                                     (a no-op on the 14,775 has = 0 holdout feeds, whose rows are all NaN); every other row unchanged
        nn_x/w12/nx3/perm/make_perm.txt             the checks below
Permutation: numpy default_rng(SEED) with SEED = 20261116, one uniform permutation of the M rows of each feed, drawn in feed order
(val: all 14,423 feeds; train: the 30,000 holdout feeds in sorted order), applied to BOTH columns. Exclusive create: existing files are never overwritten.
Checks printed: (z, rk) pairs of every feed are the same multiset before and after; NaN pattern equal; rows outside the permuted feeds bitwise equal;
share of rows that moved; the two columns moved by the same permutation.
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '12')
sys.dont_write_bytecode = True
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path('<repo>'); FC = ROOT / 'nn/feat_cache'; FLAT = ROOT / 'data/flat'
OUT = ROOT / 'nn_x/w12/nx3/perm'; SEED = 20261116; NAMES = ('nx_rp3_z', 'nx_rp3_rk')


def holdout_feeds(f):
    """Exactly nn_x/abl/train.py --holdout 30000 (and hz2_common.holdout_feeds): p_random > 0.9 (NaN -> 0, M < 50 -> 0), M >= 50, rng 12345."""
    M = f['M'].to_numpy().astype(np.int64)
    pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')
    assert (pr['fidx'] == np.arange(len(f))).all()
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    return np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))


def main():
    lines = []

    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)
    for sp in ('val', 'train'):
        (OUT / sp).mkdir(parents=True, exist_ok=True)
        for n in NAMES:
            assert not (OUT / sp / f'{n}.npy').exists(), f'{OUT / sp / n}.npy exists: nothing is overwritten (delete nothing; use the existing files)'
    P(f'nx3_make_perm: SEED {SEED}; one permutation per feed for both columns {NAMES}')
    for sp in ('val', 'train'):
        f = pl.read_parquet(FLAT / sp / 'feeds.parquet')
        M = f['M'].to_numpy().astype(np.int64); off = f['cand_off'].to_numpy().astype(np.int64)
        feeds = np.arange(len(f)) if sp == 'val' else holdout_feeds(f)
        z = np.load(FC / sp / 'nx_rp3_z.npy'); rk = np.load(FC / sp / 'nx_rp3_rk.npy')
        assert z.dtype == np.float32 and rk.dtype == np.float32 and len(z) == len(rk) == int(M.sum())
        rng = np.random.default_rng(SEED)
        src = np.arange(len(z), dtype=np.int64)           # src[i] = row whose value goes to row i
        for fi in feeds:
            o, m = off[fi], M[fi]
            src[o:o + m] = o + rng.permutation(m)
        zp, rp = z[src], rk[src]
        rows = np.concatenate([np.arange(off[fi], off[fi] + M[fi]) for fi in feeds]); inmask = np.zeros(len(z), bool); inmask[rows] = True
        # checks
        nan_same = bool((np.isnan(zp) == np.isnan(z)).all() and (np.isnan(rp) == np.isnan(rk)).all())
        out_same = bool((zp[~inmask].view(np.uint32) == z[~inmask].view(np.uint32)).all() and (rp[~inmask].view(np.uint32) == rk[~inmask].view(np.uint32)).all())
        fid = np.repeat(np.arange(len(M)), M)
        assert (fid[src] == fid).all(), 'a value left its feed'
        fs = fid[rows]; a0, b0, a1, b1 = (np.ascontiguousarray(x[rows]).view(np.uint32) for x in (z, rk, zp, rp))   # rows of the permuted feeds only
        k0, k1 = np.lexsort((b0, a0, fs)), np.lexsort((b1, a1, fs))                   # sort rows by (feed, z bits, rk bits)
        pairs_same = bool((a0[k0] == a1[k1]).all() and (b0[k0] == b1[k1]).all())
        moved = float((src[inmask] != np.arange(len(z))[inmask]).mean())
        has_rows = inmask & ~np.isnan(z)
        chg_z = float((zp[has_rows] != z[has_rows]).mean()) if has_rows.any() else float('nan')
        chg_rk = float((rp[has_rows] != rk[has_rows]).mean()) if has_rows.any() else float('nan')
        n_has_feeds = int((~np.isnan(z[off[feeds]])).sum())
        P(f'[{sp}] rows {len(z):,}; permuted feeds {len(feeds):,} ({n_has_feeds:,} with the view, {len(feeds) - n_has_feeds:,} all-NaN), rows in them {int(inmask.sum()):,}')
        P(f'[{sp}] NaN pattern unchanged: {nan_same}; rows outside the permuted feeds bitwise equal: {out_same}; per-feed multiset of (z, rk) pairs preserved: {pairs_same}')
        P(f'[{sp}] share of rows of the permuted feeds whose source row changed: {moved:.4f}; share of non-NaN rows whose value changed: z {chg_z:.4f}, rk {chg_rk:.4f}')
        assert nan_same and out_same and pairs_same
        for n, a in (('nx_rp3_z', zp), ('nx_rp3_rk', rp)):
            p = OUT / sp / f'{n}.npy'
            with open(p, 'xb') as fh:                       # exclusive create
                np.save(fh, a.astype(np.float32))
            P(f'[{sp}] wrote {p} {a.shape} float32')
        del z, rk, zp, rp, src
    (OUT / 'make_perm.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
