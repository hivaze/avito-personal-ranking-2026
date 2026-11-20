"""Build a KD target file with label-conditional (co-click) smoothing, for a zero-patch screen of bet CO-CLICK:
   logit'_j = teacher_mean11_j + B1 * z_j + B2 * (sc_j - mean sc)
   z_j  = within-feed z-score of max cosine(item emb j, item emb of the OTHER clicked items of the feed)
   sc_j = 1 if j shares its category with another clicked item of the feed
 Coefficients B1 0.33, B2 0.15 = conditional-logit fit on the 30K holdout (p2_coclick.py, leave-one-positive-out, user halves).
 Train-only target; no model input changes; nothing from val/test. Output: out/teacher_mean11_coclick.npy (float32, flat train order).
 usage: build_coclick_kd.py [n_proc=8]   (CPU only)"""
import sys, os, json, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
import numpy as np, polars as pl
from multiprocessing import Pool
ROOT = '<repo>'; B1, B2 = 0.33, 0.15
f = pl.read_parquet(f'{ROOT}/data/flat/train/feeds.parquet', columns=['M', 'cand_off'])
M = f['M'].to_numpy().astype(np.int64); off = f['cand_off'].to_numpy().astype(np.int64); n = len(M)
dd = f'{ROOT}/data/competition/train/item_embeddings'; mt = json.loads(open(f'{dd}/data.meta').read())
def work(rng_):
    a, b = rng_
    table = np.memmap(f'{dd}/data.memmap', dtype=np.int8, mode='r', shape=tuple(mt['shape']))
    lo, hi = off[a], off[b - 1] + M[b - 1]
    ei = np.load(f'{ROOT}/data/flat/train/cand_emb_idx.npy', mmap_mode='r')[lo:hi]; ct = np.load(f'{ROOT}/data/flat/train/cand_category_id.npy', mmap_mode='r')[lo:hi]
    y = np.asarray(np.load(f'{ROOT}/data/flat/train/cand_target.npy', mmap_mode='r')[lo:hi]) > 0
    ei = np.asarray(ei); ct = np.asarray(ct); add = np.zeros(hi - lo, np.float32)
    for i in range(a, b):
        s0 = off[i] - lo; sl = slice(s0, s0 + M[i]); yy = y[sl]; Pp = np.where(yy)[0]
        if len(Pp) == 0: continue
        e = ei[sl]; E = np.asarray(table[np.maximum(e, 0)], np.float32); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6); E[e < 0] = 0
        c = ct[sl]; S = E @ E[Pp].T; sim = S.max(1).astype(np.float64); sc = np.isin(c, c[Pp]).astype(np.float64)
        for k, ip in enumerate(Pp):
            if len(Pp) > 1:
                Rk = [kk for kk in range(len(Pp)) if kk != k]; sim[ip] = S[ip, Rk].max(); sc[ip] = float(c[ip] in set(c[Pp[Rk]].tolist()))
            else:
                sim[ip] = np.nan; sc[ip] = 0.0
        ok = ~np.isnan(sim)
        if ok.sum() < 2: continue
        z = np.zeros(len(sim)); z[ok] = (sim[ok] - sim[ok].mean()) / (sim[ok].std() + 1e-9)
        v = B1 * z + B2 * (sc - sc[ok].mean()); add[sl] = (v - v.mean()).astype(np.float32)
    return a, lo, add
if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t0 = time.time(); step = 20000; jobs = [(a, min(a + step, n)) for a in range(0, n, step)]
    t = np.load(f'{ROOT}/nn_x/w12/kd/out/teacher_mean11.npy').astype(np.float32); out = t.copy(); tot = 0.0
    with Pool(nproc) as p:
        for k, (a, lo, add) in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
            out[lo:lo + len(add)] += add; tot += float((add.astype(np.float64) ** 2).sum())
            if k % 20 == 0: print(f'{k}/{len(jobs)} chunks {time.time()-t0:.0f}s', flush=True)
    print('rms of the added term', (tot / len(out)) ** 0.5, 'teacher sd', float(t[:5000000].std()), flush=True)
    np.save(f'{ROOT}/nn_x/w12/recheck/long_bets/out/teacher_mean11_coclick.npy', out); print('DONE', time.time() - t0)
