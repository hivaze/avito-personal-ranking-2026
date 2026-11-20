"""witness: shared helpers of the staleness-witness tools (CPU only; read-only on every file outside nn_x/w12/stale/witness/).

KEY = witness (reports/STALE_1115/witness.md). Everything here is label-free except `Labels` (used by the read-outs only).
Conventions: per-feed full-list NDCG, binary gain, pessimistic ties; member score = per-feed z (src/stack_val.feed_rank_z[:, 1]);
pool = member mean of z; a weighted pool = sum_g w_g * z(pool_g); SE = sd(per-feed paired difference, ddof 1) / sqrt(n feeds).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
NT = str(min(12, int(os.environ.get('WT_THREADS', '12'))))          # at most 12; light read-outs set WT_THREADS=4
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = NT
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = '<repo>/nn_x/w12/stale/witness/cache/numba'   # numba caches of imported project modules land here
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import numba as nb

ROOT = Path('<repo>')
FLAT = ROOT / 'data/flat'
FC = ROOT / 'nn/feat_cache'
HERE = ROOT / 'nn_x/w12/stale/witness'
CACHE = HERE / 'cache'
LOGS = HERE / 'logs'
SUB = CACHE / 'sub'                      # mini split of the SUBSET feeds (holdout days >= Sep 13)
PRED = CACHE / 'pred'                    # pred/<view>/<member>.npy   (subset rows, CPU fp32)
SV = CACHE / 'staleval'                  # staleval/nn_<member>_staleval.npy, staleval/nn_<member>_valbase.npy (val rows, CPU fp32)
STUDY = ROOT / 'nn_x/w12/next/poplag/cache'            # lagged columns + staged predictions of the poplag study (read-only)
VERIFY = ROOT / 'nn_x/w12/next/poplag_verify/cache'    # reviewer's public-like predictions on the subset (read-only)
for _p in (CACHE, LOGS, SUB, PRED, SV, CACHE / 'numba', CACHE / 'locks'):
    _p.mkdir(parents=True, exist_ok=True)
TZ = 3 * 3600
DAY = 86400
SEP6 = 20702                             # (ts + TZ) // DAY of 2026-09-06 MSK
N_SUB_FEEDS, N_SUB_ROWS = 13424, 2089648
LAGCOLS = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo',
           'it_hist_et4_luo', 'xp_imp_c50']
VIEWS = {'lag0': None, 'priv': 'priv_v3', 'pub': 'pub_v3'}    # my view name -> the study's view tag

S4 = (951, 952, 953, 954)
S8 = tuple(range(951, 959))
G = {
    'W8': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4],
    'W16': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'W16new': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in (955, 956, 957, 958)],
    'N8': [f'w5_e10_{a}_s{s}' for a in ('kd05', 'kd05t25') for s in S4],                 # the 8 narrow (d128) students of S16
    'S16': [f'w5_e10_{a}_s{s}' for a in ('kd05', 'kd05t25', 'kd05w192', 'kd05t25w192') for s in S4],
    'base4': [f'w5_e10_base_s{s}' for s in S4],
    'pop1kd': [f'abl_e10_pop1kd05t25_s{s}' for s in S4],                                  # popularity-light students, d128 (W8P20 members)
    'pop1kdw': [f'abl_e10_pop1kd05w192_s{s}' for s in S4],                                # popularity-light students, d192
    'pop1x8': [f'abl_e10_pop1_s{s}' for s in S8],                                         # hard-label popularity-light members of C3
    'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)],
    'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
    'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)],
}
G['G3'] = G['F10'] + G['F7x'] + G['F7x70']
G['C3'] = G['G3'] + G['pop1x8']
# reduced, same-proportion pools of pair (iv) (DECLARATION block 4 (6)): the first seeds of each group
G['F10h'] = G['F10'][:3]; G['F7xh'] = G['F7x'][:3]; G['F7x70h'] = G['F7x70'][:3]; G['pop1h'] = G['pop1x8'][:4]
G3H = '0.3~F10h+0.35~F7xh+0.35~F7x70h'                                   # raw-weighted group means = member-equal G3 with fewer seeds
C3H = '0.2142857143~F10h+0.25~F7xh+0.25~F7x70h+0.2857142857~pop1h'        # = (6 F10 + 7 F7x + 7 F7x70 + 8 pop1) / 28
for _a in ('hz', 'ref', 'ref2'):
    G[_a] = [f'w5_{st}cd_{_a}_s{s}' for st in 'kl' for s in S8]


def fam(v):
    """(family dir, bare variant) of a member name; '@last' stays part of the variant."""
    if v.startswith('w5_'):
        return ROOT / 'nn_x/w5', v
    if v.startswith('abl_'):
        return ROOT / 'nn_x/abl', v
    if v.startswith('r6_'):
        return ROOT / 'nn_x/r6', v
    return ROOT / 'nn', v                 # v23_* and other nn/ members


def ckpt_path(v):
    d, b = fam(v)
    return d / 'ckpt' / f'{b}.pt'


def done_json(v):
    """A run is finished when <family>/logs/<variant>.json exists (variant without '@last')."""
    d, b = fam(v)
    return d / 'logs' / f"{b.replace('@last', '')}.json"


def stored_val(v):
    """Stored GPU bf16 train-only val predictions of a member (project layout, as pc_common.paths)."""
    d, b = fam(v)
    if v.startswith(('w5_', 'abl_e10_pop1kd')):
        return d / 'preds' / f'nn_{b}_trainonly_val.npy'
    if v.startswith('abl_'):
        p = d / 'preds' / f'nn_{b}_trainonly_val.npy'
        return p if p.exists() else ROOT / 'preds' / f'nn_{b}_trainonly_val.npy'
    return ROOT / 'preds' / f'nn_{b}_trainonly_val.npy'


def stored_holdout(v):
    d, b = fam(v)
    if v.startswith(('w5_', 'abl_')):
        return d / 'preds/holdout' / f'nn_{b}_holdout.npy'
    return ROOT / 'preds/holdout' / f'nn_{b}_holdout.npy'


def stored_stale(v):
    """Stored GPU stale-val file (nn_x/<fam>/preds/stale) if one exists."""
    d, b = fam(v)
    return d / 'preds/stale' / f'nn_{b}_staleval.npy'


def xt_transform(name, x):
    """nn_x/w5/data.py SplitData.attach_xt transform of one raw column (same branches, same order), float32 numpy."""
    x = np.asarray(x).astype(np.float32)
    if name.startswith('has:') or name.endswith('_rk'):
        pass
    elif name.endswith('_z'):
        x = np.clip(x, -5, 5)
    elif name.startswith(('seq_cv', 'it_hist')):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


def holdout_feeds():
    """nn_x/w5/train.py --holdout 30000: p_random > 0.9 (NaN -> 0, M < 50 -> 0), M >= 50, rng 12345. Returns (feeds frame, hold, p_random)."""
    import polars as pl
    f = pl.read_parquet(FLAT / 'train/feeds.parquet')
    M = f['M'].to_numpy().astype(np.int64)
    pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')
    assert (pr['fidx'] == np.arange(len(f))).all()
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    return f, hold, p


# ----------------------------------------------------------------------------------------------------------------- metric
@nb.njit(parallel=True, cache=False)
def _ndcg(s, y, off, M):
    """Copy of nn_x/w12/recheck/pool_composition/pc_common._ndcg (numba form of src/metrics.ndcg_per_feed, pessimistic ties)."""
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m]; yv = y[o:o + m]
        npos = 0
        for k in range(m):
            if yv[k] > 0: npos += 1
        if npos == 0: continue
        order = np.argsort(-sv.astype(np.float64) + 0.0, kind='mergesort')
        dcg = 0.0
        i = 0
        while i < m:
            j = i
            while j + 1 < m and sv[order[j + 1]] == sv[order[i]]: j += 1
            if j == i:
                if yv[order[i]] > 0: dcg += 1.0 / np.log2(i + 2.0)
            else:
                cp = 0
                for k in range(i, j + 1):
                    if yv[order[k]] > 0: cp += 1
                for k in range(j - cp + 1, j + 1): dcg += 1.0 / np.log2(k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


def ndcg_fast(s, y, off, M):
    return _ndcg(np.ascontiguousarray(s, np.float32), np.ascontiguousarray(y, np.float32), np.ascontiguousarray(off, np.int64),
                 np.ascontiguousarray(M, np.int64))


@nb.njit(parallel=True, cache=False)
def _feed_z(off, M, x):
    """Column 1 of src/stack_val.feed_rank_z: (v - mean) / (sd + 1e-9) per feed, float32."""
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m):
            out[o + k] = (v[k] - mu) / sd
    return out


def feed_z(off, M, x):
    return _feed_z(np.ascontiguousarray(off, np.int64), np.ascontiguousarray(M, np.int64), np.ascontiguousarray(x, np.float32))


def se(d):
    d = np.asarray(d, np.float64)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def ms(d, w=None):
    """(mean, SE); with weights: weighted mean and the n_eff SE of src/pmix.wmean_se."""
    d = np.asarray(d, np.float64)
    if w is None:
        return float(d.mean()), se(d)
    m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum()
    return float(m), float(np.sqrt(np.average((d - m) ** 2, weights=w) / neff))


def fmt(d, w=None):
    m, s = ms(d, w)
    return f'{m:+.5f} +- {s:.5f}'


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def save_atomic(path, arr):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('tmp_' + str(os.getpid()) + '_' + path.name)
    np.save(tmp, arr); os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------------- CPU batching
SUB_BUDGET, SUB_MAXF = 2048, 32      # measured 2026-11-15T02:40Z (logs/bench2.log): 3.5x the throughput of the 65536 / 1024 batches on 12 CPU threads


def anchored_batches(s, feed_ids=None, sub_budget=SUB_BUDGET, sub_maxf=SUB_MAXF):
    """Batches for CPU inference with the SAME padding as common.predict's batches (cand_budget 65536, max_feeds 1024, no shuffle):
    every reference batch (one M bucket) is cut into sub-batches of about sub_budget candidates, and each sub-batch holds at least one feed whose
    history pads to the reference batch's Hx (SplitData.gather pads the history to the batch maximum, and the model output depends on that
    padding: logs/bench.log, max |diff| 5.6e-2 without the anchor, 1.5e-5 with it). Label-free; uses SplitData.batches unmodified."""
    feed_ids = np.arange(s.n) if feed_ids is None else feed_ids
    out = []
    for fb in s.batches(feed_ids, 65536, 1024, shuffle=False):
        fb = np.asarray(fb)
        hx = np.minimum(np.maximum(-(-np.minimum(s.H_np[fb], s.max_hist) // 64) * 64, 64), s.max_hist)
        anc = fb[hx == hx.max()]; rest = fb[hx != hx.max()]
        Mp = int(-(-s.M_np[fb].max() // 16) * 16)
        B = max(1, min(sub_maxf, sub_budget // Mp))
        nsub = min(max(1, -(-len(fb) // B)), len(anc))
        order = np.concatenate([anc, rest])
        out += [order[i::nsub] for i in range(nsub)]
    return out
