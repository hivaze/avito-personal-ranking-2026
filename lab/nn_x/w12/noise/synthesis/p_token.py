"""CPU probe: how much does the domain token (production row 0 vs random-order row 1) change a member's train scores?

Question behind it: the teacher was dumped with the training-time domain mix (production row for production feeds and for
every feed with M < 50). Would a teacher scored with the random-order row (the row used on val/test) give a materially
different, position-debiased soft target?  And does the token already absorb the production click-by-rank excess?

Usage: CUDA_VISIBLE_DEVICES="" python p_token.py <checkpoint name> [n per group]   -> out/p_token_<ckpt>.txt
"""
import sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, '<repo>')
from avito_ranker.model import load_ranker
from avito_ranker.data.loader import load_split
from avito_ranker.training.predict import predict
from avito_ranker.metrics import ndcg_per_feed

torch.set_num_threads(12)
ROOT = Path('<repo>')
OUT = ROOT / 'lab/nn_x/w12/noise/synthesis/out'
name = sys.argv[1]
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
t0 = time.time()
model = load_ranker(ROOT / f'artifacts/members/checkpoints/{name}.pt', 'cpu')
split = load_split('train', model.config.view, 'cpu')
print('loaded', split.summary(), f'{time.time()-t0:.0f}s', flush=True)
M = split.feed_sizes; off = split.candidate_offsets
p_old = np.nan_to_num(np.load(ROOT / 'artifacts/exploration/train_prandom.npz')['p_random'].astype(np.float32), nan=-1)
p_v2 = np.load(ROOT / 'lab/eda/labels_position/out/train_prandom_v2.npz')['p_random'].astype(np.float32)
rng = np.random.default_rng(20261118)
groups = {
    'prod_M50+ (old p<0.05, v2 p<0.05)': np.flatnonzero((M >= 50) & (p_old >= 0) & (p_old < 0.05) & (p_v2 < 0.05)),
    'rand_M50+ (old p>0.9, v2 p>0.9)': np.flatnonzero((M >= 50) & (p_old > 0.9) & (p_v2 > 0.9)),
    'prod_M<50 (v2 p<0.05)': np.flatnonzero((M < 50) & (p_v2 < 0.05)),
    'rand_M<50 (v2 p>0.9; trainer gives production row)': np.flatnonzero((M < 50) & (p_v2 > 0.9)),
}
tgt = split.targets
pos = np.load(ROOT / 'artifacts/data/flat/train/cand_pos.npy', mmap_mode='r')
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 199), (200, 10**6)]
lines = [f'checkpoint {name} (inference_domain {model.inference_domain}); n per group {n_per}; seed 20261118']


def rank_bins(fids):
    """Per row of the feeds `fids` (rows laid end to end): display-rank bin, last displayed item = len(BINS)."""
    fids = np.asarray(fids)
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fids])
    fidx = np.repeat(np.arange(len(fids)), M[fids])
    order = np.lexsort((np.asarray(pos[rows]), fidx))
    start = np.concatenate([[0], np.cumsum(M[fids])[:-1]])
    rk = np.empty(len(rows), np.int64); rk[order] = np.arange(len(rows)) - np.repeat(start, M[fids])
    out = np.full(len(rows), -1)
    for k, (lo, hi) in enumerate(BINS):
        out[(rk >= lo) & (rk <= hi)] = k
    out[rk == np.repeat(M[fids], M[fids]) - 1] = len(BINS)
    return out


def softmax(s):
    e = np.exp(s - s.max()); return e / e.sum()


for gname, pool in groups.items():
    ids = np.sort(rng.choice(pool, size=min(n_per, len(pool)), replace=False))
    sc = {}
    for row in (0, 1):
        mix = np.zeros((split.n_feeds, 3), np.float32); mix[:, row] = 1.0
        split.domain_mix = torch.from_numpy(mix)
        t1 = time.time(); sc[row] = predict(model, split, ids, tf32=False); print(gname, row, f'{time.time()-t1:.0f}s', flush=True)
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in ids])
    sub_off = np.concatenate([[0], np.cumsum(M[ids])[:-1]])
    nd = {r: ndcg_per_feed(sc[r][rows], tgt[rows], sub_off, M[ids]) for r in (0, 1)}
    # examination oracle: add log e(display-rank bin), e from random-order feeds NOT in this sample (test-like groups only)
    if gname.startswith('rand'):
        ref = np.setdiff1d(groups[gname], ids)
        rb_ref, y_ref = rank_bins(ref), tgt[np.concatenate([np.arange(off[i], off[i] + M[i]) for i in ref])]
        rate = np.array([y_ref[rb_ref == k].mean() for k in range(len(BINS) + 1)])
        loge = np.log(rate / rate.mean())
        rb = rank_bins(ids)
        nd_or = ndcg_per_feed(sc[1][rows] + loge[rb], tgt[rows], sub_off, M[ids])
        do = nd_or - nd[1]
        oracle_line = (f'  examination oracle (row1 score + log e(rank bin), e from {len(ref):,} other feeds of the group; '
                       f'rates by bin {np.round(rate, 4).tolist()}): NDCG {nd_or.mean():.5f} vs {nd[1].mean():.5f}, '
                       f'gain {do.mean():+.5f} +- {do.std(ddof=1)/np.sqrt(len(do)):.5f}')
    else:
        oracle_line = None
    corr, tv, shift_sd, within_sd = [], [], [], []
    exp_bins = {0: np.zeros(len(BINS) + 1), 1: np.zeros(len(BINS) + 1)}; act_bins = np.zeros(len(BINS) + 1); cand_bins = np.zeros(len(BINS) + 1)
    for i in ids:
        a, b = off[i], off[i] + M[i]
        s0, s1 = sc[0][a:b].astype(np.float64), sc[1][a:b].astype(np.float64)
        y = tgt[a:b]; K = y.sum()
        corr.append(np.corrcoef(s0, s1)[0, 1] if s0.std() > 0 and s1.std() > 0 else 1.0)
        q0, q1 = softmax(s0), softmax(s1)
        tv.append(0.5 * np.abs(q0 - q1).sum())
        d = (s1 - s1.mean()) - (s0 - s0.mean()); shift_sd.append(d.std()); within_sd.append(s0.std())
        rk = np.argsort(np.argsort(np.asarray(pos[a:b]), kind='stable'), kind='stable')
        last = rk == (M[i] - 1)
        bidx = np.full(M[i], -1)
        for k, (lo, hi) in enumerate(BINS):
            bidx[(rk >= lo) & (rk <= hi) & ~last] = k
        bidx[last] = len(BINS)
        for k in range(len(BINS) + 1):
            m = bidx == k
            exp_bins[0][k] += K * q0[m].sum(); exp_bins[1][k] += K * q1[m].sum(); act_bins[k] += y[m].sum(); cand_bins[k] += m.sum()
    corr, tv, shift_sd, within_sd = map(np.array, (corr, tv, shift_sd, within_sd))
    dn = nd[1] - nd[0]
    lines.append(f'\n== {gname}: {len(ids)} feeds of {len(pool):,}; mean M {M[ids].mean():.0f}')
    lines.append(f'  within-feed Pearson(score row0, score row1): mean {corr.mean():.4f}, 5th pct {np.percentile(corr,5):.4f}')
    lines.append(f'  TV(softmax row0, softmax row1) per feed: mean {tv.mean():.4f}, median {np.median(tv):.4f}, 95th pct {np.percentile(tv,95):.4f}')
    lines.append(f'  sd of centred score shift row1-row0: {shift_sd.mean():.4f} vs within-feed score sd {within_sd.mean():.4f}')
    lines.append(f'  NDCG on these (seen) train feeds: row0 {nd[0].mean():.5f}, row1 {nd[1].mean():.5f}, diff {dn.mean():+.5f} +- {dn.std(ddof=1)/np.sqrt(len(dn)):.5f}')
    lines.append('  display-rank bin: actual click rate | expected rate row0 | row1 | ratio row1/row0')
    for k in range(len(BINS) + 1):
        lab = 'last item' if k == len(BINS) else (f'{BINS[k][0]}' if BINS[k][0] == BINS[k][1] else f'{BINS[k][0]}-{BINS[k][1] if BINS[k][1] < 10**6 else "+"}')
        if cand_bins[k] == 0: continue
        lines.append(f'    {lab:>9}: {act_bins[k]/cand_bins[k]:.4f} | {exp_bins[0][k]/cand_bins[k]:.4f} | {exp_bins[1][k]/cand_bins[k]:.4f} | {exp_bins[1][k]/max(exp_bins[0][k],1e-12):.3f}   (cands {int(cand_bins[k]):,})')
    da = act_bins / act_bins.sum(); d0 = exp_bins[0] / exp_bins[0].sum(); d1 = exp_bins[1] / exp_bins[1].sum()
    lines.append(f'  TV over rank bins of click mass: actual vs row0 {0.5*np.abs(da-d0).sum():.4f}; actual vs row1 {0.5*np.abs(da-d1).sum():.4f}; row0 vs row1 {0.5*np.abs(d0-d1).sum():.4f}')
    if oracle_line: lines.append(oracle_line)
    np.savez(OUT / f'p_token_{name}_{gname.split()[0]}.npz', ids=ids, s0=sc[0][rows], s1=sc[1][rows], nd0=nd[0], nd1=nd[1])
    print('\n'.join(lines[-(len(BINS) + 8):]), flush=True)

lines.append(f'\nwall {time.time()-t0:.0f}s')
(OUT / f'p_token_{name}.txt').write_text('\n'.join(lines) + '\n')
print('DONE', flush=True)
