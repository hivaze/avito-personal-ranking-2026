"""poplag_verify step K (LABEL-FREE): parity checks the declared gate does NOT contain, on the SUBSET (days >= Sep 13), private-like view:
 (1) JOINT zero pattern of the two W8 families (it_hist_all_luo = 0 / > 0) x (seq_cv_pop = 0 / > 0) by item-age bucket, lagged holdout rows vs private test rows (M >= 50);
 (2) magnitude of the non-zero counts (median / q90 of the raw count) by bucket.
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_joint.py"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *

OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


EDGES = np.array([0, 1, 3, 7, 10, 24, 76, np.inf]); BN = ['<1d', '1-3d', '3-7d', '7-10d', '10-24d', '24-76d', '>76d+cap']
KN = np.load(ROOT / 'data/feats/life_f4_age_knots.npy')


def bucket(f4):
    u = f4.astype(np.float64) * 64.0; k = np.clip(np.floor(u).astype(np.int64), 1, 62)
    a = np.where(f4.astype(np.float64) < 63 / 64, KN[k] + (u - k) * (KN[k + 1] - KN[k]), np.nan)
    return np.clip(np.searchsorted(EDGES, np.where(np.isnan(a), 1e9, a / 24.0), side='right') - 1, 0, 6)


ftr, hold, _ = my_holdout()
fM = ftr['M'].to_numpy().astype(np.int64); fco = ftr['cand_off'].to_numpy().astype(np.int64)
hd = (ftr['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY
hM = fM[hold]; fid = np.repeat(np.arange(len(hold)), hM)
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(fco[hold], hM)])
bk = bucket(np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[rows]))
fte = pl.read_parquet(FLAT / 'test/feeds.parquet'); tts = fte['timestamp'].to_numpy().astype(np.int64); tM = fte['M'].to_numpy().astype(np.int64)
t_fx = np.load(FLAT / 'test/cand_fidx.npy').astype(np.int64); t_bk = bucket(np.load(FLAT / 'test/cand_f4.npy'))
ref = ((tts >= PUB_TS) & (tM >= 50))[t_fx]
for scope, hs in (('SUBSET days >= Sep 13', (hd >= SEP6 + 7)[fid]), ('  Sep 17-19 only', (hd >= SEP6 + 11)[fid]), ('complement days < Sep 13', (hd < SEP6 + 7)[fid])):
    P(f'\n=== {scope}: {int(hs.sum())} holdout rows vs {int(ref.sum())} private test rows (M >= 50) ===')
    for tag, a_h, b_h in (('lag 0 (stored)', np.asarray(np.load(FC / 'train/it_hist_all_luo.npy', mmap_mode='r')[rows]), np.asarray(np.load(FC / 'train/seq_cv_pop.npy', mmap_mode='r')[rows])),
                          ('private-like v3', np.load(STUDY / 'lag_priv_v3_it_hist_all_luo.npy'), np.load(STUDY / 'lag_priv_v3_seq_cv_pop.npy'))):
        a_t = np.load(FC / 'test/it_hist_all_luo.npy')[ref]; b_t = np.load(FC / 'test/seq_cv_pop.npy')[ref]
        ch = (a_h[hs] == 0) * 2 + (b_h[hs] == 0); ct = (a_t == 0) * 2 + (b_t == 0)            # 0: both warm, 1: hist warm / covis cold, 2: hist cold / covis warm, 3: both cold
        w = np.bincount(bk[hs], minlength=7) / hs.sum(); tv = np.zeros(7); worst = (0.0, '')
        for b in range(7):
            ph = np.bincount(ch[bk[hs] == b], minlength=4) / max((bk[hs] == b).sum(), 1); pt = np.bincount(ct[t_bk[ref] == b], minlength=4) / max((t_bk[ref] == b).sum(), 1)
            tv[b] = 0.5 * np.abs(ph - pt).sum()
            if w[b] >= 0.05 and np.abs(ph - pt).max() > worst[0]:
                worst = (float(np.abs(ph - pt).max()), f'{BN[b]} cell {int(np.abs(ph - pt).argmax())}')
        P(f' {tag:16s}: joint zero pattern (it_hist_all_luo x seq_cv_pop): row-weighted total-variation distance over age buckets {float((w * tv).sum()):.4f}; by bucket ' + ' '.join(f'{BN[b]} {tv[b]:.3f}' for b in range(7)) + f'; largest cell gap (buckets >= 5% rows) {worst[0]:.4f} ({worst[1]})')
        q = []
        for b in range(7):
            xh = a_h[hs][(bk[hs] == b) & (a_h[hs] > 0)]; xt = a_t[(t_bk[ref] == b) & (a_t > 0)]
            q.append(f'{BN[b]} {np.median(xh):.0f}/{np.quantile(xh, 0.9):.0f} vs {np.median(xt):.0f}/{np.quantile(xt, 0.9):.0f}')
        P(f' {tag:16s}: non-zero it_hist_all_luo, median / q90 holdout vs test: ' + ' | '.join(q))
(HERE / 'logs' / 'v_joint.txt').write_text('\n'.join(OUT) + '\n')
