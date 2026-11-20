"""poplag_verify step J (LABEL-FREE): remaining claims of the study re-computed with my own code from its cached columns / predictions:
gate history (v1; v2 calibration J(g); v2 at g = 3), feed-level cold share, label-free prediction shift of the staged private-like predictions.
The v1 / v2 column files themselves are the study's (not rebuilt by me); only v3 was rebuilt (v_gate.py, v_ithist.py, v_sample.py).
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_claims.py"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *
import time

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
hd = (ftr['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY; late = hd >= SEP6 + 7
hM = fM[hold]; oh = np.concatenate([[0], np.cumsum(hM)[:-1]]); fid = np.repeat(np.arange(len(hold)), hM)
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(fco[hold], hM)])
bk = bucket(np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[rows]))
fte = pl.read_parquet(FLAT / 'test/feeds.parquet'); tts = fte['timestamp'].to_numpy().astype(np.int64); tM = fte['M'].to_numpy().astype(np.int64)
t_fx = np.load(FLAT / 'test/cand_fidx.npy').astype(np.int64); t_bk = bucket(np.load(FLAT / 'test/cand_f4.npy'))
tpub = (tts < PUB_TS)[t_fx]; tbig = (tM >= 50)[t_fx]
P(f'=== poplag_verify remaining claims {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} (label-free) ===')


def crit(name, hraw, traw, tb):
    def pb(raw, b):
        x = np.asarray(raw, np.float32); x = (np.log1p(np.maximum(x, 0)) if name.startswith(('seq_cv', 'it_hist')) else np.clip(x, -10, 10)).astype(np.float64)
        zz = (np.asarray(raw) == 0).astype(np.float64); n = np.bincount(b, minlength=7).astype(np.float64)
        return n, np.bincount(b, weights=zz, minlength=7) / np.maximum(n, 1), np.bincount(b, weights=x, minlength=7) / np.maximum(n, 1), x.std()
    n, zh, mh, _ = pb(hraw, bk); nt, zt, mt, sd = pb(traw, tb)
    w = n / n.sum(); gz = np.abs(zh - zt); big = np.where(w >= 0.05, gz, 0.0)
    return float((w * gz).sum()), float(big.max()), BN[int(big.argmax())], float((w * np.abs(mh - mt)).sum() / sd)


PRIM = ['it_hist_all_luo', 'seq_cv_pop', 'xp_imp_c50']
TR = {cn: np.load(FC / 'test' / f'{cn}.npy') for cn in PRIM}
for nm in ('priv', 'pub'):
    ref = (tpub if nm == 'pub' else ~tpub) & tbig
    P(f'\n gate history, view {nm} (study\'s cached v1 / v2 columns, my gate code); primaries dZ / largest bucket / dM:')
    line = []
    for cn in PRIM:
        a = crit(cn, np.load(STUDY / f'lag_{nm}_{cn}.npy'), TR[cn][ref], t_bk[ref]) if (STUDY / f'lag_{nm}_{cn}.npy').exists() else None
        line.append(f'{cn} ' + (f'{a[0]:.4f} / {a[1]:.4f} ({a[2]}) / {a[3]:.4f}' if a else 'file missing'))
    P('   v1: ' + ' | '.join(line))
    J = []
    for g in range(7):
        r = [crit(cn, np.load(STUDY / f'lag_{nm}_g{g}_{cn}.npy'), TR[cn][ref], t_bk[ref]) for cn in PRIM]
        J.append(sum(x[0] for x in r))
        if g == 3:
            P('   v2 at g = 3: ' + ' | '.join(f'{cn} {x[0]:.4f} / {x[1]:.4f} ({x[2]}) / {x[3]:.4f}' for cn, x in zip(PRIM, r)))
    P(f'   J(g) = sum of the three primary dZ, g = 0..6: ' + ', '.join(f'{x:.4f}' for x in J) + f'  -> argmin g = {int(np.argmin(J))}')

# feed-level cold share
P('\n feed-level cold share (per-feed share of rows with it_hist_all_luo = 0): q10 / q50 / q90 / share of feeds >= 0.45')
def fl(x, M):
    o = np.concatenate([[0], np.cumsum(M)[:-1]]); cs = np.add.reduceat((x == 0).astype(np.float64), o) / M
    return f'{np.quantile(cs, 0.1):.3f} / {np.quantile(cs, 0.5):.3f} / {np.quantile(cs, 0.9):.3f} / {np.mean(cs >= 0.45):.3f}'
P('   holdout stored (lag 0): ' + fl(np.asarray(np.load(FC / 'train/it_hist_all_luo.npy', mmap_mode='r')[rows]), hM))
P('   holdout private-like v3: ' + fl(np.load(STUDY / 'lag_priv_v3_it_hist_all_luo.npy'), hM))
P('   holdout public-like v3: ' + fl(np.load(STUDY / 'lag_pub_v3_it_hist_all_luo.npy'), hM))
tx = np.load(FC / 'test/it_hist_all_luo.npy'); assert (np.diff(t_fx) >= 0).all()
for nm, sel in (('private test feeds, M >= 50', (tts >= PUB_TS) & (tM >= 50)), ('public test feeds, M >= 50', (tts < PUB_TS) & (tM >= 50))):
    r = sel[t_fx]; P(f'   {nm}: ' + fl(tx[r], tM[sel]))

# label-free prediction shift
z = lambda x: zfeed(x, oh, hM).astype(np.float64)
pool = lambda view, mem: np.mean([z(np.load(STUDY / f'pred_{view}_{v}.npy')) for v in mem], 0)
def fcorr(a, b):
    return np.add.reduceat(z(a) * z(b), oh) / hM
def top1(x):
    return np.lexsort((-x, fid))[oh]
def top10(a, b):
    ra = np.empty(len(a), np.int64); oa = np.lexsort((-a, fid)); ra[oa] = np.arange(len(a)) - oh[fid[oa]]
    rb = np.empty(len(b), np.int64); ob = np.lexsort((-b, fid)); rb[ob] = np.arange(len(b)) - oh[fid[ob]]
    return np.add.reduceat(((ra < 10) & (rb < 10)).astype(np.float64), oh) / 10.0
w0, w1, p0, p1 = pool('lag0', W8), pool('priv_v3', W8), pool('lag0', P4), pool('priv_v3', P4)
P('\n label-free prediction shift, private-like v3 vs lag 0 (CPU both), 30,000 feeds:')
for nm, a, b in (('W8 pool', w0, w1), ('pop1kd pool', p0, p1)):
    c = fcorr(a, b); t = (top1(a) != top1(b)).astype(np.float64); o = top10(a, b)
    P(f'   {nm}: per-feed corr {c.mean():.4f} +- {c.std(ddof=1) / np.sqrt(len(c)):.4f}; top-1 changed {t.mean():.4f} +- {t.std(ddof=1) / np.sqrt(len(t)):.4f}; top-10 overlap {o.mean():.4f} | subset: corr {c[late].mean():.4f}, top-1 changed {t[late].mean():.4f}')
c0, c1 = fcorr(w0, p0), fcorr(w1, p1); dd = c1 - c0
P(f'   per-feed corr(W8 pool, pop1kd pool): lag 0 {c0.mean():.4f} -> lagged {c1.mean():.4f}; paired change {dd.mean():+.5f} +- {dd.std(ddof=1) / np.sqrt(len(dd)):.5f}')
(HERE / 'logs' / 'v_claims.txt').write_text('\n'.join(OUT) + '\n')
