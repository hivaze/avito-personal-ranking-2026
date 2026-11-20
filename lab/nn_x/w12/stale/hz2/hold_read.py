"""hz2 LABELLED HOLDOUT READ (DECLARATION.txt ADDENDUM 2; train-holdout labels only, no val / test label). CPU replay vs CPU replay.
Pools = member mean of per-feed z; S(w) = NDCG((1 - w) z(W8 pool) + w z(pop1kd pool)) - NDCG(W8 pool); NDCG = src/metrics.py ndcg_per_feed.
Usage: python hold_read.py <view dir> [<view dir> ...]   -> logs/hold_read_<views>.txt
"""
import sys
from hz2_common import *
sys.path.insert(0, str(ROOT / 'src'))
from metrics import ndcg_per_feed  # noqa: E402

PL = ROOT / 'nn_x/w12/next/poplag/cache'
W8 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in (951, 952, 953, 954)]
POP1KD = [f'abl_e10_pop1kd05t25_s{s}' for s in (951, 952, 953, 954)]
Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
fid = np.repeat(np.arange(len(Mh)), Mh)
OUTL = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUTL.append(s)


def se(d):
    d = np.asarray(d, np.float64); return float(d.std(ddof=1) / np.sqrt(len(d)))


def fmt(d):
    d = np.asarray(d, np.float64); return f'{d.mean():+.5f} +- {se(d):.5f}'


def z(x):
    x = np.asarray(x, np.float32).astype(np.float64)
    mu = np.bincount(fid, weights=x) / Mh
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x * x) / Mh - mu * mu, 0)) + 1e-9
    return ((x - mu[fid]) / sd[fid]).astype(np.float32)


def pred(view, v):
    return np.load(PL / f'pred_lag0_{v}.npy') if view == 'lag0' else np.load(CACHE / f'hpred_{view}_{v}.npy')


def pool(view, mem):
    return np.mean([z(pred(view, v)) for v in mem], 0)


def main():
    yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
    ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)
    stat = lambda pw, pp, w: ev((1 - w) * z(pw).astype(np.float64) + w * z(pp).astype(np.float64)) - ev(pw)
    ts = np.load(PL / 'mini' / 'ts.npy'); hold = np.load(PL / 'mini' / 'hold.npy')
    d = day_of(ts) - DAY0
    dg = np.where(d <= 2, 0, np.where(d <= 6, 1, np.where(d <= 9, 2, 3)))
    NM = ['Sep 6-8', 'Sep 9-12', 'Sep 13-14', 'Sep 17-19']
    P(f'=== hz2 holdout read {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; holdout feeds {len(Mh)}, rows {int(Mh.sum())}; +- = 1 SE paired per feed ===')
    cw0, cp0 = pool('lag0', W8), pool('lag0', POP1KD); e_w0, e_p0 = ev(cw0), ev(cp0)
    s0 = {w: stat(cw0, cp0, w) for w in (0.1, 0.2, 0.3)}
    P(f' stored inputs (CPU replay, poplag cache): W8 pool {e_w0.mean():.5f} | pop1kd pool {e_p0.mean():.5f} | S(0.2) {fmt(s0[0.2])}')
    for view in sys.argv[1:]:
        P(f'\n##### view {view} #####')
        part_pub = np.load(HERE / view / 'train' / 'hz2_part_pub.npy')[hold]; Lf = np.load(HERE / view / 'train' / 'hz2_lag.npy')[hold]
        have_p = all((CACHE / f'hpred_{view}_{v}.npy').exists() for v in POP1KD)
        cw = pool(view, W8); e_w = ev(cw)
        for v in W8 + (POP1KD if have_p else []):
            P(f'  member {v:28s} view - stored: {fmt(ev(pred(view, v)) - ev(pred("lag0", v)))}')
        if have_p:
            cp = pool(view, POP1KD); e_p = ev(cp); S = {w: stat(cw, cp, w) for w in (0.1, 0.2, 0.3)}
        subsets = [('all 30,000', np.ones(len(Mh), bool)), ('days >= Sep 13', dg >= 2)] + [(NM[g], dg == g) for g in range(4)] \
            + [('private-like feeds', ~part_pub), ('public-like feeds', part_pub), ('private-like, days >= Sep 13', (~part_pub) & (dg >= 2)),
               ('private-like, Sep 17-19', (~part_pub) & (dg == 3))] + [(f'lag {l}', Lf == l) for l in np.unique(Lf)]
        for nm, m in subsets:
            line = f' {nm:30s} n {int(m.sum()):5d} | W8 pool view - stored {fmt((e_w - e_w0)[m])}'
            if have_p:
                line += (f' | pop1kd pool {fmt((e_p - e_p0)[m])} | pop1kd - W8: stored {fmt((e_p0 - e_w0)[m])} view {fmt((e_p - e_w)[m])}'
                         f' | S(0.2) stored {fmt(s0[0.2][m])} view {fmt(S[0.2][m])} D {fmt((S[0.2] - s0[0.2])[m])}')
            P(line)
        if have_p:
            for w in (0.1, 0.3):
                P(f' w {w}: all: S stored {fmt(s0[w])} view {fmt(S[w])} D {fmt(S[w] - s0[w])} | days >= Sep 13: view {fmt(S[w][dg >= 2])} D {fmt((S[w] - s0[w])[dg >= 2])}')
    (LOGS / f'hold_read_{"_".join(sys.argv[1:])}.txt').write_text('\n'.join(OUTL) + '\n')


if __name__ == '__main__':
    main()
