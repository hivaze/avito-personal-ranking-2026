"""hz2 LABELLED HOLDOUT READ, late subset (DECLARATION.txt ADDENDUM 2, 3 and 5; train-holdout labels only, no val / test label). CPU replay vs CPU replay.
Reads cache/hpred_<view>_late_<member>.npy (holdout feeds of days >= Sep 13 scored through the view; other rows NaN), poplag's lag-0 predictions of the same
code path and, for information, poplag's v3 predictions (private-like lags for every feed).
Pools = member mean of per-feed z; S(w) = NDCG((1 - w) z(W8 pool) + w z(pop1kd pool)) - NDCG(W8 pool); NDCG = src/metrics.py ndcg_per_feed; +- = 1 SE paired per feed.
Usage: python hold_read_late.py feats feats_L [nW8 nP]   -> logs/hold_read_late[_partial].txt
  nW8 / nP given = PARTIAL read of the first nW8 W8 members and nP pop1kd members (progress only; the declared read is the full one, 8 + 4).
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
    d = np.asarray(d, np.float64); return f'{d.mean():+.5f} +- {se(d):.5f} (z {d.mean() / se(d):+.2f})'


def z(x):
    x = np.asarray(x, np.float32).astype(np.float64)
    mu = np.bincount(fid, weights=x) / Mh
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x * x) / Mh - mu * mu, 0)) + 1e-9
    return ((x - mu[fid]) / sd[fid]).astype(np.float32)


def main():
    views = [a for a in sys.argv[1:] if not a.isdigit()]
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
    w8 = W8[:nums[0]] if nums else W8
    pk = POP1KD[:nums[1]] if nums else POP1KD
    partial = bool(nums) and (len(w8) < 8 or len(pk) < 4)
    FAKE = os.environ.get('HZ2_FAKE_Y') == '1'      # plumbing test only: random labels, no real label is opened, output to logs/TEST_*.txt
    if FAKE:
        yh = (np.random.default_rng(0).random(int(Mh.sum())) < 0.03).astype(np.float32)
        if os.environ.get('HZ2_TEST_MEMBERS'):
            a_, b_ = os.environ['HZ2_TEST_MEMBERS'].split('|'); w8 = a_.split(','); pk = b_.split(',') if b_ else []
        partial = True
    else:
        yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
    assert len(yh) == int(Mh.sum()) and np.array_equal(Mh, np.load(PL / 'mini' / 'M.npy').astype(np.int64))
    ts = np.load(PL / 'mini' / 'ts.npy'); hold = np.load(PL / 'mini' / 'hold.npy')
    d = day_of(ts) - DAY0
    late = d >= 7
    late_row = late[fid]
    dg = np.where(d <= 2, 0, np.where(d <= 6, 1, np.where(d <= 9, 2, 3)))

    def pred(view, v):
        if view == 'lag0':
            return np.load(PL / f'pred_lag0_{v}.npy')
        if view == 'v3':
            return np.load(PL / f'pred_priv_v3_{v}.npy')
        x = np.load(CACHE / f'hpred_{view}_late_{v}.npy')
        assert np.array_equal(np.isnan(x), ~late_row), (view, v, 'NaN rows are not exactly the rows of feeds before Sep 13')
        return np.nan_to_num(x, nan=0.0)

    pool = lambda view, mem: np.mean([z(pred(view, v)) for v in mem], 0)
    ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)
    stat = lambda pw, pp, w: ev((1 - w) * z(pw).astype(np.float64) + w * z(pp).astype(np.float64)) - ev(pw)
    P(f'=== hz2 holdout read (late subset) {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; holdout feeds of days >= Sep 13: {int(late.sum())} of {len(Mh)}; '
      f'rows {int(late_row.sum())}; +- = 1 SE paired per feed ===')
    P(f' members: W8 {len(w8)} of 8, pop1kd {len(pk)} of 4' + ('   ##### FAKE RANDOM LABELS: PLUMBING TEST, NUMBERS MEANINGLESS #####' if FAKE else '') + ('   *** PARTIAL READ (progress only, not the declared read) ***' if partial else '   (the declared read)'))
    cw0 = pool('lag0', w8); e_w0 = ev(cw0)
    have_p = len(pk) > 0
    if have_p:
        cp0 = pool('lag0', pk); e_p0 = ev(cp0); s0 = {w: stat(cw0, cp0, w) for w in (0.1, 0.2, 0.3)}
    P(f' stored inputs (CPU replay, poplag cache), days >= Sep 13: W8 pool {e_w0[late].mean():.5f}' + (f' | pop1kd pool {e_p0[late].mean():.5f} | S(0.2) {fmt(s0[0.2][late])}' if have_p else ''))
    EW = {}
    for view in views:
        P(f'\n##### view {view} #####')
        part_pub = np.load(HERE / view / 'train' / 'hz2_part_pub.npy')[hold]; Lf = np.load(HERE / view / 'train' / 'hz2_lag.npy')[hold]
        cw = pool(view, w8); e_w = ev(cw); EW[view] = e_w
        for v in w8 + pk:
            P(f'  member {v:28s} view - stored: {fmt((ev(pred(view, v)) - ev(pred("lag0", v)))[late])}')
        if have_p:
            cp = pool(view, pk); e_p = ev(cp); S = {w: stat(cw, cp, w) for w in (0.1, 0.2, 0.3)}
        subsets = [('days >= Sep 13', late), ('Sep 13-14', dg == 2), ('Sep 17-19', dg == 3), ('private-like', (~part_pub) & late), ('public-like', part_pub & late),
                   ('private-like, Sep 17-19', (~part_pub) & (dg == 3))] + [(f'lag {l}', (Lf == l) & late) for l in np.unique(Lf)]
        for nm, m in subsets:
            line = f' {nm:26s} n {int(m.sum()):5d} | dW (W8 pool view - stored) {fmt((e_w - e_w0)[m])}'
            if have_p:
                line += (f' | dP (pop1kd pool) {fmt((e_p - e_p0)[m])} | pop1kd - W8: stored {fmt((e_p0 - e_w0)[m])} view {fmt((e_p - e_w)[m])}'
                         f' | S(0.2) stored {fmt(s0[0.2][m])} view {fmt(S[0.2][m])} D {fmt((S[0.2] - s0[0.2])[m])}')
            P(line)
        if have_p:
            for w in (0.1, 0.3):
                P(f' w {w}: days >= Sep 13: S stored {fmt(s0[w][late])} view {fmt(S[w][late])} D {fmt((S[w] - s0[w])[late])}')
            P(f' dP - dW (pop1kd pool loss minus W8 pool loss), days >= Sep 13: {fmt(((e_p - e_p0) - (e_w - e_w0))[late])}')
    if len(views) == 2:
        a, b = views
        P(f'\n##### E4: W8 pool, {b} minus {a} (paired per feed) #####')
        for nm, m in (('Sep 17-19', dg == 3), ('Sep 13-14', dg == 2), ('days >= Sep 13', late)):
            P(f' {nm:16s} n {int(m.sum()):5d}: {fmt((EW[b] - EW[a])[m])}')
    # information: poplag v3 (private-like lags for every feed) on the same feeds
    try:
        P('\n##### information: poplag v3 view (private-like lags for every feed; non-causal on early days, causal on Sep 17-19) #####')
        c3 = pool('v3', w8); e3 = ev(c3)
        line = f' days >= Sep 13 n {int(late.sum())}: W8 pool v3 - stored {fmt((e3 - e_w0)[late])}'
        if have_p:
            p3 = pool('v3', pk); ep3 = ev(p3); S3 = stat(c3, p3, 0.2)
            line += f' | pop1kd pool {fmt((ep3 - e_p0)[late])} | S(0.2) v3 {fmt(S3[late])} D {fmt((S3 - s0[0.2])[late])}'
        P(line)
        P(f' Sep 17-19 n {int((dg == 3).sum())}: W8 pool v3 - stored {fmt((e3 - e_w0)[dg == 3])}')
        for view in views:
            part_pub = np.load(HERE / view / 'train' / 'hz2_part_pub.npy')[hold]
            for nm, m in (('private-like, days >= Sep 13', (~part_pub) & late), ('private-like, Sep 17-19', (~part_pub) & (dg == 3))):
                P(f' {view:8s} {nm:30s} n {int(m.sum()):5d}: W8 pool {view} - v3 {fmt((EW[view] - e3)[m])} | v3 - stored {fmt((e3 - e_w0)[m])} | {view} - stored {fmt((EW[view] - e_w0)[m])}')
    except FileNotFoundError as e:
        P(f' poplag v3 predictions not available: {e}')
    (LOGS / ('TEST_hold_read_late.txt' if FAKE else 'hold_read_late_partial.txt' if partial else 'hold_read_late.txt')).write_text('\n'.join(OUTL) + '\n')


if __name__ == '__main__':
    main()
