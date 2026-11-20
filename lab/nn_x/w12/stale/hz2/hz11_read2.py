"""hz2 ADDENDUM 6 (HZ11) read: the 11-12 half-dose hz models (H = w5_cd_hzcd_r101..104) against their references (R = w5_cd_m02ref_r101..104) through the causal hz2
view on the holdout feeds of days >= Sep 13 (train-holdout labels only). Quantities, expectation and reading: DECLARATION.txt ADDENDUM 6.
Usage: python hz11_read.py [view = feats]   -> logs/hz11_read_<view>.txt
"""
import sys
from hz2_common import *
sys.path.insert(0, str(ROOT / 'src'))
from metrics import ndcg_per_feed  # noqa: E402

PL = ROOT / 'nn_x/w12/next/poplag/cache'
VIEW = sys.argv[1] if len(sys.argv) > 1 else 'feats'
ROOTS = [101, 102, 103, 104]
SUF = os.environ.get('HZ11_SUF', '')   # '' = selected checkpoints (ADDENDUM 6), '@last' = ADDENDUM 8
H = [f'w5_cd_hzcd_r{r}{SUF}' for r in ROOTS]
R = [f'w5_cd_m02ref_r{r}{SUF}' for r in ROOTS]
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
    yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
    ts = np.load(PL / 'mini' / 'ts.npy'); hold = np.load(PL / 'mini' / 'hold.npy')
    d = day_of(ts) - DAY0
    late = d >= 7; late_row = late[fid]
    dg = np.where(d <= 2, 0, np.where(d <= 6, 1, np.where(d <= 9, 2, 3)))
    part_pub = np.load(HERE / (VIEW if VIEW != 'stored' else 'feats') / 'train' / 'hz2_part_pub.npy')[hold]
    ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)

    def pred(view, v):
        x = np.load(CACHE / f'hpred_{view}_late_{v}.npy')
        assert np.array_equal(np.isnan(x), ~late_row), (view, v)
        return np.nan_to_num(x, nan=0.0)

    P(f'=== HZ11 read (checkpoints: {SUF or "selected"}) {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; view {VIEW}; holdout feeds of days >= Sep 13: {int(late.sum())}; +- = 1 SE paired per feed ===')
    # ---- path control: CPU stored replay vs the stored GPU bf16 holdout file
    okc = True
    P(' path control (CPU fp32 "stored" replay vs nn_x/w5/preds/holdout/nn_<v>_holdout.npy, same feeds):')
    for v in R + H:
        g = np.load(ROOT / 'nn_x/w5/preds/holdout' / f'nn_{v}_holdout.npy').astype(np.float32)
        c = pred('stored', v)
        eg, ec = ev(g)[late], ev(c)[late]
        cr = float(np.corrcoef(z(g)[late_row], z(c)[late_row])[0, 1])
        good = abs(float(ec.mean() - eg.mean())) <= 0.0005 and cr >= 0.999
        okc = okc and good
        P(f'   {v:22s} NDCG GPU file {eg.mean():.5f} CPU replay {ec.mean():.5f} diff {fmt(ec - eg)} corr(z) {cr:.5f} -> {"ok" if good else "NOT VALID"}')
    P(f' PATH CONTROL {"VALID" if okc else "NOT VALID"}')
    pool = lambda view, mem: np.mean([z(pred(view, v)) for v in mem], 0)
    eH_v, eR_v, eH_s, eR_s = ev(pool(VIEW, H)), ev(pool(VIEW, R)), ev(pool('stored', H)), ev(pool('stored', R))
    P(f' levels, days >= Sep 13: pool R stored {eR_s[late].mean():.5f} view {eR_v[late].mean():.5f} | pool H stored {eH_s[late].mean():.5f} view {eH_v[late].mean():.5f}')
    for nm, m in (('days >= Sep 13', late), ('Sep 13-14', dg == 2), ('Sep 17-19', dg == 3), ('private-like', (~part_pub) & late), ('public-like', part_pub & late)):
        qv, qs = (eH_v - eR_v)[m], (eH_s - eR_s)[m]
        P(f' {nm:16s} n {int(m.sum()):5d} | Q_view {fmt(qv)} | Q_stored {fmt(qs)} | I = Q_view - Q_stored {fmt(qv - qs)} | dR {fmt((eR_v - eR_s)[m])} | dH {fmt((eH_v - eH_s)[m])}')
    # member level
    mv = np.mean([(ev(pred(VIEW, h)) - ev(pred(VIEW, r)))[late] for h, r in zip(H, R)], 0)
    ms = np.mean([(ev(pred('stored', h)) - ev(pred('stored', r)))[late] for h, r in zip(H, R)], 0)
    P(f' member level (mean of the 4 paired single-model differences), days >= Sep 13: Q_view {fmt(mv)} | Q_stored {fmt(ms)} | I {fmt(mv - ms)}')
    for h, r in zip(H, R):
        P(f'   root {h.split("_r")[-1]}: Q_view {fmt((ev(pred(VIEW, h)) - ev(pred(VIEW, r)))[late])} | Q_stored {fmt((ev(pred("stored", h)) - ev(pred("stored", r)))[late])}')
    # the GPU-file reading of Q_stored on the same feeds and on all 30,000 (reference to the 11-12 number)
    gp = lambda mem: np.mean([z(np.load(ROOT / 'nn_x/w5/preds/holdout' / f'nn_{v}_holdout.npy')) for v in mem], 0)
    qg = ev(gp(H)) - ev(gp(R))
    P(f' Q_stored from the stored GPU files: days >= Sep 13 {fmt(qg[late])} | all 30,000 {fmt(qg)} | days < Sep 13 {fmt(qg[~late])}')
    qv, qs = (eH_v - eR_v)[late], (eH_s - eR_s)[late]
    i_ = qv - qs
    sup = qv.mean() >= se(qv) and i_.mean() >= 2 * se(i_)
    ag = i_.mean() <= 0 or qv.mean() <= -se(qv)
    P(f' READING (ADDENDUM 6): {"SUPPORTS M" if sup else ("AGAINST M at half dose" if ag else "not separated at 4 models / 13,424 feeds")}'
      f'{"" if okc else "   [path control NOT VALID: reading not to be used]"}')
    (LOGS / f'hz11_read_{VIEW}{SUF.replace("@", "_")}.txt').write_text('\n'.join(OUTL) + '\n')


if __name__ == '__main__':
    main()
