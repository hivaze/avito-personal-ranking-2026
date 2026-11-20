"""hz2: generic CAUSAL LAGGED-HOLDOUT read of arm A against arm B (witness W4 of the 02:15Z staleness policy), train-holdout labels only.
Both arms are scored through the SAME hz2 view on the holdout feeds of days >= Sep 13 (13,424 feeds; the subset where the label-free gate holds), CPU fp32 on both sides.
  step 1 (label-free, 3-8 min per model at 12 threads):  python hold_infer2.py feats  <A1,A2,...,B1,B2,...> late
                                                         python hold_infer2.py stored <A1,A2,...,B1,B2,...> late      (optional: the in-period arm through the same CPU path)
  step 2:  python lagged_pair_read.py feats <A1,A2,...> <B1,B2,...> [tag]     -> logs/lagged_pair_<tag>.txt
Member names = checkpoint names without ".pt" (nn_x/w5/ckpt/<name>.pt for names starting with "w5_", nn_x/abl/ckpt otherwise); "<variant>@last" is the last checkpoint.
A and B must have the same number of members if a member-level (paired by position) number is wanted; pools = mean of per-feed z.
Quantities: Q_view = NDCG(pool A | view) - NDCG(pool B | view); Q_stored (if the stored replays exist) = the same under stored inputs; I = Q_view - Q_stored;
dA, dB = view - stored per pool. +- = 1 SE paired per feed. The script applies NO rule: write the rule down before running step 2.
"""
import sys
from hz2_common import *
sys.path.insert(0, str(ROOT / 'src'))
from metrics import ndcg_per_feed  # noqa: E402

PL = ROOT / 'nn_x/w12/next/poplag/cache'
VIEW = sys.argv[1]
A = sys.argv[2].split(','); B = sys.argv[3].split(',')
TAG = sys.argv[4] if len(sys.argv) > 4 else f'{A[0]}_vs_{B[0]}'.replace('@', '_')
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
        p = CACHE / f'hpred_{view}_late_{v}.npy'
        if not p.exists():
            raise SystemExit(f'missing {p.name}: run  python hold_infer2.py {view} {v} late')
        x = np.load(p)
        assert np.array_equal(np.isnan(x), ~late_row), (view, v)
        return np.nan_to_num(x, nan=0.0)

    have_s = all((CACHE / f'hpred_stored_late_{v}.npy').exists() for v in A + B)
    P(f'=== causal lagged-holdout read {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; view {VIEW}; A = {A}; B = {B}; holdout feeds of days >= Sep 13: {int(late.sum())}; +- = 1 SE paired per feed ===')
    pool = lambda view, mem: np.mean([z(pred(view, v)) for v in mem], 0)
    eA_v, eB_v = ev(pool(VIEW, A)), ev(pool(VIEW, B))
    if have_s:
        eA_s, eB_s = ev(pool('stored', A)), ev(pool('stored', B))
    P(f' levels: pool A view {eA_v[late].mean():.5f} | pool B view {eB_v[late].mean():.5f}' + (f' | pool A stored {eA_s[late].mean():.5f} | pool B stored {eB_s[late].mean():.5f}' if have_s else ' | (no stored replay: Q_view only)'))
    for nm, m in (('days >= Sep 13', late), ('Sep 13-14', dg == 2), ('Sep 17-19', dg == 3), ('private-like', (~part_pub) & late), ('public-like', part_pub & late)):
        line = f' {nm:16s} n {int(m.sum()):5d} | Q_view {fmt((eA_v - eB_v)[m])}'
        if have_s:
            line += f' | Q_stored {fmt((eA_s - eB_s)[m])} | I {fmt(((eA_v - eB_v) - (eA_s - eB_s))[m])} | dA {fmt((eA_v - eA_s)[m])} | dB {fmt((eB_v - eB_s)[m])}'
        P(line)
    if len(A) == len(B):
        mv = np.mean([(ev(pred(VIEW, a)) - ev(pred(VIEW, b)))[late] for a, b in zip(A, B)], 0)
        line = f' member level (mean of the {len(A)} paired single-model differences, by position): Q_view {fmt(mv)}'
        if have_s:
            ms = np.mean([(ev(pred('stored', a)) - ev(pred('stored', b)))[late] for a, b in zip(A, B)], 0)
            line += f' | Q_stored {fmt(ms)} | I {fmt(mv - ms)}'
        P(line)
        for a, b in zip(A, B):
            P(f'   {a} - {b}: Q_view {fmt((ev(pred(VIEW, a)) - ev(pred(VIEW, b)))[late])}')
    (LOGS / f'lagged_pair_{TAG}.txt').write_text('\n'.join(OUTL) + '\n')


if __name__ == '__main__':
    main()
