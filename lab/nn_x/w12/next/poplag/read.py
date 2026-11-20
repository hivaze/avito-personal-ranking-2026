"""poplag CONTROL + READ (labelled; run only after the gate verdict).
  python read.py control                 -> CPU fp32 lag-0 replay vs the stored bf16 files (member NDCG, statistic)
  python read.py read <view> [<view>..]  -> lagged statistic S(w), pool level changes, slices (CPU replay vs CPU replay)
Pools = member mean of per-feed z (src/stack_val.py feed_rank_z[:, 1] = (x - feed mean) / (feed sd + 1e-9)); statistic as nn_x/w12/recheck/leak_overfit/r3_pop1.py:
S(w) = NDCG((1 - w) z(W8 pool) + w z(pop1kd pool)) - NDCG(W8 pool). NDCG = src/metrics.py ndcg_per_feed (official tie rule)."""
import sys, time
from pl_common import *
sys.path.insert(0, str(ROOT / 'src'))
from metrics import ndcg_per_feed  # noqa: E402  (pure numpy)

Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
fid = np.repeat(np.arange(len(Mh)), Mh)
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def z(x):
    x = np.asarray(x, np.float32).astype(np.float64)
    mu = np.bincount(fid, weights=x) / Mh
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x * x) / Mh - mu * mu, 0)) + 1e-9
    return ((x - mu[fid]) / sd[fid]).astype(np.float32)


def pool(view, mem, src='cpu'):
    return np.mean([z(np.load(CACHE / f'pred_{view}_{v}.npy') if src == 'cpu' else np.load(stored_holdout(v))) for v in mem], 0)


def main():
    yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
    ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)
    stat = lambda pw, pp, w: ev((1 - w) * z(pw).astype(np.float64) + w * z(pp).astype(np.float64)) - ev(pw)
    mode = sys.argv[1]
    P(f'=== poplag {mode} {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; holdout feeds {len(Mh)}, rows {int(Mh.sum())}; +- = 1 SE paired per feed ===')
    if mode == 'control':
        ok = True
        for v in W8 + POP1KD:
            es, ec = ev(np.load(stored_holdout(v))), ev(np.load(CACHE / f'pred_lag0_{v}.npy'))
            d = ec - es; flag = abs(d.mean()) <= 0.0010; ok &= flag
            P(f' {v:28s} stored bf16 GPU {es.mean():.5f} | CPU fp32 replay {ec.mean():.5f} | paired {fmt(d)} | within 0.0010: {flag}')
        sw, sp = pool('lag0', W8, 'stored'), pool('lag0', POP1KD, 'stored'); cw, cp = pool('lag0', W8), pool('lag0', POP1KD)
        P(f' W8 pool: stored {ev(sw).mean():.5f} CPU {ev(cw).mean():.5f} paired {fmt(ev(cw) - ev(sw))} | pop1kd pool: stored {ev(sp).mean():.5f} CPU {ev(cp).mean():.5f} paired {fmt(ev(cp) - ev(sp))}')
        for w in (0.1, 0.2, 0.3):
            a, b = stat(sw, sp, w), stat(cw, cp, w)
            P(f' statistic w {w}: stored files {fmt(a)} | CPU lag-0 replay {fmt(b)} | CPU - stored {fmt(b - a)}')
        a, b = stat(sw, sp, 0.2), stat(cw, cp, 0.2)
        f2 = abs(b.mean() - (-0.00063)) <= 0.0003
        P(f' CONTROL: members within 0.0010: {ok}; statistic (w 0.2) CPU {b.mean():+.5f} vs declared -0.00063: within +-0.0003: {f2}  ->  {"VALID" if ok and f2 else "INVALID"}')
        (LOGS / 'control.txt').write_text('\n'.join(OUT) + '\n')
        return
    mini = CACHE / 'mini'; ts = np.load(mini / 'ts.npy'); hday = day_of(ts); dow = (hday + 3) % 7
    cw0, cp0 = pool('lag0', W8), pool('lag0', POP1KD); e_w0, e_p0 = ev(cw0), ev(cp0)
    s0 = {w: stat(cw0, cp0, w) for w in (0.1, 0.2, 0.3)}
    P(f' lag 0 (CPU replay): W8 pool {e_w0.mean():.5f} | pop1kd pool {e_p0.mean():.5f} | S(0.1) {fmt(s0[0.1])} S(0.2) {fmt(s0[0.2])} S(0.3) {fmt(s0[0.3])}')
    for view in sys.argv[2:]:
        P(f'\n##### view {view} #####')
        cw, cp = pool(view, W8), pool(view, POP1KD); e_w, e_p = ev(cw), ev(cp)
        for v in W8 + POP1KD:
            d = ev(np.load(CACHE / f'pred_{view}_{v}.npy')) - ev(np.load(CACHE / f'pred_lag0_{v}.npy'))
            P(f'  member {v:28s} lagged - lag 0: {fmt(d)}')
        P(f' W8 pool: lagged {e_w.mean():.5f} vs lag 0 {e_w0.mean():.5f}: {fmt(e_w - e_w0)}')
        P(f' pop1kd pool: lagged {e_p.mean():.5f} vs lag 0 {e_p0.mean():.5f}: {fmt(e_p - e_p0)}')
        P(f' pop1kd pool - W8 pool: lagged {fmt(e_p - e_w)} | lag 0 {fmt(e_p0 - e_w0)}')
        S = {w: stat(cw, cp, w) for w in (0.1, 0.2, 0.3)}
        for w in (0.1, 0.2, 0.3):
            P(f' STATISTIC w {w}: lagged {fmt(S[w])} (z {S[w].mean() / se(S[w]):+.2f}) | lag 0 {fmt(s0[w])} | lagged - lag 0 {fmt(S[w] - s0[w])}')
        d = S[0.2]
        rule = 'bias explanation holds' if d.mean() >= -0.0002 else ('view closed for student pools' if d.mean() <= -0.0005 else 'pick 2 at most')
        P(f' RULE on S(0.2) = {d.mean():+.5f}: >= -0.0002 holds / <= -0.0005 closed / between pick 2 at most  ->  "{rule}"')
        # slices (pre-declared robustness column: holdout days >= Sep 13)
        late = hday >= 20709
        P(f'  days >= Sep 13 (n {int(late.sum())}): S(0.2) lagged {fmt(d[late])} | lag 0 {fmt(s0[0.2][late])};  days < Sep 13 (n {int((~late).sum())}): lagged {fmt(d[~late])} | lag 0 {fmt(s0[0.2][~late])}')
        wk = dow >= 5
        P(f'  weekend feeds (n {int(wk.sum())}): lagged {fmt(d[wk])} | lag 0 {fmt(s0[0.2][wk])};  weekday feeds (n {int((~wk).sum())}): lagged {fmt(d[~wk])} | lag 0 {fmt(s0[0.2][~wk])}')
        Lf = np.load(CACHE / f'view_{view.split("_")[0]}_L.npy')
        P('  by imposed lag: ' + ' | '.join(f'L {l} (n {int((Lf == l).sum())}) {fmt(d[Lf == l])}' for l in np.unique(Lf)))
        for nm, col in (('it_hist_all_luo', 'cold share = rows with lagged it_hist_all_luo = 0'), ('xp_imp_c50', 'cold share = rows with lagged xp_imp_c50 = 0')):
            x = np.load(CACHE / f'lag_{view}_{nm}.npy'); cs = np.bincount(fid, weights=(x == 0)) / Mh
            x0 = np.asarray(np.load(FC / 'train' / f'{nm}.npy', mmap_mode='r')[np.load(mini / 'crow.npy')]); cs0 = np.bincount(fid, weights=(x0 == 0)) / Mh
            P(f'  {col}: feed mean lagged {cs.mean():.3f} (lag 0 {cs0.mean():.3f}); share of feeds >= 0.45: lagged {float((cs >= 0.45).mean()):.3f} (lag 0 {float((cs0 >= 0.45).mean()):.3f})')
            for lo, hi in ((0, 0.3), (0.3, 0.45), (0.45, 1.01)):
                m = (cs >= lo) & (cs < hi)
                if m.sum() > 100:
                    P(f'    bin [{lo}, {min(hi, 1.0)}) n {int(m.sum())}: S(0.2) lagged {fmt(d[m])} | lag 0 same feeds {fmt(s0[0.2][m])} | W8 pool lagged - lag 0 {fmt((e_w - e_w0)[m])} | pop1kd pool lagged - lag 0 {fmt((e_p - e_p0)[m])}')
    (LOGS / f'read_{"_".join(sys.argv[2:])}.txt').write_text('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
