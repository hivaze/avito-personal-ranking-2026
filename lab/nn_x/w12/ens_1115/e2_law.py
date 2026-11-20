"""e2 part A: pool-size law N(n) = N_inf - c / n for the wide students (random subsets), train-only branch, fine-tuned branch, final (0.5 z FT + 0.5 z TO), holdout.
Sets: W16 (16 held-out students), FD16 (16 full-data students), W32 (both).  Output out/law.json + printed table.
Per n: up to R distinct random subsets (all subsets when fewer); the subset mean of the per-feed NDCG is an unbiased estimate of E[N(n)] over seeds.
Fit by least squares over n in a declared range; SE of c and of the forecast = sqrt(feed SE^2 + member-jackknife SE^2):
  feed SE: the fit is linear in the per-n means, so c has a per-feed contribution vector; member jackknife: curves from the subsets that exclude member j (fit range n <= N/2, where enough
  subsets exclude j), var = (N-1)/N sum (c_-j - mean)^2.
Forecast: gain(16 -> K) = c (1/16 - 1/K)."""
from ens_common import *
import itertools, math, time
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
m = Meta(); st = Store(m); rng = np.random.default_rng(1115); t0 = time.time()
import pmix; WTS = pmix.weights()

def subsets(N, n, R):
    if math.comb(N, n) <= R: return [np.array(c) for c in itertools.combinations(range(N), n)]
    seen = set(); out = []
    while len(out) < R:
        c = tuple(sorted(rng.choice(N, n, replace=False)))
        if c not in seen: seen.add(c); out.append(np.array(c))
    return out

def run(tag, names, hold):
    N = len(names); TO = st.mat('to_val', names).astype(np.float64); FT = st.mat('ft_val', names).astype(np.float64)
    H = st.mat('hold', names).astype(np.float64) if hold else None
    recs = []   # (n, subset, {kind: per-feed ndcg float32})
    for n in range(1, N + 1):
        for c in subsets(N, n, R):
            to = TO[c].mean(0); ft = FT[c].mean(0); d = dict(to=m.ev(to).astype(np.float32), ft=m.ev(ft).astype(np.float32),
                                                             fin=m.ev(0.5 * m.z('val', ft) + 0.5 * m.z('val', to)).astype(np.float32))
            if hold: d['hold'] = m.eh(H[c].mean(0)).astype(np.float32)
            recs.append((n, c, d))
        print(tag, n, f'{time.time() - t0:.0f}s', flush=True)
    out = {}
    for kind in (('to', 'ft', 'fin', 'hold') if hold else ('to', 'ft', 'fin')):
        def curve(excl=None, nmax=N):
            E = {}
            for n in range(1, nmax + 1):
                v = [d[kind] for (n_, c, d) in recs if n_ == n and (excl is None or excl not in c)]
                if v: E[n] = np.mean(v, 0, dtype=np.float64)
            return E
        def fit(E, lo, hi, w=None):
            ns = np.array([n for n in E if lo <= n <= hi]); A = np.stack([np.ones(len(ns)), -1.0 / ns], 1); G = np.linalg.pinv(A)      # (2, len(ns)): [a; c] = G @ means
            Y = np.stack([E[n] for n in ns])                                                                                              # (len(ns), feeds)
            pf = G @ Y                                                                                                                    # per-feed contributions to a and c
            if w is None: a, c = pf.mean(1); sa, sc = pf.std(1) / np.sqrt(pf.shape[1])
            else:
                a, sa = pmix.wmean_se(pf[0], w); c, sc = pmix.wmean_se(pf[1], w)
            res = max(abs(a - c / n - (E[n].mean() if w is None else np.average(E[n], weights=w))) for n in ns)
            return float(a), float(c), float(sa), float(sc), float(res)
        E = curve(); r = dict(curve={n: float(E[n].mean()) for n in E})
        for nm, lo, hi in (('all', 1, N), ('n>=2', 2, N), ('n>=4', 4, N), ('half', 1, N // 2)):
            a, c, sa, sc, res = fit(E, lo, hi); r[nm] = dict(a=a, c=c, se_a_feed=sa, se_c_feed=sc, max_resid=res)
        for wn in ('public', 'private'):
            a, c, sa, sc, res = fit(E, 1, N, WTS[wn] if kind != 'hold' else None); r['all_' + wn] = dict(a=a, c=c, se_c_feed=sc)
        # member jackknife on the 'half' range
        cj = []; aj = []
        for j in range(N):
            a, c, *_ = fit(curve(excl=j, nmax=N // 2), 1, N // 2); cj.append(c); aj.append(a)
        cj = np.array(cj); aj = np.array(aj)
        r['jk_se_c'] = float(np.sqrt((N - 1) / N * ((cj - cj.mean()) ** 2).sum())); r['jk_se_a'] = float(np.sqrt((N - 1) / N * ((aj - aj.mean()) ** 2).sum()))
        c = r['all']['c']; sc = float(np.hypot(r['all']['se_c_feed'], r['jk_se_c'])); r['se_c_total'] = sc
        r['gain_from_16'] = {K: dict(gain=c * (1 / 16 - 1 / K), se=sc * (1 / 16 - 1 / K)) for K in (32, 64, 128)}
        r['gain_from_16']['inf'] = dict(gain=c / 16, se=sc / 16)
        r['gain_from_N'] = {K: dict(gain=c * (1 / N - 1 / K), se=sc * (1 / N - 1 / K)) for K in (2 * N, 4 * N, 8 * N)}
        # out-of-range check: fit on n <= N/2, predict n = N
        a, c2, *_ = fit(E, 1, N // 2); r['oor_pred_N'] = float(a - c2 / N); r['oor_err'] = float(a - c2 / N - E[N].mean())
        # direct: realised pool of N against the mean of the random half-size pools (paired per feed)
        d = E[N] - E[N // 2]; r['direct_N_minus_half'] = dict(mean=float(d.mean()), se=se(d))
        out[kind] = r
        print(f'{tag} {kind}: a {r["all"]["a"]:.5f} c {c:.5f} (feed SE {r["all"]["se_c_feed"]:.5f}, member jackknife SE {r["jk_se_c"]:.5f}) max resid {r["all"]["max_resid"]:.5f} | '
              f'N({N}) {E[N].mean():.5f} E[N({N // 2})] {E[N // 2].mean():.5f} E[N(1)] {E[1].mean():.5f} | 16->32 {r["gain_from_16"][32]["gain"]:+.5f}+-{r["gain_from_16"][32]["se"]:.5f} '
              f'16->64 {r["gain_from_16"][64]["gain"]:+.5f}+-{r["gain_from_16"][64]["se"]:.5f} 16->128 {r["gain_from_16"][128]["gain"]:+.5f}+-{r["gain_from_16"][128]["se"]:.5f} 16->inf {c / 16:+.5f}+-{sc / 16:.5f} | '
              f'fit n>=4: a {r["n>=4"]["a"]:.5f} c {r["n>=4"]["c"]:.5f} | pmix public c {r["all_public"]["c"]:.5f} private c {r["all_private"]["c"]:.5f} | half-range fit predicts N({N}) with error {r["oor_err"]:+.5f} | '
              f'direct N({N}) - E[N({N // 2})] {d.mean():+.5f}+-{se(d):.5f}', flush=True)
    return out

res = {}
for tag, names, hold in (('W16', W16, True), ('FD16', FD16, False), ('W32', W16 + FD16, False)):
    res[tag] = run(tag, names, hold)
    json.dump(res, open(OUT / 'law.json', 'w'), indent=0)
print(f'done {time.time() - t0:.0f}s')
