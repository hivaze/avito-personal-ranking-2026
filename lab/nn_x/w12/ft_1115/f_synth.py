"""ft_1115 step F: synthesis numbers.
 F1  fold-averaging gain from the real-noise SIMEX arrays (out/c3_simex.npz): linear-through-origin and quadratic fits, per w_ft and view.
 F2  estimated TEST-TIME curve of the file against w_ft: d_test(w) = d_view(w) + [G(w) - G(0.5)], G(w) = gain of removing 4/5 of the fold noise at weight w
     (measured per w where available, G1 w^2 otherwise), view = pmix-public / pmix-private; pools W8P20, FD16P20 and the four-disjoint-pool mean; bootstrap of the vertex.
 F3  FT half on public: prior (pmix-public + fold term) against the two public pairs; residual = what val cannot measure.
 F4  value of a public probe at w_ft 1.0 or 0.75 on the W8 pool (three-point public curve): normal-normal update, decision thresholds, expected gain.
Output: out/f_synth.txt"""
from lib import *
from math import erf, sqrt
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
Z = np.load(OUT / 'c3_simex.npz'); lam = Z['lam']; SG = list(np.round(Z['grid'], 2))
w_('## F1. Fold averaging (test = mean of 5 fold models; val out-of-fold = 1): gain of removing 4/5 of the fold noise, real-noise SIMEX (12 draws per lambda)')
w_('pool | w_ft | linear through 0: val / pmix-public / pmix-private | quadratic fit (a lam + b lam^2 at lam = -0.8): val / pmix-public / pmix-private | SE over draws (linear, val)')
G = {}
for name in ('W8P20', 'FD16P20', 'W8'):
    L = Z[name]                                                    # lam x draws x w x view
    for wi, w in enumerate(SG):
        lin = []; quad = []
        for v in range(3):
            y = L[:, :, wi, v].mean(1); sl = (lam * y).sum() / (lam ** 2).sum(); lin.append(-0.8 * sl)
            A = np.stack([lam, lam ** 2], 1); a, b = np.linalg.lstsq(A, y, rcond=None)[0]; quad.append(a * -0.8 + b * 0.64)
        sld = (lam[:, None] * L[:, :, wi, 0]).sum(0) / (lam ** 2).sum(); sse = 0.8 * sld.std(ddof=1) / np.sqrt(len(sld))
        G[(name, w)] = (lin, quad, sse)
        w_(f'{name} | {w:.2f} | {lin[0]:+.5f} / {lin[1]:+.5f} / {lin[2]:+.5f} | {quad[0]:+.5f} / {quad[1]:+.5f} / {quad[2]:+.5f} | {sse:.5f}')
g1 = {n: (G[(n, 1.0)][0][0], G[(n, 1.0)][1][0]) for n in ('W8P20', 'FD16P20', 'W8')}
w_('G1 (FT branch alone), linear / quadratic: ' + ' | '.join(f'{n} {a:+.5f} / {b:+.5f}' for n, (a, b) in g1.items()) + f' ; mean linear {np.mean([a for a, b in g1.values()]):+.5f}, mean quadratic {np.mean([b for a, b in g1.values()]):+.5f}')
w_('file at w_ft 0.5, linear / quadratic: ' + ' | '.join(f'{n} {G[(n, 0.5)][0][0]:+.5f} / {G[(n, 0.5)][1][0]:+.5f}' for n in ('W8P20', 'FD16P20', 'W8')))
G1lo, G1hi = float(np.mean([a for a, b in g1.values()])), float(np.mean([b for a, b in g1.values()]))
w_(f'-> used below: G(w) = G1 w^2 with G1 between {G1lo:+.5f} (linear) and {G1hi:+.5f} (quadratic); at w_ft 0.5: {G1lo / 4:+.5f} .. {G1hi / 4:+.5f}')
w_()
# ------------------------------------------------------------------ F2
w_('## F2. Estimated test-time curve against w_ft (delta against 0.5): view curve + fold-averaging term G1 (w^2 - 0.25)')
BG = None; CUR = {}
for name in ('W8P20', 'FD16P20'):
    z = np.load(OUT / f'b_perfeed_{name}.npz'); BG = list(np.round(z['grid'], 2)); CUR[name] = z['e'] - z['e'][BG.index(0.5)]
CUR['mean of 4 disjoint 8-pools'] = np.load(OUT / 'd_recipe_mean_delta.npy')
g = user_groups(); ug, gi = np.unique(g, return_inverse=True); ng = len(ug); rng = np.random.default_rng(11)
gw = np.array(BG); X = np.stack([gw - 0.5, (gw - 0.5) ** 2], 1)
for name, Dm in CUR.items():
    w_(f'### {name}')
    w_('w_ft | plain val (out-of-fold) | pmix-public | pmix-public + fold term (G1 linear .. quadratic) | pmix-private | pmix-private + fold term')
    for i, w in enumerate(BG):
        t = (w * w - 0.25); a, sa = ms(Dm[i], WPUB); b, sb = ms(Dm[i], WPRIV)
        w_(f'{w:.2f} | {f(Dm[i])} | {a:+.5f}+-{sa:.5f} | {a + G1lo * t:+.5f} .. {a + G1hi * t:+.5f} | {b:+.5f}+-{sb:.5f} | {b + G1lo * t:+.5f} .. {b + G1hi * t:+.5f}')
    for view, wt in (('pmix-public', WPUB), ('pmix-private', WPRIV)):
        num = np.stack([np.bincount(gi, weights=Dm[i] * wt, minlength=ng) for i in range(len(BG))]); den = np.bincount(gi, weights=wt, minlength=ng)
        for lab, G1 in (('no fold term', 0.0), ('G1 linear', G1lo), ('G1 quadratic', G1hi)):
            corr = G1 * (gw ** 2 - 0.25); vs = []; gains = []; am = []
            def fit(m):
                p, q = np.linalg.lstsq(X, m + corr, rcond=None)[0]
                v = 0.5 - p / (2 * q) if q < 0 else (1.0 if p > 0 else 0.0); v = float(np.clip(v, 0, 1)); return v, p * (v - 0.5) + q * (v - 0.5) ** 2
            v0, g0 = fit(num.sum(1) / den.sum())
            for b_ in range(2000):
                c = np.bincount(rng.integers(0, ng, ng), minlength=ng).astype(np.float64); m = (num @ c) / (den @ c); v, gg = fit(m); vs.append(v); gains.append(gg); am.append(gw[int(np.argmax(m + corr))])
            vs = np.array(vs); am = np.array(am)
            w_(f'  {view}, {lab}: fitted optimum {v0:.2f} (bootstrap 16-50-84%: {np.percentile(vs, 16):.2f} {np.percentile(vs, 50):.2f} {np.percentile(vs, 84):.2f}; 2.5-97.5%: {np.percentile(vs, 2.5):.2f} {np.percentile(vs, 97.5):.2f}) | fitted gain of the optimum over 0.5 {g0:+.5f}'
               f' | P(optimum > 0.6) {(vs > 0.6).mean():.2f} P(optimum < 0.4) {(vs < 0.4).mean():.2f} | raw-grid argmax: P(<0.5) {(am < 0.5).mean():.2f} P(=0.5) {(am == 0.5).mean():.2f} P(>0.5) {(am > 0.5).mean():.2f}')
w_()
# ------------------------------------------------------------------ F3
w_('## F3. The FT half (file at w_ft 0.5 minus train-only pool) at the public mix: estimate from val against the public readings')
rows = []
for name, pub, se_pub in (('W8', 0.000477, None), ('G3 (scored, 1-seed FT)', 0.000474, None)):
    if name == 'W8':
        z = np.load(OUT / 'b_perfeed_W8.npz'); gr = list(np.round(z['grid'], 2)); d = z['e'][gr.index(0.5)] - z['e'][gr.index(0.0)]
    else:
        g_ft, g_to = stored('G3', 'val'); d = ev(0.5 * zv(g_ft) + 0.5 * zv(g_to)) - ev(zv(g_to))
    v, sv = ms(d); a, sa = ms(d, WPUB); b, sb = ms(d, WPRIV); sp = d.std() / np.sqrt(8562)
    fold = (G1lo / 4, G1hi / 4); est = a + np.mean(fold); res = pub - est; sres = np.sqrt(sa ** 2 + sp ** 2)
    rows.append((d, pub, sp, est, sa))
    w_(f'{name}: val out-of-fold {v:+.5f}+-{sv:.5f} | feed mix (pmix-public) {a:+.5f}+-{sa:.5f} | + fold averaging {fold[0]:+.5f}..{fold[1]:+.5f} -> estimate {est:+.5f}+-{sa:.5f} | public reading {pub:+.6f}+-{sp:.5f}'
       f' | public minus estimate {res:+.5f}+-{sres:.5f} | private-mix estimate {b + np.mean(fold):+.5f}+-{sb:.5f}')
rho = float(np.corrcoef(rows[0][0], rows[1][0])[0, 1])
est_m = 0.5 * (rows[0][3] + rows[1][3]); se_est = 0.5 * np.sqrt(rows[0][4] ** 2 + rows[1][4] ** 2 + 2 * rho * rows[0][4] * rows[1][4])
pub_m = 0.5 * (rows[0][1] + rows[1][1]); se_pub = 0.5 * np.sqrt(rows[0][2] ** 2 + rows[1][2] ** 2 + 2 * rho * rows[0][2] * rows[1][2])
res = pub_m - est_m; sres = np.sqrt(se_est ** 2 + se_pub ** 2)
w_(f'two pairs pooled (per-feed correlation of the two deltas {rho:.3f}): estimate {est_m:+.5f}+-{se_est:.5f} | public {pub_m:+.6f}+-{se_pub:.5f} | residual = everything val cannot measure (extra 20% of data seen by the 15 models, pmix error, noise) {res:+.5f}+-{sres:.5f}'
   f' -> one-sided 95% upper bound {res + 1.645 * sres:+.5f}, lower {res - 1.645 * sres:+.5f}')
post = (est_m / se_est ** 2 + pub_m / se_pub ** 2) / (1 / se_est ** 2 + 1 / se_pub ** 2); spost = 1 / np.sqrt(1 / se_est ** 2 + 1 / se_pub ** 2)
w_(f'precision-weighted value of the FT half on public (val-based estimate and public readings as two independent witnesses): {post:+.5f}+-{spost:.5f} (z {post / spost:.1f}); P(FT half > 0 on public) = {Phi(post / spost):.2f}')
for name in ('W8P20', 'FD16P20'):
    z = np.load(OUT / f'b_perfeed_{name}.npz'); gr = list(np.round(z['grid'], 2)); d = z['e'][gr.index(0.5)] - z['e'][gr.index(0.0)]
    a, sa = ms(d, WPUB); b, sb = ms(d, WPRIV); v, sv = ms(d)
    w_(f'{name}: FT half val out-of-fold {v:+.5f}+-{sv:.5f} | test-time estimate at the public mix {a + (G1lo + G1hi) / 8:+.5f}+-{sa:.5f} | at the private mix {b + (G1lo + G1hi) / 8:+.5f}+-{sb:.5f}')
w_()
# ------------------------------------------------------------------ F4
w_('## F4. What one public probe on the W8 pool can do (public points on record: w_ft 0 -> 0.394009, w_ft 0.5 -> 0.394486)')
z = np.load(OUT / 'b_perfeed_W8.npz'); gr = list(np.round(z['grid'], 2)); e = z['e']
dA = e[gr.index(0.5)] - e[gr.index(0.0)]
a_pri, sa_pri = ms(dA, WPUB); a_pri += (G1lo + G1hi) / 8; sa_pub = dA.std() / np.sqrt(8562)
a_post = (a_pri / sa_pri ** 2 + 0.000477 / sa_pub ** 2) / (1 / sa_pri ** 2 + 1 / sa_pub ** 2); sa_post = 1 / np.sqrt(1 / sa_pri ** 2 + 1 / sa_pub ** 2)
w_(f'a = y(0.5) - y(0): prior from val (pmix-public + fold term) {a_pri:+.5f}+-{sa_pri:.5f}; public {0.000477:+.6f}+-{sa_pub:.5f}; posterior {a_post:+.5f}+-{sa_post:.5f}')
MODEL_SD = 0.0003            # allowance for the error of pmix as a predictor of a public delta (judgement)
for wprobe in (1.0, 0.75):
    dB = e[gr.index(wprobe)] - e[gr.index(0.5)]; b_pri, sb = ms(dB, WPUB); b_pri += 0.5 * (G1lo + G1hi) * (wprobe ** 2 - 0.25); sb_pri = np.sqrt(sb ** 2 + MODEL_SD ** 2); sb_pub = dB.std() / np.sqrt(8562)
    kk = sb_pri ** 2 / (sb_pri ** 2 + sb_pub ** 2); sd_pred = np.sqrt(sb_pri ** 2 + sb_pub ** 2)
    # quadratic through t = -0.5 (0), 0 (0.5), h (probe): f(t) = p t + q t^2 ; a = 0.5 p - 0.25 q ; b = p h + q h^2
    h = wprobe - 0.5
    def vertex(a, b):
        q = (b - 2 * h * a) / (h * h + 0.5 * h); p = 2 * a + 0.5 * q
        if q >= 0: return (1.0 if p > 0 else 0.0), max(p * 0.5 + q * 0.25, 0.0)
        t = float(np.clip(-p / (2 * q), -0.5, 0.5)); return 0.5 + t, p * t + q * t * t
    v0, g0 = vertex(a_post, b_pri)
    w_(f'probe at w_ft {wprobe}: predicted public delta against W8 {b_pri:+.5f} (prior sd {sb_pri:.5f}, incl. {MODEL_SD} model allowance); paired public SE of the probe {sb_pub:.5f}; predictive sd of the reading {sd_pred:.5f}; weight of the reading in the posterior {kk:.2f}')
    w_(f'   prior optimum (quadratic through the three points) {v0:.2f}, gain over 0.5 {g0:+.6f}')
    # thresholds on the observed delta for the posterior optimum to pass 0.6 / fall below 0.4; expected gain of acting on the probe
    obs = np.linspace(b_pri - 4 * sd_pred, b_pri + 4 * sd_pred, 4001); dens = np.exp(-0.5 * ((obs - b_pri) / sd_pred) ** 2); dens /= dens.sum()
    vv = np.array([vertex(a_post, (1 - kk) * b_pri + kk * o) for o in obs])
    up = obs[vv[:, 0] >= 0.6]; dn = obs[vv[:, 0] <= 0.4]
    w_(f'   posterior optimum >= 0.6 iff the probe reads >= {up.min() if len(up) else float("nan"):+.5f} against W8 (probability {dens[vv[:, 0] >= 0.6].sum():.2f}); <= 0.4 iff it reads <= {dn.max() if len(dn) else float("nan"):+.5f} (probability {dens[vv[:, 0] <= 0.4].sum():.2f})')
    w_(f'   expected gain of moving the weight to the posterior optimum after the probe (against keeping 0.5), averaged over readings: {(dens * vv[:, 1]).sum():+.6f}; gain if the reading is +1 predictive sd above the forecast: {vertex(a_post, (1 - kk) * b_pri + kk * (b_pri + sd_pred))[1]:+.6f} (optimum {vertex(a_post, (1 - kk) * b_pri + kk * (b_pri + sd_pred))[0]:.2f}); at +2 sd: {vertex(a_post, (1 - kk) * b_pri + kk * (b_pri + 2 * sd_pred))[1]:+.6f} (optimum {vertex(a_post, (1 - kk) * b_pri + kk * (b_pri + 2 * sd_pred))[0]:.2f})')
open(OUT / 'f_synth.txt', 'w').write('\n'.join(out) + '\n')
