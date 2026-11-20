"""Lane A c03: gates g1 and g2 of RULE.txt (labels of the witnesses are opened here, AFTER c02_parity.py).
 g1  cz = condlib.Null.score, null model = pool + control block + nx_rp3_z + nx_rp3_rk; H = holdout (form 0 outside the 15,225 lag-view feeds), T = val with the
     private-like twin (nx3 stale-val twin in the null). Holm over the two forms on p_form = max(p_H, p_T); bars scaled by max(1, sd of 100 within-feed permutations);
     no permutation may reach the read.
 g2  exposure-preserving placebos, 100 draws: item values x permuted inside the feed within strata, entered value recomputed and scored.
     P1 strata = 8 bins of imp3 (CRITIC2 4.1); P2 strata = 8 bins of the form's own expected clicks e.
 Information (cannot promote): N = natural val; halves and lags; pmix weights; null model without nx3.
The entered value is multiplied by ONE constant per form before the float32 cast (25 for F1, 300 for F2): cz does not depend on it (both values are printed).
A first launch stopped on an exact-equality assertion (float64 against float32 statistic) before any form was scored: out/c03_info.failed1.txt.
usage: c03_info.py   output out/c03_info.txt, out/c03_info.json"""
import json
from scipy.stats import norm
from cs_common import *
lens, condlib = load_lens()
import pmix
P = Log(OUT / 'c03_info.txt')
P('RULE.txt md5 ' + md5(CS + '/RULE.txt') + f'; lens md5 {md5("<repo>/reports/INPUTS_1116/residual_rows/lens.py")}, condlib md5 {md5("<repo>/reports/INPUTS_1116/residual_rows/condlib.py")}')
V = Views(P)
FORMS = [('F1 cs3a_c', 'cs3a', 25.0), ('F2 cs3x_c', 'cs3x', 300.0)]
ND = 100
W = pmix.weights()


class MyNull:
    """copy of critic/cc_common.py MyNull (per-feed efficient scores, needed for the weighted cz); fit = condlib.fit_big"""

    def __init__(self, D, cols, theta0):
        self.D = D; n = D['n']; off = D['off']
        self.X = np.ascontiguousarray(np.stack([D['base']] + list(cols), 1).astype(np.float32)); K = self.X.shape[1]
        r = condlib.fit_big(D, self.X, theta0=theta0)
        self.theta = r['theta']; self.fid = np.repeat(np.arange(n), D['M'])
        s = (self.X @ self.theta.astype(np.float32)).astype(np.float64)
        mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[self.fid]); Z = np.add.reduceat(e, off); self.p = e / Z[self.fid]
        self.npos = np.bincount(self.fid, D['y'], n); self.w = self.npos[self.fid] * self.p; self.res = D['y'].astype(np.float64) - self.w
        A = np.zeros((K, K)); self.MU = np.zeros((n, K))
        step = 400000; b = np.searchsorted(off, np.arange(0, len(self.p), step)); b = np.unique(np.concatenate([b, [n]]))
        for i in range(len(b) - 1):
            f0_, f1_ = b[i], b[i + 1]; r0 = off[f0_]; r1 = off[f1_] if f1_ < n else len(self.p)
            Xc = self.X[r0:r1].astype(np.float64)
            A += (Xc * self.w[r0:r1, None]).T @ Xc
            self.MU[f0_:f1_] = np.add.reduceat(Xc * self.p[r0:r1, None], off[f0_:f1_] - r0, axis=0)
        A -= (self.MU * self.npos[:, None]).T @ self.MU
        self.Ainv = np.linalg.inv(A + 1e-6 * np.eye(K) * np.trace(A) / K)

    def score_f(self, x):
        D = self.D; off = D['off']; x = np.asarray(x, np.float64)
        mx = np.add.reduceat(self.p * x, off)
        b = (self.X.astype(np.float32).T @ (self.w * x).astype(np.float32)).astype(np.float64) - self.MU.T @ (self.npos * mx)
        g = self.Ainv @ b
        xt = x - (self.X @ g.astype(np.float32)).astype(np.float64)
        return np.add.reduceat(self.res * xt, off)


def czw(D, Uf, wf=None):
    u = Uf if wf is None else Uf * wf
    Uc = np.bincount(D['clus'], u); s = np.sqrt((Uc ** 2).sum())
    return float(u.sum() / s) if s > 0 else 0.0


RES = {}
for wit in ('H', 'T', 'N'):
    sp = 'hold' if wit == 'H' else 'val'; q = {'H': 'hold', 'T': 'val_twin', 'N': 'val_nat'}[wit]
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']
    rows, fl, off_q, M_q = V.rows(q)
    assert (off == off_q).all() and (M == M_q).all() and (D['rows'] is None or (D['rows'] == rows).all())
    if wit == 'H':
        nzr = np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32)
        nz = np.nan_to_num(nzr, nan=0.0); nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
    elif wit == 'T':
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
    else:
        nz = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nr = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
    N = condlib.Null(D, cn + ['nx_rp3_z', 'nx_rp3_rk'], cl + [nz, nr])
    chk = N.score(nz)
    N0 = condlib.Null(D, cn, cl); c0 = N0.score(nz)
    P(f'\n=== witness {wit}: feeds {D["n"]}, rows {len(D["y"])}; control block {len(cn)} columns + nx_rp3_z, nx_rp3_rk; pool NDCG {D["ndcg0"].mean():.5f} ===')
    P(f'  checks: nx_rp3_z under the null model that holds it: cz {chk["cz"]:+.3f} (must be 0); under the null model without nx3: cz {c0["cz"]:+.2f} (record: H +4.98, T +6.46, N +6.78)')
    Nm = MyNull(D, cl + [nz, nr], theta0=N.r['theta'])
    fid = N.fid; slot = V.row_slot(q); lag = np.load(arr_path(q, 'lag_row')).astype(np.int64)
    imp = get(q, 'imp3')
    rng = np.random.default_rng({'H': 7101, 'T': 7102, 'N': 7103}[wit])
    for fn, k, scale in FORMS:
        raw = get(q, f'{k}_x'); view = ~np.isnan(raw); e_ = get(q, f'{k}_e')
        st1 = strata(imp, view, off, M); st2 = strata(e_, view, off, M)
        ent = centre_in_strata(raw, st1, off, M, NB)
        # the READ is on the stored entered array (built from the float64 statistic); the placebos recompute it from the stored float32 statistic
        stored = np.nan_to_num(np.load(arr_path(q, f'{k}_c')).astype(np.float64)); dmax = float(np.abs(np.nan_to_num(ent).astype(np.float64) - stored).max())
        assert dmax < 1e-6, f'entered value differs from the stored array by {dmax}'
        x = (stored * scale).astype(np.float32)
        r = N.score(x); read = float(r['cz']); r_un = float(N.score(stored.astype(np.float32))['cz']); r_my = czw(D, Nm.score_f(x))
        r_re = float(N.score((np.nan_to_num(ent).astype(np.float64) * scale).astype(np.float32))['cz'])
        P(f'  {fn}: stored entered array against its recomputation from the stored float32 statistic: max abs difference {dmax:.2e}; cz of the recomputation {r_re:+.3f}')
        o = dict(cz=read, cz_unscaled=r_un, cz_second_path=r_my, b1=float(r['b1']) * scale, dnats=float(r['dnats']), r2=float(r['r2']), cz_without_nx3=float(N0.score(x)['cz']))
        P(f'  {fn} | cz {read:+.3f} (without the constant factor {r_un:+.3f}; second score path {r_my:+.3f}) | coefficient per unit of the entered value {o["b1"]:+.4f} | nats per feed {o["dnats"]:.6f} | R2 with the null model {o["r2"]:.3f} | without nx3 in the null {o["cz_without_nx3"]:+.2f}')
        # P0: within-feed permutations of the entered value
        p0 = np.array([float(N.score(condlib.perm_within(x, off, M, {'H': 3000, 'T': 4000, 'N': 5000}[wit] + s))['cz']) for s in range(ND)])
        o['P0'] = dict(sd=float(p0.std()), mean=float(p0.mean()), max_abs=float(np.abs(p0).max()), n_beyond=int((np.abs(p0) >= abs(read)).sum()), draws=p0.tolist())
        P(f'      P0 within-feed permutations ({ND}): mean {p0.mean():+.2f}, sd {p0.std():.2f}, max |cz| {np.abs(p0).max():.2f}; at or beyond the read: {o["P0"]["n_beyond"]} of {ND}')
        for pn, st in (('P1', st1), ('P2', st2)):
            key = fid * NB + st; cz_ = []; cor = []
            for s in range(ND):
                xp = perm_in_strata(raw, key, rng); xe = (np.nan_to_num(centre_in_strata(xp, st1, off, M, NB)).astype(np.float64) * scale).astype(np.float32)
                cz_.append(float(N.score(xe)['cz']))
                if s < 5:
                    cor.append(float(np.corrcoef(xe[view], x[view])[0, 1]))
            cz_ = np.array(cz_)
            o[pn] = dict(mean=float(cz_.mean()), sd=float(cz_.std()), min=float(cz_.min()), max=float(cz_.max()), click_part=float((read - cz_.mean()) / cz_.std()), n_beyond=int((np.abs(cz_) >= abs(read)).sum()), corr=float(np.mean(cor)), draws=cz_.tolist())
            P(f'      {pn} placebo within strata of {"imp3" if pn == "P1" else "the form own expected clicks"} ({ND}): mean {cz_.mean():+.2f}, sd {cz_.std():.2f}, min {cz_.min():+.2f}, max {cz_.max():+.2f}; (read - mean) / sd = {o[pn]["click_part"]:+.2f}; at or beyond the read {o[pn]["n_beyond"]} of {ND}; corr(placebo, entered) {np.mean(cor):+.3f}')
        # information
        info = {}
        Uf = Nm.score_f(x)
        if wit == 'H':
            regs = [('slots 6-8', (slot >= 6) & (slot <= 8)), ('slots 11-13', slot >= 11)] + [(f'lag {l} d', lag == l) for l in (3, 5, 8, 9, 10)]
        else:
            regs = [('slot 16', slot == 16), ('slot 18', slot == 18)] + [(f'lag {l} d', lag == l) for l in np.unique(lag) if (lag == l).sum() > 10000]
            for wn in ('public', 'private'):
                info[f'pmix-{wn} weights'] = czw(D, Uf, W[wn])
        for rn, m in regs:
            info[rn] = float(N.score(np.where(m, x, 0.0).astype(np.float32))['cz'])
        o['info'] = info
        P('      information: ' + ' | '.join(f'{a} {b:+.2f}' for a, b in info.items()))
        RES.setdefault(fn, {})[wit] = o
    del N, N0, Nm, D

# ------------------------------------------------------------------ verdicts
ALPHA = float(norm.sf(2.5))
P(f'\n=== g1 (Holm over the two forms, alpha = 1 - Phi(2.5) = {ALPHA:.5f}; z bar for the smaller p_form {norm.isf(ALPHA / 2):.3f}, then {norm.isf(ALPHA):.3f}) ===')
pf = {}
for fn, _, _ in FORMS:
    zs = {}
    for wit in ('H', 'T'):
        o = RES[fn][wit]; zs[wit] = o['cz'] / max(1.0, o['P0']['sd'])
    pf[fn] = max(float(norm.sf(zs['H'])), float(norm.sf(zs['T'])))
    RES[fn]['g1'] = dict(z_eff_H=zs['H'], z_eff_T=zs['T'], p_form=pf[fn], no_perm_beyond=bool(RES[fn]['H']['P0']['n_beyond'] == 0 and RES[fn]['T']['P0']['n_beyond'] == 0))
order = sorted(pf, key=pf.get); alive = True
for i, fn in enumerate(order):
    bar = ALPHA / (len(order) - i); holm = alive and pf[fn] <= bar; alive = holm
    g = RES[fn]['g1']; g['holm_bar_p'] = bar; g['holm_bar_z'] = float(norm.isf(bar)); g['holm'] = bool(holm); g['pass'] = bool(holm and g['no_perm_beyond'])
    P(f'  {fn}: cz H {RES[fn]["H"]["cz"]:+.2f} (P0 sd {RES[fn]["H"]["P0"]["sd"]:.2f} -> scaled {g["z_eff_H"]:+.2f}), T {RES[fn]["T"]["cz"]:+.2f} (P0 sd {RES[fn]["T"]["P0"]["sd"]:.2f} -> scaled {g["z_eff_T"]:+.2f}); p_form {pf[fn]:.4f} against the Holm bar {bar:.5f} (z {g["holm_bar_z"]:.2f} on both): {"ok" if holm else "MISS"}; '
      f'within-feed permutations at or beyond the read H / T: {RES[fn]["H"]["P0"]["n_beyond"]} / {RES[fn]["T"]["P0"]["n_beyond"]} of {ND} ({"ok" if g["no_perm_beyond"] else "MISS"}) -> g1 {"PASS" if g["pass"] else "MISS"}; N (information) {RES[fn]["N"]["cz"]:+.2f}')
P('=== g2 (both witnesses: |mean P1| < 1.5, |mean P2| < 1.5, (read - mean) / sd >= 2.5 against P1 and against P2) ===')
for fn, _, _ in FORMS:
    ok = True; cells = []
    for wit in ('H', 'T'):
        o = RES[fn][wit]
        for pn in ('P1', 'P2'):
            a = abs(o[pn]['mean']) < 1.5; b = o[pn]['click_part'] >= 2.5; ok &= a and b
            cells.append(f'{wit} {pn}: mean {o[pn]["mean"]:+.2f} ({"ok" if a else "MISS"}), click part {o[pn]["click_part"]:+.2f} ({"ok" if b else "MISS"})')
    RES[fn]['g2'] = dict(**{'pass': bool(ok)})
    P(f'  {fn}: ' + '; '.join(cells) + f' -> g2 {"PASS" if ok else "MISS"}')
P('=== numbers of the HOLDOUT-ONLY clause (printed in any case): cz on T at the pmix-private weights (bar +1.7), N cz (bar +2.0) ===')
for fn, _, _ in FORMS:
    P(f'  {fn}: T at pmix-private weights {RES[fn]["T"]["info"]["pmix-private weights"]:+.2f}; N {RES[fn]["N"]["cz"]:+.2f}; N at pmix-private weights {RES[fn]["N"]["info"]["pmix-private weights"]:+.2f}')
json.dump(RES, open(OUT / 'c03_info.json', 'x'), indent=1, default=float)
P('DONE')
