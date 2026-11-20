"""gateB step 3 core: gate families fitted on a training index set, nested driver."""
from lib2 import *
import json, time
J5 = IF[0.5]; QS = [0.2, 0.35, 0.5, 0.65, 0.8]; LAMS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]; MCUTS = [25, 50, 100]
ORDER = ['CONST2', 'MSTEP', 'STUMP1', 'PSTUMP', 'LINM', 'LINp', 'LIN2']


def smooth(E):
    k = np.array([1, 2, 3, 2, 1], np.float64) / 9
    def ax(A, axis):
        A = np.moveaxis(A, axis, -1); Pd = np.concatenate([A[..., :1], A[..., :1], A, A[..., -1:], A[..., -1:]], -1)
        out = sum(k[t] * Pd[..., t:t + A.shape[-1]] for t in range(5)); return np.moveaxis(out, -1, axis)
    return np.ascontiguousarray(ax(ax(E, 1), 2))


@nb.njit(parallel=True, cache=False)
def lin_obj(Es, idx, Xs, a0, a, b0, b):
    n = len(idx); d = Xs.shape[1]; val = np.empty(n); gp = np.empty(n); gf = np.empty(n)
    for t in nb.prange(n):
        f = idx[t]; sp = a0; sf = b0
        for k in range(d):
            sp += a[k] * Xs[f, k]; sf += b[k] * Xs[f, k]
        p = 1.0 / (1.0 + np.exp(-sp)); r = 1.0 / (1.0 + np.exp(-sf)); u = 16.0 * p; v = 12.0 * r
        i = int(u); j = int(v)
        if i > 15: i = 15
        if j > 11: j = 11
        fa = u - i; fb = v - j; e00 = Es[f, i, j]; e10 = Es[f, i + 1, j]; e01 = Es[f, i, j + 1]; e11 = Es[f, i + 1, j + 1]
        val[t] = (1 - fa) * (1 - fb) * e00 + fa * (1 - fb) * e10 + (1 - fa) * fb * e01 + fa * fb * e11
        gp[t] = ((1 - fb) * (e10 - e00) + fb * (e11 - e01)) * 16.0 * p * (1 - p); gf[t] = ((1 - fa) * (e01 - e00) + fa * (e11 - e10)) * 12.0 * r * (1 - r)
    return val, gp, gf


def lin_out(Xs, a0, a, b0, b):
    return 0.8 / (1 + np.exp(-(a0 + Xs @ a))), 0.2 + 0.6 / (1 + np.exp(-(b0 + Xs @ b)))


class Ctx:
    """holds tensor, smoothed tensor, features for one composition."""
    def __init__(self, E, X, q, y):
        self.E = E; self.Es = smooth(E); self.X = X; self.q = q; self.y = y; self.n = len(E); self.E2 = E.reshape(self.n, -1)

    def prep(self, tr):
        """winsorise + standardise with training-fold statistics -> Xs for ALL feeds."""
        lo = np.quantile(self.X[tr], 0.01, axis=0); hi = np.quantile(self.X[tr], 0.99, axis=0); Xc = np.clip(self.X, lo, hi)
        mu = Xc[tr].mean(0); sd = Xc[tr].std(0); sd[sd < 1e-9] = 1.0
        return np.ascontiguousarray((Xc - mu) / sd), dict(lo=lo, hi=hi, mu=mu, sd=sd)

    def masks(self, tr):
        """stump masks (rows = candidate stumps) over ALL feeds from training-fold quantiles; list of (feature, cut)."""
        defs = []; rows = []
        for k in range(self.X.shape[1]):
            cuts = [0.5] if FEATS[k] == 'u0' else list(np.unique(np.quantile(self.X[tr, k], QS)))
            for c in cuts:
                m = self.X[:, k] > c
                if 0 < m[tr].sum() < len(tr): defs.append((k, float(c))); rows.append(m)
        return defs, np.array(rows)

    # ---- grid families: return (wp, wf) arrays over ALL feeds + description
    def const2(self, tr):
        m = self.E[tr].mean(0); i, j = np.unravel_index(m.argmax(), m.shape)
        return np.full(self.n, WP[i]), np.full(self.n, WF[j]), dict(wp=float(WP[i]), wf=float(WF[j]))

    def constp(self, tr):
        i = self.E[tr][:, :, J5].mean(0).argmax(); return np.full(self.n, WP[i]), np.full(self.n, 0.5), dict(wp=float(WP[i]), wf=0.5)

    def mstep(self, tr):
        best = None; Et = self.E[tr][:, :, J5]; lm = self.X[:, 0]
        for c in MCUTS:
            hi = lm >= np.log(c) - 1e-9; h = hi[tr]; lo_s = Et[~h].sum(0); hi_s = Et[h].sum(0); v = (lo_s.max() + hi_s.max()) / len(tr)
            if best is None or v > best[0] + 1e-12: best = (v, c, int(lo_s.argmax()), int(hi_s.argmax()), hi)
        v, c, il, ih, hi = best
        return np.where(hi, WP[ih], WP[il]), np.full(self.n, 0.5), dict(cut=c, lo=float(WP[il]), hi=float(WP[ih]))

    def stump1(self, tr):
        defs, R = self.masks(tr); Rt = R[:, tr].astype(np.float64); Et = self.E2[tr]; tot = Et.sum(0); H = Rt @ Et; L = tot[None] - H
        v = (H.max(1) + L.max(1)) / len(tr); s = int(v.argmax()); ih, jh = np.unravel_index(H[s].argmax(), (len(WP), len(WF))); il, jl = np.unravel_index(L[s].argmax(), (len(WP), len(WF)))
        m = R[s]; k, c = defs[s]
        return np.where(m, WP[ih], WP[il]), np.where(m, WF[jh], WF[jl]), dict(feat=FEATS[k], cut=c, lo=(float(WP[il]), float(WF[jl])), hi=(float(WP[ih]), float(WF[jh])))

    def pstump(self, tr):
        defs, R = self.masks(tr); Rt = R[:, tr].astype(np.float64); Et = self.E[tr]; n = len(tr); ar = np.arange(n)
        m0 = Et.mean(0); i0, j0 = np.unravel_index(m0.argmax(), m0.shape); ii = np.full(n, i0); jj = np.full(n, j0); cur = m0.max()
        dp = dict(feat=None, cut=None, lo=float(WP[i0]), hi=float(WP[i0])); df = dict(feat=None, cut=None, lo=float(WF[j0]), hi=float(WF[j0])); mp = np.zeros(self.n, bool); mf = np.zeros(self.n, bool)
        wpl = wph = WP[i0]; wfl = wfh = WF[j0]
        for it in range(4):
            Ep = Et[ar, :, jj]; tot = Ep.sum(0); H = Rt @ Ep; L = tot[None] - H; v = (H.max(1) + L.max(1)) / n; s = int(v.argmax())
            if v[s] > cur + 1e-12:
                cur = v[s]; mp = R[s]; wph = WP[H[s].argmax()]; wpl = WP[L[s].argmax()]; ii = np.where(mp[tr], H[s].argmax(), L[s].argmax()); dp = dict(feat=FEATS[defs[s][0]], cut=defs[s][1], lo=float(wpl), hi=float(wph)); ch = True
            else: ch = False
            Ef = Et[ar, ii, :]; tot = Ef.sum(0); H = Rt @ Ef; L = tot[None] - H; v = (H.max(1) + L.max(1)) / n; s = int(v.argmax())
            if v[s] > cur + 1e-12:
                cur = v[s]; mf = R[s]; wfh = WF[H[s].argmax()]; wfl = WF[L[s].argmax()]; jj = np.where(mf[tr], H[s].argmax(), L[s].argmax()); df = dict(feat=FEATS[defs[s][0]], cut=defs[s][1], lo=float(wfl), hi=float(wfh)); ch = True
            if not ch: break
        return np.where(mp, wph, wpl), np.where(mf, wfh, wfl), dict(pop=dp, ft=df)

    # ---- linear families
    def lin(self, tr, kind, lam, Xs=None):
        if Xs is None: Xs, _ = self.prep(tr)
        d = Xs.shape[1]; n = len(tr); tr = np.ascontiguousarray(tr.astype(np.int64))
        if kind == 'LINM':
            i0 = self.E[tr][:, :, J5].mean(0).argmax(); wp0, wf0 = WP[i0], 0.5; ma = np.zeros(d); ma[0] = 1; mb = np.zeros(d); tb0 = 0.0
        else:
            m = self.E[tr].mean(0); i0, j0 = np.unravel_index(m.argmax(), m.shape); wp0, wf0 = WP[i0], WF[j0]; ma = np.ones(d); mb = np.ones(d) if kind == 'LIN2' else np.zeros(d); tb0 = 1.0
        wp0 = min(max(wp0, 0.02), 0.78); wf0 = wf0 if kind == 'LINM' else min(max(wf0, 0.22), 0.78)
        th = np.zeros(2 * d + 2); th[0] = np.log(wp0 / 0.8 / (1 - wp0 / 0.8)); r0 = (wf0 - 0.2) / 0.6; th[d + 1] = np.log(r0 / (1 - r0))
        msk = np.concatenate([[1.0], ma, [tb0], mb]); pen = np.concatenate([[0.0], np.ones(d), [0.0], np.ones(d)])
        mom = np.zeros_like(th); vel = np.zeros_like(th); best = (-1e9, th.copy()); lr = 0.05; b1, b2 = 0.9, 0.999
        for t in range(301):
            val, gp, gf = lin_obj(self.Es, tr, Xs, th[0], th[1:d + 1], th[d + 1], th[d + 2:])
            obj = val.mean() - lam * (pen * th * th).sum()
            if obj > best[0]: best = (obj, th.copy())
            if t == 300: break
            Xt = Xs[tr]; gr = np.concatenate([[gp.mean()], Xt.T @ gp / n, [gf.mean()], Xt.T @ gf / n]) - 2 * lam * pen * th; gr *= msk
            mom = b1 * mom + (1 - b1) * gr; vel = b2 * vel + (1 - b2) * gr * gr; th = th + lr * (mom / (1 - b1 ** (t + 1))) / (np.sqrt(vel / (1 - b2 ** (t + 1))) + 1e-12)
        th = best[1]; wp, wf = lin_out(Xs, th[0], th[1:d + 1], th[d + 1], th[d + 2:])
        return wp, wf, dict(a0=float(th[0]), a=[float(x) for x in th[1:d + 1]], b0=float(th[d + 1]), b=[float(x) for x in th[d + 2:]], lam=lam)

    def grid_val(self, wp, wf, idx):
        """exact tensor value for grid-valued weights on feeds idx."""
        i = np.rint(wp[idx] / 0.05).astype(int); j = np.rint((wf[idx] - 0.2) / 0.05).astype(int); return self.E[idx, i, j]

    def exact(self, wp, wf):
        return ev_w(self.q, wp, wf, self.y)

    def fit_all(self, tr, g, inner_seed):
        """fit every family on tr (inner 4-fold CV for ridge + AUTO). Returns {family: (wp, wf, desc)} over ALL feeds."""
        fin = folds(g, inner_seed, 4)[tr]; cvv = {}; Xs, _ = self.prep(tr)
        grid_f = {'CONST2': self.const2, 'MSTEP': self.mstep, 'STUMP1': self.stump1, 'PSTUMP': self.pstump}
        acc = {k: np.zeros(len(tr)) for k in grid_f}; wl = {(k, l): (np.zeros(self.n), np.full(self.n, 0.5)) for k in ('LINM', 'LINp', 'LIN2') for l in LAMS}
        for c in range(4):
            itr = tr[fin != c]; iva = tr[fin == c]; Xi, _ = self.prep(itr)
            for k, fn in grid_f.items():
                wp, wf, _ = fn(itr); acc[k][fin == c] = self.grid_val(wp, wf, iva)
            for k in ('LINM', 'LINp', 'LIN2'):
                for l in LAMS:
                    wp, wf, _ = self.lin(itr, k, l, Xi); wl[(k, l)][0][iva] = wp[iva]; wl[(k, l)][1][iva] = wf[iva]
        for k in grid_f: cvv[k] = acc[k].mean()
        lam_best = {}
        for k in ('LINM', 'LINp', 'LIN2'):
            vs = [self.exact(*wl[(k, l)])[tr].mean() for l in LAMS]; b = int(np.argmax(vs)); lam_best[k] = LAMS[b]; cvv[k] = vs[b]
        out = {'CONSTp': self.constp(tr)}
        for k, fn in grid_f.items(): out[k] = fn(tr)
        for k in ('LINM', 'LINp', 'LIN2'): out[k] = self.lin(tr, k, lam_best[k], Xs)
        pick = ORDER[0]
        for k in ORDER[1:]:
            if cvv[k] > cvv[pick] + 1e-9: pick = k
        out['AUTO'] = (out[pick][0], out[pick][1], dict(pick=pick, cv={k: float(v) for k, v in cvv.items()}, lam=lam_best))
        return out


FAMS = ['CONSTp', 'CONST2', 'MSTEP', 'STUMP1', 'PSTUMP', 'LINM', 'LINp', 'LIN2', 'AUTO']


def nested(ctx, g, M, plain=(0.2, 0.5), seeds=range(5), tag='', log=print):
    """returns dict with per-feed nested arrays (mean over fold seeds), in-sample arrays, descriptions."""
    n = ctx.n; t0 = time.time(); allf = np.arange(n)
    nest = {k: np.zeros(n) for k in FAMS}; desc = {k: [] for k in FAMS}; wsum = {k: [np.zeros(n), np.zeros(n)] for k in FAMS}
    for s in seeds:
        fo = folds(g, s, 5); W = {k: [np.zeros(n), np.zeros(n)] for k in FAMS}
        for c in range(5):
            tr = allf[fo != c]; te = fo == c; out = ctx.fit_all(tr, g, 1000 + 10 * s + c)
            for k in FAMS:
                W[k][0][te] = out[k][0][te]; W[k][1][te] = out[k][1][te]; desc[k].append(out[k][2])
        for k in FAMS:
            nest[k] += ctx.exact(W[k][0], W[k][1]) / len(seeds); wsum[k][0] += W[k][0] / len(seeds); wsum[k][1] += W[k][1] / len(seeds)
        log(f'[{tag}] seed {s} done {time.time() - t0:.0f}s  AUTO-CONST2 so far (x{s + 1}/{len(seeds)} scale) {(nest["AUTO"] - nest["CONST2"]).mean() * len(seeds) / (s + 1):+.5f}')
    full = ctx.fit_all(allf, g, 999); ins = {k: ctx.exact(full[k][0], full[k][1]) for k in FAMS}
    pl_ = ctx.E[:, IP[plain[0]], IF[plain[1]]]; mr = np.where(M >= 50, ctx.E[:, IP[0.4], J5], ctx.E[:, IP[0.2], J5])
    return dict(nest=nest, ins=ins, desc=desc, full={k: full[k][2] for k in FAMS}, fullw={k: (full[k][0], full[k][1]) for k in FAMS}, plain=pl_, mrule=mr, wmean=wsum)
