"""CCB-1113 tester1 harness: nested LIN residual on G3 (PREDECLARED_ccb.txt B1-B8). CPU only, read-only on the project.
Import: from h import *  ->  H = Harness(); H.run(name, xs_val, xs_hold, ...)
"""
import os, sys, json, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib_av import *  # noqa  (local copy of nn_x/audit/features/lib_av.py; numba cache stays in this dir)
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
sys.path.insert(1, "<repo>/src")
from metrics import ndcg_per_feed  # noqa
import pmix  # noqa

T0 = time.time()
NOISE_BAND = (-0.00047, 0.00022)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def holdout_ids():
    ftr, Mtr, offtr = meta("train")
    p = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    p[Mtr < 50] = 0.0
    hold = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
    return hold, ftr, Mtr, offtr, p


G3_HOLD = [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + \
          [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)]


class Harness:
    def __init__(self):
        f, M, off = meta("val")
        self.fv, self.M, self.off = f, M, off
        self.user = f["user_id"].to_numpy()
        self.y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
        nF = len(M); self.nF = nF
        self.fid = np.repeat(np.arange(nF), M)
        self.npos_f = np.bincount(self.fid, weights=self.y, minlength=nF)
        day = (f["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
        self.vday = day
        luo = np.load(feat_file("val", "it_hist_all_luo")); cold = luo == 0
        coldpos_f = np.bincount(self.fid, weights=self.y * cold, minlength=nF)
        self.SL = {"M<50": M < 50, "M>=50": M >= 50, "tue": day <= day.min() + 1, "thu": day >= day.min() + 2,
                   "coldpos": (coldpos_f == self.npos_f) & (self.npos_f > 0),
                   "warmpos": (coldpos_f == 0) & (self.npos_f > 0)}
        zz = self.zv
        self.b = zz(0.5 * zz(np.load(ROOT / "preds/ens_G3_ft_val.npy")) + 0.5 * zz(np.load(ROOT / "preds/ens_G3_to_val.npy")))
        self.nd = lambda s: ndcg_per_feed(s, self.y, off, M)
        bv = self.nd(self.b).mean()
        assert abs(bv - 0.40032) < 2e-5, bv
        self.folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=self.user))
        self.rowfold = np.empty(len(self.y), np.int64)
        for k, (trf, vaf) in enumerate(self.folds):
            self.rowfold[np.isin(self.fid, vaf)] = k
        # holdout
        hm = np.load(ROOT / "preds/holdout/holdout_meta.npy")
        self.Mh, self.oh = hm[0].astype(np.int64), hm[1].astype(np.int64)
        self.yh = np.load(ROOT / "preds/holdout/holdout_y.npy").astype(np.float64)
        hold, ftr, Mtr, offtr, p = holdout_ids()
        assert np.array_equal(Mtr[hold], self.Mh)
        self.hold = hold
        self.hrows = rows_of(offtr, Mtr, hold)   # rows of data/flat/train for the holdout candidates (holdout order)
        ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
        assert np.array_equal(np.asarray(ytr[self.hrows]).astype(np.float64), self.yh)
        ts = ftr["timestamp"].to_numpy()[hold].astype(np.int64)
        hday = (ts + 10800) // 86400
        self.hday_abs = hday
        dd = hday - hday.min()
        self.H_LATE, self.H_EARLY = dd >= 11, dd <= 2
        assert self.H_LATE.sum() == 8279 and self.H_EARLY.sum() == 7823
        acc = None
        for k in G3_HOLD:
            z = self.zh(np.load(ROOT / f"preds/holdout/nn_{k}_holdout.npy"))
            acc = z if acc is None else acc + z
        self.bh = self.zh(acc / len(G3_HOLD))
        self.ndh = lambda s: ndcg_per_feed(s, self.yh, self.oh, self.Mh)
        bhv = self.ndh(self.bh).mean()
        assert abs(bhv - 0.32379) < 2e-5, bhv
        self.base_h = self.ndh(self.bh)
        self.fidh = np.repeat(np.arange(len(self.Mh)), self.Mh)
        self.pw = pmix.weights()["private"]
        log(f"sanity: base val {bv:.5f} (14,423 feeds, {len(self.y)} rows) | base holdout {bhv:.5f} (30,000 feeds, {len(self.yh)} rows)")
        self._ctrl = {}

    def zv(self, x):
        return feed_z_nanmin(np.asarray(x, np.float64), self.off, self.M)

    def zh(self, x):
        return feed_z_nanmin(np.asarray(x, np.float64), self.oh, self.Mh)

    def nll(self, theta, G, rows):
        s = G @ theta; ff = self.fid[rows]; nF = self.nF
        e = np.exp(np.clip(s, -50, 20.0)); Z = np.bincount(ff, weights=e, minlength=nF)
        ok = Z > 0; lZ = np.log(np.where(ok, Z, 1.0)); yy = self.y[rows]
        npos = np.bincount(ff, weights=yy, minlength=nF)
        L = -(yy * s).sum() + (npos * lZ).sum()
        p = e / np.where(ok, Z, 1.0)[ff]; w = npos[ff] * p - yy
        return L, G.T @ w

    def fit_lin(self, G, tr):
        th0 = np.zeros(G.shape[1]); th0[0] = 0.94
        rows = np.flatnonzero(tr)
        Gt = np.ascontiguousarray(G[tr])
        r = minimize(lambda t: self.nll(t, Gt, rows), th0, jac=True, method="L-BFGS-B", options=dict(maxiter=200))
        return r.x

    def cv_lin(self, G):
        pred = np.empty(len(self.y))
        for k in range(5):
            tr = self.rowfold != k; pred[~tr] = G[~tr] @ self.fit_lin(G, tr)
        return pred

    def cols(self, x, split, isn_flag=None, z_only=False):
        x = np.asarray(x, np.float64)
        if split == "v":
            z = self.zv(x); rk = pctrank(x, self.off, self.M)
        else:
            z = self.zh(x); rk = pctrank(x, self.oh, self.Mh)
        isn = np.isnan(x).astype(np.float64)
        if isn_flag is None:
            isn_flag = bool(isn.any())
        out = [z] if z_only else [z, rk]
        if isn_flag:
            out.append(isn)
        return out, isn_flag

    def design(self, xs_v, xs_h, z_only=False, base_v=None, base_h=None):
        Gv = [self.b if base_v is None else base_v]
        Gh = [self.bh if base_h is None else base_h]
        names = ["b"]
        for j, xv in enumerate(xs_v):
            cv, fl = self.cols(xv, "v", z_only=z_only)
            Gv += cv
            if xs_h is not None:
                ch, _ = self.cols(xs_h[j], "h", isn_flag=fl, z_only=z_only)
                Gh += ch
        return np.stack(Gv, 1), (np.stack(Gh, 1) if xs_h is not None else None)

    def stats(self, d):
        se = d.std() / np.sqrt(len(d))
        return float(d.mean()), float(se), float(d.mean() / se) if se > 0 else 0.0

    def evaluate(self, name, Gv, Gh, Cv=None, Ch=None, primary=True):
        """LIN on Gv vs control LIN on Cv (default [b]). Holdout: theta_full applied to Gh vs control (Ch theta_full or b_h)."""
        t = time.time()
        if Cv is None:
            key = "b"
            if key not in self._ctrl:
                self._ctrl[key] = self.nd(self.cv_lin(self.b[:, None]))
            c_nd = self._ctrl[key]
        else:
            c_nd = self.nd(self.cv_lin(Cv))
        p_nd = self.nd(self.cv_lin(Gv))
        d = p_nd - c_nd
        m, se, z = self.stats(d)
        pval = float(1 - norm.cdf(z))
        pm, pse = pmix.wmean_se(d, self.pw)
        res = dict(name=name, primary=primary, ncols=int(Gv.shape[1]), val_delta=m, val_se=se, z=z, p=pval,
                   pmix=float(pm), pmix_se=float(pse), lin_val=float(p_nd.mean()), ctrl_val=float(c_nd.mean()))
        sl = {}
        for sn, msk in self.SL.items():
            mm, ss, _ = self.stats(d[msk]); sl[sn] = [mm, ss]
        res["slices"] = sl
        if Gh is not None:
            th = self.fit_lin(Gv, np.ones(len(self.y), bool))
            res["theta_full_head"] = [float(v) for v in th[:min(len(th), 12)]]
            hp = self.ndh(Gh @ th)
            if Ch is None:
                hc = self.base_h
            else:
                thc = self.fit_lin(Cv, np.ones(len(self.y), bool))
                hc = self.ndh(Ch @ thc)
            dh = hp - hc
            hm_, hse, hz = self.stats(dh)
            lm, lse, lz = self.stats(dh[self.H_LATE]); em, ese, ez = self.stats(dh[self.H_EARLY])
            res.update(hold_delta=hm_, hold_se=hse, hold_z=hz, late_delta=lm, late_se=lse, early_delta=em, early_se=ese,
                       hold_lin=float(hp.mean()), hold_ctrl=float(hc.mean()))
        line = (f"RESULT {name} [{'PRIMARY' if primary else 'INFO'}] cols {Gv.shape[1]}: nested val {m:+.5f} ({se:.5f}) z {z:+.2f} "
                f"p {pval:.4f} | lin {p_nd.mean():.5f} ctrl {c_nd.mean():.5f} | pmix {pm:+.5f} ({pse:.5f}) z {pm/pse:+.2f}")
        if Gh is not None:
            line += (f" | holdout {hm_:+.5f} ({hse:.5f}) z {hz:+.2f} late {lm:+.5f} ({lse:.5f}) early {em:+.5f} ({ese:.5f})"
                     f" [hold lin {hp.mean():.5f} ctrl {hc.mean():.5f}]")
        line += " | slices " + " ".join(f"{k} {v[0]:+.5f}" for k, v in sl.items())
        line += f" | noise band {NOISE_BAND} | {time.time() - t:.0f}s"
        log(line)
        return res, d

    def run(self, name, xs_v, xs_h, z_only=False, primary=True, out_jsonl=None):
        Gv, Gh = self.design(xs_v, xs_h, z_only=z_only)
        res, d = self.evaluate(name, Gv, Gh, primary=primary)
        if out_jsonl:
            with open(out_jsonl, "a") as fh:
                fh.write(json.dumps(res) + "\n")
        return res, d


def coverage(name, x):
    x = np.asarray(x, np.float64)
    fin = np.isfinite(x)
    nz = fin & (x != 0)
    log(f"coverage {name}: finite {fin.mean():.4f} nonzero {nz.mean():.4f} mean {np.nanmean(x):.4g} sd {np.nanstd(x):.4g}")
