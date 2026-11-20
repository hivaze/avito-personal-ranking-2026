"""hzcv mechanism diagnostic (not a gate): does training on lagged co-vis rows fix the val calibration of censored
zeros (RESEARCH-2: train-only NN actual/expected 1.055 on cv=0 & censored rows)?
Re-trains the hz_gate.py arms R0 and R1 (seed 0, same data/params) and reports, on val, actual/expected positives by
group, expected = n_pos(feed) * softmax(s / T) with one global T per model fitted by listwise likelihood on val
(a single calibration scalar; it does not change any ranking).
Groups (as featrev RESEARCH-2): cv=0 & censored, cv=0 & exposed 0-72 h, cv=0 & exposed > 72 h, cv>0.
"""
import sys, time
from pathlib import Path
import numpy as np, lightgbm as lgb
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hz_gate as G  # noqa: E402
import hz_common as hz  # noqa: E402


def main():
    trs = np.load(G.SCR / "train_sample_feeds.npy")
    tr = G.block("train", trs)
    va = G.block("val", np.arange(len(G.feeds("val"))))
    Xv, yv, offv, Mv, rows_v, fr_v = va
    nF = len(Mv)
    fid = np.repeat(np.arange(nF), Mv)
    npos = np.bincount(fid, weights=yv, minlength=nF)
    E = np.asarray(np.load(hz.FEATS_NEW / "val/src_expo_h.npy", mmap_mode="r")[rows_v])
    cv0 = Xv["seq_cv_pop"] == 0
    cens = Xv["hzcv_censored"] > 0
    groups = {"cv0&censored": cv0 & cens, "cv0&exp0-72h": cv0 & ~cens & (E <= 72), "cv0&exp>72h": cv0 & (E > 72),
              "cv>0": ~cv0}
    R0 = G.BASE + G.LUO + ["it_hist_all_luo_rk", "it_hist_all_luo_zz"]
    BL = [G.LAG.get(n, n) for n in G.BASE]
    arms = {"R0": R0, "R1": BL + G.LUO + ["it_hist_all_luo_rk", "it_hist_all_luo_zz"] + G.OFFS}
    for arm, names in arms.items():
        ds = lgb.Dataset(np.stack([tr[0][n] for n in names], 1), tr[1], group=tr[3], feature_name=names,
                         categorical_feature=["category_id"])
        b = lgb.train(dict(G.PARAMS, seed=0), ds, 350)
        s = b.predict(np.stack([Xv[n] for n in names], 1), num_threads=G.TH).astype(np.float64)

        def probs(T):
            z = s / T
            mx = np.maximum.reduceat(z, offv)
            e = np.exp(z - mx[fid])
            Z = np.add.reduceat(e, offv)
            return e / Z[fid]

        def nll(T):
            p = probs(T)
            return -(yv * np.log(p + 1e-300)).sum()

        T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
        exp = npos[fid] * probs(T)
        line = f"{arm}: T {T:.3f} |"
        for g, m in groups.items():
            a, e_ = yv[m].sum(), exp[m].sum()
            line += f" {g} share {m.mean():.3f} A/E {a / e_:.3f} (+-{1 / np.sqrt(max(a, 1)):.3f}) |"
        print(line, flush=True)


if __name__ == "__main__":
    main()
