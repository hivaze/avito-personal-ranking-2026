"""Cold-item calibration of v11 by day: actual / expected positives where expected = npos * softmax(c * z(v11)) per feed
(c fitted on all val by listwise CE). Feed-clustered bootstrap SE. Also for FT5 and TO5 separately."""
from common_e import meta, ROOT, SCR, fz
import numpy as np
from scipy.optimize import minimize_scalar

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
cold = np.load(f"{ROOT}/eda/errors/synthesis/derived/val/cold_item.npy").astype(bool)
luo0 = np.load(f"{ROOT}/data/feats/val/it_hist_all_luo.npy") == 0
ts = f["timestamp"].to_numpy()
day = (ts + 3 * 3600) // 86400
day = np.where(day <= day.min() + 1, 0, 1)
npos = np.bincount(fid, y, n)
f4 = np.load(f"{ROOT}/data/feats/val/f4.npy")
young = f4 < 20 / 64
rng = np.random.default_rng(0)
B = 300
boots = [rng.integers(0, n, n) for _ in range(B)]


def probs(z, c):
    e = np.exp(c * z - np.repeat(np.maximum.reduceat(c * z, off), M))
    return e / np.repeat(np.add.reduceat(e, off), M) * npos[fid]


for nm, raw in (("v11", np.load(f"{SCR}/out/v11_val.npy")), ("FT5", np.load(f"{ROOT}/preds/nn_final5_val_oof.npy")),
                ("TO5", np.load(f"{ROOT}/preds/nn_final5_trainonly_val.npy"))):
    z = fz(raw, fid, M)

    def ce(c):
        p = probs(z, c) / npos[fid]
        return -(y * np.log(p + 1e-12)).sum() / n
    c = minimize_scalar(ce, bounds=(0.3, 3), method="bounded").x
    p = probs(z, c)
    out = [f"{nm} (c={c:.2f})"]
    for lab, sel in (("cold", cold), ("luo0", luo0), ("young<6.8d", young), ("young&luo0", young & luo0)):
        for dl, dm in (("Tue", day == 0), ("Thu", day == 1)):
            m = sel & dm[fid]
            a = np.bincount(fid, y * m, n)
            e = np.bincount(fid, p * m, n)
            fm = dm
            r = a[fm].sum() / e[fm].sum()
            idx = np.where(fm)[0]
            bs = [a[idx[b % len(idx)]].sum() / e[idx[b % len(idx)]].sum() for b in boots[:B]]
            out.append(f"{lab}-{dl} {r:.3f}+-{np.std(bs):.3f}")
    print("  ".join(out))
