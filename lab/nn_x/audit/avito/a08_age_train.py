"""a08: positive rate by listing age in random-like vs production-like train feeds (M>=50), and by production pos bucket.
Prior check for a training-only freshness offset on production rows."""
from common_av import *

tr = meta("train")
hold, p = holdout_index()
rl = np.where((p > 0.9) & (tr["M"] >= 50))[0]
pr = np.where((p < 0.05) & (tr["M"] >= 50))[0]
rng = np.random.default_rng(5)
rl = np.sort(rng.choice(rl, 60000, replace=False))
pr = np.sort(rng.choice(pr, 60000, replace=False))
knots = np.load(FEATS / "life_f4_age_knots.npy")
K = 64
CAP = 63 / 64 - 1e-6


def age_hours(x):
    x = x.astype(np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    return np.where(x < CAP, knots[k] * (1 - w) + knots[k + 1] * w, np.nan)


edges = [0, 3, 12, 24, 72, 336, 2160, 8760, 1e9]
labels = ["<3h", "3-12h", "12-24h", "1-3d", "3-14d", "2w-3m", "3m-1y", ">1y"]


def rows_of(feeds):
    return np.concatenate([np.arange(o, o + m) for o, m in zip(tr["off"][feeds], tr["M"][feeds])])


out = {}
for name, feeds in [("random-like", rl), ("production-like", pr)]:
    rows = rows_of(feeds)
    y = np.asarray(cand("train", "target")[rows]).astype(np.float64)
    a = age_hours(np.asarray(cand("train", "f4")[rows]))
    pos = np.asarray(cand("train", "pos")[rows]).astype(np.int64)
    b = np.digitize(np.nan_to_num(a, nan=1e8), edges) - 1
    base = y.mean()
    share = np.bincount(b, minlength=8) / len(b)
    lift = np.bincount(b, weights=y, minlength=8) / np.maximum(np.bincount(b, minlength=8), 1) / base
    out[name] = lift
    log(f"{name:16s} base {base:.4f}; share by age {np.round(share, 3)}; lift by age {np.round(lift, 2)}")
    if name == "production-like":
        for lo, hi in [(0, 5), (5, 20), (20, 60), (60, 400)]:
            m = (pos >= lo) & (pos < hi)
            l2 = np.bincount(b[m], weights=y[m], minlength=8) / np.maximum(np.bincount(b[m], minlength=8), 1) / y[m].mean()
            log(f"   pos [{lo},{hi}): rows {m.sum()} lift by age {np.round(l2, 2)}; age share {np.round(np.bincount(b[m], minlength=8) / m.sum(), 3)}")
log(f"log(production lift / random-like lift) by age {labels}: {np.round(np.log(out['production-like'] / out['random-like']), 3)}")
# val reference
v = meta("val")
yv = np.asarray(cand("val", "target"), dtype=np.float64)
bv = np.digitize(np.nan_to_num(age_hours(np.asarray(cand("val", "f4"))), nan=1e8), edges) - 1
log(f"val lift by age: {np.round(np.bincount(bv, weights=yv, minlength=8) / np.maximum(np.bincount(bv, minlength=8), 1) / yv.mean(), 2)}")
log("done")
