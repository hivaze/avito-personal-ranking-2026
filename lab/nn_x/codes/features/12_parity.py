"""Split parity of f1..f5: PSI (20 val-quantile bins, floors as their own bin) of train random-like / production / uniform,
test public (first 30% of test feeds by time) / private, against val; plus the within-feed z (feed-standardised) PSI."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *

S = {"val": load("val"), "test": load("test"),
     "train_uniform": load("train", train_sample("uniform", 100_000, 6)),
     "train_rl": load("train", train_sample("rl", 60_000, 6)), "train_prod": load("train", train_sample("prod", 60_000, 6))}
ft = feeds("test"); tsf = ft["timestamp"].to_numpy(); cut = np.quantile(tsf, 0.3)
S["test_public"] = {k: (v[S["test"]["timestamp"] <= cut] if isinstance(v, np.ndarray) and len(v) == len(S["test"]["f1"]) else v) for k, v in S["test"].items()}
S["test_private"] = {k: (v[S["test"]["timestamp"] > cut] if isinstance(v, np.ndarray) and len(v) == len(S["test"]["f1"]) else v) for k, v in S["test"].items()}

def fz(d, k):
    feed = d["feed"]; x = d[k].astype(np.float64); nf = feed.max() + 1
    M = np.bincount(feed, minlength=nf)
    m = np.bincount(feed, x, minlength=nf) / np.maximum(M, 1)
    s = np.sqrt(np.maximum(np.bincount(feed, x * x, minlength=nf) / np.maximum(M, 1) - m ** 2, 0))
    return (x - m[feed]) / np.where(s[feed] > 0, s[feed], 1)

def psi(a, b, edges):
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6; pb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))

names = ["train_uniform", "train_rl", "train_prod", "test", "test_public", "test_private"]
print("PSI vs val (raw value) | (within-feed z)")
print("f".ljust(4) + "".join(f"{n:>24}" for n in names))
for k in FS:
    ref = S["val"][k].astype(np.float64)
    e = np.unique(np.quantile(ref, np.linspace(0, 1, 21))); e[0] = -1e9; e[-1] = 1e9
    e = np.unique(np.r_[e, [1e-7, 0.0001]])
    zr = fz(S["val"], k); ez = np.quantile(zr, np.linspace(0, 1, 21)); ez[0] = -1e9; ez[-1] = 1e9; ez = np.unique(ez)
    cells = []
    for n in names:
        cells.append(f"{psi(S[n][k], ref, e):.4f} | {psi(fz(S[n], k) if n not in ('test_public','test_private') else fz_sub(n, k), zr, ez):.4f}".rjust(24) if False else f"{psi(S[n][k], ref, e):.4f}".rjust(24))
    print(k.ljust(4) + "".join(cells))
print("\nwithin-feed z PSI vs val (train/test full splits only)")
for k in FS:
    zr = fz(S["val"], k); ez = np.unique(np.quantile(zr, np.linspace(0, 1, 21))); ez[0] = -1e9; ez[-1] = 1e9
    print(k.ljust(4) + "".join(f"{n}:{psi(fz(S[n], k), zr, ez):.4f}  " for n in ["train_uniform", "train_rl", "train_prod", "test"]))
print("\nmeans")
for n in ["val"] + names:
    print(n.ljust(14) + " ".join(f"{k} {S[n][k].mean():.4f}" for k in FS))
