"""Critic check 5 (CPU): 5-fold user cross-fit over 4 hashes for the stream's novelty and familiarity offsets
(items 2, 3, 5), same feature definitions as systematic/s3_correct.py and s4_novelty.py (global-T base)."""
import sys
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/systematic")
from core import layout, load_table, _feed_lse, clogit_fit  # noqa: E402
sys.path.insert(0, "<repo>/src")
import pmix  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

P = "<repo>/preds/"
F, M, off = layout()
fid = np.repeat(np.arange(len(M)), M)
D = load_table()
y = D["label"].to_numpy().astype(np.float64)
W = pmix.weights()
s = 0.5 * np.load(P + "ens_FD16P20_m04_02kd_ft_val.npy").astype(np.float64) + 0.5 * np.load(P + "ens_FD16P20_m04_02kd_to_val.npy").astype(np.float64)
nd0 = ndcg_per_feed(s, y, off, M)


def col(c):
    return D[c].to_numpy().astype(np.float64)


def zc(x):
    x = np.asarray(x, np.float64).copy(); ok = ~np.isnan(x)
    x[ok] = (x[ok] - x[ok].mean()) / x[ok].std(); x[~ok] = 0.0
    return x


Mr = M[fid].astype(float)
cni = col("cat_newest_idx"); nov8 = ((cni < 0) | (cni >= 8)).astype(float)
feat = {"nov8": nov8, "nov8_focus": nov8 * (col("h_brk10") <= 1), "nov8_hent": nov8 * zc(col("h_ent")),
        "hmean5_z": zc(col("hmean5")), "h1_z": zc(col("h1")), "f1_z": zc(col("f1")), "f3_z": zc(col("f3")),
        "cat_hist_share_z": zc(col("cat_hist_share")), "cat_share_feed_z": zc(col("cat_n_in_feed") / Mr),
        "seqcv_z": zc(np.log1p(1e4 * col("seq_cv_invd20"))), "ucos_z": zc(col("ucos")), "hmax_z": zc(col("hmax")),
        "cat50": (col("cat") == 50).astype(float), "nodup": (col("n_cand_dup95") == 0).astype(float)}
specs = {"nov8": ["nov8"], "nov8 x focused": ["nov8_focus"], "nov8 + nov8 x h_ent": ["nov8", "nov8_hent"],
         "familiarity set (12)": ["hmean5_z", "h1_z", "f1_z", "f3_z", "cat_hist_share_z", "cat_share_feed_z", "seqcv_z",
                                  "ucos_z", "hmax_z", "nov8", "cat50", "nodup"]}


def folds(seed, K=5):
    u = F["user_id"].to_numpy().astype(np.uint64)
    key = np.where(u == 0, np.arange(len(u), dtype=np.uint64) + np.uint64(10 ** 12), u) + np.uint64(seed * 7919)
    with np.errstate(over="ignore"):
        z = key + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(K)).astype(np.int64)


def rep(d):
    return " | ".join(f"{k} {v[0]:+.5f}±{v[1]:.5f}" for k, v in [
        ("val", pmix.wmean_se(d)), ("pub", pmix.wmean_se(d, W["public"])), ("priv", pmix.wmean_se(d, W["private"])),
        ("M<50", pmix.wmean_se(d[M < 50])), ("M>=50", pmix.wmean_se(d[M >= 50]))])


def nll(x, fm):
    lse = _feed_lse(x, off, M)
    return float((-(x - lse[fid]) * y)[fm[fid]].sum())


FS = {sd: folds(sd) for sd in (1, 2, 3, 4)}
TS = {}
for sd, fo in FS.items():
    for k in range(5):
        tr = fo != k
        TS[(sd, k)] = float(np.exp(minimize_scalar(lambda lt: nll(s / np.exp(lt), tr), bounds=(-1, 1), method="bounded").x))
for name, fs in specs.items():
    X = np.column_stack([feat[f] for f in fs])
    allD = []
    for sd, fo in FS.items():
        new = s.copy()
        for k in range(5):
            tr = fo != k; r = (fo == k)[fid]; T = TS[(sd, k)]
            b = clogit_fit(s / T, X, y, off, M, tr, l2=1e-4)
            new[r] = s[r] / T + X[r] @ b
        allD.append(ndcg_per_feed(new, y, off, M) - nd0)
    per = [d.mean() for d in allD]
    print(f"{name:28s} 4-hash mean: {rep(np.mean(allD, 0))}   per-hash val {np.round(per, 5).tolist()}", flush=True)
