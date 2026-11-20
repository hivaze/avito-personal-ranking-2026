"""Verifier: is the transductive PMI table a leak channel? Rebuild T/T1 (K=1024) from (a) train histories only,
(b) val+test histories only; recompute clu_pmi1024_mean / clu_pmi11024_mean for val/test; compare with the stored files
(correlation, single-feature NDCG on val). Train-only T contains no information from val/test time at all."""
import sys
sys.path.insert(0, "<repo>/feats_v2/clu_"); sys.path.insert(0, "<repo>/src")
from clu_common import *
from pmi_table import trans_accum
from build import pair_query, HALF, L_CF
from metrics import ndcg
nb.set_num_threads(24)
K = 1024
parts = {}
for split in ["train", "val", "test"]:
    fe = feeds(split)
    u = fe["uid"]
    if split == "train":
        cnt = np.bincount(u); fw = np.where(u > 0, 1.0 / cnt[u], 1.0)
    else:
        fw = np.ones(len(u))
    t, t1 = trans_accum(fe["ho"], fe["H"], np.asarray(hist_cl(split, K)), np.load(FLAT / split / "hist_event_type.npy"), fw, K, 24)
    parts[split] = (t.reshape(K, K), t1.reshape(K, K)); log("T", split, t.sum(), t1.sum())
z = np.load(WORK / f"pmi_{K}.npz")
Tall = parts["train"][0] + parts["val"][0] + parts["test"][0]
print("stored T == recomputed sum:", np.allclose(z["T"], Tall), np.allclose(z["T1"], parts["train"][1] + parts["val"][1] + parts["test"][1]))
print("share of T mass from val+test:", (parts["val"][0].sum() + parts["test"][0].sum()) / Tall.sum())
def S_of(T):
    pb = T.sum(0) / T.sum(); pba = (T + 10.0 * pb[None, :]) / (T.sum(1, keepdims=True) + 10.0)
    return np.log(pba / np.maximum(pb[None, :], 1e-12)).astype(np.float64)
wtab = 0.5 ** (np.arange(256) / HALF)
fv, Mv, offv = None, None, None
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
import polars as pl
fvv = pl.read_parquet(FLAT / "val/feeds.parquet"); Mv = fvv["M"].to_numpy().astype(np.int64); offv = fvv["cand_off"].to_numpy()
rng = np.random.default_rng(0); noise = [rng.random(len(yv)) for _ in range(3)]
def single(x):
    v = np.where(np.isnan(x), -np.inf, x); fin = np.isfinite(v); lo, hi = v[fin].min(), v[fin].max()
    v = np.where(fin, (v - lo) / (hi - lo + 1e-12), -1.0)
    return float(np.mean([ndcg(v + 1e-7 * nz, yv, offv, Mv) for nz in noise]))
for variant, idx in (("train_only", ["train"]), ("valtest_only", ["val", "test"])):
    for ti, nm in ((0, "pmi"), (1, "pmi1")):
        T = sum(parts[s][ti] for s in idx)
        S = S_of(T)
        for split in ["val", "test"]:
            fe = feeds(split); N = int(fe["M"].sum())
            out = np.empty((N, 2), np.float32)
            pair_query(fe["co"], fe["M"], fe["ho"], fe["H"], np.asarray(cand_cl(split, K)), np.asarray(hist_cl(split, K)), S, L_CF, wtab, out)
            got = np.load(FEATS / split / f"clu_{nm}{K}_mean.npy")
            ok = ~np.isnan(got)
            r = np.corrcoef(out[ok, 0], got[ok])[0, 1]
            msg = f"{variant:12s} clu_{nm}{K}_mean {split}: corr with stored = {r:.5f}"
            if split == "val":
                msg += f"  single NDCG(+) variant={single(out[:, 0].astype(np.float64)):.4f} stored={single(got.astype(np.float64)):.4f}"
                np.save(WORK / f"verify_{nm}{K}_mean_{variant}_val.npy", out[:, 0])
            print(msg, flush=True)
