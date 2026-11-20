"""EMB-USABILITY stage 3 (descriptive, label-free except the drift-vs-click rate which is a plain rate): what drives the
val-vs-train vector change?  Bit-identical / quantisation-level drift / real change, vs item activity and category edits.
Also the fine in-sample w curve for proj_mu (descriptive only; the gate number is the nested one in eval.log).
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
import sys, json, time
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
sys.path.insert(0, "<repo>/nn_x/emb/usability")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from qt import leak_mask
from s1_eval_lib import feed_z_nan

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; COMP = ROOT / "data/competition"; OUT = ROOT / "nn_x/emb/usability/out"
LOG = open(OUT / "desc.log", "a"); T0 = time.time()
def log(*a):
    s = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def table(split):
    meta = json.load(open(COMP / split / "item_embeddings/data.meta"))
    return np.memmap(COMP / split / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
TR, VA = table("train"), table("val")
S = np.load(OUT / "sig_val.npz")
c_iid = np.load(FLAT / "val/cand_item_id.npy"); c_emb = np.load(FLAT / "val/cand_emb_idx.npy")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
pop = np.log1p(np.maximum(np.load(ROOT / "data/feats/val/it_hist_all_luo.npy").astype(np.float64), 0))
poptr = np.log1p(np.maximum(np.load(ROOT / "data/feats/val/it_cand_cnt_train.npy").astype(np.float64), 0))
# distinct items (first occurrence) among val candidates with a train vector
has = S["in_train"] == 1
_, first = np.unique(c_iid, return_index=True)
sel = np.zeros(len(c_iid), bool); sel[first] = True; sel &= has
log(f"distinct val candidate items with a train-table vector: {sel.sum():,}")
# rebuild the train rows for these items
mapTR = None
ids_tr = []; em_tr = []
for pre in ["cand", "hist"]:
    ids = np.load(FLAT / f"train/{pre}_item_id.npy"); em = np.load(FLAT / f"train/{pre}_emb_idx.npy")
    ok = em >= 0
    if mapTR is None: mapTR = np.full(111_400_000, -1, np.int32)
    mapTR[ids[ok]] = em[ok]
r_tr = mapTR[c_iid[sel]]; r_va = c_emb[sel]
o = np.argsort(r_tr); A = np.asarray(TR[r_tr[o]]).astype(np.int16); inv = np.empty_like(o); inv[o] = np.arange(len(o)); A = A[inv]
B = np.asarray(VA[r_va]).astype(np.int16)
D = np.abs(A - B); ndiff = (D > 0).sum(1); maxd = D.max(1)
cs = S["cos_tv"][sel]
cls = np.where(ndiff == 0, 0, np.where(maxd <= 1, 1, np.where(cs >= 0.99, 2, 3)))
names = ["bit-identical", "drift |d|<=1 (quantisation level)", "drift |d|>1, cos>=0.99", "changed cos<0.99"]
for c in range(4):
    m = cls == c
    log(f"  {names[c]:36s} share {m.mean():.4f}  n_dims_diff mean {ndiff[m].mean():6.1f}  max|d| median {np.median(maxd[m]) if m.any() else 0:5.1f}  "
        f"pop(LUO) {pop[sel][m].mean():.3f}  train-cand-count {poptr[sel][m].mean():.3f}  row click rate {y[sel][m].mean():.5f}")
# drift magnitude vs train popularity decile
dec = np.digitize(poptr[sel], np.unique(np.quantile(poptr[sel], np.linspace(0, 1, 11)[1:-1])))
log("  share non-identical / changed(cos<.99) by train-candidate-count bin (low->high):")
for b in np.unique(dec):
    m = dec == b
    log(f"    bin {b}: n {m.sum():7d}  non-identical {(cls[m] > 0).mean():.3f}  changed {(cls[m] == 3).mean():.4f}  mean 1-cos {np.mean(1 - cs[m]):.5f}")
# direction of the drift: is the val-minus-train difference aligned with the mean direction / PC1 ?
fit = np.load(OUT / "fit.npz"); mu = fit["mu"]; V = fit["V"]
dv = (B - A).astype(np.float64)[cls > 0]; dvn = dv / np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-9)
log(f"  drift direction: mean |cos(drift, mu_dir)| {np.abs(dvn @ (mu / np.linalg.norm(mu))).mean():.4f}; mean |cos(drift, PC1)| {np.abs(dvn @ V[:, 0]).mean():.4f}; "
    f"random-direction expectation ~{np.sqrt(2/np.pi/128):.4f}; norm of mean drift / mean norm {np.linalg.norm(dv.mean(0))/np.linalg.norm(dv,axis=1).mean():.4f}")

# fine in-sample w curve for proj_mu (descriptive)
f = pl.read_parquet(FLAT / "val/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
keep = ~leak_mask()
zG3 = feed_rank_z(off, M, np.load(ROOT / "preds/ens_G3_final_val.npy").astype(np.float32))[:, 1].astype(np.float64)
x = S["proj_mu"].astype(np.float64); zx = feed_z_nan(off, M, x)
b = ndcg_per_feed(zG3, y, off, M)[keep]
for w in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
    d = ndcg_per_feed(zG3 + w * zx, y, off, M)[keep] - b
    log(f"  proj_mu in-sample w {w:.3f}: {d.mean():+.5f} +- {d.std()/np.sqrt(len(d)):.5f}")
log("done")
