"""Label-free distribution check of the S3 block columns (lead's notice of 02:19Z, point 2): train rows (xm) against holdout, val and test rows.
For every column: mean / sd on xm, standardised mean difference (SMD = (mean_set - mean_xm) / sd_xm), sd ratio and a two-sample KS statistic on
200,000-row samples. Prints the summary over columns and the worst columns; writes reads/dist_<variant>.json.
usage: CUDA_VISIBLE_DEVICES="" CBX_THREADS=4 python -u fs_dist.py [pca|cand|raw]"""
import sys, json
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/famS")
import fs_common as fs
from fs_common import cbx, log

variant = sys.argv[1] if len(sys.argv) > 1 else "pca"
names = fs.block_names(variant)
rg = np.random.default_rng(11)
NS = 200_000


def stats(ds):
    B = fs.build_block(ds, variant)
    idx = np.sort(rg.choice(ds.n_rows, min(NS, ds.n_rows), replace=False))
    S = np.asarray(B[idx], np.float32)
    mu = np.array([np.asarray(B[:, j], np.float64).mean() for j in range(B.shape[1])])
    sd = np.array([np.asarray(B[:, j], np.float64).std() for j in range(B.shape[1])])
    return mu, sd, S


def ks(a, b):
    a = np.sort(a); b = np.sort(b); g = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, g, side="right") / len(a) - np.searchsorted(b, g, side="right") / len(b)).max())


mu0, sd0, S0 = stats(cbx.load("xm"))
out = {"variant": variant, "columns": names, "xm": {"mean": mu0.tolist(), "sd": sd0.tolist()}}
for nm in ("hold", "val", "test"):
    mu, sd, S = stats(cbx.load(nm))
    smd = (mu - mu0) / np.maximum(sd0, 1e-9); sr = sd / np.maximum(sd0, 1e-9)
    k = np.array([ks(S0[:, j], S[:, j]) for j in range(S.shape[1])])
    out[nm] = {"smd": smd.tolist(), "sd_ratio": sr.tolist(), "ks": k.tolist()}
    w = np.argsort(-np.abs(smd))[:5]; wk = np.argsort(-k)[:5]
    log(f"[{variant}] {nm:4s} vs xm: |SMD| median {np.median(np.abs(smd)):.4f} p90 {np.quantile(np.abs(smd), 0.9):.4f} max {np.abs(smd).max():.4f} | sd ratio min {sr.min():.3f} max {sr.max():.3f} | "
        f"KS median {np.median(k):.4f} p90 {np.quantile(k, 0.9):.4f} max {k.max():.4f}")
    log(f"      worst SMD: " + ", ".join(f"{names[j]} {smd[j]:+.3f}" for j in w) + " | worst KS: " + ", ".join(f"{names[j]} {k[j]:.3f}" for j in wk))
    for grp, sl in (("cand part", slice(0, len(names) // 4 if variant != "cand" else len(names))), ("product part", slice(len(names) // 4, len(names)))):
        if variant == "cand" and grp == "product part":
            continue
        log(f"      {grp}: |SMD| max {np.abs(smd[sl]).max():.4f}, KS max {k[sl].max():.4f}")
(fs.HERE / "reads" / f"dist_{variant}.json").write_text(json.dumps(out))
log("DIST_DONE")
