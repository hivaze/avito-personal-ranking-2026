"""(a) earlier labelled feeds of the same user: LightGBM lambdarank residual on W8, user-split 2-fold, val + holdout.
Arms: ctx (null control: base + feed context only), a_all (ctx + bank signals, all feeds), a_cov (fit + scored on covered feeds only).
Usage: s02_resid_a.py val|hold_g0|hold_g2 [seed]"""
from hx import *
import json
tag = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
D = load_val() if tag == "val" else load_hold()
Z = np.load(OUT / f"bank_{tag}.npz"); Xb = Z["X"]; nb_ = [str(s) for s in Z["names"]]; fh = Z["feed_has"]; fp = Z["feed_pair"]
ctx = ctx_cols(D); Xc = np.stack([ctx[k] for k in ctx], 1).astype(np.float32); nc = list(ctx)
print(f"== {tag} seed {seed}: base NDCG {ndcg_per_feed(D['base'], D['y'], D['off'], D['M']).mean():.5f}; covered feeds {fh.mean():.4f}; re-shown-pair feeds {fp.mean():.4f}", flush=True)
res = {}
# linear nested replication (the report's protocol) for the row-varying z signals
for nm in ("ui_shown_fz", "ui_skip_fz", "uc_ctr_log_fz", "uc_clickshare_fz", "uc_expo_log_fz", "uc_skiprate_fz", "uc_y_fz", "uc_n_fz"):
    d, wc = nested_lin(D, Xb[:, nb_.index(nm)].astype(np.float64), seed=seed); print(f"  linear nested {nm:18s} {fmt(d)}  w={wc}  covered-only mean {d[fh].mean():+.5f}", flush=True)
    res["lin_" + nm] = (float(d.mean()), float(se(d)))
d0, det0, _ = resid_cv(D, Xc, nc, "ctx-only (null control)", seed=seed); res["ctx"] = (float(np.nanmean(d0)), float(se(d0)), det0)
d1, det1, s1 = resid_cv(D, np.hstack([Xc, Xb]), nc + nb_, "a_all = ctx + bank", seed=seed); res["a_all"] = (float(np.nanmean(d1)), float(se(d1)), det1)
print(f"  a_all minus ctx-only: {fmt(d1 - d0)}; on covered feeds {fmt((d1 - d0)[fh])}; on uncovered {fmt((d1 - d0)[~fh])}", flush=True)
d2, det2, s2 = resid_cv(D, np.hstack([Xc, Xb]), nc + nb_, "a_cov = ctx + bank, covered feeds only", seed=seed, feed_mask=fh)
dd = np.where(fh, d2, 0.0); print(f"  a_cov as an all-feed delta (0 on uncovered): {fmt(dd)} (z {dd.mean() / se(dd):+.2f}); re-shown-pair feeds {fmt(d2[fp])}", flush=True)
res["a_cov"] = (float(dd.mean()), float(se(dd)), det2)
d3, det3, _ = resid_cv(D, Xc, nc, "ctx-only on covered feeds (control for a_cov)", seed=seed, feed_mask=fh)
dc = np.where(fh, d2 - d3, 0.0); print(f"  a_cov minus its control: all-feed {fmt(dc)}", flush=True)
res["ctx_cov"] = (float(np.where(fh, d3, 0).mean()), float(se(np.where(fh, d3, 0))), det3)
np.savez(OUT / f"s02_{tag}_s{seed}.npz", d_ctx=d0, d_all=d1, d_cov=d2, d_ctxcov=d3)
json.dump(res, open(OUT / f"s02_{tag}_s{seed}.json", "w"), indent=1, default=str)
