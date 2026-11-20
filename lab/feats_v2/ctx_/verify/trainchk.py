import sys, json
sys.path.insert(0, "<repo>/src")
from common import *
K = "ctx_hn_soft05,ctx_hn_rrw,ctx_hn_e1_rrw,ctx_hn_top1w,ctx_hn_margin,ctx_hn_bestrk,ctx_hs_t10,ctx_hs_d7,ctx_hs_k128,ctx_hs_e1,ctx_hs_k256,ctx_hs_e1k64,ctx_hs_k128t10,ctx_hs_mutual,ctx_nd_knn_u_dev,ctx_nd_ucent,ctx_nd_knn_h_dev,ctx_nd_knn_h,ctx_nd_max,ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_u,ctx_nd_top5u,ctx_nd_top5h".split(",")
f, M, off = feed_meta("train")
N = int(M.sum()); n = len(M)
day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
ud = np.unique(day)
# sample: 60K feeds from first day, 60K feeds from last day
rng = np.random.default_rng(0)
sel = {}
for nm, d in [("first", ud[0]), ("last", ud[-2])]:
    fe = np.where(day == d)[0]; sel[nm] = np.sort(rng.choice(fe, 60000, replace=False))
y = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
pos = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r")
def rows_of(fs):
    return np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fs])
R = {k: rows_of(v) for k, v in sel.items()}
yv = np.load(FLAT / "val/cand_target.npy"); fv, Mv, offv = feed_meta("val")
rs = np.sort(rng.choice(N, 5_000_000, replace=False))
lens = {s: len(np.load(FLAT / s / "cand_fidx.npy", mmap_mode="r")) for s in ["train", "val", "test"]}
print("cand rows", lens, flush=True)
def nd(x, yy, o, m):
    noise = np.random.default_rng(1).random(len(x)) * 1e-7
    xx = np.where(np.isnan(x), -1e9, x)
    up = ndcg(xx * (1 + 0) + noise * np.nanstd(x), yy, o, m); dn = ndcg(np.where(np.isnan(x), -1e9, -x) + noise * np.nanstd(x), yy, o, m)
    return up, dn
out = []
for k in K:
    shp = {s: np.load(FEATS / s / f"{k}.npy", mmap_mode="r").shape for s in ["train", "val", "test"]}
    dt = np.load(FEATS / "train" / f"{k}.npy", mmap_mode="r").dtype
    ok = all(shp[s] == (lens[s],) for s in shp)
    xt = np.asarray(np.load(FEATS / "train" / f"{k}.npy", mmap_mode="r")[rs])
    xv = np.load(FEATS / "val" / f"{k}.npy")
    r = dict(f=k, shape_ok=ok, dtype=str(dt), nan_tr=float(np.isnan(xt).mean()), nan_val=float(np.isnan(xv).mean()),
             nuniq_tr_sample=int(len(np.unique(xt[~np.isnan(xt)][:1_000_000]))),
             q_tr=[round(float(z), 4) for z in np.nanquantile(xt, [.1, .5, .9])], q_val=[round(float(z), 4) for z in np.nanquantile(xv, [.1, .5, .9])])
    xall = np.load(FEATS / "train" / f"{k}.npy", mmap_mode="r")
    for nm, rr in R.items():
        x = np.asarray(xall[rr]); fs = sel[nm]; mm = M[fs]; oo = np.concatenate([[0], np.cumsum(mm)[:-1]])
        yy = np.asarray(y[rr]).astype(np.float64)
        r[f"nd_tr_{nm}"] = [round(z, 4) for z in nd(x, yy, oo, mm)]
        # position-debiased-ish view: restrict to rows with pos<10 ? keep simple: corr with pos
        p = np.asarray(pos[rr]).astype(np.float64); g = ~np.isnan(x)
        r[f"spearman_pos_{nm}"] = round(float(np.corrcoef(np.argsort(np.argsort(x[g])), np.argsort(np.argsort(p[g])))[0, 1]), 3)
    r["nd_val"] = [round(z, 4) for z in nd(xv, yv, offv, Mv)]
    out.append(r)
    print(json.dumps(r), flush=True)
json.dump(out, open("/tmp/scratchpad/trainchk.json", "w"))
