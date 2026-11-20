"""(2) anisotropy / common direction and (3) special dims, on the 1.5M train candidate-item sample
(+ the 500K uniform sample of the full train table for value statistics)."""
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
import numpy as np, torch
sys.path.insert(0, '<repo>/nn_x/emb/structure')
from knn_gpu import knn_ip
from scipy import stats

OUT = "<repo>/nn_x/emb/structure/"
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
z = np.load(OUT + "cache/train_sample.npz")
Q = z["E"]; X = Q.astype(np.float32) / 256.0; n = len(X)
cat = z["cat"].astype(np.int64); ucat = np.unique(cat); cmap = np.zeros(cat.max() + 1, np.int64); cmap[ucat] = np.arange(len(ucat)); C = cmap[cat]; K = len(ucat)
ncand = z["n_cand"].astype(np.float64); npos = z["n_pos"].astype(np.float64); hc = z["hcnt"].astype(np.float64)
attrs = {"f1": z["f1"], "f2": z["f2"], "f3": z["f3"], "f4": z["f4"], "f5": z["f5"], "log_n_cand": np.log1p(ncand),
         "log_n_hist": np.log1p(z["n_hist"].astype(np.float64)), "log_hist_et1": np.log1p(hc[:, 0]),
         "log_age_h": np.log1p((z["t_feed"] - z["t_post"]) / 3600.0), "ctr_smoothed": (npos + 0.2) / (ncand + 10)}
res = {}
U = np.load(OUT + "cache/train_uniform.npz")
QU = U["E"]

# ---------------- quantisation ----------------
nq2 = (QU.astype(np.int64) ** 2).sum(1)
res["quant"] = dict(mean_sq_norm_minus_65536=float(nq2.mean() - 65536), expected_round_nearest=128 / 12,
                    norm_mean=float(np.sqrt(nq2).mean()), norm_sd=float(np.sqrt(nq2).std()), expected_norm_sd_round=float(np.sqrt(65536 / 12) / 256),
                    share_norm_outside_255_257=float(((np.sqrt(nq2) < 255) | (np.sqrt(nq2) > 257)).mean()),
                    min_val=int(QU.min()), max_val=int(QU.max()), p_zero=float((QU == 0).mean()))
vals, cnts = np.unique(QU, return_counts=True)
h = dict(zip(vals.tolist(), cnts.tolist()))
ev = sum(c for v, c in h.items() if v % 2 == 0 and -40 <= v <= 40); od = sum(c for v, c in h.items() if v % 2 != 0 and -40 <= v <= 40)
res["quant"]["even_odd_ratio_|v|<=40"] = ev / od
# zero excess per dim: c(0) vs mean of c(-1), c(1)
zx = []
for d in range(128):
    col = QU[:, d]
    c0 = (col == 0).sum(); cm1 = (col == -1).sum(); cp1 = (col == 1).sum()
    zx.append(c0 / max((cm1 + cp1) / 2, 1))
zx = np.array(zx)
res["quant"]["zero_excess_ratio_per_dim_min_med_max"] = [float(zx.min()), float(np.median(zx)), float(zx.max())]
# row-level zero count vs independent-dims expectation
pz = (QU == 0).mean(0); rz = (QU == 0).sum(1)
sim = (np.random.default_rng(0).random((len(QU), 128)) < pz).sum(1)
res["quant"]["row_zero_count_obs_mean_sd_q999"] = [float(rz.mean()), float(rz.std()), float(np.quantile(rz, 0.999))]
res["quant"]["row_zero_count_indep_mean_sd_q999"] = [float(sim.mean()), float(sim.std()), float(np.quantile(sim, 0.999))]
# exact duplicates within the uniform sample (different rows, identical vectors)
_, inv, cc = np.unique(QU, axis=0, return_inverse=True, return_counts=True)
res["quant"]["uniform500k_rows_with_exact_twin_in_sample"] = float((cc[inv.ravel()] > 1).mean())
log("quant", res["quant"])

# ---------------- per-dim stats ----------------
Xf = QU.astype(np.float64)
mu = Xf.mean(0); sd = Xf.std(0); sk = stats.skew(Xf, 0); ku = stats.kurtosis(Xf, 0)
bc = (sk ** 2 + 1) / (ku + 3)
ndist = np.array([len(np.unique(QU[:, d])) for d in range(128)])
# number of modes of a smoothed histogram
modes = []
for d in range(128):
    hh = np.bincount(QU[:, d].astype(np.int64) + 128, minlength=256).astype(np.float64)
    g = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2); g /= g.sum()
    s = np.convolve(hh, g, "same")
    pk = [i for i in range(1, 255) if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > 0.05 * s.max()]
    modes.append(len(pk))
modes = np.array(modes)
# R^2 of each dim from the other 127 (linear)
Cv = np.cov(Xf.T); P = np.linalg.inv(Cv); r2_others = 1 - 1 / (np.diag(P) * np.diag(Cv))
res["dims"] = dict(mean_min_max=[float(mu.min()), float(mu.max())], sd_min_med_max=[float(sd.min()), float(np.median(sd)), float(sd.max())],
                   skew_min_max=[float(sk.min()), float(sk.max())], exkurt_min_med_max=[float(ku.min()), float(np.median(ku)), float(ku.max())],
                   bimodality_coef_min_med_max=[float(bc.min()), float(np.median(bc)), float(bc.max())], n_dims_bc_gt_0555=int((bc > 0.555).sum()),
                   n_distinct_min_med_max=[int(ndist.min()), int(np.median(ndist)), int(ndist.max())], dims_with_2plus_modes=np.nonzero(modes >= 2)[0].tolist(),
                   p_zero_min_med_max=[float(pz.min()), float(np.median(pz)), float(pz.max())],
                   r2_from_other_dims_min_med_max=[float(r2_others.min()), float(np.median(r2_others)), float(r2_others.max())],
                   r2_from_other_dims_lowest5=[(int(i), float(r2_others[i])) for i in np.argsort(r2_others)[:5]],
                   top_kurtosis_dims=[(int(i), float(ku[i]), float(sk[i])) for i in np.argsort(-ku)[:5]])
# contiguity of the correlation structure (concatenated blocks would show near-diagonal blocks)
R = np.corrcoef(Xf.T); A = np.abs(R - np.eye(128))
ii, jj = np.triu_indices(128, 1); dd = jj - ii
res["dims"]["mean_abs_corr_offdiag"] = float(A[ii, jj].mean())
res["dims"]["mean_abs_corr_|i-j|<=4"] = float(A[ii, jj][dd <= 4].mean()); res["dims"]["mean_abs_corr_|i-j|>32"] = float(A[ii, jj][dd > 32].mean())
blocks = {}
for b in (2, 4, 8):
    w = 128 // b; lab = np.arange(128) // w
    same = lab[ii] == lab[jj]
    blocks[b] = [float(A[ii, jj][same].mean()), float(A[ii, jj][~same].mean())]
res["dims"]["blockwise_abs_corr_within_vs_across"] = blocks
# canonical correlations first 64 vs last 64 (an independent appended block would give small CCs)
def cca(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    qa, _ = np.linalg.qr(a); qb, _ = np.linalg.qr(b)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)
sub = Xf[:200000]
cc1 = cca(sub[:, :64], sub[:, 64:]); cc2 = cca(sub[:, :96], sub[:, 96:]); cc3 = cca(sub[:, :112], sub[:, 112:])
res["dims"]["cca_0-63_vs_64-127_top/median/min"] = [float(cc1[0]), float(np.median(cc1)), float(cc1[-1])]
res["dims"]["cca_0-95_vs_96-127_top/median/min"] = [float(cc2[0]), float(np.median(cc2)), float(cc2[-1])]
res["dims"]["cca_0-111_vs_112-127_top/median/min"] = [float(cc3[0]), float(np.median(cc3)), float(cc3[-1])]
log("dims", res["dims"])

# per-dim association with attributes (candidate sample)
Xs = X.astype(np.float64)
gm = Xs.mean(0)
cnt = np.bincount(C, minlength=K).astype(np.float64)
assoc = {}
rk = stats.rankdata
Xr = np.apply_along_axis(rk, 0, Xs[:300000])
for nm, y in attrs.items():
    y = np.asarray(y, np.float64)[:300000]; ok = np.isfinite(y)
    yr = rk(y[ok])
    rho = np.array([np.corrcoef(Xr[ok, d], yr)[0, 1] for d in range(128)])
    # full linear R^2 for comparison (in-sample, 300K rows, 129 params)
    A1 = np.c_[Xs[:300000][ok], np.ones(ok.sum())]; w = np.linalg.lstsq(A1, y[ok], rcond=None)[0]
    full = 1 - ((y[ok] - A1 @ w) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    o = np.argsort(-np.abs(rho))
    assoc[nm] = {"best_dim": int(o[0]), "best_spearman": float(rho[o[0]]), "second_dim": int(o[1]), "second_spearman": float(rho[o[1]]),
                 "n_dims_abs_rho_gt_0.2": int((np.abs(rho) > 0.2).sum()),
                 "best_dim_rho2_over_full_linear_r2": float(rho[o[0]] ** 2 / max(full, 1e-9)), "full_linear_r2_insample": float(full)}
res["dim_attr_assoc"] = assoc
log("assoc", assoc)
res["dims"]["eta2_cat_note"] = "see s02_probes.json eta2_per_dim"

# ---------------- anisotropy ----------------
m = Xs.mean(0); mn = m / np.linalg.norm(m)
p = Xs @ mn
res["aniso"] = dict(mean_vec_norm=float(np.linalg.norm(m)), proj_mean=float(p.mean()), proj_sd=float(p.std()), proj_q01_q50_q99=np.quantile(p, [0.01, 0.5, 0.99]).tolist())
cmj = np.bincount(C, weights=p, minlength=K) / cnt
res["aniso"]["eta2_cat_of_proj"] = float((cnt * (cmj - p.mean()) ** 2).sum() / (p.var() * n))
res["aniso"]["proj_by_cat_min_max"] = [float(cmj.min()), float(cmj.max())]
# PC1 vs mean direction
Vt = np.load(OUT + "cache/pca_Vt.npy")
res["aniso"]["cos_meandir_PC1"] = float(abs(Vt[0] @ mn)); res["aniso"]["cos_meandir_PC1..3"] = [float(abs(Vt[k] @ mn)) for k in range(3)]
cor = {}
for nm, y in attrs.items():
    y = np.asarray(y, np.float64); ok = np.isfinite(y)
    rho_all = stats.spearmanr(p[ok], y[ok])[0]
    # within-category (demeaned by category) spearman
    pr = p - cmj[C]; ym = np.bincount(C[ok], weights=y[ok], minlength=K) / np.maximum(np.bincount(C[ok], minlength=K), 1)
    rho_w = stats.spearmanr(pr[ok], (y - ym[C])[ok])[0]
    cor[nm] = dict(spearman=float(rho_all), spearman_within_cat=float(rho_w))
res["aniso"]["proj_vs_attr"] = cor
# user mean direction (val users, non-zero)
ue = np.load("<repo>/data/flat/val/user_emb.npy").astype(np.float64); ue = ue[np.abs(ue).sum(1) > 0] / 256.0
um = ue.mean(0)
res["aniso"]["user_mean_vec_norm"] = float(np.linalg.norm(um)); res["aniso"]["cos_item_mean_user_mean"] = float(um @ m / np.linalg.norm(um) / np.linalg.norm(m))
log("aniso", res["aniso"])

# ---------------- raw vs centred vs whitened neighbour structure ----------------
NS = 300000
Xa = X[:NS].astype(np.float32)
Cc = np.cov((Xa - m).T); ev_, evec = np.linalg.eigh(Cc)
W = evec / np.sqrt(ev_ + 1e-6)  # PCA whitening
spaces = {"raw": Xa, "centered": (Xa - m).astype(np.float32), "whitened": ((Xa - m) @ W).astype(np.float32),
          "whitened_top64": ((Xa - m) @ W[:, -64:]).astype(np.float32), "drop_PC1": ((Xa - m) - np.outer((Xa - m) @ Vt[0], Vt[0])).astype(np.float32)}
nb = {}
for nm, F in spaces.items():
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    D, I = knn_ip(np.ascontiguousarray(F), np.ascontiguousarray(F[:50000]), 11)
    nb[nm] = I[:, 1:]
    rcos = (F[:50000, None, :] * F[I[:, 1:]]).sum(-1)
    nb[nm + "_cos"] = rcos
cs = C[:NS]
geo = {}
for nm in spaces:
    I = nb[nm]
    ov = np.mean([len(set(I[i]) & set(nb["raw"][i])) / 10 for i in range(50000)])
    knnr = {}
    for an in ("f2", "f5", "log_n_cand", "log_age_h", "ctr_smoothed"):
        y = np.asarray(attrs[an], np.float64)[:NS]; okk = np.isfinite(y)
        pr = np.nanmean(np.where(okk[I], y[I], np.nan), 1); yy = y[:50000]; o2 = np.isfinite(yy) & np.isfinite(pr)
        knnr[an] = float(np.corrcoef(yy[o2], pr[o2])[0, 1])
    geo[nm] = dict(top10_overlap_with_raw=float(ov), knn10_same_cat=float((cs[I] == cs[:50000, None]).mean()), knn1_same_cat=float((cs[I[:, 0]] == cs[:50000]).mean()),
                   knn10_corr_with_attr=knnr)
    log(nm, geo[nm])
res["neighbour_spaces"] = geo
res["random_pair_cos_raw_centered"] = [float((Xa[:100000] * Xa[100000:200000]).sum(1).mean()),
                                       float(((spaces["centered"][:100000] / np.linalg.norm(spaces["centered"][:100000], axis=1, keepdims=True)) *
                                              (spaces["centered"][100000:200000] / np.linalg.norm(spaces["centered"][100000:200000], axis=1, keepdims=True))).sum(1).mean())]
np.save(OUT + "cache/item_mean_vec.npy", m); np.save(OUT + "cache/whiten_W.npy", W)
json.dump(res, open(OUT + "s03_aniso_dims.json", "w"), indent=1, default=float)
log("saved")
