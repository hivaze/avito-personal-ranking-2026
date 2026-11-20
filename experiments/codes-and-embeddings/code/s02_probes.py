# Linear and non-linear probes from the 128-d item vector on 1.5M train candidate items (5 random
# folds): category, k-means cluster, features f1..f5, log popularity, listing age, item click rate,
# each with and without category one-hots.
# Run without arguments on GPU N (logistic probes, MLP and nearest neighbours) after s01_build.py,
# which writes cache/train_sample.npz. The command line is not on record.
# Produced s02_probes.json and s02_probes.log: category accuracy 0.983, click-rate R2 0.003.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/emb/structure/s02_probes.py
#
"""(1) Linear / nonlinear probes from the 128-d item vector (train sample, 1.5M candidate items, 5-fold random item folds).
Category (53-way acc, top-3; which dims), k-means cluster id, f1..f5 (item means), log popularity, posting time / age, item CTR,
with and without category one-hots. GPU N for the logistic probes, MLP and kNN.
"""
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
import numpy as np, torch
sys.path.insert(0, '<repo>/nn_x/emb/structure')
from knn_gpu import knn_ip

OUT = "<repo>/nn_x/emb/structure/"
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
dev = "cuda"
z = np.load(OUT + "cache/train_sample.npz")
S = z["idx"]; X = z["E"].astype(np.float32) / 256.0
n = len(S)
cat = z["cat"].astype(np.int64)
res = {}
rng = np.random.default_rng(1)
fold = rng.integers(0, 5, n)
ucat = np.unique(cat); log("categories present", len(ucat), "cat==0", (cat == 0).sum())
cmap = np.zeros(cat.max() + 1, np.int64); cmap[ucat] = np.arange(len(ucat)); C = cmap[cat]
K = len(ucat)
Xg = torch.tensor(X, device=dev)


def logreg_cv(F, y, ncls, l2=1e-4, iters=150, topk=(1, 3)):
    """multinomial logistic regression, full-batch LBFGS on GPU, 5-fold; returns top-k accuracies and OOF predicted class."""
    F = torch.as_tensor(F, device=dev, dtype=torch.float32)
    mu = F.mean(0, keepdim=True); sd = F.std(0, keepdim=True) + 1e-6
    F = (F - mu) / sd
    yt = torch.as_tensor(y, device=dev)
    acc = {k: 0 for k in topk}
    oof = torch.zeros(len(y), dtype=torch.long, device=dev)
    for f in range(5):
        tr = torch.as_tensor(fold != f, device=dev); te = ~tr
        W = torch.zeros(F.shape[1], ncls, device=dev, requires_grad=True)
        b = torch.zeros(ncls, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
        Ft, yt_ = F[tr], yt[tr]
        def closure():
            opt.zero_grad()
            l = torch.nn.functional.cross_entropy(Ft @ W + b, yt_) + l2 * (W ** 2).sum()
            l.backward(); return l
        opt.step(closure)
        with torch.no_grad():
            lo = F[te] @ W + b
            tk = lo.topk(max(topk), 1).indices
            for k in topk:
                acc[k] += (tk[:, :k] == yt[te][:, None]).any(1).sum().item()
            oof[te] = tk[:, 0]
    return {k: acc[k] / len(y) for k in topk}, oof.cpu().numpy()


# ---------------- category ----------------
r, oofc = logreg_cv(X, C, K)
res["cat_logreg_acc"] = r[1]; res["cat_logreg_top3"] = r[3]
maj = np.bincount(C).max() / n; res["cat_majority"] = maj
log("category logreg", r, "majority", maj)
# nearest category centroid (cosine), 5-fold
acc = 0
for f in range(5):
    tr = fold != f
    M = np.stack([X[tr & (C == k)].mean(0) for k in range(K)])
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    acc += (np.argmax(X[~tr] @ M.T, 1) == C[~tr]).sum()
res["cat_nearest_centroid_acc"] = acc / n; log("nearest centroid", acc / n)
# per-dim eta^2 (between-category share of variance)
gm = X.mean(0)
cnt = np.bincount(C, minlength=K).astype(np.float64)
cm = np.stack([np.bincount(C, weights=X[:, d], minlength=K) for d in range(128)], 1) / cnt[:, None]
eta2 = ((cnt[:, None] * (cm - gm) ** 2).sum(0)) / (X.var(0) * n)
res["eta2_per_dim"] = eta2.round(4).tolist()
o = np.argsort(-eta2)
log("eta2 q0/q50/q100", eta2.min(), np.median(eta2), eta2.max(), "top dims", o[:10].tolist(), eta2[o[:10]].round(3).tolist())
res["eta2_summary"] = dict(min=float(eta2.min()), median=float(np.median(eta2)), max=float(eta2.max()), top10=o[:10].tolist(),
                           block_means=[float(eta2[i:i + 32].mean()) for i in range(0, 128, 32)],
                           spearman_dim=float(np.corrcoef(np.argsort(np.argsort(eta2)), np.arange(128))[0, 1]))
# subsets: top-k eta2 dims vs random k dims vs prefix; blocks of 32
sub = {}
for k in (4, 8, 16, 32, 64):
    a_top = logreg_cv(X[:, o[:k]], C, K, topk=(1,))[0][1]
    a_pre = logreg_cv(X[:, :k], C, K, topk=(1,))[0][1]
    a_rnd = np.mean([logreg_cv(X[:, np.random.default_rng(s).choice(128, k, replace=False)], C, K, topk=(1,))[0][1] for s in range(2)])
    sub[k] = dict(top_eta=a_top, prefix=a_pre, random=a_rnd); log("subset", k, sub[k])
res["cat_subset_acc"] = sub
blk = {}
for i in range(4):
    d = np.arange(i * 32, i * 32 + 32)
    blk[f"only_{i*32}-{i*32+31}"] = logreg_cv(X[:, d], C, K, topk=(1,))[0][1]
    blk[f"drop_{i*32}-{i*32+31}"] = logreg_cv(np.delete(X, d, 1), C, K, topk=(1,))[0][1]
res["cat_block_acc"] = blk; log("blocks", blk)
# PCA: accuracy from top-k PCs
Xc = X - gm
U, s_, Vt = np.linalg.svd(Xc[:300000], full_matrices=False)
P = Xc @ Vt.T
pcs = {}
for k in (1, 2, 3, 5, 10, 20, 32):
    pcs[k] = logreg_cv(P[:, :k], C, K, topk=(1,))[0][1]
res["cat_acc_topPCs"] = pcs; log("PCs", pcs)
eta_pc = []
for j in range(10):
    cmj = np.bincount(C, weights=P[:, j], minlength=K) / cnt
    eta_pc.append(float((cnt * (cmj - P[:, j].mean()) ** 2).sum() / (P[:, j].var() * n)))
res["eta2_PC1_10"] = eta_pc; log("eta2 PCs", np.round(eta_pc, 3))
np.save(OUT + "cache/pca_Vt.npy", Vt); np.save(OUT + "cache/pca_mu.npy", gm)
# confusion: worst categories
per = {int(ucat[k]): float((oofc[C == k] == k).mean()) for k in range(K)}
res["cat_acc_per_category_worst5"] = sorted(per.items(), key=lambda t: t[1])[:5]
res["cat_n_items"] = {int(ucat[k]): int(cnt[k]) for k in range(K)}

# ---------------- k-means cluster id ----------------
for kk in (1024, 4096):
    cl = np.load(f"<repo>/data/clu_work/item_cl_{kk}_train.npy", mmap_mode="r")[S].astype(np.int64)
    cent = np.load(f"<repo>/data/clu_work/item_cent_{kk}.npy")
    nc = (np.argmax(X @ cent.T, 1) == cl).mean()
    # category purity of clusters
    tab = np.zeros((kk, K)); np.add.at(tab, (cl, C), 1)
    pur = tab.max(1).sum() / n
    res[f"cl{kk}"] = dict(nearest_centroid_reproduces=float(nc), category_purity=float(pur))
    if kk == 1024:
        a, _ = logreg_cv(X, cl, kk, l2=1e-6, iters=300, topk=(1, 3))
        res[f"cl{kk}"]["logreg_acc"] = a[1]; res[f"cl{kk}"]["logreg_top3"] = a[3]
    log("cluster", kk, res[f"cl{kk}"])

# ---------------- regressions ----------------
ncand = z["n_cand"].astype(np.float64); npos = z["n_pos"].astype(np.float64); hc = z["hcnt"].astype(np.float64)
tpost = z["t_post"]; tfeed = z["t_feed"]
targets = {
    "f1": z["f1"], "f2": z["f2"], "f3": z["f3"], "f4": z["f4"], "f5": z["f5"],
    "log_n_cand_train": np.log1p(ncand), "log_n_hist_train": np.log1p(z["n_hist"].astype(np.float64)),
    "log_hist_et1": np.log1p(hc[:, 0]), "log_hist_et2": np.log1p(hc[:, 1]), "log_hist_et3": np.log1p(hc[:, 2]),
    "share_hist_et1": hc[:, 0] / np.maximum(hc.sum(1), 1),
    "log_age_h": np.log1p((tfeed - tpost) / 3600.0), "t_post_days": (tpost - np.nanmin(tpost)) / 86400.0,
    "age_capped": (z["f4_rep"] >= 63 / 64 - 1e-6).astype(np.float64),
    "ctr_train_ncand>=20": np.where(ncand >= 20, npos / np.maximum(ncand, 1), np.nan),
    "ctr_train_smoothed": (npos + 0.02 * 10) / (ncand + 10),
    "hist_per_cand": np.log1p(z["n_hist"]) - np.log1p(ncand),
}
Oh = np.zeros((n, K), np.float32); Oh[np.arange(n), C] = 1
Xe = np.concatenate([X, np.ones((n, 1), np.float32)], 1)


def ridge_cv(F, y, lam=1e-2):
    ok = np.isfinite(y)
    pred = np.full(n, np.nan)
    Fg = torch.tensor(F, device=dev, dtype=torch.float64); yg = torch.tensor(np.nan_to_num(y), device=dev, dtype=torch.float64)
    for f in range(5):
        tr = torch.tensor((fold != f) & ok, device=dev); te = (fold == f) & ok
        A = Fg[tr]; G = A.T @ A; G += lam * len(A) * torch.eye(G.shape[0], device=dev, dtype=torch.float64) * (G.diag().mean() / len(A))
        w = torch.linalg.solve(G, A.T @ yg[tr])
        pred[te] = (Fg[torch.tensor(te, device=dev)] @ w).cpu().numpy()
    return pred


def r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    return float(1 - ((y[ok] - p[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())


# per-category interaction model: separate ridge per category (emb within category)
def ridge_percat_cv(y, lam=1e-2):
    pred = np.full(n, np.nan)
    for k in range(K):
        m = C == k
        if m.sum() < 500:
            continue
        sub = np.nonzero(m)[0]
        Fk = Xe[sub]; yk = y[sub]; okk = np.isfinite(yk)
        for f in range(5):
            tr = (fold[sub] != f) & okk; te = (fold[sub] == f) & okk
            if tr.sum() < 200 or te.sum() == 0:
                continue
            A = Fk[tr].astype(np.float64); G = A.T @ A; G += lam * np.eye(129) * G.diagonal().mean()
            w = np.linalg.solve(G, A.T @ yk[tr]); pred[sub[te]] = Fk[te] @ w
    return pred


# kNN regression (cosine, k=20, faiss GPU, 5-fold) -- one pass gives neighbour ids, reused for every target
nbr = np.zeros((n, 20), np.int64)
for f in range(5):
    tr = np.nonzero(fold != f)[0]; te = np.nonzero(fold == f)[0]
    _, I = knn_ip(X[tr], X[te], 20); nbr[te] = tr[I]
log("kNN done")
# kNN category accuracy (majority of 20 / top-1)
res["cat_knn1_acc"] = float((C[nbr[:, 0]] == C).mean())
res["cat_knn10_same_cat_share"] = float((C[nbr[:, :10]] == C[:, None]).mean())
log("knn cat", res["cat_knn1_acc"], res["cat_knn10_same_cat_share"])


def mlp_cv(F, y, epochs=6):
    ok = np.isfinite(y); pred = np.full(n, np.nan)
    Fg = torch.tensor(F, device=dev); ym = np.nanmean(y); ys = np.nanstd(y)
    yg = torch.tensor(np.nan_to_num((y - ym) / ys), device=dev, dtype=torch.float32)
    for f in range(5):
        torch.manual_seed(f)
        net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
        tr = torch.tensor(np.nonzero((fold != f) & ok)[0], device=dev); te = np.nonzero((fold == f) & ok)[0]
        steps = epochs * (len(tr) // 4096); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
        for st in range(steps):
            bi = tr[torch.randint(0, len(tr), (4096,), device=dev)]
            l = ((net(Fg[bi]).squeeze(1) - yg[bi]) ** 2).mean()
            opt.zero_grad(); l.backward(); opt.step(); sch.step()
        with torch.no_grad():
            pred[te] = net(Fg[torch.tensor(te, device=dev)]).squeeze(1).cpu().numpy() * ys + ym
    return pred


reg = {}
for name, y in targets.items():
    y = np.asarray(y, np.float64)
    ok = np.isfinite(y)
    pe = ridge_cv(Xe, y); pc = ridge_cv(np.concatenate([Oh, np.ones((n, 1), np.float32)], 1), y)
    pec = ridge_cv(np.concatenate([Xe, Oh], 1), y)
    ppc = ridge_percat_cv(y)
    yk = np.where(ok, y, np.nan); kn = np.nanmean(yk[nbr], 1)
    row = dict(n=int(ok.sum()), mean=float(np.nanmean(y)), sd=float(np.nanstd(y)), ridge_emb=r2(y, pe), cat_only=r2(y, pc), ridge_emb_cat=r2(y, pec),
               ridge_per_category=r2(y, ppc), knn20=r2(y, kn))
    sse_c = np.nansum((y - pc) ** 2); sse_ec = np.nansum((y - pec) ** 2)
    row["partial_r2_emb_given_cat"] = float(1 - sse_ec / sse_c)
    if name in ("f1", "f2", "f3", "f4", "f5", "log_n_cand_train", "log_n_hist_train", "log_age_h", "t_post_days", "ctr_train_smoothed", "log_hist_et1"):
        row["mlp_emb"] = r2(y, mlp_cv(X, y))
    reg[name] = row
    log(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
res["regressions"] = reg
# ctr as classification: AUC of ridge-predicted item ctr for items' click-or-not in train (row-level is production-biased)
json.dump(res, open(OUT + "s02_probes.json", "w"), indent=1, default=float)
log("saved")
