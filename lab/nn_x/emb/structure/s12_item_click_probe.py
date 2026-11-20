"""(1b) Does the item vector carry item-level 'attractiveness' (click propensity) beyond category, on unbiased VAL traffic?
Logistic probes on val candidate rows, user-grouped 5-fold (user_id 0 = each feed its own group), fixed L2 (no tuning => no inner loop).
Feature sets: category one-hot; emb; emb + category; f1..f5 (reference). Also: a train-fitted item-level binomial probe
(1.5M train items, clicks / impressions, production traffic) applied to val rows. Reports pooled AUC and mean within-feed AUC (OOF).
GPU N."""
import os, json, numpy as np, polars as pl, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
from sklearn.metrics import roc_auc_score

OUT = "<repo>/nn_x/emb/structure/"
V = "<repo>/data/flat/val/"
dev = "cuda"
NV = 2_516_711
E = np.fromfile("<repo>/data/competition/val/item_embeddings/data.memmap", dtype=np.int8).reshape(NV, 128)
feeds = pl.read_parquet(V + "feeds.parquet"); F = len(feeds)
ce = np.load(V + "cand_emb_idx.npy"); cf = np.load(V + "cand_fidx.npy").astype(np.int64); y = np.load(V + "cand_target.npy").astype(np.float32)
cc = np.load(V + "cand_category_id.npy").astype(np.int64)
ok = ce >= 0
X = E[np.maximum(ce, 0)].astype(np.float32) / 256
uid = feeds["user_id"].to_numpy().astype(np.int64)
grp = np.where(uid > 0, uid, -(np.arange(F) + 1))
ug, ginv = np.unique(grp, return_inverse=True)
gfold = np.random.default_rng(0).integers(0, 5, len(ug))
fold = gfold[ginv][cf]
K = 54
OH = np.zeros((len(y), K), np.float32); OH[np.arange(len(y)), cc] = 1
FF = np.stack([np.load(V + f"cand_f{k}.npy") for k in range(1, 6)], 1).astype(np.float32)


def within_auc(s):
    rm = ok
    o = np.lexsort((s, cf)); yy = y[o]; ff = cf[o]; r_ = rm[o]; yy = yy[r_]; ff = ff[r_]
    rank = np.arange(len(yy)) - np.searchsorted(ff, ff)
    npos = np.bincount(ff, weights=yy, minlength=F); nn = np.bincount(ff, minlength=F) - npos
    sr = np.bincount(ff, weights=(rank + 1) * yy, minlength=F); okf = (npos > 0) & (nn > 0)
    return float(((sr[okf] - npos[okf] * (npos[okf] + 1) / 2) / (npos[okf] * nn[okf])).mean())


def logit_fit(A, t, w=None, l2=1e-3, iters=200):
    A = torch.tensor(A, device=dev); t = torch.tensor(t, device=dev)
    w = torch.ones_like(t) if w is None else torch.tensor(w, device=dev, dtype=torch.float32)
    mu = A.mean(0, keepdim=True); sd = A.std(0, keepdim=True) + 1e-6
    An = (A - mu) / sd
    W = torch.zeros(A.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad()
        l = (torch.nn.functional.binary_cross_entropy_with_logits(An @ W + b, t, reduction="none") * w).sum() / w.sum() + l2 * (W ** 2).sum()
        l.backward(); return l
    opt.step(cl)
    return lambda B: ((torch.tensor(B, device=dev) - mu) / sd @ W + b).detach().cpu().numpy()


res = {}
sets = {"category": OH, "emb": X, "emb+category": np.concatenate([X, OH], 1), "f1..f5": FF, "f1..f5+category": np.concatenate([FF, OH], 1),
        "f1..f5+category+emb": np.concatenate([FF, OH, X], 1)}
for nm, A in sets.items():
    oof = np.zeros(len(y), np.float32)
    for f in range(5):
        tr = (fold != f) & ok; te = (fold == f) & ok
        pred = logit_fit(A[tr], y[tr])
        oof[te] = pred(A[te])
    res[nm] = dict(pooled_auc=float(roc_auc_score(y[ok], oof[ok])), within_feed_auc=within_auc(oof + np.random.default_rng(0).random(len(y)).astype(np.float32) * 1e-6))
    print(nm, res[nm], flush=True)
# train-fitted item-level probe (production traffic), applied to val
z = np.load(OUT + "cache/train_sample.npz")
Xt = z["E"].astype(np.float32) / 256; nc = z["n_cand"].astype(np.float32); npos = z["n_pos"].astype(np.float32)
OHt = np.zeros((len(Xt), K), np.float32); OHt[np.arange(len(Xt)), z["cat"].astype(np.int64)] = 1
for nm, At, Av in [("train_item_probe_emb", Xt, X), ("train_item_probe_category", OHt, OH), ("train_item_probe_emb+category", np.concatenate([Xt, OHt], 1), np.concatenate([X, OH], 1))]:
    # binomial likelihood: duplicate each item as (1, n_pos) and (0, n_cand - n_pos) weighted rows
    A2 = np.concatenate([At, At]); t2 = np.r_[np.ones(len(At), np.float32), np.zeros(len(At), np.float32)]; w2 = np.r_[npos, nc - npos]
    pred = logit_fit(A2, t2, w2)
    s = pred(Av)
    res[nm] = dict(val_pooled_auc=float(roc_auc_score(y[ok], s[ok])), val_within_feed_auc=within_auc(s + np.random.default_rng(0).random(len(y)).astype(np.float32) * 1e-6))
    print(nm, res[nm], flush=True)
json.dump(res, open(OUT + "s12_item_click_probe.json", "w"), indent=1)
