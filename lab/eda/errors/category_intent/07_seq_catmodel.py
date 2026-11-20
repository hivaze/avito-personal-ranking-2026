"""Small torch category-intent model (GPU N): history sequence encoder + per-(feed, category) query.

History tokens (last K events, newest first): category, event_type, event_source, recency position (+ user-embedding token).
Group query: category embedding + MLP(standardised handcrafted group features from 02_build_feats).
Score = MLP([q, attn(q, hist), q*attn]). Loss: listwise softmax CE over the feed's categories,
target = share of the feed's positives in each category. Train: same train subset as the GBDT (R + S);
early stopping on the same 10% user-disjoint train holdout. Outputs val/test group logits (log Q up to a constant).
"""
import json, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numba as nb
from catlib import *

K = 64
D = 96
torch.manual_seed(0)
dev = torch.device("cuda")
t0 = time.time()
tag = sys.argv[1] if len(sys.argv) > 1 else "S1"
use_hand = (sys.argv[2] if len(sys.argv) > 2 else "1") == "1"


@nb.njit(parallel=True, cache=True)
def _pad_hist(hoff, H, hc, he, hs, K, oc, oe, os_):
    for f in nb.prange(len(H)):
        n = min(H[f], K)
        for j in range(n):
            oc[f, j] = hc[hoff[f] + j] + 1
            oe[f, j] = he[hoff[f] + j]
            os_[f, j] = hs[hoff[f] + j] + 1


def hist_tensor(split, feeds):
    v = load_view(split, feeds=feeds, with_hist=True)
    n = len(v["H"])
    oc = np.zeros((n, K), np.int64); oe = np.zeros((n, K), np.int64); os_ = np.zeros((n, K), np.int64)
    _pad_hist(v["hoff"], v["H"], v["h_category_id"].astype(np.int64), v["h_event_type"].astype(np.int64),
              v["h_event_source"].astype(np.int64), K, oc, oe, os_)
    return oc, oe, os_, v["user_emb"]


grp = {s: dict(np.load(OUT / f"grp_{s}.npz", allow_pickle=True)) for s in ["train", "val", "test"]}
names = list(grp["train"]["names"])
hand_idx = [i for i, n in enumerate(names) if n != "cat" and not n.startswith("upca")]
# standardisation: signed log1p for heavy-tailed columns, then z with train stats
Xtr = grp["train"]["X"][:, hand_idx].astype(np.float64)
heavy = np.abs(Xtr).max(0) > 20
def prep(X):
    X = X[:, hand_idx].astype(np.float64)
    X[:, heavy] = np.sign(X[:, heavy]) * np.log1p(np.abs(X[:, heavy]))
    return X
mu = prep(grp["train"]["X"]).mean(0); sd = prep(grp["train"]["X"]).std(0) + 1e-6
data = {}
for s in ["train", "val", "test"]:
    g = grp[s]
    feeds = g["feeds"] if s == "train" else None
    oc, oe, os_, ue = hist_tensor(s, feeds)
    Xs = ((prep(g["X"]) - mu) / sd).clip(-6, 6).astype(np.float32)
    tgt = g["g_npos"].astype(np.float32) if "g_npos" in g else np.zeros(len(g["g_cat"]), np.float32)
    data[s] = dict(hc=torch.tensor(oc, device=dev), he=torch.tensor(oe, device=dev), hs=torch.tensor(os_, device=dev),
                   ue=torch.tensor(ue / 64.0, device=dev), X=torch.tensor(Xs, device=dev),
                   cat=torch.tensor(g["g_cat"], device=dev), tgt=torch.tensor(tgt, device=dev),
                   goff=torch.tensor(g["goff"], device=dev), G=torch.tensor(g["G"], device=dev))
    print(s, "ready", Xs.shape, oc.shape, f"{time.time()-t0:.0f}s", flush=True)
NF = len(hand_idx)


class CatIntent(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.Embedding(N_CAT + 1, D); self.ee = nn.Embedding(N_ET + 1, D); self.se = nn.Embedding(40, D)
        self.pe = nn.Embedding(K, D); self.ue = nn.Linear(128, D)
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.qcat = nn.Embedding(N_CAT, D)
        self.qx = nn.Sequential(nn.Linear(NF, 2 * D), nn.GELU(), nn.Linear(2 * D, D)) if use_hand else None
        self.att = nn.MultiheadAttention(D, 4, dropout=0.1, batch_first=True)
        self.out = nn.Sequential(nn.Linear(3 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, 1))

    def forward(self, hc, he, hs, ue, gcat, gx, gmask):
        pad = hc == 0
        h = self.ce(hc) + self.ee(he) + self.se(hs) + self.pe.weight[None]
        h = torch.cat([self.ue(ue)[:, None], h], 1)
        pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        h = self.enc(h, src_key_padding_mask=pad)
        q = self.qcat(gcat)
        if self.qx is not None:
            q = q + self.qx(gx)
        a, _ = self.att(q, h, h, key_padding_mask=pad)
        z = self.out(torch.cat([q, a, q * a], -1)).squeeze(-1)
        return z.masked_fill(~gmask, -1e4)


def batch(d, fb):
    G = d["G"][fb]; Gm = int(G.max())
    ar = torch.arange(Gm, device=dev)
    gm = ar[None] < G[:, None]
    gi = (d["goff"][fb][:, None] + ar[None]).clamp(max=len(d["cat"]) - 1)
    return gi, gm


def predict(model, d):
    model.eval(); n = len(d["G"]); out = torch.zeros(len(d["cat"]), device=dev)
    with torch.no_grad():
        for i in range(0, n, 2048):
            fb = torch.arange(i, min(i + 2048, n), device=dev)
            gi, gm = batch(d, fb)
            z = model(d["hc"][fb], d["he"][fb], d["hs"][fb], d["ue"][fb], d["cat"][gi], d["X"][gi], gm)
            out[gi[gm]] = z[gm]
    return out.cpu().numpy().astype(np.float64)


tr = data["train"]
users = grp["train"]["user_id"]
rng = np.random.default_rng(1)
uu = np.unique(users)
hold_u = set(rng.choice(uu, int(0.1 * len(uu)), replace=False).tolist())
nf = len(users)
hold_f = np.array([u in hold_u for u in users]) & (users != 0) | ((users == 0) & (rng.random(nf) < 0.1))
npos_f = np.bincount(grp["train"]["g_feed"], weights=grp["train"]["g_npos"], minlength=nf)
tr_feeds = np.where(~hold_f & (npos_f > 0) & (grp["train"]["G"] >= 2))[0]
ho_mask = hold_f
leak = leak_mask_val(); vmask = (~leak) & (grp["val"]["G"] >= 2)
nn_ref = np.load(OUT / "01_val_catndcg_perfeed.npz")["nn_trainonly_LSE_T1.0"]
hv = dict(g_y=(grp["train"]["g_npos"] > 0).astype(np.float32))

model = CatIntent().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
B = 512
EP = 6
steps = EP * (len(tr_feeds) // B)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
best = (-1, None); log = []
ho_idx = np.where(ho_mask)[0]
# holdout view for category ndcg
ho_G = grp["train"]["G"][ho_idx]
ho_goff = np.concatenate([[0], np.cumsum(ho_G)[:-1]])
ho_gsel = np.concatenate([np.arange(grp["train"]["goff"][f], grp["train"]["goff"][f] + grp["train"]["G"][f]) for f in ho_idx])
for ep in range(EP):
    model.train(); perm = torch.tensor(rng.permutation(tr_feeds), device=dev); tl = 0.0; t1 = time.time()
    for i in range(0, len(perm) - B + 1, B):
        fb = perm[i:i + B]
        gi, gm = batch(tr, fb)
        z = model(tr["hc"][fb], tr["he"][fb], tr["hs"][fb], tr["ue"][fb], tr["cat"][gi], tr["X"][gi], gm)
        t = tr["tgt"][gi] * gm
        t = t / t.sum(1, keepdim=True)
        loss = -(t * F.log_softmax(z, 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tl += loss.item()
    p_all = predict(model, tr)
    ndh = cat_ndcg(dict(g_y=hv["g_y"][ho_gsel], goff=ho_goff, G=ho_G), p_all[ho_gsel]).mean()
    pv = predict(model, data["val"])
    ndv = cat_ndcg(dict(g_y=grp["val"]["g_y"], goff=grp["val"]["goff"], G=grp["val"]["G"]), pv)
    d = paired(ndv, nn_ref, vmask)
    print(f"ep {ep} loss {tl/(len(perm)//B):.4f} holdout catNDCG {ndh:.4f} | val catNDCG {ndv[vmask].mean():.4f} (vs NN-LSE {d[0]:+.4f} ± {d[1]:.4f}) | {time.time()-t1:.0f}s", flush=True)
    log.append(dict(ep=ep, holdout=float(ndh), val=float(ndv[vmask].mean()), d_vs_nn=d))
    if ndh > best[0]:  # select epoch on the train holdout only
        best = (ndh, ep, pv, predict(model, data["test"]), ndv)
print("best epoch (by train holdout):", best[1], flush=True)
np.save(OUT / f"cat_{tag}_val_grp.npy", best[2].astype(np.float32))
np.save(OUT / f"cat_{tag}_test_grp.npy", best[3].astype(np.float32))
d = paired(best[4], nn_ref, vmask)
json.dump(dict(log=log, best_ep=best[1], val_catndcg=float(best[4][vmask].mean()), d_vs_nn=d, use_hand=use_hand),
          open(OUT / f"07_seq_{tag}.json", "w"), indent=1)
print("total", time.time() - t0)
