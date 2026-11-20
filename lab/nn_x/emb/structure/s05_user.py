"""(5) user_emb side. Is user_emb a function of the history item vectors, or does it carry extra information?
Maps history -> user_emb are fitted on TRAIN feeds (target = user_emb, label-free) and evaluated on all 14,423 VAL feeds.
Val labels are used only to evaluate fixed (unfitted) scores: AUC / NDCG of cos(user, item), cos(history pools, item), residual.
GPU N."""
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
import numpy as np, polars as pl, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

OUT = "<repo>/nn_x/emb/structure/"
FL = "<repo>/data/flat/"
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
dev = "cuda"
NTAB = {"train": 81_557_944, "val": 2_516_711}
L = 64


def table(split):
    return np.memmap(f"<repo>/data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NTAB[split], 128))


def build(split, fsel):
    """pooled history features + last-64 token tensors for the selected feeds"""
    f = pl.read_parquet(FL + split + "/feeds.parquet")
    H = f["H"].to_numpy()[fsel].astype(np.int64); ho = f["hist_off"].to_numpy()[fsel]
    rows = np.repeat(ho, H) + (np.arange(H.sum()) - np.repeat(np.cumsum(H) - H, H))
    seg = np.repeat(np.arange(len(fsel)), H)
    he = np.load(FL + split + "/hist_emb_idx.npy", mmap_mode="r")[rows]
    et = np.load(FL + split + "/hist_event_type.npy", mmap_mode="r")[rows].astype(np.int64)
    es = np.load(FL + split + "/hist_event_source.npy", mmap_mode="r")[rows].astype(np.int64)
    hp = np.load(FL + split + "/hist_hpos.npy", mmap_mode="r")[rows].astype(np.int64)
    ok = he >= 0
    T = table(split)
    srt = np.argsort(he[ok]); ids = he[ok][srt]
    V = np.empty((ok.sum(), 128), np.int8); V[srt] = T[ids]
    Vg = torch.tensor(V, device=dev).float() / 256.0
    sg = torch.tensor(seg[ok], device=dev); hpg = torch.tensor(hp[ok], device=dev).float(); etg = torch.tensor(et[ok], device=dev)
    F = len(fsel)
    def pool(w):
        out = torch.zeros(F, 128, device=dev); out.index_add_(0, sg, Vg * w[:, None]); return out
    one = torch.ones_like(hpg)
    P = {"all": pool(one), "l20": pool((hpg < 20).float()), "rec5": pool(torch.exp(-hpg / 5)), "et1": pool((etg == 1).float()),
         "l5": pool((hpg < 5).float())}
    # last valid event
    hmin = torch.full((F,), 10 ** 9, device=dev, dtype=torch.long); hmin.scatter_reduce_(0, sg, hpg.long(), "amin")
    last = torch.zeros(F, 128, device=dev); m = hpg.long() == hmin[sg]; last[sg[m]] = Vg[m]
    P["l1"] = last
    nval = torch.zeros(F, device=dev); nval.index_add_(0, sg, one)
    for k in P:
        P[k] = P[k] / P[k].norm(dim=1, keepdim=True).clamp_min(1e-9)
    # tokens for the sequence model (most recent L events, valid only)
    tok = np.zeros((F, L, 128), np.int8); tet = np.zeros((F, L), np.int64); tes = np.zeros((F, L), np.int64); tm = np.zeros((F, L), bool)
    sel = ok & (hp < L)

    pos_in_ok = np.cumsum(ok) - 1
    r = np.nonzero(sel)[0]
    tok[seg[r], hp[r]] = V[pos_in_ok[r]]; tet[seg[r], hp[r]] = et[r]; tes[seg[r], hp[r]] = np.minimum(es[r], 15); tm[seg[r], hp[r]] = True
    return P, nval, H, (tok, tet, tes, tm)


rng = np.random.default_rng(0)
ft = pl.read_parquet(FL + "train/feeds.parquet")
U_tr = np.load(FL + "train/user_emb.npy", mmap_mode="r")
Ht = ft["H"].to_numpy()
cand = np.nonzero(Ht > 0)[0]
fsel = np.sort(rng.choice(cand, 300_000, replace=False))
ut = np.asarray(U_tr[fsel]).astype(np.float32) / 256.0
keep = np.abs(ut).sum(1) > 0
fsel, ut = fsel[keep], ut[keep]
Ptr, nvtr, Htr, toktr = build("train", fsel)
log("train built", len(fsel))
fv = pl.read_parquet(FL + "val/feeds.parquet")
Pva, nvva, Hva, tokva = build("val", np.arange(len(fv)))
uv = np.load(FL + "val/user_emb.npy").astype(np.float32) / 256.0
uvalid = np.abs(uv).sum(1) > 0
log("val built")
res = {}
keys = ["all", "l20", "rec5", "l5", "l1", "et1"]
# cosine of user with each pool (val, feeds with valid user and history)
vm = uvalid & (nvva.cpu().numpy() > 0)
Uv = torch.tensor(uv, device=dev); Ut = torch.tensor(ut, device=dev)
res["cos_user_pool_val"] = {k: float((Uv[vm] * Pva[k][vm]).sum(1).mean()) for k in keys}
res["cos_user_pool_val_shuffled_l20"] = float((Uv[vm] * Pva["l20"][vm][torch.randperm(int(vm.sum()), device=dev)]).sum(1).mean())
log("cos", res["cos_user_pool_val"])


def feats(P, nv):
    lh = torch.log1p(nv)[:, None]
    return torch.cat([P[k] for k in keys] + [lh / 6, torch.ones_like(lh)], 1)


Ftr = feats(Ptr, nvtr); Fva = feats(Pva, nvva)
umean = Ut.mean(0)
def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - umean) ** 2).sum())
def fit_ridge(F, Y, lam=1e-3):
    G = F.T.double() @ F.double(); G += lam * G.diag().mean() * torch.eye(G.shape[0], device=dev, dtype=torch.float64)
    return torch.linalg.solve(G, F.T.double() @ Y.double()).float()
ridge = {}
for nm, cols in [("mean_all", ["all"]), ("mean_last20", ["l20"]), ("all_pools", keys)]:
    ci = sum([[keys.index(k) * 128 + j for j in range(128)] for k in cols], []) + [len(keys) * 128, len(keys) * 128 + 1]
    W = fit_ridge(Ftr[:, ci], Ut)
    pv = Fva[:, ci] @ W
    pvn = pv / pv.norm(dim=1, keepdim=True)
    ridge[nm] = dict(val_r2=r2(pv[vm], Uv[vm]), val_cos=float((pvn[vm] * Uv[vm]).sum(1).mean()), train_r2=r2((Ftr[:, ci] @ W), Ut))
    if nm == "all_pools":
        Uhat_ridge = pvn
log("ridge", ridge)
res["ridge_history_to_user"] = ridge


# sequence model upper bound: transformer over the last 64 events (+ event type / source), cosine loss
class Seq(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.inp = torch.nn.Linear(128, 256); s.et = torch.nn.Embedding(11, 256); s.es = torch.nn.Embedding(16, 256); s.pos = torch.nn.Embedding(L, 256)
        s.enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(256, 4, 512, 0.1, batch_first=True, norm_first=True), 2)
        s.q = torch.nn.Parameter(torch.randn(256) * 0.02); s.out = torch.nn.Sequential(torch.nn.Linear(256 + 6 * 128 + 1, 512), torch.nn.GELU(), torch.nn.Linear(512, 128))
    def forward(s, tok, et, es, m, pooled):
        x = s.inp(tok) + s.et(et) + s.es(es) + s.pos.weight[None]
        h = s.enc(x, src_key_padding_mask=~m)
        a = (h @ s.q).masked_fill(~m, -1e9).softmax(1)
        z = (a[..., None] * h).sum(1)
        return s.out(torch.cat([z, pooled], 1))
def tens(tk, i):
    tok, tet, tes, tm = tk
    return (torch.tensor(tok[i], device=dev).float() / 256.0, torch.tensor(tet[i], device=dev), torch.tensor(tes[i], device=dev), torch.tensor(tm[i], device=dev))
torch.manual_seed(0)
net = Seq().to(dev)
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
ntr = len(fsel); bs = 1024; ep = 4
steps = ep * (ntr // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
hasany = toktr[3].any(1)
st = 0
for e in range(ep):
    perm = np.random.default_rng(e).permutation(ntr)
    for b in range(ntr // bs):
        i = perm[b * bs:(b + 1) * bs]
        tk, et_, es_, m_ = tens(toktr, i)
        m_[:, 0] |= ~m_.any(1)  # avoid fully masked rows
        pr = net(tk, et_, es_, m_, Ftr[i, :6 * 128 + 1])
        loss = 1 - torch.nn.functional.cosine_similarity(pr, Ut[i]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); st += 1
    log("seq epoch", e, float(loss))
net.eval()
outs = []
with torch.no_grad():
    for s in range(0, len(fv), 4096):
        i = np.arange(s, min(s + 4096, len(fv)))
        tk, et_, es_, m_ = tens(tokva, i); m_[:, 0] |= ~m_.any(1)
        outs.append(net(tk, et_, es_, m_, Fva[i, :6 * 128 + 1]))
Useq = torch.cat(outs); Useqn = Useq / Useq.norm(dim=1, keepdim=True)
res["seq_model_history_to_user"] = dict(val_cos=float((Useqn[vm] * Uv[vm]).sum(1).mean()), val_r2=r2(Useq[vm], Uv[vm]),
                                        val_cos_q10_q50_q90=np.quantile((Useqn[vm] * Uv[vm]).sum(1).cpu().numpy(), [.1, .5, .9]).tolist(),
                                        n_train_feeds=int(ntr), epochs=ep)
log("seq", res["seq_model_history_to_user"])

# ---------------- candidate-level evaluation on val ----------------
M = fv["M"].to_numpy().astype(np.int64); off = fv["cand_off"].to_numpy()
ce = np.load(FL + "val/cand_emb_idx.npy"); tg = np.load(FL + "val/cand_target.npy").astype(np.int64); ccat = np.load(FL + "val/cand_category_id.npy").astype(np.int64)
cfi = np.load(FL + "val/cand_fidx.npy")
Tv = table("val")
Cv = torch.tensor(np.asarray(Tv[np.maximum(ce, 0)]), device=dev).float() / 256.0
fi = torch.tensor(cfi, device=dev).long()
mvec = torch.tensor(np.load(OUT + "cache/item_mean_vec.npy"), device=dev).float()
Wh = torch.tensor(np.load(OUT + "cache/whiten_W.npy"), device=dev).float()
uhat_s = Useqn
resid = Uv - Uhat_ridge * (Uhat_ridge * Uv).sum(1, keepdim=True)  # component of u orthogonal to the history prediction
resid_seq = Uv - uhat_s * (uhat_s * Uv).sum(1, keepdim=True)
def cosq(q):
    qn = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return (qn[fi] * Cv).sum(1).cpu().numpy()
scores = {"cos_user": cosq(Uv), "cos_hist_mean_all": cosq(Pva["all"]), "cos_hist_last20": cosq(Pva["l20"]), "cos_hist_rec5": cosq(Pva["rec5"]),
          "cos_ridge_uhat": cosq(Uhat_ridge), "cos_seq_uhat": cosq(uhat_s), "dot_user_resid_ridge": cosq(resid), "dot_user_resid_seq": cosq(resid_seq)}
# transformed spaces (label-free; whitening fitted on train items)
def tr(x, mode):
    if mode == "centered":
        return x - mvec
    return (x - mvec) @ Wh
Cc = {md: tr(Cv, md) for md in ("centered", "whitened")}
for md in ("centered", "whitened"):
    Cn = Cc[md] / Cc[md].norm(dim=1, keepdim=True)
    for nm, q in [("user", Uv), ("hist_rec5", Pva["rec5"])]:
        qq = tr(q, md); qq = qq / qq.norm(dim=1, keepdim=True).clamp_min(1e-9)
        scores[f"{md}_cos_{nm}"] = (qq[fi] * Cn).sum(1).cpu().numpy()
validf = {"user": uvalid, "hist": nvva.cpu().numpy() > 0}
need = {k: ("user" if ("user" in k and "resid" not in k) else "both" if "resid" in k else "hist") for k in scores}
both = uvalid & validf["hist"]


def eval_score(s, rowmask=None, feedmask=None):
    s = s.astype(np.float64).copy()
    fm = np.ones(len(M), bool) if feedmask is None else feedmask
    rm = fm[cfi] & (ce >= 0) & np.isfinite(s)
    if rowmask is not None:
        rm &= rowmask
    out = {"pooled_auc": float(roc_auc_score(tg[rm], s[rm])) if tg[rm].min() < tg[rm].max() else np.nan}
    # within-feed AUC (mean over feeds with both classes)
    o = np.lexsort((s, cfi)); y = tg[o]; ff = cfi[o]; r_ = rm[o]
    y = y[r_]; ff = ff[r_]
    rank = np.arange(len(y)) - np.searchsorted(ff, ff)  # 0-based rank within feed (by score ascending)
    npos = np.bincount(ff, weights=y, minlength=len(M)); nn = np.bincount(ff, minlength=len(M)) - npos
    sr = np.bincount(ff, weights=(rank + 1) * y, minlength=len(M))
    okf = (npos > 0) & (nn > 0)
    auc_f = (sr[okf] - npos[okf] * (npos[okf] + 1) / 2) / (npos[okf] * nn[okf])
    out["within_feed_auc"] = float(auc_f.mean()); out["n_feeds"] = int(okf.sum())
    if rowmask is None:
        sc = np.where(np.isfinite(s), s, -1e9) + np.random.default_rng(0).random(len(s)) * 1e-9
        nd = ndcg_per_feed(sc, tg, off, M)
        out["ndcg"] = float(np.asarray(nd)[fm].mean())
    return out


ev = {}
for k, s in scores.items():
    fm = uvalid if need[k] == "user" else validf["hist"] if need[k] == "hist" else both
    if k.startswith(("centered", "whitened")):
        fm = uvalid if k.endswith("user") else validf["hist"]
    ev[k] = eval_score(s, feedmask=fm & both)  # common feed set for comparability: valid user AND history
    log(k, ev[k])
res["val_eval_common_feeds"] = ev
res["val_eval_common_feeds_n"] = int(both.sum())
nd0 = ndcg_per_feed(np.random.default_rng(1).random(len(tg)), tg, off, M)
res["random_ndcg_common"] = float(np.asarray(nd0)[both].mean())


# fixed (unfitted) fusions: z-scores within feed
def fz(s):
    s = s.astype(np.float64)
    mu = np.bincount(cfi, weights=s, minlength=len(M)) / M; sd = np.sqrt(np.maximum(np.bincount(cfi, weights=s * s, minlength=len(M)) / M - mu ** 2, 1e-12))
    return (s - mu[cfi]) / sd[cfi]
fus = {"z(user)+z(hist_rec5)": fz(scores["cos_user"]) + fz(scores["cos_hist_rec5"]),
       "z(seq_uhat)+z(resid_seq)": fz(scores["cos_seq_uhat"]) + fz(scores["dot_user_resid_seq"]),
       "z(hist_rec5)+z(resid_seq)": fz(scores["cos_hist_rec5"]) + fz(scores["dot_user_resid_seq"]),
       "z(hist_rec5)+z(seq_uhat)": fz(scores["cos_hist_rec5"]) + fz(scores["cos_seq_uhat"])}
res["val_fusions_common_feeds"] = {k: eval_score(v, feedmask=both) for k, v in fus.items()}
np.savez(OUT + "cache/s05_scores.npz", both=both, **{k.replace("(", "_").replace(")", "_").replace("+", "P"): v for k, v in {**scores, **fus}.items()})
log("fusions", res["val_fusions_common_feeds"])

# by history length (user cos vs history control); H=0 feeds use only the user vector
Hb = [(0, 1), (1, 10), (10, 50), (50, 150), (150, 256), (256, 257)]
byH = {}
for lo, hi in Hb:
    fm = (Hva >= lo) & (Hva < hi)
    row = {"n_feeds": int(fm.sum())}
    row["cos_user"] = eval_score(scores["cos_user"], feedmask=fm & uvalid)
    if lo > 0:
        for k in ("cos_hist_mean_all", "cos_hist_rec5", "cos_seq_uhat", "dot_user_resid_seq"):
            row[k] = eval_score(scores[k], feedmask=fm & (both if "resid" in k else validf["hist"]))
    byH[f"H{lo}-{hi - 1}"] = row
    log("H", lo, hi, row)
res["val_by_H"] = byH
# by candidate category (pooled AUC within the category's rows + within-feed AUC on that subset)
byC = {}
for c in np.unique(ccat):
    rm = ccat == c
    if rm.sum() < 20000:
        continue
    byC[int(c)] = {"n_rows": int(rm.sum()), "pos_rate": float(tg[rm].mean())}
    for k in ("cos_user", "cos_hist_mean_all", "cos_hist_rec5", "dot_user_resid_seq"):
        e_ = eval_score(scores[k], rowmask=rm, feedmask=both)
        byC[int(c)][k] = [round(e_["pooled_auc"], 4), round(e_["within_feed_auc"], 4), e_["n_feeds"]]
res["val_by_category"] = byC
log("by cat", len(byC))

# retrieval test: is user_emb closer to the feed's candidate centroid than the history is?
cbar = torch.zeros(len(M), 128, device=dev); cbar.index_add_(0, fi, Cv); cbar = cbar / cbar.norm(dim=1, keepdim=True)
def cbm(q):
    qn = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9); return float((qn[torch.tensor(both, device=dev)] * cbar[torch.tensor(both, device=dev)]).sum(1).mean())
res["cos_to_candidate_centroid"] = {"user": cbm(Uv), "hist_all": cbm(Pva["all"]), "hist_rec5": cbm(Pva["rec5"]), "hist_l20": cbm(Pva["l20"]),
                                    "ridge_uhat": cbm(Uhat_ridge), "seq_uhat": cbm(uhat_s), "resid_seq_dir": cbm(resid_seq)}
# ridge: predict candidate centroid from [history pools] vs [history pools + user] (fit on train feeds' candidates, eval val)
Ttr = table("train")
Mt = ft["M"].to_numpy().astype(np.int64)[fsel]; cot = ft["cand_off"].to_numpy()[fsel]
sub = np.arange(min(len(fsel), 150_000))
rows = np.concatenate([np.arange(cot[i], cot[i] + Mt[i]) for i in sub])
segs = np.repeat(sub, Mt[sub])
cet = np.load(FL + "train/cand_emb_idx.npy", mmap_mode="r")[rows]
okr = cet >= 0
srt = np.argsort(cet[okr]); Vc = np.empty((okr.sum(), 128), np.int8); Vc[srt] = Ttr[cet[okr][srt]]
cbt = torch.zeros(len(sub), 128, device=dev); cbt.index_add_(0, torch.tensor(segs[okr], device=dev), torch.tensor(Vc, device=dev).float() / 256)
cbt = cbt / cbt.norm(dim=1, keepdim=True)
cr = {}
for nm, Ftr_, Fva_ in [("history_pools", Ftr[sub], Fva), ("history_pools+user", torch.cat([Ftr[sub], Ut[sub]], 1), torch.cat([Fva, Uv], 1)),
                        ("user_only", torch.cat([Ut[sub], torch.ones(len(sub), 1, device=dev)], 1), torch.cat([Uv, torch.ones(len(Uv), 1, device=dev)], 1))]:
    W = fit_ridge(Ftr_, cbt)
    pv = Fva_ @ W; pvn = pv / pv.norm(dim=1, keepdim=True)
    bm = torch.tensor(both, device=dev)
    cmean = cbt.mean(0)
    cr[nm] = dict(val_cos=float((pvn[bm] * cbar[bm]).sum(1).mean()), val_r2=float(1 - ((cbar[bm] - pv[bm]) ** 2).sum() / ((cbar[bm] - cmean) ** 2).sum()))
res["predict_candidate_centroid"] = cr
log("centroid", res["cos_to_candidate_centroid"], cr)
json.dump(res, open(OUT + "s05_user.json", "w"), indent=1, default=float)
log("saved")
