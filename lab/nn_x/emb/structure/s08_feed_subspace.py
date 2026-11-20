"""(3d) The 'feed-level' (category-independent) part of the item vector: is it a discrete attribute such as location?
VAL candidates + history. Tests:
 a) per-feed share of candidates in the minor mode of the bimodal dims (6: mode at +21; 96: mode at +65).
    A location-like flag gives a U-shaped per-feed share (near 0 or near 1); a content attribute gives a binomial-like share.
 b) PCA of feed means of category-residualised candidate vectors: spectrum and multimodality of feed-level PC scores.
 c) same user across splits (val feed vs the user's last train feed): correlation of the feed-level projection.
 d) pairs posted in the same 128-s slot (same-seller proxy) vs random same-category pairs: agreement in the feed-level subspace.
 e) does the feed-level subspace carry within-feed ranking signal (cos(candidate, history) restricted to the subspace)?
 f) within-feed near-duplicate pairs: P(both clicked | one clicked) vs random same-feed pairs."""
import sys, json, numpy as np, polars as pl
from scipy import stats
sys.path.insert(0, "<repo>/feats_v2/life_")
from post_time import age_hours

OUT = "<repo>/nn_x/emb/structure/"
V = "<repo>/data/flat/val/"
NV = 2_516_711
E = np.fromfile("<repo>/data/competition/val/item_embeddings/data.memmap", dtype=np.int8).reshape(NV, 128)
feeds = pl.read_parquet(V + "feeds.parquet"); M = feeds["M"].to_numpy().astype(np.int64); off = feeds["cand_off"].to_numpy(); F = len(M)
ce = np.load(V + "cand_emb_idx.npy"); cc = np.load(V + "cand_category_id.npy").astype(np.int64); cf = np.load(V + "cand_fidx.npy").astype(np.int64)
tg = np.load(V + "cand_target.npy").astype(np.float64)
he = np.load(V + "hist_emb_idx.npy"); hc = np.load(V + "hist_category_id.npy").astype(np.int64); hf = np.load(V + "hist_fidx.npy").astype(np.int64)
hp = np.load(V + "hist_hpos.npy")
res = {}
ok = ce >= 0
Q = E[np.maximum(ce, 0)].astype(np.float32)
# a) minor modes
for d, thr in [(6, 12), (96, 40)]:
    inm = (Q[:, d] > thr) & ok
    share_item = float(inm[ok].mean())
    sh = np.bincount(cf[ok], weights=inm[ok], minlength=F) / np.maximum(np.bincount(cf[ok], minlength=F), 1)
    big = M >= 50
    # binomial null with the same overall rate
    p = share_item; null = np.random.default_rng(0).binomial(M[big], p) / M[big]
    hist_sh = np.histogram(sh[big], bins=10, range=(0, 1))[0] / big.sum()
    hist_null = np.histogram(null, bins=10, range=(0, 1))[0] / big.sum()
    # is the mode membership category-driven? share by top categories
    res[f"dim{d}_minor_mode"] = dict(threshold=thr, item_share=share_item, feed_share_sd=float(sh[big].std()), binomial_null_sd=float(null.std()),
                                     feed_share_hist_10bins=hist_sh.round(3).tolist(), null_hist_10bins=hist_null.round(3).tolist(),
                                     pos_rate_in_mode=float(tg[inm].mean()), pos_rate_out=float(tg[ok & ~inm].mean()))
    # same feed: history share vs candidate share
    hok = he >= 0
    hin = np.zeros(len(he), bool); hin[hok] = E[he[hok], d] > thr
    shh = np.bincount(hf[hok], weights=hin[hok], minlength=F) / np.maximum(np.bincount(hf[hok], minlength=F), 1)
    both = big & (np.bincount(hf[hok], minlength=F) >= 20)
    res[f"dim{d}_minor_mode"]["corr_feed_share_hist_vs_cand"] = float(np.corrcoef(sh[both], shh[both])[0, 1])
    print(d, res[f"dim{d}_minor_mode"], flush=True)

# b) feed-level PCA of category-residualised candidate vectors
K = cc.max() + 1
nk = np.bincount(cc[ok], minlength=K).astype(np.float64)
CM = np.stack([np.bincount(cc[ok], weights=Q[ok, d], minlength=K) for d in range(128)], 1) / np.maximum(nk, 1)[:, None]
R = np.where(ok[:, None], Q - CM[cc], 0)
nf = np.bincount(cf[ok], minlength=F).astype(np.float64)
FMn = np.stack([np.bincount(cf[ok], weights=R[ok, d], minlength=F) for d in range(128)], 1) / np.maximum(nf, 1)[:, None]
big = nf >= 50
G = np.cov(FMn[big].T); ev, evec = np.linalg.eigh(G); ev, evec = ev[::-1], evec[:, ::-1]
Wtot = np.cov(R[ok][::5].T); tot = np.trace(Wtot)
res["feedlevel_pca"] = dict(between_feed_var_share_of_catresid_total=float(np.trace(G) / tot), top_eig_share=(ev[:10] / ev.sum()).round(4).tolist(),
                            n_eig_for_80pct=int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.8) + 1))
sc = FMn[big] @ evec[:, :4]
res["feedlevel_pca"]["feed_score_bimodality_coef_PC1_4"] = [float((stats.skew(sc[:, k]) ** 2 + 1) / (stats.kurtosis(sc[:, k]) + 3)) for k in range(4)]
res["feedlevel_pca"]["feed_score_PC1_hist20"] = np.histogram(sc[:, 0], bins=20)[0].tolist()
# loadings of PC1 on the skewed dims
res["feedlevel_pca"]["PC1_top_loading_dims"] = [(int(i), round(float(evec[i, 0]), 3)) for i in np.argsort(-np.abs(evec[:, 0]))[:8]]
res["feedlevel_pca"]["PC2_top_loading_dims"] = [(int(i), round(float(evec[i, 1]), 3)) for i in np.argsort(-np.abs(evec[:, 1]))[:8]]
print(res["feedlevel_pca"], flush=True)
Pf = evec[:, :16]  # feed-level subspace (top 16)
np.save(OUT + "cache/feedlevel_subspace16.npy", Pf); np.save(OUT + "cache/val_cat_means.npy", CM)

# c) same user, val feed vs that user's train feeds (projection of the history mean onto the feed-level subspace, cat-residualised)
hok = he >= 0
HR = E[he[hok]].astype(np.float32) - CM[np.minimum(hc[hok], K - 1)]
nh = np.bincount(hf[hok], minlength=F).astype(np.float64)
HMn = np.stack([np.bincount(hf[hok], weights=HR[:, d], minlength=F) for d in range(128)], 1) / np.maximum(nh, 1)[:, None]
# train feeds of the same users
ft = pl.read_parquet("<repo>/data/flat/train/feeds.parquet")
uv = feeds["user_id"].to_numpy(); ut = ft["user_id"].to_numpy()
last_tr = pl.DataFrame({"u": ut, "i": np.arange(len(ut)), "t": ft["timestamp"].to_numpy()}).filter(pl.col("u") > 0).sort("t").group_by("u").agg(pl.col("i").last())
mp = dict(zip(last_tr["u"].to_list(), last_tr["i"].to_list()))
pairs = [(i, mp[u]) for i, u in enumerate(uv) if u > 0 and u in mp and nh[i] >= 20]
vi = np.array([p[0] for p in pairs]); ti = np.array([p[1] for p in pairs])
# train history means for those train feeds (train table)
NT = 81_557_944
ET = np.memmap("<repo>/data/competition/train/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NT, 128))
Ht = ft["H"].to_numpy()[ti].astype(np.int64); hot = ft["hist_off"].to_numpy()[ti]
rows = np.repeat(hot, Ht) + (np.arange(Ht.sum()) - np.repeat(np.cumsum(Ht) - Ht, Ht)); seg = np.repeat(np.arange(len(ti)), Ht)
the = np.load("<repo>/data/flat/train/hist_emb_idx.npy", mmap_mode="r")[rows]; thc = np.load("<repo>/data/flat/train/hist_category_id.npy", mmap_mode="r")[rows].astype(np.int64)
okt = the >= 0
srt = np.argsort(the[okt]); TV = np.empty((okt.sum(), 128), np.int8); TV[srt] = ET[the[okt][srt]]
TR = TV.astype(np.float32) - CM[np.minimum(thc[okt], K - 1)]
nt = np.bincount(seg[okt], minlength=len(ti)).astype(np.float64)
TMn = np.stack([np.bincount(seg[okt], weights=TR[:, d], minlength=len(ti)) for d in range(128)], 1) / np.maximum(nt, 1)[:, None]
okp = nt >= 20
a = HMn[vi][okp] @ Pf; b = TMn[okp] @ Pf
rnd = np.random.default_rng(0).permutation(okp.sum())
def mcos(x, y):
    return float(((x / np.linalg.norm(x, axis=1, keepdims=True)) * (y / np.linalg.norm(y, axis=1, keepdims=True))).sum(1).mean())
# complement (content) subspace for comparison
Pc = evec[:, 16:]
a2 = HMn[vi][okp] @ Pc; b2 = TMn[okp] @ Pc
res["same_user_val_vs_train"] = dict(n_users=int(okp.sum()), feedsub_cos_same_user=mcos(a, b), feedsub_cos_random_user=mcos(a, b[rnd]),
                                     restsub_cos_same_user=mcos(a2, b2), restsub_cos_random_user=mcos(a2, b2[rnd]),
                                     feedsub_PC1_corr=float(np.corrcoef(a[:, 0], b[:, 0])[0, 1]), restsub_PC1_corr=float(np.corrcoef(a2[:, 0], b2[:, 0])[0, 1]))
print(res["same_user_val_vs_train"], flush=True)

# d) same-128s-slot pairs (same seller proxy) vs random same-category pairs: agreement in the feed-level subspace (cat-residualised items)
knots = np.load("<repo>/data/feats/life_f4_age_knots.npy")
ts = feeds["timestamp"].to_numpy().astype(np.float64)
tp = ts[cf] - age_hours(np.load(V + "cand_f4.npy"), knots) * 3600.0
slot = np.where(np.isfinite(tp) & ok, np.round(tp / 128), -1).astype(np.int64)
df = pl.DataFrame({"e": ce, "slot": slot, "c": cc}).filter(pl.col("slot") >= 0).unique(["e"])
g = df.group_by("slot").agg(pl.col("e"), pl.col("c")).filter(pl.col("e").list.len().is_between(2, 5))
pa, pb, samec = [], [], []
for es, cs in zip(g["e"].to_list(), g["c"].to_list()):
    pa.append(es[0]); pb.append(es[1]); samec.append(cs[0] == cs[1])
pa, pb, samec = np.array(pa), np.array(pb), np.array(samec)
cat_e = np.zeros(NV, np.int64); cat_e[ce[ok]] = cc[ok]
def resid(e):
    return E[e].astype(np.float32) - CM[cat_e[e]]
ra, rb = resid(pa), resid(pb)
rng = np.random.default_rng(1)
items = np.unique(ce[ok]); xa = rng.choice(items, 600000); xb = rng.choice(items, 600000); m = (cat_e[xa] == cat_e[xb]) & (xa != xb)
xa, xb = xa[m][:100000], xb[m][:100000]
qa, qb = resid(xa), resid(xb)
res["same_slot_pairs"] = dict(n_pairs=int(len(pa)), same_cat_share=float(samec.mean()),
                              feedsub_cos_same_slot=mcos(ra @ Pf, rb @ Pf), feedsub_cos_random_samecat=mcos(qa @ Pf, qb @ Pf),
                              restsub_cos_same_slot=mcos(ra @ Pc, rb @ Pc), restsub_cos_random_samecat=mcos(qa @ Pc, qb @ Pc),
                              full_cos_same_slot=mcos(E[pa].astype(np.float32), E[pb].astype(np.float32)), full_cos_random_samecat=mcos(E[xa].astype(np.float32), E[xb].astype(np.float32)))
print(res["same_slot_pairs"], flush=True)

# e) ranking signal of the feed-level subspace: cos(candidate, last-20 history mean) in the subspace vs in the complement (cat-residualised)
l20 = hok & (hp < 20)
HR20 = np.stack([np.bincount(hf[l20], weights=(E[he[l20], d].astype(np.float32) - CM[np.minimum(hc[l20], K - 1), d]), minlength=F) for d in range(128)], 1)
def within_auc(s):
    rm = ok & np.isfinite(s)
    o = np.lexsort((s, cf)); y = tg[o]; ff = cf[o]; r_ = rm[o]; y = y[r_]; ff = ff[r_]
    rank = np.arange(len(y)) - np.searchsorted(ff, ff)
    npos = np.bincount(ff, weights=y, minlength=F); nn = np.bincount(ff, minlength=F) - npos
    sr = np.bincount(ff, weights=(rank + 1) * y, minlength=F); okf = (npos > 0) & (nn > 0)
    return float(((sr[okf] - npos[okf] * (npos[okf] + 1) / 2) / (npos[okf] * nn[okf])).mean())
def cos_sub(P):
    a = R @ P; b = HR20[cf] @ P
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
Xn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
Hraw = np.stack([np.bincount(hf[l20], weights=E[he[l20], d].astype(np.float32), minlength=F) for d in range(128)], 1)
Hraw /= np.linalg.norm(Hraw, axis=1, keepdims=True) + 1e-9
res["ranking_signal_within_feed_auc"] = dict(raw_cos_last20=within_auc((Xn * Hraw[cf]).sum(1)), catresid_full=within_auc(cos_sub(np.eye(128))),
                                             catresid_feedsub16=within_auc(cos_sub(Pf)), catresid_rest112=within_auc(cos_sub(Pc)))
print(res["ranking_signal_within_feed_auc"], flush=True)

# f) near-duplicate pairs inside a feed vs random same-feed pairs: P(both clicked | at least one clicked)
Xv = np.where(ok[:, None], Xn, 0)
nd_both = nd_one = rd_both = rd_one = sc_both = sc_one = 0
for f in range(F):
    s, e_ = off[f], off[f] + M[f]
    y = tg[s:e_]
    if y.sum() == 0:
        continue
    Z = Xv[s:e_]; S = Z @ Z.T
    iu, ju = np.triu_indices(len(y), 1)
    v = S[iu, ju]; yy = y[iu] + y[ju]
    sameitem = ce[s:e_][iu] == ce[s:e_][ju]
    nd = (v > 0.98) & ~sameitem
    samecat = (cc[s:e_][iu] == cc[s:e_][ju]) & ~nd & ~sameitem
    nd_one += (yy[nd] >= 1).sum(); nd_both += (yy[nd] == 2).sum()
    rd_one += (yy >= 1).sum(); rd_both += (yy == 2).sum()
    sc_one += (yy[samecat] >= 1).sum(); sc_both += (yy[samecat] == 2).sum()
res["feed_pair_coclick"] = dict(neardup_P_both_given_one=nd_both / max(nd_one, 1), samecat_nondup_P_both_given_one=sc_both / max(sc_one, 1),
                                all_pairs_P_both_given_one=rd_both / max(rd_one, 1), neardup_pairs_with_a_click=int(nd_one))
print(res["feed_pair_coclick"], flush=True)
json.dump(res, open(OUT + "s08_feed_subspace.json", "w"), indent=1, default=float)
