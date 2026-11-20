"""O3b: is the train->val vector change item-level (edits) or shared by listings with identical inputs?
Train bit-identical pairs (same-seller batch listings, o02b) whose both items are also in the val table:
P(still bit-identical in val) vs P(each member changed). Independent item-level edits at the base change rate
would split ~1/3 of changed pairs; a shared cause (seller-level input such as seller rating / company flag, a global
re-embed or model refresh) keeps them identical. Also: the systematic mean drift direction vs item age.
Outputs out/o03b.json."""
import json, sys, time
import numpy as np, polars as pl

sys.path.insert(0, "<repo>/feats_v2/life_")
from post_time import age_hours

T0 = time.time()
R = "<repo>"; OUT = f"{R}/nn_x/emb/origin/out"
NV, NT, NS = 2516711, 81557944, 4710948
res = {}


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def l(split, n):
    return np.load(f"{R}/data/flat/{split}/{n}.npy", mmap_mode="r")


def row_item(split, n):
    it = np.full(n, -1, np.int64)
    for kind in ("cand", "hist"):
        E = l(split, f"{kind}_emb_idx"); I = l(split, f"{kind}_item_id")
        for s in range(0, len(E), 100_000_000):
            e = np.asarray(E[s:s + 100_000_000]); i = np.asarray(I[s:s + 100_000_000]); m = e >= 0
            it[e[m]] = i[m]
    return it


MAXI = 111_400_000
it_tr = row_item("train", NT); log("train rows mapped", int((it_tr >= 0).sum()))
it_v = row_item("val", NV); it_s = row_item("test", NS)
vrow = np.full(MAXI, -1, np.int64); vrow[it_v] = np.arange(NV)
srow = np.full(MAXI, -1, np.int64); srow[it_s] = np.arange(NS)
Ht = np.load(f"{OUT}/hash_train.npy")[:, 0]; Hv = np.load(f"{OUT}/hash_val.npy")[:, 0]; Hs = np.load(f"{OUT}/hash_test.npy")[:, 0]
o = np.argsort(Ht, kind="stable"); hs = Ht[o]
st = np.r_[0, np.nonzero(hs[1:] != hs[:-1])[0] + 1]; en = np.r_[st[1:], len(hs)]; sz = en - st
g = np.nonzero(sz > 1)[0]
A, B = [], []
for k in range(1, 6):
    gg = g[sz[g] > k]; A.append(o[st[gg]]); B.append(o[st[gg] + k])
A = np.concatenate(A); B = np.concatenate(B)
for tag, rowmap, Hx in (("val", vrow, Hv), ("test", srow, Hs)):
    ra = rowmap[it_tr[A]]; rb = rowmap[it_tr[B]]
    m = (ra >= 0) & (rb >= 0)
    a_ch = Hx[ra[m]] != Ht[A[m]]; b_ch = Hx[rb[m]] != Ht[B[m]]
    same = Hx[ra[m]] == Hx[rb[m]]
    any_ch = a_ch | b_ch
    # base change rate for single items in this split (all shared items)
    sh = rowmap[it_tr[it_tr >= 0]] if False else None
    r = {"pairs_both_present": int(m.sum()), "share_a_changed": float(a_ch.mean()), "share_b_changed": float(b_ch.mean()),
         "share_both_changed": float((a_ch & b_ch).mean()), "share_exactly_one_changed": float((a_ch ^ b_ch).mean()),
         "share_still_identical": float(same.mean()),
         "share_still_identical_given_any_changed": float(same[any_ch].mean()) if any_ch.any() else None,
         "independence_pred_exactly_one": float(2 * a_ch.mean() * (1 - a_ch.mean()))}
    res[f"dup_pairs_{tag}"] = r
    log(tag, r)
# systematic drift direction vs age (val, shared items)
trow = np.full(MAXI, -1, np.int64); trow[it_tr[it_tr >= 0]] = np.nonzero(it_tr >= 0)[0]
S = np.nonzero(trow[it_v] >= 0)[0]; TR = trow[it_v[S]]
mv = np.memmap(f"{R}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NV, 128))
mt = np.memmap(f"{R}/data/competition/train/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NT, 128))
xv = np.asarray(mv[S]).astype(np.float32); oo = np.argsort(TR); xt = np.empty_like(xv); xt[oo] = np.asarray(mt[TR[oo]]).astype(np.float32)
ch = ~(xv == xt).all(1)
d = xv[ch] - xt[ch]; d /= np.linalg.norm(d, axis=1, keepdims=True)
dbar = d.mean(0); res["mean_unit_drift_norm"] = float(np.linalg.norm(dbar)); dbar /= np.linalg.norm(dbar)
proj = d @ dbar
res["share_drift_positive_on_mean_dir"] = float((proj > 0).mean())
# age of the item at its first val appearance
ce = np.asarray(l("val", "cand_emb_idx")); ok = ce >= 0
fts = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet")["timestamp"].to_numpy().astype(np.float64)
ct = fts[np.asarray(l("val", "cand_fidx"))]
ag = age_hours(np.asarray(l("val", "cand_f4")), np.load(f"{R}/data/feats/life_f4_age_knots.npy"))
tpost = np.full(NV, np.nan); okk = ok & np.isfinite(ag); tpost[ce[okk]] = ct[okk] - ag[okk] * 3600
tp = tpost[S][ch]; fin = np.isfinite(tp)
q = np.quantile(tp[fin], np.linspace(0, 1, 6))
res["mean_dir_projection_by_posting_time_quintile"] = [[float(q[i] / 86400), float(q[i + 1] / 86400),
                                                         float(proj[fin][(tp[fin] >= q[i]) & (tp[fin] <= q[i + 1])].mean())] for i in range(5)]
# projection of static vectors onto the drift direction vs posting time (is dbar an "age" axis?)
allv = np.asarray(mv).astype(np.float32); allv /= np.linalg.norm(allv, axis=1, keepdims=True)
pv = allv @ dbar
fin2 = np.isfinite(tpost)
res["spearman_val_vectors_on_drift_dir_vs_tpost"] = float(pl.DataFrame({"a": pv[fin2], "b": tpost[fin2]}).select(pl.corr("a", "b", method="spearman")).item())
mu_item = allv.mean(0); mu_item /= np.linalg.norm(mu_item)
res["cos_drift_dir_mean_item"] = float(dbar @ mu_item)
log({k: v for k, v in res.items() if not k.startswith("dup")})
json.dump(res, open(f"{OUT}/o03b.json", "w"), indent=1)
log("done")
