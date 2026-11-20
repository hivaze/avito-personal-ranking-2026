"""(2) Profile low-ranked positives vs high-ranked positives vs negatives; local (+-K rank) pairwise residual AUC."""
import os
from common_e import meta, ROOT, SCR
import numpy as np
import numba as nb
import polars as pl

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
r = np.load(f"{SCR}/out/v11_rank.npy")
loss = np.load(f"{SCR}/out/v11_loss.npy")
pf = pl.read_parquet(f"{SCR}/out/perfeed.parquet")
pct = r / np.maximum(M[fid] - 1, 1)


def ld(nm):
    for d in (f"{ROOT}/data/feats/val", f"{ROOT}/data/feats_new/val", f"{ROOT}/eda/errors/synthesis/derived/val"):
        p = f"{d}/{nm}.npy"
        if os.path.exists(p):
            return np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
    raise FileNotFoundError(nm)


hoff = f["hist_off"].to_numpy().astype(np.int64)
H = f["H"].to_numpy()
het = np.load(f"{ROOT}/data/flat/val/hist_event_type.npy").astype(np.int64)
hes = np.load(f"{ROOT}/data/flat/val/hist_event_source.npy").astype(np.int64)
hcat = np.load(f"{ROOT}/data/flat/val/hist_category_id.npy").astype(np.int64)
ccat = np.load(f"{ROOT}/data/flat/val/cand_category_id.npy").astype(np.int64)
am = ld("seq_argmax_hpos")
ok = ~np.isnan(am) & (H[fid] > 0)
ami = np.where(ok, hoff[fid] + np.nan_to_num(am).astype(np.int64), 0)
et_am = np.where(ok, het[ami], -1)
es_am = np.where(ok, hes[ami], -1)
# feed-level history shares
fh = np.repeat(np.arange(n), H)
et_share = np.zeros((n, 11))
np.add.at(et_share, (fh, het), 1)
et_share /= np.maximum(H, 1)[:, None]
es_share = np.zeros((n, 31))
np.add.at(es_share, (fh, hes), 1)
es_share /= np.maximum(H, 1)[:, None]
last_et = np.where(H > 0, het[np.minimum(hoff, len(het) - 1)], -1)

A = {}
for nm in ["maxsim32", "seq_max_all", "cos_u", "cos_h10", "cat_cnt_all", "cat_share_all", "cat_min_hpos", "it_hist_all_luo",
           "seq_cv_pop", "seq_cv_invd20", "it_tr_ctr", "it_cand_cnt_train", "f1", "f2", "f3", "f4", "f5", "ctx_nd_max", "ctx_nd_c90",
           "ctx_nd_c95", "ctx_nd_dupb_h", "feed_cat_share", "centrality", "pre2_fe", "pre_fe", "cold_item", "ag_age",
           "hist_n_uniq_cat", "seq_argmax_hpos"] + [f"cat_cnt_et{k}" for k in range(1, 11)]:
    A[nm] = ld(nm)
A["cat_in_hist"] = (A["cat_cnt_all"] > 0).astype(float)
A["cat_recent3"] = (A["cat_min_hpos"] < 3).astype(float)
A["maxsim_lt07"] = (A["maxsim32"] < 0.7).astype(float)
A["luo_zero"] = (A["it_hist_all_luo"] == 0).astype(float)
A["cvpop_zero"] = (A["seq_cv_pop"] == 0).astype(float)
A["nd95_any"] = (A["ctx_nd_c95"] > 0).astype(float)
for k in (1, 2, 3, 4, 5):
    A[f"argmax_et{k}"] = (et_am == k).astype(float)
for k in (0, 1, 2, 3):
    A[f"argmax_es{k}"] = (es_am == k).astype(float)
for k in (1, 2, 3, 4, 5, 6):
    A[f"hist_et{k}_share"] = et_share[fid, k]
for k in (0, 1, 2, 3, 4):
    A[f"hist_es{k}_share"] = es_share[fid, k]
A["last_et"] = last_et[fid].astype(float)
A["H"] = H[fid].astype(float)
A["M"] = M[fid].astype(float)
A["user0"] = pf["u0"].to_numpy()[fid].astype(float)
A["new_user"] = pf["cold_user"].to_numpy()[fid].astype(float)
# same-category candidates in the feed
key = fid * 64 + ccat
_, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
A["n_samecat_cand"] = cnt[inv].astype(float)
np.savez_compressed(f"{SCR}/out/attrs.npz", **{k: v.astype(np.float32) for k, v in A.items()}, et_am=et_am, es_am=es_am)

pos = y > 0
G = {"pos_top10": pos & (pct <= 0.1), "pos_10_50": pos & (pct > 0.1) & (pct < 0.5), "pos_bot50": pos & (pct >= 0.5),
     "neg_all": ~pos, "neg_top10": (~pos) & (pct <= 0.1)}
print("group sizes:", {k: int(v.sum()) for k, v in G.items()})

# local pairwise residual AUC: positive vs negatives within +-K ranks in the same feed
order = np.lexsort((r, fid))  # rows sorted by feed then rank


@nb.njit(parallel=False)
def local_pairs(order, off, M, y, fidx, K):
    pa = []
    pb = []
    for f_ in range(len(M)):
        o = off[f_]
        m = M[f_]
        for i in range(m):
            a = order[o + i]
            if y[a] <= 0:
                continue
            for j in range(max(0, i - K), min(m, i + K + 1)):
                b = order[o + j]
                if y[b] > 0:
                    continue
                pa.append(a)
                pb.append(b)
    return np.array(pa), np.array(pb)


pa, pb = local_pairs(order, off, M, y, fid, 5)
print("local pairs (K=5):", len(pa))
res = []
for k, x in A.items():
    xa, xb = x[pa], x[pb]
    okp = ~(np.isnan(xa) | np.isnan(xb))
    gt = (xa[okp] > xb[okp]).mean()
    eq = (xa[okp] == xb[okp]).mean()
    lauc = gt + 0.5 * eq
    # feed-clustered SE of the local AUC
    d = np.where(xa > xb, 1.0, np.where(xa == xb, 0.5, 0.0)) - 0.5
    d = np.where(okp, d, 0)
    fs = np.bincount(fid[pa], d, n)
    fc = np.bincount(fid[pa], okp.astype(float), n)
    mm = fc > 0
    se = np.sqrt(((fs[mm] - fc[mm] * (lauc - 0.5)) ** 2).sum()) / fc.sum()
    row = {"attr": k}
    for g, m in G.items():
        row[g] = np.nanmean(x[m])
    row["local_auc"] = lauc
    row["local_z"] = (lauc - 0.5) / se
    row["frac_eq"] = eq
    res.append(row)
T = pl.DataFrame(res)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(4)
print(T)
T.write_csv(f"<repo>/solvers/out/errors/s02_profile.csv")

# loss share by attribute bins for positives
print("\nloss share by positive attribute (share of positives / share of loss / mean pct rank / top10 rate):")
L = loss[pos]
for nm, bins in (("maxsim32", [0.6, 0.7, 0.8, 0.9, 0.95]), ("cat_min_hpos", [1, 3, 10, 50, 998]), ("it_hist_all_luo", [1, 3, 10, 100]),
                 ("ctx_nd_c90", [1, 3, 10]), ("f4", [0.15, 0.3, 0.5, 0.7]), ("n_samecat_cand", [2, 5, 10, 30]),
                 ("pre2_fe", [5, 10, 15, 20]), ("H", [1, 11, 51, 129, 256])):
    x = A[nm][pos]
    b = np.digitize(np.nan_to_num(x, nan=-1), bins)
    edges = ["<" + str(bins[0])] + [f"[{bins[i]},{bins[i + 1]})" for i in range(len(bins) - 1)] + [">=" + str(bins[-1])]
    print(f" {nm}: " + "; ".join(f"{edges[i]} {np.mean(b == i):.3f}/{L[b == i].sum() / L.sum():.3f}/{pct[pos][b == i].mean():.3f}/{(pct[pos][b == i] <= 0.1).mean():.3f}" for i in range(len(edges)) if (b == i).any()))
for nm in ("et_am", "es_am"):
    x = {"et_am": et_am, "es_am": es_am}[nm][pos]
    print(f" {nm}: " + "; ".join(f"{v}: {np.mean(x == v):.3f}/{L[x == v].sum() / L.sum():.3f}/{pct[pos][x == v].mean():.3f}" for v in np.unique(x) if np.mean(x == v) > 0.005))
# positive rate by argmax event type among all candidates, and by cat_cnt_etk>0
print("\nP(pos) lift by argmax-history-event type (all candidates):", {int(v): round(float(y[et_am == v].mean() / y.mean()), 3) for v in np.unique(et_am) if (et_am == v).mean() > 0.005})
print("P(pos) lift by argmax-history-event source:", {int(v): round(float(y[es_am == v].mean() / y.mean()), 3) for v in np.unique(es_am) if (es_am == v).mean() > 0.005})
print("P(pos) lift if candidate category has event type k in history:", {k: round(float(y[A[f'cat_cnt_et{k}'] > 0].mean() / y.mean()), 3) for k in range(1, 11)})
