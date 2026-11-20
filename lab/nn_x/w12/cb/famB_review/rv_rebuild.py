"""famB_review: rebuild of key B3 columns for a random sample of TRAIN (xm) rows and VAL rows from RAW data with own code
(no import of cbx / fb_common / any famB builder). Compares with famB/feat/{a,b,c}_{xm,val}.npy.
  block c: kn_cos1, kn_cos20 (label-free), kn_ex20, kn_coec20 (neighbour clicks, other user folds, earlier days)
  block a: ua_ns, ua_npos, ua_cat_{imp,pos,coec}, ua_c1024_{imp,pos,coec}, ua_item_{imp,pos}, ue_k20_coec, ua_dt_last (the user's strictly earlier non-holdout train feeds)
  block b: tr_cat_base, tr_cat_l1, tr_cat_l20m (category transition lifts, other user folds, earlier days)
usage: python -u rv_rebuild.py [n_random_rows=2500]
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import polars as pl

ROOT = "<repo>"; F = f"{ROOT}/data/flat"; CLU = f"{ROOT}/data/clu_work"; FEAT = f"{ROOT}/nn_x/w12/cb/famB/feat"
HERE = f"{ROOT}/nn_x/w12/cb/famB_review"
NR = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ sets (own derivation)
ft = pl.read_parquet(f"{F}/train/feeds.parquet"); fv = pl.read_parquet(f"{F}/val/feeds.parquet")
Mt = ft["M"].to_numpy().astype(np.int64); offt = np.r_[0, np.cumsum(Mt)[:-1]]; uidt = ft["user_id"].to_numpy().astype(np.int64); tst = ft["timestamp"].to_numpy().astype(np.int64)
Ht = ft["H"].to_numpy().astype(np.int64); hofft = np.r_[0, np.cumsum(Ht)[:-1]]
Mv = fv["M"].to_numpy().astype(np.int64); offv = np.r_[0, np.cumsum(Mv)[:-1]]; uidv = fv["user_id"].to_numpy().astype(np.int64); tsv = fv["timestamp"].to_numpy().astype(np.int64)
Hv = fv["H"].to_numpy().astype(np.int64); hoffv = np.r_[0, np.cumsum(Hv)[:-1]]
NF = len(Mt)
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32)
p = np.nan_to_num(pr, nan=0.0); p[Mt < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (Mt >= 50))[0], 30000, replace=False))
xm = np.setdiff1d(np.where((p > 0.5) & (Mt >= 50))[0], hold)
is_hold = np.zeros(NF, bool); is_hold[hold] = True
xm_off = np.r_[0, np.cumsum(Mt[xm])[:-1]]
log("xm feeds", len(xm), "rows", int(Mt[xm].sum()), "| val feeds", len(Mv), "rows", int(Mv.sum()))
day_t = (tst + 3 * 3600) // 86400 - 20702
assert day_t.min() == 0 and day_t.max() == 14


def xfold(uid, fidx):
    h = (uid.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(4294967296)
    return np.where(uid == 0, fidx % 5, (h % np.uint64(5)).astype(np.int64)).astype(np.int64)


fold_t = xfold(uidt, np.arange(NF)); fold_v = xfold(uidv, np.arange(len(Mv)))

# ------------------------------------------------------------------ study columns + sample
A = {s: np.load(f"{FEAT}/a_{s}.npy", mmap_mode="r") for s in ("xm", "val")}
B = {s: np.load(f"{FEAT}/b_{s}.npy", mmap_mode="r") for s in ("xm", "val")}
C = {s: np.load(f"{FEAT}/c_{s}.npy", mmap_mode="r") for s in ("xm", "val")}
an, bn, cn = (json.load(open(f"{FEAT}/{b}_names.json")) for b in "abc")
rg = np.random.default_rng(777)
sample = {}
for s, n_rows in (("xm", int(Mt[xm].sum())), ("val", int(Mv.sum()))):
    r1 = rg.choice(n_rows, NR, replace=False)
    cov = np.where(~np.isnan(np.asarray(A[s][:, 0])))[0]             # rows the study says have a user source (block a): extra sample there
    r2 = rg.choice(cov, NR, replace=False)
    sample[s] = np.unique(np.r_[r1, r2])
    log(s, "sample rows", len(sample[s]), "| study block-a coverage of rows", round(len(cov) / n_rows, 4))

# raw train row arrays
item = np.load(f"{F}/train/cand_item_id.npy"); y = np.load(f"{F}/train/cand_target.npy"); pos = np.load(f"{F}/train/cand_pos.npy")
cat = np.load(f"{F}/train/cand_category_id.npy"); ei = np.load(f"{F}/train/cand_emb_idx.npy")
cl4096 = np.load(f"{CLU}/cand_cl_4096_train.npy"); cl1024 = np.load(f"{CLU}/cand_cl_1024_train.npy")
NROW = len(item)
row_hold = np.repeat(is_hold, Mt)
row_fid = np.repeat(np.arange(NF, dtype=np.int32), Mt)
log("raw train arrays loaded", NROW)
# position prior (own): non-holdout train rows
nh = ~row_hold
cntp = np.bincount(pos[nh], minlength=512); clp = np.bincount(pos[nh], weights=y[nh], minlength=512)
pc = (clp + 0.0159 * 50) / (cntp + 50)
pc_study = np.load(f"{FEAT}/pos_prior.npy")
log("position prior: max abs diff to the study's", float(np.abs(pc - pc_study).max()))
ex = pc[pos]
r0g = float(y[nh].sum() / ex[nh].sum())
log("global COEC r0", r0g)
embT = np.memmap(f"{ROOT}/data/competition/train/item_embeddings/data.memmap", np.int8, "r").reshape(-1, 128)
embV = np.memmap(f"{ROOT}/data/competition/val/item_embeddings/data.memmap", np.int8, "r").reshape(-1, 128)


def emb_rows(tab, idx):
    out = np.zeros((len(idx), 128), np.float64); k = idx >= 0
    out[k] = tab[idx[k]]
    return out


# target descriptors of the sampled rows
T = {}
for s in ("xm", "val"):
    r = sample[s]
    if s == "xm":
        fpos = np.searchsorted(xm_off, r, "right") - 1; fidx = xm[fpos]; flat = offt[fidx] + (r - xm_off[fpos])
        T[s] = dict(r=r, fidx=fidx, uid=uidt[fidx], ts=tst[fidx], fold=fold_t[fidx], cut=day_t[fidx], item=item[flat].astype(np.int64), cat=cat[flat].astype(np.int64),
                    c4096=cl4096[flat].astype(np.int64), c1024=cl1024[flat].astype(np.int64), emb=emb_rows(embT, ei[flat]), hoff=hofft[fidx], H=Ht[fidx])
    else:
        fidx = np.searchsorted(offv, r, "right") - 1
        g = lambda n: np.load(f"{F}/val/{n}.npy", mmap_mode="r")
        T[s] = dict(r=r, fidx=fidx, uid=uidv[fidx], ts=tsv[fidx], fold=fold_v[fidx], cut=np.full(len(r), 15), item=np.asarray(g("cand_item_id")[r]).astype(np.int64),
                    cat=np.asarray(g("cand_category_id")[r]).astype(np.int64), c4096=np.asarray(np.load(f"{CLU}/cand_cl_4096_val.npy", mmap_mode="r")[r]).astype(np.int64),
                    c1024=np.asarray(np.load(f"{CLU}/cand_cl_1024_val.npy", mmap_mode="r")[r]).astype(np.int64), emb=emb_rows(embV, np.asarray(g("cand_emb_idx")[r])), hoff=hoffv[fidx], H=Hv[fidx])
res = {}


def cmp(tag, mine, theirs, tol=2e-3):
    mine = np.asarray(mine, np.float64); theirs = np.asarray(theirs, np.float64)
    nm, nt = np.isnan(mine), np.isnan(theirs)
    both = ~nm & ~nt
    d = np.abs(mine[both] - theirs[both]); rel = d / np.maximum(1.0, np.abs(theirs[both]))
    out = dict(n=int(len(mine)), nan_mine=int(nm.sum()), nan_study=int(nt.sum()), nan_mismatch=int((nm != nt).sum()), n_both=int(both.sum()),
               max_abs=float(d.max()) if len(d) else 0.0, n_off=int((rel > tol).sum()), corr=float(np.corrcoef(mine[both], theirs[both])[0, 1]) if both.sum() > 2 and mine[both].std() > 0 else float("nan"))
    res[tag] = out
    log(f"CMP {tag}: n {out['n']} NaN mine/study {out['nan_mine']}/{out['nan_study']} (mismatch {out['nan_mismatch']}) | both {out['n_both']}: max abs diff {out['max_abs']:.3g}, "
        f"rows off by > {tol:g} rel: {out['n_off']}, corr {out['corr']:.6f}")
    return out


# ================================================================== BLOCK C
cnt = np.bincount(item[nh]); nmax = int(item.max()) + 1; cnt = np.r_[cnt, np.zeros(nmax - len(cnt), np.int64)]
emb_of = np.full(nmax, -1, np.int64); emb_of[item] = ei
cl_of = np.full(nmax, -1, np.int64); cl_of[item] = cl4096
cand = np.where((cnt >= 5) & (emb_of >= 0) & (cl_of >= 0))[0]
o = np.lexsort((cand, -cnt[cand], cl_of[cand])); cand = cand[o]; cc = cl_of[cand]
start = np.searchsorted(cc, np.arange(4097)); rank = np.arange(len(cand)) - start[cc]
idx_item = cand[rank < 1500]; icl = cc[rank < 1500]; cptr = np.searchsorted(icl, np.arange(4097))
log("block c index (own):", len(idx_item), "items of", int((cnt > 0).sum()), "non-holdout train items; impression share", round(float(cnt[idx_item].sum() / cnt.sum()), 4))
neigh = {}
for s in ("xm", "val"):
    t = T[s]; n = len(t["r"]); cos1 = np.full(n, np.nan); cos20 = np.full(n, np.nan); nb_items = np.full((n, 20), -1, np.int64); nb_cos = np.zeros((n, 20))
    for i in range(n):
        c = t["c4096"][i]
        if c < 0 or cptr[c + 1] == cptr[c]:
            continue
        its = idx_item[cptr[c]:cptr[c + 1]]; k = its != t["item"][i]; its = its[k]
        if len(its) == 0:
            continue
        E = np.asarray(embT[emb_of[its]], np.float64); v = t["emb"][i]; nv = np.sqrt((v * v).sum())
        cs = (E @ v) / (nv * np.sqrt((E * E).sum(1))) if nv > 0 else np.zeros(len(its))
        top = np.argsort(-cs, kind="stable")[:20]
        cos1[i] = cs[top[0]]; cos20[i] = cs[top].mean(); nb_items[i, :len(top)] = its[top]; nb_cos[i, :len(top)] = cs[top]
    neigh[s] = (nb_items, nb_cos)
    cmp(f"c/{s}/kn_cos1", cos1, np.asarray(C[s][t["r"]][:, cn.index("kn_cos1")]))
    cmp(f"c/{s}/kn_cos20", cos20, np.asarray(C[s][t["r"]][:, cn.index("kn_cos20")]))
# neighbour click statistics: P, E per (neighbour item, fold, day) from non-holdout train rows
need = np.unique(np.concatenate([neigh[s][0].ravel() for s in neigh])); need = need[need >= 0]
slot = np.full(nmax, -1, np.int64); slot[need] = np.arange(len(need))
rs = np.where((slot[item] >= 0) & nh)[0]
PE = np.zeros((len(need), 5, 15, 2))
np.add.at(PE, (slot[item[rs]], fold_t[row_fid[rs]], day_t[row_fid[rs]], 0), y[rs]); np.add.at(PE, (slot[item[rs]], fold_t[row_fid[rs]], day_t[row_fid[rs]], 1), ex[rs])
log("block c: neighbour items", len(need), "source rows", len(rs))
for s in ("xm", "val"):
    t = T[s]; n = len(t["r"]); nb_items, _ = neigh[s]; e20 = np.full(n, np.nan); coec20 = np.full(n, np.nan)
    for i in range(n):
        its = nb_items[i]; its = its[its >= 0]
        if len(its) == 0 or t["cut"][i] == 0:
            continue
        m = np.ones(5, bool); m[t["fold"][i]] = False
        sub = PE[slot[its]][:, m][:, :, :t["cut"][i]].sum((0, 1, 2))
        e20[i] = sub[1]; coec20[i] = (sub[0] + 2.0) / (sub[1] + 2.0)
    cmp(f"c/{s}/kn_ex20", e20, np.asarray(C[s][t["r"]][:, cn.index("kn_ex20")]))
    cmp(f"c/{s}/kn_coec20", coec20, np.asarray(C[s][t["r"]][:, cn.index("kn_coec20")]))
del PE

# ================================================================== BLOCK A
ord_u = np.lexsort((np.arange(NF), tst, uidt)); u_sorted = uidt[ord_u]
for s in ("xm", "val"):
    t = T[s]; n = len(t["r"]); cols = ["ua_ns", "ua_npos", "ua_cat_imp", "ua_cat_pos", "ua_cat_coec", "ua_c1024_imp", "ua_c1024_pos", "ua_c1024_coec", "ua_item_imp", "ua_item_pos", "ue_k20_coec", "ua_dt_last"]
    out = {c: np.full(n, np.nan) for c in cols}; n_later = 0; max_src_ts_gap = []
    for i in range(n):
        u = t["uid"][i]
        if u == 0:
            continue
        lo, hi = np.searchsorted(u_sorted, u, "left"), np.searchsorted(u_sorted, u, "right")
        fs = ord_u[lo:hi]; fs = fs[~is_hold[fs]]
        if s == "xm":
            fs = fs[tst[fs] < t["ts"][i]]                                # strictly earlier feeds of the same user; never the target feed
            assert t["fidx"][i] not in fs
        if len(fs) == 0:
            continue
        rows = np.concatenate([np.arange(offt[f], offt[f] + Mt[f]) for f in fs])
        if len(rows) > 4000:
            rows = rows[-4000:]
        sy = y[rows].astype(np.float64); sx = ex[rows]
        out["ua_ns"][i] = len(rows); out["ua_npos"][i] = sy.sum()
        k = cat[rows] == t["cat"][i]; out["ua_cat_imp"][i] = k.sum(); out["ua_cat_pos"][i] = sy[k].sum(); out["ua_cat_coec"][i] = (sy[k].sum() + .5) / (sx[k].sum() + .5)
        k = (cl1024[rows] == t["c1024"][i]) & (t["c1024"][i] >= 0); out["ua_c1024_imp"][i] = k.sum(); out["ua_c1024_pos"][i] = sy[k].sum(); out["ua_c1024_coec"][i] = (sy[k].sum() + .5) / (sx[k].sum() + .5)
        k = item[rows] == t["item"][i]; out["ua_item_imp"][i] = k.sum(); out["ua_item_pos"][i] = sy[k].sum()
        E = emb_rows(embT, ei[rows]); v = t["emb"][i]; den = np.sqrt((v * v).sum()) * np.sqrt((E * E).sum(1))
        cs = np.where(den > 0, (E @ v) / np.where(den > 0, den, 1), 0.0); e = np.exp((cs - 1.0) / 0.2)
        out["ue_k20_coec"][i] = ((e * sy).sum() + 0.1) / ((e * sx).sum() + 0.1)
        out["ua_dt_last"][i] = (t["ts"][i] - tst[fs[-1]]) / 3600.0
    st = np.asarray(A[s][t["r"]])
    for c in cols:
        cmp(f"a/{s}/{c}", out[c], st[:, an.index(c)])
    res[f"a/{s}/coverage_mine"] = float((~np.isnan(out["ua_ns"])).mean())

# ================================================================== BLOCK B (category transitions)
hk = np.load(f"{F}/train/hist_category_id.npy")
hfid = np.repeat(np.arange(NF, dtype=np.int64), Ht)
valid = hk >= 0
cs_ = np.cumsum(valid); before = np.r_[0, cs_][hofft]                  # number of valid events before the feed's first event
rank_in = cs_ - before[hfid]                                           # 1-based rank of a valid event inside its feed (history is stored newest first)
k20 = valid & (rank_in <= 20); k1 = valid & (rank_in == 1)
H20 = np.bincount(hfid[k20] * 64 + hk[k20], minlength=NF * 64).reshape(NF, 64).astype(np.float64)
H1 = np.bincount(hfid[k1] * 64 + hk[k1], minlength=NF * 64).reshape(NF, 64).astype(np.float64)
del cs_, rank_in, k20, k1, valid, hfid
CY = np.bincount(row_fid.astype(np.int64) * 64 + cat, weights=y, minlength=NF * 64).reshape(NF, 64)
CE = np.bincount(row_fid.astype(np.int64) * 64 + cat, weights=ex, minlength=NF * 64).reshape(NF, 64)
log("block b: per-feed history / candidate category tables ready")
T20 = np.zeros((5, 15, 64, 64, 2)); T1 = np.zeros((5, 15, 64, 64, 2)); BB = np.zeros((5, 15, 64, 2))
part = fold_t * 15 + day_t
for pp in range(75):
    fs = np.where((part == pp) & ~is_hold)[0]; k, d = divmod(pp, 15)
    T20[k, d, :, :, 0] = H20[fs].T @ CY[fs]; T20[k, d, :, :, 1] = H20[fs].T @ CE[fs]
    T1[k, d, :, :, 0] = H1[fs].T @ CY[fs]; T1[k, d, :, :, 1] = H1[fs].T @ CE[fs]
    BB[k, d, :, 0] = CY[fs].sum(0); BB[k, d, :, 1] = CE[fs].sum(0)
log("block b: tables estimated; base clicks", BB[..., 0].sum())
hkv = np.load(f"{F}/val/hist_category_id.npy")
for s in ("xm", "val"):
    t = T[s]; n = len(t["r"]); base = np.full(n, np.nan); l1 = np.full(n, np.nan); l20m = np.full(n, np.nan); src = hk if s == "xm" else hkv
    for i in range(n):
        d = t["cut"][i]; c = t["cat"][i]
        if d == 0 or c < 0:
            continue
        m = np.ones(5, bool); m[t["fold"][i]] = False
        b_ = BB[m][:, :d, c].sum((0, 1)); R = (b_[0] + 20.0 * r0g) / (b_[1] + 20.0); base[i] = np.log(R)
        h = src[t["hoff"][i]:t["hoff"][i] + t["H"][i]]; h = h[h >= 0][:20].astype(np.int64)
        if len(h) == 0:
            continue
        t20 = T20[m][:, :d, :, c].sum((0, 1)); t1 = T1[m][:, :d, :, c].sum((0, 1))          # [64, 2]
        l20m[i] = np.mean(np.log((t20[h, 0] + 5.0 * R) / (t20[h, 1] + 5.0) / R))
        l1[i] = np.log((t1[h[0], 0] + 5.0 * R) / (t1[h[0], 1] + 5.0) / R)
    st = np.asarray(B[s][t["r"]])
    cmp(f"b/{s}/tr_cat_base", base, st[:, bn.index("tr_cat_base")])
    cmp(f"b/{s}/tr_cat_l1", l1, st[:, bn.index("tr_cat_l1")])
    cmp(f"b/{s}/tr_cat_l20m", l20m, st[:, bn.index("tr_cat_l20m")])
json.dump(res, open(f"{HERE}/out/rv_rebuild.json", "w"), indent=1)
log("DONE")
