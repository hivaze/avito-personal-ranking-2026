"""EMB-USABILITY stage 0: label-free signal construction (see PREDECLARED.txt). CPU only.

Outputs out/sig_val.npz (per val candidate row signals), out/test_change.npz (test descriptive), out/fit.npz, out/prep.log.
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, json
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; COMP = ROOT / "data/competition"
OUT = ROOT / "nn_x/emb/usability/out"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
LOG = open(OUT / "prep.log", "a")
def log(*a):
    s = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def table(split):
    meta = json.load(open(COMP / split / "item_embeddings/data.meta"))
    return np.memmap(COMP / split / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))

def unit(e):
    e = np.asarray(e, np.float32); n = np.linalg.norm(e, axis=1, keepdims=True); n[n == 0] = 1
    return e / n

TR, VA, TE = table("train"), table("val"), table("test")
log("tables", TR.shape, VA.shape, TE.shape)

# ------------------------------------------------------------------ item maps
MAXID = 0
for sp in ["train", "val", "test"]:
    for a in ["cand_item_id", "hist_item_id"]:
        MAXID = max(MAXID, int(np.load(FLAT / sp / f"{a}.npy", mmap_mode="r").max()))
log("max item id", MAXID)

def build_map(split):
    mp = np.full(MAXID + 1, -1, np.int32); cm = np.zeros(MAXID + 1, np.int8)
    ids_all, em_all = [], []
    for pre in ["cand", "hist"]:
        ids = np.load(FLAT / split / f"{pre}_item_id.npy"); em = np.load(FLAT / split / f"{pre}_emb_idx.npy")
        ca = np.load(FLAT / split / f"{pre}_category_id.npy")
        ok = (em >= 0) & (ids > 0)
        mp[ids[ok]] = em[ok]; cm[ids[ok]] = ca[ok]
        ids_all.append(ids[ok]); em_all.append(em[ok])
    ids = np.concatenate(ids_all); em = np.concatenate(em_all)
    mism = float((mp[ids] != em).mean())
    nitem = int((mp >= 0).sum())
    log(f"[{split}] map: {nitem:,} item_ids with a row; rows referencing a different row than the map: {mism:.2e}")
    return mp, cm

mapTR, catTR = build_map("train")
mapVA, catVA = build_map("val")

# ------------------------------------------------------------------ fit (train table sample)
rng = np.random.default_rng(0)
fit_rows = np.sort(rng.choice(TR.shape[0], 1_000_000, replace=False))
U = unit(TR[fit_rows])
mu = U.mean(0); C = np.cov(U, rowvar=False).astype(np.float64)
ev, V = np.linalg.eigh(C); ev = ev[::-1]; V = V[:, ::-1]
lam = 0.1 * np.trace(C) / 128
W = (V * (1.0 / np.sqrt(ev + lam))) @ V.T
P3 = np.eye(128) - V[:, :3] @ V[:, :3].T
log(f"fit: |mu| {np.linalg.norm(mu):.4f}, tr(C) {np.trace(C):.4f}, top eig share {ev[:5]/ev.sum()}, lam {lam:.5f}")
np.savez(OUT / "fit.npz", mu=mu, C=C, W=W, ev=ev, V=V)
mud = mu / np.linalg.norm(mu); pc1 = V[:, 0]

# ------------------------------------------------------------------ val per-candidate
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
coff = f["cand_off"].to_numpy(); hoff = f["hist_off"].to_numpy(); n = len(M)
c_iid = np.load(FLAT / "val/cand_item_id.npy"); c_emb = np.load(FLAT / "val/cand_emb_idx.npy")
c_cat = np.load(FLAT / "val/cand_category_id.npy")
h_emb = np.load(FLAT / "val/hist_emb_idx.npy"); h_et = np.load(FLAT / "val/hist_event_type.npy")
h_hpos = np.load(FLAT / "val/hist_hpos.npy")
N = len(c_iid)
assert (np.diff(h_hpos.astype(np.int64))[np.diff(np.load(FLAT / "val/hist_fidx.npy")) == 0] > 0).all(), "hist not newest-first"

UV = unit(VA[:])                                             # whole val table, unit
log("val table unit", UV.shape)
S = {}
# item 1: change vs train table
tr_row = np.where(c_iid > 0, mapTR[c_iid], -1)
has = (tr_row >= 0) & (c_emb >= 0)
ctv = np.full(N, np.nan, np.float32)
ut = unit(TR[np.sort(tr_row[has])])
order = np.argsort(tr_row[has]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
ut = ut[inv]
ctv[has] = (UV[c_emb[has]] * ut).sum(1)
S["chg_d"] = np.where(has, 1 - ctv, np.nan).astype(np.float32)
S["chg_flag"] = np.where(has, (ctv < 0.99).astype(np.float32), np.nan).astype(np.float32)
S["in_train"] = has.astype(np.float32)
S["cos_tv"] = ctv
S["cat_tr"] = np.where(has, catTR[np.maximum(c_iid, 0)], -1).astype(np.int16)
bit_ident = np.zeros(N, bool)
bit_ident[has] = (np.asarray(VA[c_emb[has]]) == np.asarray(TR[tr_row[has]])).all(1)
S["bit_ident"] = np.where(has, bit_ident, np.nan).astype(np.float32)
log(f"[item1] val cands with train-table vector {has.mean():.4f}; of those cos<0.99 {np.nanmean(S['chg_flag']):.4f}, "
    f"bit-identical {bit_ident[has].mean():.4f}, mean cos {np.nanmean(ctv):.5f}, category differs {(S['cat_tr'][has] != c_cat[has]).mean():.4f}")

# items 2/3/4: transformed val-table vectors
def tf(name):
    if name == "raw": return UV
    X = UV - mu
    if name == "cen": return unit(X)
    if name == "abtt3": return unit(X @ P3.astype(np.float32))
    if name == "zca": return unit(X @ W.astype(np.float32))
VARS = ["raw", "cen", "abtt3", "zca"]
uc = UV[np.maximum(c_emb, 0)]
S["proj_mu"] = np.where(c_emb >= 0, uc @ mud.astype(np.float32), np.nan).astype(np.float32)
S["proj_pc1"] = np.where(c_emb >= 0, (uc - mu) @ pc1.astype(np.float32), np.nan).astype(np.float32)
del uc
for v in VARS:
    T = tf(v)
    mx = np.full(N, np.nan, np.float32); n5 = np.full(N, np.nan, np.float32)
    if v == "raw":
        ma = np.full(N, np.nan, np.float32); mv = np.full(N, np.nan, np.float32)
    for g in range(n):
        s, m = coff[g], M[g]; hs, hh = hoff[g], H[g]
        ce = c_emb[s:s + m]; okc = ce >= 0
        he = h_emb[hs:hs + hh]; okh = he >= 0
        if not okc.any() or not okh.any():
            continue
        A = T[ce[okc]]; B = T[he[okh]]
        Smat = A @ B.T
        idx = np.nonzero(okc)[0] + s
        mx[idx] = Smat.max(1); n5[idx] = Smat[:, :5].mean(1)
        if v == "raw":
            et = h_et[hs:hs + hh][okh]
            act = (et >= 2) & (et <= 10); vw = et == 1
            if act.any(): ma[idx] = Smat[:, act].max(1)
            if vw.any(): mv[idx] = Smat[:, vw].max(1)
    S[f"{v}_maxcos"] = mx; S[f"{v}_new5"] = n5
    if v == "raw":
        S["maxcos_act"] = ma; S["maxcos_view"] = mv
    del T
    log(f"[item2] {v}: maxcos mean {np.nanmean(mx):.4f} cov {np.isfinite(mx).mean():.4f}; new5 mean {np.nanmean(n5):.4f}")
S["nd_act"] = (np.nan_to_num(S["maxcos_act"], nan=-1) > 0.98).astype(np.float32)
S["nd_view_only"] = ((np.nan_to_num(S["maxcos_view"], nan=-1) > 0.98) & (S["nd_act"] == 0)).astype(np.float32)
log(f"[item4] nd_act share {S['nd_act'].mean():.5f}, nd_view_only {S['nd_view_only'].mean():.5f}")
np.savez(OUT / "sig_val.npz", **S)
log("saved sig_val.npz")

# ------------------------------------------------------------------ test descriptive (item 1)
ft = pl.read_parquet(FLAT / "test/feeds.parquet")
tM = ft["M"].to_numpy().astype(np.int64)
pub = np.repeat(ft["timestamp"].to_numpy() < 1790497920, tM)
t_iid = np.load(FLAT / "test/cand_item_id.npy"); t_emb = np.load(FLAT / "test/cand_emb_idx.npy")
res = {}
for ref, mp, TAB in [("train", mapTR, TR), ("val", mapVA, VA)]:
    row = np.where(t_iid > 0, mp[t_iid], -1); h_ = (row >= 0) & (t_emb >= 0)
    cs = np.full(len(t_iid), np.nan, np.float32)
    o = np.argsort(row[h_]); rr = row[h_][o]
    ur = unit(TAB[rr]); iv = np.empty_like(o); iv[o] = np.arange(len(o)); ur = ur[iv]
    cs[h_] = (unit(TE[t_emb[h_]]) * ur).sum(1)
    res[f"cos_{ref}"] = cs
    for nm, mk in [("public", pub), ("private", ~pub)]:
        hh = h_ & mk
        log(f"[item1 test] {nm}: share with {ref}-table vector {hh.sum()/mk.sum():.4f}; of those cos<0.99 "
            f"{(cs[hh] < 0.99).mean():.4f}, cos<0.9 {(cs[hh] < 0.9).mean():.4f}, mean cos {cs[hh].mean():.5f}")
res["public"] = pub
# val candidates: the same against the val table itself is trivial; also val-vs-test for items in both (descriptive)
np.savez(OUT / "test_change.npz", **res)
log("done")
