"""Pre-training audit A1 (CPU): candidate item / category / embedding-table / ID-table coverage per split.
Full data (no sampling) except the cross-split content-identity check (sample of 50K items).
Writes nn_x/pretrain2/audit/a1_items.log (stdout)."""
import json, time, numpy as np, polars as pl
from pathlib import Path
R = Path("<repo>"); F = R / "data/flat"; P = R / "data/pretrain"; C = R / "data/competition"
BOUND = 1790497920  # src/pmix.py: public = test feeds with timestamp < BOUND
t0 = time.time()
def L(s, n, mm="r"): return np.load(F / s / f"{n}.npy", mmap_mode=mm)
cnt = np.load(P / "item_cnt.npy", mmap_mode="r"); G = len(cnt)
print("G (max item id + 1) =", G)
fe = {s: pl.read_parquet(F / s / "feeds.parquet") for s in ["train", "val", "test"]}
def bm(arr):
    b = np.zeros(G, np.bool_); b[np.asarray(arr)] = True; return b
tr_hist = bm(L("train", "hist_item_id")); print(f"train hist distinct {tr_hist.sum():,} ({time.time()-t0:.0f}s)")
tci = np.asarray(L("train", "cand_item_id")); tgt = np.asarray(L("train", "cand_target"))
tr_cand = bm(tci); tr_pos = bm(tci[tgt == 1]); print(f"train cand distinct {tr_cand.sum():,}, train pos distinct {tr_pos.sum():,}")
# exploration-only train candidates (p_random>0.9) -> 'labelled under the same policy as val/test'
pr = np.nan_to_num(np.load(R / "eda/labels_position/out/train_prandom.npz")["p_random"], nan=0.0)
print("p_random len", len(pr), "share >0.9", float((pr > 0.9).mean()))
M_tr = fe["train"]["M"].to_numpy(); fid_rows = np.repeat(np.arange(len(M_tr)), M_tr)
rnd_rows = pr[fid_rows] > 0.9
tr_cand_rnd = bm(tci[rnd_rows]); tr_pos_rnd = bm(tci[rnd_rows & (tgt == 1)])
del tci, tgt, fid_rows, rnd_rows
any_hist = tr_hist.copy()
v_hist = bm(L("val", "hist_item_id")); v_cand = bm(L("val", "cand_item_id"))
te_hist = bm(L("test", "hist_item_id")); te_cand = bm(L("test", "cand_item_id"))
all_hist = tr_hist | v_hist | te_hist
print(f"distinct history items all splits {all_hist.sum():,}; val hist {v_hist.sum():,}; test hist {te_hist.sum():,}")
# pretraining ID table (min_cnt 3): full corpus (test side) and excluding test snapshots (val side, v3A/v4A)
id_full = np.asarray(cnt) >= 3
sp = np.load(P / "seq_split.npy"); off = np.load(P / "seq_off.npy"); tok = np.load(P / "tok_item.npy", mmap_mode="r")
keep_tok = np.repeat(sp != 2, np.diff(off))
cnt_nt = np.bincount(np.asarray(tok)[keep_tok], minlength=G); id_nt = cnt_nt >= 3
print(f"ID rows full {id_full.sum():,} | excl test {id_nt.sum():,} ({time.time()-t0:.0f}s)")
corp = np.asarray(cnt) > 0
tok_cov3 = float(np.asarray(cnt)[id_full].sum() / np.asarray(cnt).sum())
print(f"corpus: distinct items {corp.sum():,}; tokens whose item has cnt>=3: {tok_cov3:.4f}; cnt==1 items {(np.asarray(cnt)==1).sum():,} cnt==2 {(np.asarray(cnt)==2).sum():,}")
del keep_tok
def cand_split(s):
    it = np.asarray(L(s, "cand_item_id")); ei = np.asarray(L(s, "cand_emb_idx")); cat = np.asarray(L(s, "cand_category_id"))
    M = fe[s]["M"].to_numpy(); ts = fe[s]["timestamp"].to_numpy(); rows_f = np.repeat(np.arange(len(M)), M)
    return it, ei, cat, rows_f, ts
def rep(name, it, mask_rows=None):
    x = it if mask_rows is None else it[mask_rows]
    u = np.unique(x)
    out = {}
    tests = [("in train hist", tr_hist), ("in train cand", tr_cand), ("in train cand|hist", tr_hist | tr_cand),
             ("in train positives", tr_pos), ("in train random-like cand", tr_cand_rnd), ("in train random-like positives", tr_pos_rnd),
             ("in val hist", v_hist), ("in val cand", v_cand), ("in test hist", te_hist),
             ("in ANY history (all splits)", all_hist), ("pretrain corpus item (cnt>0)", corp),
             ("pretrain ID row full corpus (cnt>=3)", id_full), ("pretrain ID row excl-test corpus (cnt>=3)", id_nt)]
    print(f"  [{name}] rows {len(x):,} uniq {len(u):,}")
    for lab, b in tests:
        r, q = b[x].mean(), b[u].mean(); out[lab] = (float(r), float(q))
        print(f"    {lab:44s} rows {100*r:6.2f}%  uniq {100*q:6.2f}%")
    return out
res = {}
for s in ["val", "test"]:
    it, ei, cat, rf, ts = cand_split(s)
    print(f"== {s} candidates: emb_idx==-1 share {np.mean(ei < 0):.5f}; category==0 share {np.mean(cat == 0):.5f}")
    res[s] = rep(s, it)
    if s == "test":
        pub = ts[rf] < BOUND
        res["public"] = rep("test public", it, pub); res["private"] = rep("test private", it, ~pub)
    # own-history overlap (candidate item in its own feed history)
    hf = np.repeat(np.arange(len(fe[s])), fe[s]["H"].to_numpy())
    hk = hf.astype(np.int64) * G + np.asarray(L(s, "hist_item_id")); ck = rf.astype(np.int64) * G + it
    print(f"    candidate item present in its OWN feed history: {np.isin(ck, hk).mean():.5f}")
    hc = np.asarray(L(s, "hist_category_id")); hk2 = hf.astype(np.int64) * 64 + hc; ck2 = rf.astype(np.int64) * 64 + cat
    print(f"    candidate CATEGORY present in its own history: {np.isin(ck2, hk2).mean():.4f}")
# train candidates (reference)
it_tr = np.asarray(L("train", "cand_item_id")); ei_tr = np.asarray(L("train", "cand_emb_idx"))
print(f"== train candidates: emb_idx==-1 share {np.mean(ei_tr < 0):.5f}; ID row (full corpus) {id_full[it_tr].mean():.4f}; in any history {all_hist[it_tr].mean():.4f}")
del it_tr, ei_tr
# history emb coverage, ID coverage of history tokens
for s in ["train", "val", "test"]:
    he = L(s, "hist_emb_idx"); hi = L(s, "hist_item_id")
    if s == "train":
        idx = np.random.default_rng(0).choice(len(he), 20_000_000, replace=False); idx.sort()
        he = he[idx]; hi = hi[idx]; tag = "(20M-event random sample)"
    else:
        he = np.asarray(he); hi = np.asarray(hi); tag = "(all)"
    print(f"== {s} history events {tag}: emb_idx==-1 {np.mean(he < 0):.5f}; item_id==0 {np.mean(hi == 0):.5f}; ID row full {id_full[hi].mean():.4f}; excl-test {id_nt[hi].mean():.4f}")
# categories: candidate category distribution per split (share), JS divergence vs train
def catdist(s, kind):
    c = np.asarray(L(s, f"{kind}_category_id")) if s != "train" else np.asarray(L(s, f"{kind}_category_id")[::7])
    d = np.bincount(c.astype(np.int64), minlength=54).astype(float); return d / d.sum()
dc = {s: catdist(s, "cand") for s in ["train", "val", "test"]}; dh = {s: catdist(s, "hist") for s in ["train", "val", "test"]}
def js(p, q):
    m = 0.5 * (p + q); f = lambda a, b: np.sum(np.where(a > 0, a * np.log(a / np.maximum(b, 1e-12)), 0)); return 0.5 * f(p, m) + 0.5 * f(q, m)
print("== categories: n distinct cand cats train/val/test", [(dc[s] > 0).sum() for s in dc], "hist", [(dh[s] > 0).sum() for s in dh])
print(f"   cand JS(train,val) {js(dc['train'], dc['val']):.4f} JS(train,test) {js(dc['train'], dc['test']):.4f}; hist JS(train,val) {js(dh['train'], dh['val']):.4f} JS(train,test) {js(dh['train'], dh['test']):.4f}")
print("   val/test cand categories absent from train cand:", [int(k) for k in np.where((dc['train'] == 0) & ((dc['val'] > 0) | (dc['test'] > 0)))[0]])
print("   top-8 cand categories train/val/test share:")
for k in np.argsort(-dc["train"])[:8]:
    print(f"     cat {k:2d}: {dc['train'][k]:.4f} {dc['val'][k]:.4f} {dc['test'][k]:.4f}")
# content-vector identity of the same item across split tables (50K sample of val candidate items with a train row)
def table(s):
    m = json.loads((C / s / "item_embeddings/data.meta").read_text()); return np.memmap(C / s / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(m["shape"]))
Ttr, Tv, Tte = table("train"), table("val"), table("test")
print("tables:", Ttr.shape, Tv.shape, Tte.shape)
# first train emb row per item from train candidates (sampled)
tci = np.asarray(L("train", "cand_item_id")); tce = np.asarray(L("train", "cand_emb_idx"))
vi = np.asarray(L("val", "cand_item_id")); ve = np.asarray(L("val", "cand_emb_idx"))
order = np.argsort(tci, kind="stable"); s_it = tci[order]
uv, iv = np.unique(vi, return_index=True)
pos = np.searchsorted(s_it, uv); ok = (pos < len(s_it)) & (s_it[np.minimum(pos, len(s_it) - 1)] == uv)
rng = np.random.default_rng(1); pick = rng.choice(np.where(ok)[0], min(50000, ok.sum()), replace=False)
a = Ttr[np.sort(tce[order[pos[pick]]])]; b = Tv[np.sort(ve[iv[pick]])]  # rows sorted for memmap speed; re-pair below
ra = tce[order[pos[pick]]]; rb = ve[iv[pick]]
A = Ttr[ra].astype(np.float32); B = Tv[rb].astype(np.float32)
same = np.all(A == B, axis=1); cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
print(f"== same item, train-table vs val-table content vector (n={len(pick)} val cand items also train cands): identical {same.mean():.4f}; cos q05/q50 {np.quantile(cos,0.05):.4f}/{np.quantile(cos,0.5):.4f}")
# within train: same item at two different train cand rows -> same emb row?
d_rows = np.unique(np.stack([tci[:5_000_000], tce[:5_000_000]], 1), axis=0)
_, c_ = np.unique(d_rows[:, 0], return_counts=True)
print(f"== train (first 5M cand rows): items with >1 distinct emb_idx: {np.mean(c_ > 1):.4f} of {len(c_):,}")
json.dump(res, open(R / "nn_x/pretrain2/audit/a1_items.json", "w"), indent=1)
print(f"done {time.time()-t0:.0f}s")
