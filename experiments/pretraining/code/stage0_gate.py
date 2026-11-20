# Stage-0 gate, written before any training, for using the pre-trained encoder as the starting
# weights of the ranker's history encoder: zero-shot NDCG of the encoder alone, its value as a
# residual on the best hard-label pool (weight chosen in user folds), and its pair AUC on positives
# that the pool ranks low inside the right category.
# How it was run: CPU, 12 threads, 55 s, on 2026-11-12 after the plan lab/nn_x/r3/plan.txt was
# written (21:58 UTC); the command line is not on record.
# What it produced: stage0_gate.{log,json}; the log is results/encoder_as_init_stage0_gate.txt.
# The gate failed (residual -0.00018 +- 0.00020, pair AUC 0.316 against a bar of 0.52), so no GPU
# run was made.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/r3/stage0_gate.py
"""R3 stage 0 gate (CPU only, 12 threads). See nn_x/r3/plan.txt (written first) for the pre-declared gate.

(a) inventory of the pretrained encoders (ckpt args / shapes, mmap);
(b) zero-shot val NDCG of the encoder used alone (single head logits, pre_final, z-mean ensembles over variants);
(c) ensemble value on G3: equal z-mix, nested residual (user-grouped, 2 fold seeds), slices, pmix;
    'right category, buried' cell: positive in G3's LSE-top category with >= 3 same-category candidates above it (G3);
    pairs = (that positive, the same-category negatives G3 ranks above it); pair AUC of the signal, of the blend, and the
    per-feed delta on cell feeds. The narrower P8 cell (warm + recent same-category history) is reported next to it.
Outputs: nn_x/r3/stage0_gate.log (stdout copy), nn_x/r3/stage0_gate.json.
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, json, glob
from pathlib import Path
import numpy as np, polars as pl, numba as nb
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from qt import leak_mask
from pmix import weights as pmix_weights, wmean_se
from sklearn.model_selection import GroupKFold

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat/val"; FEATS = ROOT / "data/feats/val"; P = ROOT / "preds"
ERR = ROOT / "nn_x/insights/errors/out"; OUT = ROOT / "nn_x/r3"
T0 = time.time()
LOG = open(OUT / "stage0_gate.log", "w")
def log(*a):
    s = f"[{time.time()-T0:6.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
RES = {}

# ---------------- (a) inventory ----------------
import torch
for tag in ["v3A", "v4A", "v3", "v4"]:
    p = ROOT / f"data/pretrain/ckpt_{tag}.pt"
    try:
        ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        sd = ck["model"]; a = ck["args"]
        shapes = {k: tuple(v.shape) for k, v in sd.items() if k.startswith(("id_emb", "id_bias", "content.0", "blocks.0.qkv", "heads", "bos"))}
        n_dense = sum(v.numel() for k, v in sd.items() if not k.startswith(("id_emb", "id_bias")))
        log(f"ckpt {tag}: d={a['d']} nl={a['nl']} nh={a['nh']} d_id={a['d_id']} min_cnt={a['min_cnt']} epochs={a['epochs']} "
            f"exclude_test={a.get('exclude_test')} vt_rep={a.get('vt_rep')} id_bias={a.get('id_bias')} multi={a.get('multi')} "
            f"dense params {n_dense/1e6:.2f}M; id table {shapes.get('id_emb.weight')}; keys {len(sd)}; size {p.stat().st_size/2**30:.1f} GB")
        RES[f"ckpt_{tag}"] = dict(args=a, id_table=shapes.get("id_emb.weight"), n_dense=n_dense)
        del ck, sd
    except Exception as e:
        log(f"ckpt {tag}: load failed: {e}")
sp = np.load(ROOT / "data/pretrain/seq_split.npy"); off = np.load(ROOT / "data/pretrain/seq_off.npy")
log(f"corpus: {len(sp)} sequences (split counts train/val/test = {np.bincount(sp).tolist()}), {off[-1]:,} events")

# ---------------- val structure ----------------
f = pl.read_parquet(FLAT / "feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); n = len(M)
fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "cand_target.npy").astype(np.float64)
cat = np.load(FLAT / "cand_category_id.npy").astype(np.int64)
users = f["user_id"].to_numpy().copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
keep = ~leak_mask()
T = pl.read_parquet(ERR / "feed_table.parquet")
tue = T["tue"].to_numpy().astype(bool); assert (T["M"].to_numpy() == M).all()
W = pmix_weights(); wpriv = W["private"]
SL = {"all": np.ones(n, bool), "leakfree": keep, "M<50": keep & (M < 50), "M>=50": keep & (M >= 50), "Tue": keep & tue, "Thu": keep & ~tue,
      "coldpos": keep & T["coldpos"].to_numpy().astype(bool), "warmpos": keep & T["warmpos"].to_numpy().astype(bool)}
log(f"val feeds {n}, leak-free {keep.sum()}, rows {len(y):,}")

def z(x): return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
def nd(s): return ndcg_per_feed(np.asarray(s, np.float64), y, off, M)
def se(d): d = np.asarray(d, np.float64); return d.std(ddof=1) / np.sqrt(len(d))

@nb.njit(cache=True)
def feed_z_nan(off, M, x):
    out = np.zeros(len(x))
    for g in range(len(M)):
        o, m = off[g], M[g]
        s = 0.0; s2 = 0.0; c = 0
        for k in range(m):
            v = x[o + k]
            if v == v:
                s += v; s2 += v * v; c += 1
        if c >= 2:
            mu = s / c; sd = np.sqrt(max(s2 / c - mu * mu, 0.0))
            if sd > 0:
                for k in range(m):
                    v = x[o + k]
                    if v == v:
                        out[o + k] = (v - mu) / sd
    return out

zG3 = 0.5 * z(np.load(P / "ens_G3_ft_val.npy")) + 0.5 * z(np.load(P / "ens_G3_to_val.npy"))
zTO = z(np.load(P / "ens_G3_to_val.npy"))
BASES = {"G3": zG3, "G3_to": zTO}
base_per = {b: nd(v) for b, v in BASES.items()}
log("base NDCG all / leak-free:", {b: (round(float(v.mean()), 5), round(float(v[keep].mean()), 5)) for b, v in base_per.items()})

# ---------------- (b) zero-shot scores ----------------
VARS = ["v1", "v2", "v3", "v3A", "v4", "v4A", "v4s1"]
HEADS = ["fe", "new", "fe_noid", "new_noid", "hmax32", "hmax8", "idmax32"]
raw = {}
log("--- zero-shot val NDCG of single head scores (all feeds | leak-free) ---")
single = {}
for v in VARS:
    row = []
    for h in HEADS:
        p = P / f"pre_{v}_{h}_val.npy"
        if not p.exists():
            continue
        x = np.load(p).astype(np.float64); raw[(v, h)] = x
        per = nd(x); single[f"{v}_{h}"] = (float(per.mean()), float(per[keep].mean()))
        row.append(f"{h} {per.mean():.4f}/{per[keep].mean():.4f}")
    log(f"  {v:5s} " + "  ".join(row))
RES["single"] = single
best_single = max((k for k in single if k.split("_", 1)[1] in ("fe", "new")), key=lambda k: single[k][0])
log(f"best single head: {best_single} {single[best_single]}")

def zmean(keys):
    return np.mean([z(raw[k]) for k in keys], 0)
ENS = {}
ENS["pre_final"] = np.load(P / "pre_final_val.npy").astype(np.float64)
ENS["pre2_final"] = np.load(P / "pre2_final_val.npy").astype(np.float64)
ENS["ENS_clean"] = zmean([(v, h) for v in ["v3A", "v4A"] for h in ["fe", "new"]])
ENS["ENS_clean+h"] = zmean([(v, h) for v in ["v3A", "v4A"] for h in ["fe", "new"]] + [(v, "hmax32") for v in ["v3A", "v4A"]])
ALLV = ["v2", "v3", "v3A", "v4", "v4A", "v4s1"]
ENS["ENS_all"] = zmean([(v, h) for v in ALLV for h in ["fe", "new"]])
ENS["ENS_all+h"] = zmean([(v, h) for v in ALLV for h in ["fe", "new"]] + [(v, "hmax32") for v in ALLV])
ENS["ENS_all+h+h8"] = zmean([(v, h) for v in ALLV for h in ["fe", "new"]] + [(v, h) for v in ALLV for h in ["hmax32", "hmax8"]])
ENS["ENS_all_noid"] = zmean([(v, h) for v in ALLV for h in ["fe_noid", "new_noid"]])
ENS["best_single"] = raw[tuple(best_single.split("_", 1))]
ENS["pre2_id_contrib"] = np.load(FEATS / "pre2_id_contrib.npy").astype(np.float64)
ENS["pre2_idmax32"] = np.load(FEATS / "pre2_idmax32.npy").astype(np.float64)
log("--- zero-shot val NDCG of encoder-only ensembles (all | leak-free | M<50 | M>=50 | Tue | Thu) ---")
RES["ens"] = {}
for k, x in ENS.items():
    per = nd(np.where(np.isfinite(x), x, np.nanmin(x[np.isfinite(x)]) - 1))
    RES["ens"][k] = {s: float(per[m].mean()) for s, m in SL.items()}
    log(f"  {k:16s} " + " ".join(f"{s} {per[m].mean():.4f}" for s, m in SL.items() if s in ("all", "leakfree", "M<50", "M>=50", "Tue", "Thu")))

# ---------------- cells and pairs ----------------
ck = fid * 64 + cat; uk, inv = np.unique(ck, return_inverse=True)
lse = np.log(np.bincount(inv, np.exp(zG3)))
kf = uk // 64; o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
pcr = cr[inv]                                                            # G3 LSE rank of the candidate's category within the feed
LV = np.load(ERR / "feed_label_vars.npz", allow_pickle=True)
o2 = np.lexsort((y, -zG3, ck)); wr = np.empty(len(y), np.int64); wr[o2] = np.arange(len(y)) - np.searchsorted(ck[o2], ck[o2])
cold = np.load(ERR / "cold_luo.npy").astype(bool); minh = np.load(ERR / "cand_cat_minh.npy")
cellpos = (y > 0) & (pcr == 0) & (wr >= 3)                               # task cell: right category, >= 3 same-cat items above
cellpos_p8 = (y > 0) & (pcr == 0) & (~cold) & (minh < 8)                # P8 / probe cell positives (any beater count)
log(f"cell positives (task): {cellpos.sum():,} of {int((y>0).sum()):,} positives in {len(np.unique(fid[cellpos])):,} feeds "
    f"(leak-free {cellpos[keep[fid]].sum():,}); P8-cell positives {cellpos_p8.sum():,}")
# pairs: (cell positive, same-category negative ranked above it by G3)
pi, ni = [], []
for g in np.nonzero(keep)[0]:
    s, m = off[g], M[g]
    pp = np.nonzero(cellpos[s:s + m] | cellpos_p8[s:s + m])[0] + s
    if len(pp) == 0:
        continue
    neg = np.nonzero(y[s:s + m] == 0)[0] + s
    for p_ in pp:
        q = neg[(cat[neg] == cat[p_]) & (zG3[neg] > zG3[p_])]
        pi.append(np.full(len(q), p_)); ni.append(q)
pi = np.concatenate(pi); ni = np.concatenate(ni)
cell = cellpos[pi]; cellp8 = cellpos_p8[pi] & (~cold[ni])
cell_feeds = np.zeros(n, bool); cell_feeds[np.unique(fid[pi[cell]])] = True
log(f"pairs: task cell {cell.sum():,} (feeds {cell_feeds.sum():,}), P8 cell {cellp8.sum():,}; "
    f"loss share of task-cell positives: see 05_mass ('cat#1|w3-9' + 'cat#1|w10+')")
def auc(d): return float(((d > 0) + 0.5 * (d == 0)).mean()) if len(d) else float("nan")

# ---------------- (c) blends on G3 ----------------
WG = [0, .05, .1, .15, .2, .3, .5]
folds = {seed: list(GroupKFold(5, shuffle=True, random_state=seed).split(np.arange(n), groups=users)) for seed in range(2)}
RES["blend"] = {}
for name in ["pre_final", "ENS_clean+h", "ENS_all", "ENS_all+h", "ENS_all+h+h8", "best_single", "ENS_all_noid", "pre2_id_contrib", "pre2_idmax32"]:
    x = ENS[name]; dfn = np.isfinite(x)
    zx = feed_z_nan(off, M, np.where(dfn, x, np.nan))
    both = dfn[pi] & dfn[ni]; d = zx[pi] - zx[ni]
    a_cell, a_p8 = auc(d[both & cell]), auc(d[both & cellp8])
    rec = dict(auc_cell=a_cell, auc_cell_p8=a_p8, cell_cov=float(both[cell].mean()))
    log(f"=== {name}: pair AUC on task cell {a_cell:.4f} (cov {both[cell].mean():.3f}), P8 cell {a_p8:.4f}")
    for b, zb in BASES.items():
        # equal z-mix
        eq = nd(0.5 * zb + 0.5 * zx) - base_per[b]
        rec[f"eqmix_{b}"] = (float(eq[keep].mean()), float(se(eq[keep])))
        pers = {w: nd(zb + w * zx) for w in WG}
        deltas = []
        for seed in range(2):
            outd = np.zeros(n)
            for trn, va in folds[seed]:
                trk = trn[keep[trn]]
                wbest = max(WG, key=lambda w: pers[w][trk].mean())
                outd[va] = pers[wbest][va] - base_per[b][va]
            deltas.append(outd)
        dn = np.mean(deltas, 0)
        wfull = max(WG, key=lambda w: pers[w][keep].mean())
        pm, pse = wmean_se(dn[keep], wpriv[keep])
        sl = {s: (float(dn[m].mean()), float(se(dn[m]))) for s, m in SL.items()}
        slc = (float(dn[cell_feeds & keep].mean()), float(se(dn[cell_feeds & keep])))
        zbl = zb + wfull * zx
        dbl = zbl[pi] - zbl[ni]
        rec[f"nested_{b}"] = dict(slices=sl, pmix=(float(pm), float(pse)), w_full=wfull, cell_feeds=slc,
                                  auc_cell_blend=auc(dbl[cell]), fixed_w={w: float((pers[w] - base_per[b])[keep].mean()) for w in WG})
        log(f"  base {b:5s}: equal z-mix {eq[keep].mean():+.5f} ± {se(eq[keep]):.5f} | nested {sl['leakfree'][0]:+.5f} ± {sl['leakfree'][1]:.5f} "
            f"(z {sl['leakfree'][0]/max(sl['leakfree'][1],1e-9):+.2f}; w_full {wfull}) pmix {pm:+.5f} ± {pse:.5f} | "
            f"M<50 {sl['M<50'][0]:+.5f} M>=50 {sl['M>=50'][0]:+.5f} Tue {sl['Tue'][0]:+.5f} Thu {sl['Thu'][0]:+.5f} "
            f"coldpos {sl['coldpos'][0]:+.5f} warmpos {sl['warmpos'][0]:+.5f} | cell feeds {slc[0]:+.5f} ± {slc[1]:.5f} "
            f"blend cell AUC {auc(dbl[cell]):.4f} | fixed-w: " + " ".join(f"{w}:{(pers[w]-base_per[b])[keep].mean():+.5f}" for w in WG))
    RES["blend"][name] = rec

# ---------------- gate ----------------
zs_clean = RES["ens"]["pre_final"]["all"]; zs_best = max(RES["ens"][k]["all"] for k in ["pre_final", "ENS_clean+h", "ENS_all", "ENS_all+h", "ENS_all+h+h8", "best_single"])
best_name = max(["pre_final", "ENS_clean+h", "ENS_all", "ENS_all+h", "ENS_all+h+h8", "best_single"], key=lambda k: RES["ens"][k]["all"])
nest = {k: RES["blend"][k]["nested_G3"]["slices"]["leakfree"] for k in RES["blend"]}
auc_c = {k: RES["blend"][k]["auc_cell"] for k in RES["blend"]}
cond_a = zs_best >= 0.36
cond_b = any(v[0] > 0.0005 and v[0] > 2 * v[1] for v in nest.values())
cond_c = any(a - 0.5 > 0.02 for a in auc_c.values())
RES["gate"] = dict(zero_shot_clean=zs_clean, zero_shot_best=(best_name, zs_best), cond_zero_shot_ge_036=bool(cond_a),
                   nested_G3=nest, cond_nested=bool(cond_b), cell_auc=auc_c, cond_cell_auc=bool(cond_c), proceed=bool(cond_a and (cond_b or cond_c)))
log(f"GATE: zero-shot clean (pre_final) {zs_clean:.4f}; best encoder-only {best_name} {zs_best:.4f} -> >= 0.36: {cond_a}")
log(f"GATE: nested residual on G3 (leak-free): " + "; ".join(f"{k} {v[0]:+.5f}±{v[1]:.5f}" for k, v in nest.items()) + f" -> any > +0.0005 and > 2 SE: {cond_b}")
log(f"GATE: task-cell pair AUC: " + "; ".join(f"{k} {a:.4f}" for k, a in auc_c.items()) + f" -> any > 0.52: {cond_c}")
log(f"GATE DECISION: {'PROCEED to stage 1' if RES['gate']['proceed'] else 'STOP (gate failed)'}")
json.dump(RES, open(OUT / "stage0_gate.json", "w"), indent=1, default=str)
log("done")
