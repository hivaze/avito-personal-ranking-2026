"""Alt-approach audit, probe 1: which *structurally different* signals still carry residual information on top of G3?

CPU only. Val, leak-free feeds (the 117 feeds whose non-zero user has a later val/test feed are dropped, as in
eda/errors/synthesis/02_screen.py). Base = G3 final = 0.5 z(ens_G3_ft) + 0.5 z(ens_G3_to) (val 0.40032), and G3_to.

For every candidate signal x:
  * within-feed z (NaN -> 0, i.e. "unknown" sits at the feed mean);
  * pairwise AUC on all (pos, neg) pairs, on same-category pairs, and on the INSIGHTS_1112 item-5 cell:
      warm positive (it_hist_all_luo > 0) with a recent same-category history event (min hpos < 8), whose category is
      G3's top category (LSE rank 0), against the warm same-category negatives that G3 ranks ABOVE it ("beaters").
      G3 itself has AUC 0 there by construction; a signal that separates these pairs is new information.
  * nested linear blend z(base) + w z(x), w in {0,.05,.1,.2,.3,.5} chosen on 4/5 of the feeds (GroupKFold by user,
    user-0 feeds as singletons), applied to the held-out 1/5, 2 seeds -> paired per-feed delta vs w = 0, SE, and the
    same delta reweighted to the private-test (M x cold-share) mix (src/pmix.py, report only).

Signals (all computed without val labels; the ALS / item2vec arrays were fitted on train+val+test histories, which is
leak-free on the kept feeds because same-user later snapshots are exactly what the leak mask removes):
  A  transductive co-visitation from train+val+test histories (eda/history_leak/05_covis.py cache) vs train-only;
  B  other-user transductive popularity in val+test histories, earlier / later than the feed (08_item_transductive cache);
  C  ALS (f128, evt weights) and item2vec profiles on all histories (eda/collab_stats cache), incl. the 'clean' ALS;
  D  pretrained-encoder ID-space signals (pre2_idmax32, pre2_id_contrib) = the P8 CPU gate, cell-targeted;
  E  category-level set signal: log-sum-exp of G3 z within (feed, category) (does G3 under-use category aggregation?);
  R  references already inside the NN (seq_cv_invd20, it_hist_all_luo, pre2_fe, cos_u, seq_max_all, f1) to calibrate
     what "absorbed" looks like.
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time
from pathlib import Path
import numpy as np, polars as pl, numba as nb
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from qt import leak_mask
from pmix import weights as pmix_weights, wmean_se
from sklearn.model_selection import GroupKFold

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat/val"; FEATS = ROOT / "data/feats/val"; P = ROOT / "preds"
HL = ROOT / "eda/history_leak"; CS = ROOT / "eda/collab_stats/out"; ERR = ROOT / "nn_x/insights/errors/out"
OUT = ROOT / "nn_x/audit/altapproach"
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

f = pl.read_parquet(FLAT / "feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); n = len(M)
fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "cand_target.npy").astype(np.float64)
cat = np.load(FLAT / "cand_category_id.npy").astype(np.int64)
users = f["user_id"].to_numpy().copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
keep = ~leak_mask(); log(f"val feeds {n}, leak-free {keep.sum()}")
W = pmix_weights(); wpriv = W["private"]

def z(x): return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
zG3 = 0.5 * z(np.load(P / "ens_G3_ft_val.npy")) + 0.5 * z(np.load(P / "ens_G3_to_val.npy"))
zTO = z(np.load(P / "ens_G3_to_val.npy"))
BASES = {"G3": zG3, "G3_to": zTO}
cold = np.load(ERR / "cold_luo.npy").astype(bool)          # it_hist_all_luo == 0
minh = np.load(ERR / "cand_cat_minh.npy")                   # min hpos of a same-category history event (1e6 = none)
log("base NDCG", {k: round(float(ndcg_per_feed(v, y, off, M)[keep].mean()), 5) for k, v in BASES.items()})


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


# ---------------- signals ----------------
S = {}
COLS = ['cv_raw5', 'cv_rec5', 'cv_fwd5', 'cv_invd20', 'cv_raw20', 'pop_hist', 'cv_users5', 'cv_rec5_norm', 'cv_fwd5_t1', 'cv_bwd5']
cvT = np.load(HL / "out_covis_val_train.npy").astype(np.float64)
cvVT = np.load(HL / "out_covis_val_train+val+test.npy").astype(np.float64)
S["A_cv_invd20_trainsrc"] = np.log1p(cvT[:, 3])
S["A_cv_invd20_tvt"] = np.log1p(cvVT[:, 3])
S["A_cv_invd20_vt_only"] = np.log1p(np.maximum(cvVT[:, 3] - cvT[:, 3], 0))     # the transductive increment alone
S["A_cv_rec5norm_tvt"] = cvVT[:, 7]
S["A_cv_users5_tvt"] = np.log1p(cvVT[:, 6])
S["A_cv_fwd5_tvt"] = np.log1p(cvVT[:, 2])
S["A_pop_hist_tvt"] = np.log1p(cvVT[:, 5])
tr = pl.read_parquet(HL / "out_item_trans_val.parquet").sort("row")
assert (tr["row"].to_numpy() == np.arange(len(y))).all()
for c in ["ho_later_all", "ho_later_fc", "ho_earlier_all", "ho_earlier_fc", "cand_other", "cand_other_later"]:
    S["B_" + c] = np.log1p(tr[c].to_numpy().astype(np.float64))
als = np.load(CS / "alsgpu_f128_i12_evt_a5_val.npz"); alsp = np.load(CS / "cf_alsgpu_f128_i12_evt_a5_profiles_val.npz")
alsc = np.load(CS / "alsclean_f64_i12_evt_a5_val.npz")
i2v = np.load(CS / "i2v_d128_e10_nf0_val.npz"); i2vp = np.load(CS / "cf_i2v_d128_e10_nf0_profiles_val.npz")
nan = lambda v, cov: np.where(cov, v.astype(np.float64), np.nan)
S["C_als128_dot"] = nan(als["dot"], als["cov"]); S["C_als128_lastk20"] = nan(als["lastk20"], als["cov"])
for k in ["exp5", "last10", "samecat_exp10", "et1_exp5"]:
    S["C_als128_prof_" + k] = nan(alsp[k], als["cov"])
S["C_alsclean64_dot"] = nan(alsc["dot"], alsc["cov"]); S["C_alsclean64_lastk20"] = nan(alsc["lastk20"], alsc["cov"])
S["C_i2v_cos_k5"] = nan(i2v["cos_k5"], i2v["cov"]); S["C_i2v_maxcos_k5"] = nan(i2v["maxcos_k5"], i2v["cov"])
for k in ["exp5", "samecat_exp10"]:
    S["C_i2v_prof_" + k] = nan(i2vp[k], i2v["cov"])
S["C_feat_als_exp5"] = np.load(FEATS / "als_exp5.npy").astype(np.float64)   # the on-disk family (als_v1 harness +0.0014)
for k in ["pre2_idmax32", "pre2_id_contrib", "pre_idmax32", "pre_id_contrib_v2", "pre2_hmax32"]:
    S["D_" + k] = np.load(FEATS / f"{k}.npy").astype(np.float64)
# E: category-level LSE of G3 z (set-level category aggregation), and feed share of the candidate's category
ck = fid * 64 + cat; uk, inv = np.unique(ck, return_inverse=True)
lse = np.log(np.bincount(inv, np.exp(zG3))); S["E_cat_lse_G3"] = lse[inv]
cnt = np.bincount(inv); S["E_cat_mean_G3"] = (np.bincount(inv, zG3) / cnt)[inv]
S["E_cat_max_G3"] = np.full(len(uk), -np.inf); np.maximum.at(S["E_cat_max_G3"], inv, zG3); S["E_cat_max_G3"] = S["E_cat_max_G3"][inv]
for k in ["seq_cv_invd20", "it_hist_all_luo"]:
    S["R_" + k] = np.log1p(np.maximum(np.load(FEATS / f"{k}.npy").astype(np.float64), 0))
for k in ["pre2_fe", "cos_u", "seq_max_all", "f1"]:
    S["R_" + k] = np.load((FLAT / f"cand_{k}.npy") if k == "f1" else (FEATS / f"{k}.npy")).astype(np.float64)
log("signals", len(S))

# ---------------- pairs ----------------
pi, ni, pf = [], [], []
for g in np.nonzero(keep)[0]:
    s, m = off[g], M[g]; yy = y[s:s + m]
    p = np.nonzero(yy == 1)[0] + s; q = np.nonzero(yy == 0)[0] + s
    if len(p) and len(q):
        pi.append(np.repeat(p, len(q))); ni.append(np.tile(q, len(p))); pf.append(np.full(len(p) * len(q), g))
pi = np.concatenate(pi); ni = np.concatenate(ni); pf = np.concatenate(pf)
same = cat[pi] == cat[ni]
# cell: warm|recent positive whose category is G3's top category, vs warm same-category beaters
kf = uk // 64; o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
pcr = cr[inv]
cell = same & (~cold[pi]) & (~cold[ni]) & (minh[pi] < 8) & (pcr[pi] == 0) & (zG3[ni] > zG3[pi])
cell_any = same & (~cold[ni]) & (zG3[ni] > zG3[pi])          # any positive vs same-cat warm beaters
log(f"pairs {len(pi):,} same-cat {same.mean():.3f} cell {cell.sum():,} ({cell.sum()/len(pi):.4f}) cell_any {cell_any.sum():,}")
def auc(d): return float(((d > 0) + 0.5 * (d == 0)).mean()) if len(d) else float("nan")

# ---------------- screen ----------------
WG = [0, .05, .1, .2, .3, .5]
folds = {seed: list(GroupKFold(5, shuffle=True, random_state=seed).split(np.arange(n), groups=users)) for seed in range(2)}
base_per = {b: ndcg_per_feed(v, y, off, M) for b, v in BASES.items()}
rows = []
hdr = f"{'signal':26s} {'cov':>5s} | {'AUCall':>6s} {'same':>6s} {'cell':>6s} {'cellA':>6s} | " + " | ".join(f"{b}: nested±SE  pmix±SE  w*" for b in BASES)
print(hdr, flush=True)
for name, x in S.items():
    x = np.asarray(x, np.float64); dfn = np.isfinite(x)
    zx = feed_z_nan(off, M, np.where(dfn, x, np.nan))
    d = zx[pi] - zx[ni]; both = dfn[pi] & dfn[ni]
    a_all, a_same, a_cell, a_cellA = auc(d[both]), auc(d[both & same]), auc(d[both & cell]), auc(d[both & cell_any])
    rec = dict(signal=name, cov=float(dfn.mean()), auc_all=a_all, auc_same=a_same, auc_cell=a_cell, auc_cell_any=a_cellA,
               cell_cov=float(both[cell].mean()))
    parts = []
    for b, zb in BASES.items():
        pers = {w: ndcg_per_feed(zb + w * zx, y, off, M) for w in WG}
        deltas = []
        for seed in range(2):
            out = np.zeros(n)
            for trn, va in folds[seed]:
                trk = trn[keep[trn]]
                wbest = max(WG, key=lambda w: pers[w][trk].mean())
                out[va] = pers[wbest][va] - base_per[b][va]
            deltas.append(out)
        dn = np.mean(deltas, 0)
        m_, se_ = dn[keep].mean(), dn[keep].std() / np.sqrt(keep.sum())
        pm, pse = wmean_se(dn[keep], wpriv[keep])
        wfull = max(WG, key=lambda w: pers[w][keep].mean())
        parts.append(f"{m_:+.5f}±{se_:.5f} {pm:+.5f}±{pse:.5f} w{wfull:.2f}")
        rec[f"nested_{b}"] = m_; rec[f"nested_se_{b}"] = se_; rec[f"pmix_{b}"] = pm; rec[f"pmix_se_{b}"] = pse; rec[f"wfull_{b}"] = wfull
    rows.append(rec)
    print(f"{name:26s} {dfn.mean():5.3f} | {a_all:6.4f} {a_same:6.4f} {a_cell:6.4f} {a_cellA:6.4f} | " + " | ".join(parts), flush=True)
pl.DataFrame(rows).write_csv(OUT / "probe_residuals.csv")
log("done")
