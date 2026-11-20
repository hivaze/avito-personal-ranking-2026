"""EMB-USABILITY stage 1: label tests exactly as PREDECLARED.txt (AUC, lifts, nested residual on G3: grid / cell-wise / GBM).
CPU only. Reads out/sig_val.npz from s0_prep.py. Writes out/eval.log, out/eval.csv.
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, argparse
from pathlib import Path
import numpy as np, polars as pl, numba as nb
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from qt import leak_mask
from pmix import weights as pmix_weights, wmean_se
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ap = argparse.ArgumentParser(); ap.add_argument("--gbm", default="primary", help="primary|all|none"); args = ap.parse_args()
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat/val"; FEATS = ROOT / "data/feats/val"
OUT = ROOT / "nn_x/emb/usability/out"
T0 = time.time()
LOG = open(OUT / "eval.log", "a")
def log(*a):
    s = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

f = pl.read_parquet(FLAT / "feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64); n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "cand_target.npy").astype(np.float64)
cat = np.load(FLAT / "cand_category_id.npy").astype(np.int64)
uid = f["user_id"].to_numpy()
users = uid.copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
keep = ~leak_mask(); keepr = keep[fid]
wpriv = pmix_weights()["private"]
def z(x): return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
zG3 = z(np.load(ROOT / "preds/ens_G3_final_val.npy"))
base_per = ndcg_per_feed(zG3, y, off, M)
log(f"feeds {n}, leak-free {keep.sum()}, base G3 NDCG all {base_per.mean():.5f} leak-free {base_per[keep].mean():.5f}")

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

# ---------------- signals
S0 = np.load(OUT / "sig_val.npz")
SIG = {}   # name -> (item, role)
for nm, it, role in [("chg_d", 1, "P"), ("chg_flag", 1, "S"), ("in_train", 1, "S"),
                     ("zca_maxcos", 2, "P"), ("cen_maxcos", 2, "S"), ("abtt3_maxcos", 2, "S"), ("cen_new5", 2, "S"),
                     ("abtt3_new5", 2, "S"), ("zca_new5", 2, "S"), ("raw_maxcos", 2, "ref"), ("raw_new5", 2, "ref"),
                     ("proj_mu", 3, "P"), ("proj_pc1", 3, "S"),
                     ("nd_act", 4, "P"), ("nd_view_only", 4, "S"), ("maxcos_act", 4, "S"),
                     ("cos_u", 5, "P")]:
    SIG[nm] = (it, role)
X = {nm: (np.load(FEATS / "cos_u.npy").astype(np.float64) if nm == "cos_u" else S0[nm].astype(np.float64)) for nm in SIG}
# cos_u: feature uses the zero user embedding for 1.4% of feeds -> leave as is (it is what G3 sees)

# ---------------- pairs (leak-free)
pi, ni = [], []
for g in np.nonzero(keep)[0]:
    s, m = off[g], M[g]; yy = y[s:s + m]
    p = np.nonzero(yy == 1)[0] + s; q = np.nonzero(yy == 0)[0] + s
    if len(p) and len(q):
        pi.append(np.repeat(p, len(q))); ni.append(np.tile(q, len(p)))
pi = np.concatenate(pi); ni = np.concatenate(ni)
same = cat[pi] == cat[ni]
pop = np.log1p(np.maximum(np.load(FEATS / "it_hist_all_luo.npy").astype(np.float64), 0))
edges = np.unique(np.quantile(pop, np.linspace(0, 1, 11)[1:-1])); pdec = np.digitize(pop, edges)
samepop = same & (pdec[pi] == pdec[ni])
log(f"pairs {len(pi):,}; same-cat {same.mean():.3f}; same-cat+same-pop-decile {samepop.mean():.3f} ({len(edges)+1} pop bins)")
def auc(d): return float(((d > 0) + 0.5 * (d == 0)).mean()) if len(d) else float("nan")
def auc_se(d, pf):
    # feed-clustered SE of the pair AUC (pairs grouped by feed)
    s = (d > 0) + 0.5 * (d == 0); a = s.mean()
    g = np.bincount(pf, weights=s - a); return a, np.sqrt((g ** 2).sum()) / len(s)
pf = fid[pi]

# ---------------- nested
WG = [0, .02, -.02, .05, -.05, .1, -.1, .15, -.15, .2, -.2, .3, -.3, .5, -.5]
folds = {seed: list(GroupKFold(5, shuffle=True, random_state=seed).split(np.arange(n), groups=users)) for seed in range(2)}
mt = np.digitize(M, np.quantile(M, [1 / 3, 2 / 3])); ht = np.digitize(H, np.quantile(H, [1 / 3, 2 / 3])); cellid = mt * 3 + ht
log("cell sizes (M tercile x H tercile)", np.bincount(cellid).tolist())

def nested(zx):
    pers = np.stack([ndcg_per_feed(zG3 + w * zx, y, off, M) for w in WG])   # (len(WG), n)
    dg, dc, wg = [], [], []
    for seed in range(2):
        og = np.zeros(n); oc = np.zeros(n)
        for trn, va in folds[seed]:
            trk = trn[keep[trn]]
            jb = int(np.argmax(pers[:, trk].mean(1))); wg.append(WG[jb])
            og[va] = pers[jb, va] - pers[0, va]
            for c in range(9):
                tc = trk[cellid[trk] == c]; vc = va[cellid[va] == c]
                if len(tc) == 0 or len(vc) == 0: continue
                jc = int(np.argmax(pers[:, tc].mean(1)))
                oc[vc] = pers[jc, vc] - pers[0, vc]
        dg.append(og); dc.append(oc)
    dg = np.mean(dg, 0)[keep]; dc = np.mean(dc, 0)[keep]
    jfull = int(np.argmax(pers[:, keep].mean(1)))
    return dict(nest=dg.mean(), nest_se=dg.std() / np.sqrt(len(dg)), pmix=wmean_se(dg, wpriv[keep])[0], pmix_se=wmean_se(dg, wpriv[keep])[1],
                cell=dc.mean(), cell_se=dc.std() / np.sqrt(len(dc)), wfull=WG[jfull], full_gain=pers[jfull, keep].mean() - pers[0, keep].mean(),
                wsel=",".join(f"{w:+.2f}" for w in wg))

# GBM variant
u0 = (uid == 0).astype(np.float64)
CTRL = np.stack([zG3, np.log(M)[fid], np.log1p(H)[fid], u0[fid]], 1)
PRM = dict(objective="lambdarank", learning_rate=0.05, num_leaves=15, min_data_in_leaf=500, seed=0, verbose=-1,
           num_threads=12, lambdarank_truncation_level=300, deterministic=True)
def gbm_oof(Xm):
    out = np.zeros(len(y))
    for trn, va in folds[0]:
        trk = np.sort(trn[keep[trn]]); vs = np.sort(va)
        rtr = np.concatenate([np.arange(off[g], off[g] + M[g]) for g in trk])
        rva = np.concatenate([np.arange(off[g], off[g] + M[g]) for g in vs])
        ds = lgb.Dataset(Xm[rtr], y[rtr], group=M[trk], free_raw_data=True)
        b = lgb.train(PRM, ds, num_boost_round=150)
        out[rva] = b.predict(Xm[rva], num_threads=12)
    return out
gbm_ctrl_per = None
def gbm_delta(x, zx):
    global gbm_ctrl_per
    if gbm_ctrl_per is None:
        gbm_ctrl_per = ndcg_per_feed(gbm_oof(CTRL), y, off, M)
        d0 = (gbm_ctrl_per - base_per)[keep]
        log(f"GBM control (zG3, logM, logH, user0) minus base G3: {d0.mean():+.5f} +- {d0.std()/np.sqrt(len(d0)):.5f}")
    per = ndcg_per_feed(gbm_oof(np.column_stack([CTRL, x, zx])), y, off, M)
    d = (per - gbm_ctrl_per)[keep]
    return d.mean(), d.std() / np.sqrt(len(d))

# ---------------- lifts (feed-clustered delta method, leak-free rows)
def rate(mask):
    mk = mask & keepr; yy = np.bincount(fid[mk], y[mk], n); nn = np.bincount(fid[mk], minlength=n).astype(float)
    r = yy.sum() / nn.sum(); return r, yy, nn
def lift(mA, mB):
    rA, yA, nA = rate(mA); rB, yB, nB = rate(mB)
    infl = (yA - rA * nA) / (rA * nA.sum()) - (yB - rB * nB) / (rB * nB.sum())
    se_log = np.sqrt((infl ** 2).sum()); L = rA / rB
    seA = np.sqrt((((yA - rA * nA) / nA.sum()) ** 2).sum()); seB = np.sqrt((((yB - rB * nB) / nB.sum()) ** 2).sum())
    return dict(rA=rA, seA=seA, nA=int(nA.sum()), rB=rB, seB=seB, nB=int(nB.sum()), lift=L, lift_lo=L * np.exp(-1.96 * se_log), lift_hi=L * np.exp(1.96 * se_log))
def plift(name, d):
    log(f"  LIFT {name}: rate A {d['rA']:.5f}+-{d['seA']:.5f} (n {d['nA']:,}) vs B {d['rB']:.5f}+-{d['seB']:.5f} (n {d['nB']:,}) "
        f"-> lift {d['lift']:.3f} [95% {d['lift_lo']:.3f}, {d['lift_hi']:.3f}]")

log("=== ITEM 1 lifts")
ch = S0["chg_flag"]; it = S0["in_train"] == 1
plift("changed (cos<0.99) vs unchanged, items with a train vector", lift(ch == 1, ch == 0))
plift("in train table vs not", lift(it, ~it))
ctv = S0["cos_tv"]
plift("strongly changed (cos<0.9) vs unchanged", lift(it & (ctv < 0.9), ch == 0))
bi = S0["bit_ident"]
plift("bit-identical vs not (but cos>=0.99)", lift(bi == 1, (bi == 0) & (ch == 0)))
log(f"  category differs between tables: changed {(S0['cat_tr'][ch == 1] != cat[ch == 1]).mean():.4f} vs unchanged {(S0['cat_tr'][ch == 0] != cat[ch == 0]).mean():.4f}")
for nm in ["f1", "f2", "f3", "f4", "f5"]:
    v = np.load(FLAT / f"cand_{nm}.npy").astype(np.float64)
    log(f"  mean {nm}: changed {v[ch == 1].mean():+.4f}  unchanged {v[ch == 0].mean():+.4f}  not-in-train {v[~it].mean():+.4f}")
log(f"  mean log1p(it_hist_all_luo): changed {pop[ch == 1].mean():.3f} unchanged {pop[ch == 0].mean():.3f} not-in-train {pop[~it].mean():.3f}")
log("=== ITEM 4 lifts")
na = S0["nd_act"] == 1; nv = S0["nd_view_only"] == 1; nei = ~na & ~nv
plift("nd_act vs neither", lift(na, nei)); plift("nd_view_only vs neither", lift(nv, nei)); plift("nd_act vs nd_view_only", lift(na, nv))
mca = S0["maxcos_act"]
for lo, hi in [(0.9, 0.95), (0.95, 0.98), (0.98, 0.995), (0.995, 1.01)]:
    mk = (mca > lo) & (mca <= hi); plift(f"maxcos_act in ({lo},{hi}] vs neither", lift(mk, nei))

# ---------------- screen
rows = []
log("=== SCREEN (leak-free val; nested = grid, cell = M x H tercile cell-wise grid; GBM = with-x minus control)")
for nm, (itm, role) in SIG.items():
    x = X[nm]; fin = np.isfinite(x)
    zx = feed_z_nan(off, M, np.where(fin, x, np.nan))
    d = x[pi] - x[ni]; both = fin[pi] & fin[ni]
    a_all, a_se = auc_se(d[both], pf[both]); a_same = auc(d[both & same]); a_sp = auc(d[both & samepop])
    r = dict(signal=nm, item=itm, role=role, cov=float(fin.mean()), auc=a_all, auc_se=a_se, auc_same=a_same, auc_samecat_pop=a_sp)
    r.update(nested(zx))
    if args.gbm == "all" or (args.gbm == "primary" and (role == "P" or nm in ("chg_flag", "maxcos_act"))):
        r["gbm"], r["gbm_se"] = gbm_delta(np.where(fin, x, np.nan), zx)
    rows.append(r)
    log(f"{nm:14s} it{itm} {role:3s} cov {r['cov']:.3f} | AUC {a_all:.4f}+-{a_se:.4f} same {a_same:.4f} same+pop {a_sp:.4f} | "
        f"nested {r['nest']:+.5f}+-{r['nest_se']:.5f} (w_full {r['wfull']:+.2f} in-sample {r['full_gain']:+.5f}; folds {r['wsel']}) "
        f"pmix {r['pmix']:+.5f}+-{r['pmix_se']:.5f} | cell {r['cell']:+.5f}+-{r['cell_se']:.5f}"
        + (f" | GBM {r['gbm']:+.5f}+-{r['gbm_se']:.5f}" if "gbm" in r else ""))
pl.DataFrame(rows).write_csv(OUT / "eval.csv")
log("done")
