"""Adversarial checks part 2: C4/C5 recompute from raw preds, M2-vs-G3 paired slices, Vp, lambda crossover,
CatBoost info row with leak-free (luo) model only. CPU only, read-only."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "12"
import sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/nn_x/pubdiag/overfit")
import numpy as np
from pathlib import Path
from common import feed_meta, FLAT
from stack_val import feed_rank_z
from metrics import ndcg_per_feed
from lib import holdout_users_ts

ROOT = Path("<repo>"); P = ROOT / "preds"; H = P / "holdout"
F = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{s}" for s in range(92, 97)],
    "F7x": [f"r6_cd_m02_r{s}" for s in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{s}" for s in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{s}" for s in range(701, 708)],
}
fv, Mv, ov = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
Mh, oh = np.load(H / "holdout_meta.npy"); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(H / "holdout_y.npy").astype(np.float64)
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]
allm = [m for f in F.values() for m in f]
Zto = {m: zv(np.load(P / f"nn_{m}_trainonly_val.npy")) for m in allm}
Zft = {m: zv(np.load(P / f"nn_{m}_n2cpv3_val_oof.npy")) for m in allm}
Zh = {m: zh(np.load(H / f"nn_{m}_holdout.npy")) for m in allm}

def pool(w):
    s = sum(w.values())
    to = (sum(x * Zto[m].astype(np.float64) for m, x in w.items()) / s).astype(np.float32)
    ft = (sum(x * Zft[m].astype(np.float64) for m, x in w.items()) / s).astype(np.float32)
    ho = (sum(x * Zh[m].astype(np.float64) for m, x in w.items()) / s).astype(np.float32)
    fin = 0.5 * zv(ft) + 0.5 * zv(to)
    return dict(val=ndcg_per_feed(fin, yv, ov, Mv), val_to=ndcg_per_feed(to, yv, ov, Mv),
                hold=ndcg_per_feed(ho, yh, oh, Mh), fin=fin)
def W(fams, dedup=False):
    w = {m: 1.0 for f in fams for m in F[f]}
    if dedup:
        for r in range(101, 108):
            if f"r6_cd_m02_r{r}" in w and f"r6_cd_m02_p70k_r{r}" in w:
                w[f"r6_cd_m02_r{r}"] = w[f"r6_cd_m02_p70k_r{r}"] = 0.5
    return w
C = {"PF7x": W(["F5cd", "F6", "F7x", "F9", "F10", "F12"]), "G3": W(["F10", "F7x", "F7x70"]), "M2": W(["F7x", "F7x70"]),
     "PX70": W(["F5cd", "F6", "F7x", "F7x70", "F9", "F10", "F12"]), "PX70r": W(["F5cd", "F6", "F7x", "F7x70", "F9", "F10", "F12"], True),
     "HG": W(["F5cd", "F9", "F10", "F12", "F7x", "F7x70"])}
R = {k: pool(w) for k, w in C.items()}

def ug(users):
    g = users.astype(np.int64).copy(); z = np.where(users == 0)[0]; g[z] = -(np.arange(len(z)) + 1)
    return np.unique(g, return_inverse=True)[1]
def clse(d, g):
    s = np.bincount(g, d - d.mean()); return np.sqrt((s ** 2).sum()) / len(d)
def wse(d, w):
    w = np.asarray(w, np.float64); m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum()
    return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)
gv = ug(fv["user_id"].to_numpy()); hu, hts = holdout_users_ts(Mh); gh = ug(hu)
Wp = np.load(ROOT / "nn_x/pubdiag/shift/p2_adv_weights.npz")["private_intrinsic+hour"]
hday = (hts + 10800) // 86400; hdow = (hday + 3) % 7
print("holdout MSK day range", hday.min(), hday.max(), "weekend share", np.mean(hdow >= 5))
tue = (fv["timestamp"].to_numpy() + 10800) // 86400 < 20720
for k, r in R.items():
    print(f"{k:6s} n={len(C[k])} slots={sum(C[k].values()):.0f} val {r['val'].mean():.5f} val_to {r['val_to'].mean():.5f} hold {r['hold'].mean():.5f}")
print("\nvs PF7x: V | Vp | Vto | H | D | E0 E1 Ehalf S | hold weekday / weekend z")
E = {}
for k in ("G3", "M2", "PX70", "PX70r", "HG"):
    dv = R[k]["val"] - R["PF7x"]["val"]; dto = R[k]["val_to"] - R["PF7x"]["val_to"]; dh = R[k]["hold"] - R["PF7x"]["hold"]
    V = (dv.mean(), clse(dv, gv)); Vp = wse(dv, Wp); Vto = (dto.mean(), clse(dto, gv)); Hh = (dh.mean(), clse(dh, gh))
    D = Vto[0] - Hh[0]; E0, E1 = Vp[0], Vp[0] - D; E[k] = (E0, D)
    wd, we = hdow < 5, hdow >= 5
    zwd = dh[wd].mean() / clse(dh[wd], ug(hu[wd])); zwe = dh[we].mean() / clse(dh[we], ug(hu[we]))
    print(f"{k:6s} V {V[0]:+.5f}±{V[1]:.5f} | Vp {Vp[0]:+.5f}±{Vp[1]:.5f} | Vto {Vto[0]:+.5f} | H {Hh[0]:+.5f}±{Hh[1]:.5f} | D {D:+.5f} | "
          f"{E0:+.5f} {E1:+.5f} {E0 - 0.5 * D:+.5f} {min(E0, E1):+.5f} | wd z {zwd:+.1f} we z {zwe:+.1f}")
# lambda crossover between M2 and G3: E(l) = Vp - l*D
(e0m, dm), (e0g, dg) = E["M2"], E["G3"]
lam = (e0m - e0g) / (dm - dg)
print(f"\nE(lambda)=Vp-lambda*D: M2 {e0m:+.5f}-{dm:.5f}l, G3 {e0g:+.5f}-{dg:.5f}l -> G3 better only for lambda > {lam:.3f}")
dv = R["M2"]["val"] - R["G3"]["val"]; dh = R["M2"]["hold"] - R["G3"]["hold"]; dto = R["M2"]["val_to"] - R["G3"]["val_to"]
print(f"M2-G3 paired: val {dv.mean():+.5f}±{clse(dv, gv):.5f}  val_to {dto.mean():+.5f}±{clse(dto, gv):.5f}  hold {dh.mean():+.5f}±{clse(dh, gh):.5f}  Vp {wse(dv, Wp)[0]:+.5f}±{wse(dv, Wp)[1]:.5f}")
cold = np.nan_to_num(np.load(ROOT / "nn_x/r6/feat_cache/val/it_hist_all_luo.npy").astype(np.float64), nan=0.0) == 0
fid = np.repeat(np.arange(len(Mv)), Mv); n = len(Mv)
npos = np.bincount(fid, yv, n); ncold = np.bincount(fid, yv * cold, n)
coldpos = (npos > 0) & (ncold == npos); warmpos = (npos > 0) & (ncold == 0)
for nm, m in [("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", tue), ("Thu", ~tue), ("coldpos", coldpos), ("warmpos", warmpos), ("mixedpos", (npos > 0) & ~coldpos & ~warmpos)]:
    print(f"  M2-G3 {nm:8s} n={m.sum():5d} {dv[m].mean():+.5f}±{clse(dv[m], ug(fv['user_id'].to_numpy()[m])):.5f}")
print("  G3-PF7x coldpos", (R["G3"]["val"] - R["PF7x"]["val"])[coldpos].mean())

# CatBoost: leak-free (luo) model only vs leaky ones
print("\nCatBoost (val, train-fit):")
for nm in ["cb_tr300k_luo", "cb_tr300k_none", "cb_tr500k_pw"]:
    x = zv(np.load(P / f"{nm}_val.npy")); base = R["G3"]["fin"]; zb = zv(base)
    rho = np.mean([np.corrcoef(np.argsort(np.argsort(x[o:o + m])), np.argsort(np.argsort(zb[o:o + m])))[0, 1] for o, m in zip(ov[:4000], Mv[:4000]) if m > 2])
    out = [f"{nm}: val {ndcg_per_feed(x, yv, ov, Mv).mean():.5f} rho~{rho:.3f}"]
    for w in (0.02, 0.05, 0.10):
        d = ndcg_per_feed(zb + w * x, yv, ov, Mv) - ndcg_per_feed(base, yv, ov, Mv)
        out.append(f"+{w}: {d.mean():+.5f}±{clse(d, gv):.5f}")
    print("  " + " | ".join(out))
