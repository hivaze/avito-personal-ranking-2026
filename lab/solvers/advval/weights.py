"""Importance-weighted val scores from the feed-level adversarial classifiers (density ratio p(B)/p(val)), compared with
src/calib.py's M-only reweighting; correlation of per-feed NDCG with the 'looks like private' score.
Pre-declared: clip normalised weights to [0.1, 10]; primary variant = 'intrinsic' (no time, no coverage features)."""
import json, sys
from common_av import *
sys.path.insert(0, f"{ROOT}/src")
import calib  # noqa: E402
from scipy.stats import spearmanr

fv, Mv, offv = meta("val")
y = np.load(f"{FLAT}/val/cand_target.npy")
ts = fv["timestamp"].to_numpy()
wd = (((ts.astype(np.int64) + MSK) // 86400) + 3) % 7
z = lambda n: feed_z(offv, Mv, load_pred(n))
nd = lambda s: ndcg_per_feed(s, y, offv, Mv)

S = {
    "v10": nd(0.5 * z("nn_final4_val_oof") + 0.5 * z("nn_final4_trainonly_val") + 0.1 * z("pre_final_val")),
    "v11": nd(0.5 * z("nn_final5_val_oof") + 0.5 * z("nn_final5_trainonly_val")),
    "f5_ft": nd(load_pred("nn_final5_val_oof")),
    "f5_to": nd(load_pred("nn_final5_trainonly_val")),
    "cd12": nd(0.5 * z("nn_final5cd12_val_oof") + 0.5 * z("nn_final5cd12_trainonly_val")),
    "cd6": nd(0.5 * z("nn_final5cd_val_oof") + 0.5 * z("nn_final5cd_trainonly_val")),
    "f4_ft": nd(load_pred("nn_final4_val_oof")),
    "f4_to": nd(load_pred("nn_final4_trainonly_val")),
    "pre_final": nd(load_pred("pre_final_val")),
    "f1": nd(np.load(f"{FLAT}/val/cand_f1.npy").astype(np.float64)),
}
# random-ranking expectation per feed (to separate the M effect): mean NDCG of 20 random permutations
rng = np.random.default_rng(0)
S_rand = np.mean([nd(rng.random(len(y))) for _ in range(20)], 0)
chk = float(S["v11"].mean())
print("v11 val", round(chk, 5), " v10 val", round(float(S["v10"].mean()), 5))
DELTAS = {"v11-v10": ("v11", "v10"), "f5_ft-f5_to": ("f5_ft", "f5_to"), "cd12-v11": ("cd12", "v11"),
          "cd6-v11": ("cd6", "v11"), "v11-f5_ft": ("v11", "f5_ft"), "v11-f5_to": ("v11", "f5_to"),
          "f4_ft-f4_to": ("f4_ft", "f4_to")}


def adv_w(comp, sname, clip=(0.1, 10.0)):
    d = np.load(f"{OUT}/feed_oof_{comp}_{sname}.npz")
    p, yy, fidx = d["oof"], d["y"], d["fidx"]
    pv = np.full(len(fv), np.nan)
    pv[fidx[yy == 0]] = p[yy == 0]
    nA, nB = (yy == 0).sum(), (yy == 1).sum()
    w = pv / np.maximum(1 - pv, 1e-6) * nA / nB
    w = w / w.mean()
    if clip:
        w = np.clip(w, *clip); w = w / w.mean()
    return w, pv


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


rows = {}
W = {"uniform": np.ones(len(fv)), "M_private": calib.weights("private"), "M_public": calib.weights("public")}
P = {}
for tgt, comp in [("private", "a_val_vs_private"), ("public", "b_val_vs_public")]:
    for sname in ["intrinsic", "notime", "all"]:
        w, pv = adv_w(comp, sname)
        W[f"adv_{tgt}_{sname}"] = w; P[f"{tgt}_{sname}"] = pv
        W[f"adv_{tgt}_{sname}_noclip"] = adv_w(comp, sname, clip=None)[0]
        W[f"adv_{tgt}_{sname}_clip5"] = adv_w(comp, sname, clip=(0.2, 5.0))[0]
# Thu-only (staleness proxy: the val day closest to test)
W["thu_only"] = (wd == 3).astype(float) / (wd == 3).mean()

res = {"levels": {}, "deltas": {}, "ess": {}}
for wn, w in W.items():
    res["ess"][wn] = ess(w)
    res["levels"][wn] = {k: wmean_se(v, w) for k, v in S.items()}
    res["deltas"][wn] = {k: wmean_se(S[a] - S[b], w) for k, (a, b) in DELTAS.items()}

order = ["uniform", "M_private", "adv_private_intrinsic", "adv_private_notime", "adv_private_notime_clip5",
         "adv_private_all", "thu_only", "M_public", "adv_public_intrinsic", "adv_public_notime", "adv_public_all"]
print("\n| weighting | ESS | " + " | ".join(S.keys()) + " |")
for wn in order:
    print(f"| {wn} | {res['ess'][wn]:.0f} | " + " | ".join(f"{res['levels'][wn][k][0]:.5f}" for k in S) + " |")
print("\n| weighting | " + " | ".join(DELTAS.keys()) + " |")
for wn in order:
    print(f"| {wn} | " + " | ".join(f"{res['deltas'][wn][k][0]:+.5f} +- {res['deltas'][wn][k][1]:.5f}" for k in DELTAS) + " |")

# ---------------- item 2: per-feed NDCG vs 'looks like private' score
print("\n### correlation of per-feed NDCG with OOF p(private) (val feeds)")
lift = {k: S[k] - S_rand for k in S}
# M-bucket residual of lift
bM = np.searchsorted(calib.EDGES, Mv, side="right")
def mres(x):
    mu = np.bincount(bM, x) / np.bincount(bM)
    return x - mu[bM]
corr = {}
for key in ["private_intrinsic", "private_notime", "public_intrinsic", "public_notime"]:
    p = P[key]
    r = {}
    for k in ["v11", "v10", "f5_ft", "f5_to"]:
        r[k] = dict(ndcg=spearmanr(p, S[k])[0], lift=spearmanr(p, lift[k])[0], lift_mres=spearmanr(p, mres(lift[k]))[0])
    for dn in ["v11-v10", "f5_ft-f5_to"]:
        a, b = DELTAS[dn]
        r[dn] = dict(delta=spearmanr(p, S[a] - S[b])[0])
    r["M"] = spearmanr(p, Mv)[0]
    corr[key] = r
    print(key, json.dumps({k: {kk: round(float(vv), 4) for kk, vv in v.items()} if isinstance(v, dict) else round(float(v), 4) for k, v in r.items()}))
# decile tables
dec = {}
for key in ["private_intrinsic", "private_notime"]:
    p = P[key]
    q = np.searchsorted(np.quantile(p, np.linspace(0, 1, 11)[1:-1]), p, side="right")
    tab = []
    for d in range(10):
        m = q == d
        tab.append(dict(dec=d, n=int(m.sum()), p=float(p[m].mean()), M=float(Mv[m].mean()), v11=float(S["v11"][m].mean()),
                        v11_lift=float(lift["v11"][m].mean()), rand=float(S_rand[m].mean()),
                        d_v11_v10=wmean_se((S["v11"] - S["v10"])[m], np.ones(m.sum())),
                        d_ft_to=wmean_se((S["f5_ft"] - S["f5_to"])[m], np.ones(m.sum())),
                        thu=float((wd[m] == 3).mean())))
    dec[key] = tab
    print(f"\n{key} deciles: dec | n | mean p | mean M | Thu share | v11 | random | v11 lift | v11-v10 | ft-to")
    for t in tab:
        print(f"{t['dec']} | {t['n']} | {t['p']:.3f} | {t['M']:.0f} | {t['thu']:.2f} | {t['v11']:.4f} | {t['rand']:.4f} | "
              f"{t['v11_lift']:.4f} | {t['d_v11_v10'][0]:+.4f} +- {t['d_v11_v10'][1]:.4f} | {t['d_ft_to'][0]:+.4f} +- {t['d_ft_to'][1]:.4f}")
# weight correlation public vs private
print("\nweight correlations (Spearman): M_public vs M_private", round(spearmanr(W["M_public"], W["M_private"])[0], 3),
      " adv intrinsic pub vs priv", round(spearmanr(W["adv_public_intrinsic"], W["adv_private_intrinsic"])[0], 3),
      " adv notime pub vs priv", round(spearmanr(W["adv_public_notime"], W["adv_private_notime"])[0], 3))
json.dump(dict(res=res, corr=corr, dec=dec), open(f"{OUT}/weights.json", "w"), indent=1, default=float)
np.savez(f"{OUT}/val_weights.npz", **{k: v for k, v in W.items()})
