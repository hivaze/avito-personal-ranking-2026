"""B: the small segments (H = 0, H 1-9, user_id 0): what is informative, how FINAL compares, nested best stored family, ceilings."""
from lib import *
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
ld = lambda n: np.load(P / n)
ND = dict(np.load(OUT / "rk_ndcg.npz")); AU = dict(np.load(OUT / "rk_auc.npz")); e = ND["FINAL"]
cov = pl.read_parquet(OUT / "feed_cov.parquet")
SG = {"H=0": Hv == 0, "H 1-9": (Hv >= 1) & (Hv <= 9), "H<=9": Hv <= 9, "uid0": uid == 0, "uid0 & H>=10": (uid == 0) & (Hv >= 10), "ALL": np.ones(NF, bool)}
print("### segment make-up (label-free)")
rows = [[k, int(m.sum()), f"{100 * m.mean():.2f}", f"{Mv[m].mean():.1f}", f"{np.median(Hv[m]):.0f}", f"{(uid[m] == 0).mean():.3f}", f"{cov['cold_share'].to_numpy()[m].mean():.3f}", f"{(cov['ue_norm'].to_numpy()[m] > 0).mean():.3f}", f"{NPOS[m].mean():.2f}"] for k, m in SG.items()]
print(tab(rows, ["segment", "feeds", "val %", "mean M", "median H", "share uid0", "cold share of cands", "share with non-zero user_emb", "n_pos"]))
print("uid0 feeds by H bucket:", {n: int(((uid == 0) & bmask[n]).sum()) for n in bmask})
# 1. rankers on the segments
keys = ["FINAL", "FD16 (16 students, no pop-light)", "FD16 TO branch", "FD16 FT branch", "POPLIGHT pool (4)", "TEACHER 11 hard-label (TO)", "G3 file (hard labels, FT+TO)", "popularity only (it_hist_all_luo)", "exposure only (xp_imp_c50)",
        "cosine to user embedding", "pre-trained encoder score (pre2_fe)", "max cosine to history (all 256)", "f1 only (+ sign, chosen by overall AUC)"]
for sg in ("H=0", "H 1-9", "uid0"):
    m = SG[sg]; rows = []
    for k in keys:
        d = ND[k] - e
        rows.append([k, "%.4f+-%.4f" % cse(ND[k], m)[:2], "%.4f" % cse(AU[k], m)[0], "" if k == "FINAL" else "%+.4f+-%.4f" % cse(d, m)[:2], "" if k == "FINAL" else "%+.4f+-%.4f" % cse(AU[k] - AU["FINAL"], m)[:2]])
    print(f"\n### segment {sg} ({int(m.sum())} feeds): rankers\n" + tab(rows, ["ranker", "NDCG", "AUC", "NDCG - FINAL (paired)", "AUC - FINAL (paired)"]))
# 2. which raw inputs are still informative (within-feed AUC, sign fixed by the ALL-feeds AUC; share of feeds where the input is not constant)
f = lambda n: np.nan_to_num(np.load(FC / f"{n}.npy").astype(np.float64), nan=0.0)
INP = {f"f{k}": np.load(FLAT / f"val/cand_f{k}.npy").astype(np.float64) for k in range(1, 6)}
for n in ("it_hist_all_luo", "it_hist_et1_luo", "xp_imp_c50", "it_tr_ctr", "it_tr_coec", "seq_cv_invd20", "seq_cv_pop", "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_u", "ctx_nd_max", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_rrw", "ctx_hs_k128",
          "pre2_fe", "pre2_new", "pre2_hmax32", "als_exp5", "centrality"):
    try: INP[n] = f(n)
    except FileNotFoundError: pass
INP["cos_u (user embedding)"] = np.load(OUT / "cos_u.npy").astype(np.float64); INP["smax (history cosine)"] = np.load(OUT / "cos_smaxw.npy")[:, 5].astype(np.float64)
ccat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
catshare = np.zeros(len(yv))
for k in range(NF):
    c = ccat[ov[k]:ov[k] + Mv[k]]; catshare[ov[k]:ov[k] + Mv[k]] = np.bincount(c, minlength=64)[c] / Mv[k]
INP["candidate-category share in the feed"] = catshare
rows = []
for n, x in INP.items():
    a = feed_auc(x); sgn = 1 if np.nanmean(a) >= 0.5 else -1; a = a if sgn == 1 else 1 - a
    mx = np.maximum.reduceat(x, ov); mn = np.minimum.reduceat(x, ov); nc = mx > mn
    rows.append([n, "+" if sgn == 1 else "-", f"{np.nanmean(a):.3f}"] + [f"{cse(a, SG[s])[0]:.3f}+-{cse(a, SG[s])[1]:.3f}" for s in ("H=0", "H 1-9", "uid0")] + [f"{nc[SG[s]].mean():.2f}" for s in ("H=0", "H 1-9", "uid0")])
print("\n### raw inputs: within-feed AUC (sign fixed on ALL feeds) and share of feeds where the input varies\n" + tab(rows, ["input", "sign", "AUC ALL", "AUC H=0", "AUC H 1-9", "AUC uid0", "varies H=0", "varies H1-9", "varies uid0"]))
# 3. largest gain any stored family shows (candidate set fixed in DECLARATION block 0), nested by user
bft, bto, pft, pto = (np.load(OUT / f"pool_{k}.npy") for k in ("b_ft", "b_to", "p_ft", "p_to"))
CAND = {}
for w in (0, 0.1, 0.3, 0.4, 0.6, 0.8, 1.0):
    CAND[f"pop-light weight {w}"] = ev(fin((1 - w) * Z(bft) + w * Z(pft), (1 - w) * Z(bto) + w * Z(pto)))
CAND["FT branch only"] = ev(ld("ens_FD16P20kd_ft_val.npy")); CAND["TO branch only"] = ev(ld("ens_FD16P20kd_to_val.npy"))
for k, t in (("W8 file", "W8kd"), ("W16 file", "W16kd"), ("S16 file (d128 students)", "S16kd"), ("G3 file (hard labels)", "G3"), ("Q1a popularity-free file", "Q1apopfree"), ("Q2a pop1 file (hard labels)", "Q2apop1")):
    CAND[k] = ev(fin(ld(f"ens_{t}_ft_val.npy"), ld(f"ens_{t}_to_val.npy")))
names = list(CAND); Dm = np.stack([CAND[k] - e for k in names], 1)   # [NF, n_cand] paired deltas vs FINAL
print(f"\n### candidates vs FINAL on ALL val: " + " | ".join(f"{k} {Dm[:, j].mean():+.5f}" for j, k in enumerate(names)))
rng = np.random.default_rng(11); REP = 20
ft_, Mt, ot = feed_meta("test"); Ht = ft_["H"].to_numpy(); ut = ft_["user_id"].to_numpy()
tshare = {"H=0": (Ht == 0).mean(), "H 1-9": ((Ht >= 1) & (Ht <= 9)).mean(), "H<=9": (Ht <= 9).mean(), "uid0": (ut == 0).mean(), "uid0 & H>=10": ((ut == 0) & (Ht >= 10)).mean()}
rows = []; rows2 = []
for sg in ("H=0", "H 1-9", "H<=9", "uid0", "uid0 & H>=10"):
    m = SG[sg]; idx = np.where(m)[0]; D = Dm[idx]; g = clu[idx]
    ins = D.mean(0); jb = int(ins.argmax()); mu_b, se_b, _ = cse(Dm[:, jb], m)
    nested = np.zeros((REP, len(idx))); picks = {}
    ug = np.unique(g)
    for r in range(REP):
        perm = rng.permutation(len(ug)); fold = dict(zip(ug, perm % 5)); fo = np.array([fold[x] for x in g])
        for k in range(5):
            tr, te = fo != k, fo == k; j = int(D[tr].mean(0).argmax())
            if D[tr].mean(0)[j] <= 0: j = -1                     # FINAL itself is a candidate (delta 0)
            nested[r, te] = 0.0 if j < 0 else D[te, j]; picks[j] = picks.get(j, 0) + 1
    nd = np.zeros(NF); nd[idx] = nested.mean(0); mu_n, se_n, _ = cse(nd, m)
    top3 = sorted(picks.items(), key=lambda t: -t[1])[:3]
    ceil = max(0.0, mu_n + 2 * se_n)
    rows.append([sg, len(idx), f"{100 * m.mean():.2f}", f"{100 * tshare[sg]:.2f}", f"{names[jb]}: {mu_b:+.4f}+-{se_b:.4f}", f"{mu_n:+.4f}+-{se_n:.4f}", "; ".join(f"{'FINAL' if j < 0 else names[j]} x{c}" for j, c in top3),
                 f"{ceil:.4f}", f"{m.mean() * ceil:.5f}", f"{tshare[sg] * ceil:.5f}"])
    for j, k in enumerate(names):
        rows2.append([sg, k, "%+.4f+-%.4f" % cse(Dm[:, j], m)[:2], f"{cse(Dm[:, j], m)[0] / cse(Dm[:, j], m)[1]:+.1f}"])
print("\n### largest gain of any stored family on a segment (vs FINAL): in-sample best and NESTED (20 x 5-fold by user; the family is chosen on 4 folds, read on the 5th)\n" +
      tab(rows, ["segment", "feeds", "val %", "test %", "IN-SAMPLE best (not a witness)", "NESTED gain", "most frequent nested picks (of 100)", "segment ceiling = max(0, nested + 2 SE)", "x val share", "x test share"]))
print("\n### every candidate on every segment (paired vs FINAL, cluster SE): 5 segments x 15 candidates = 75 looks\n" + tab(rows2, ["segment", "candidate", "delta vs FINAL", "z"]))
z = np.array([float(r[3]) for r in rows2]); print(f"cells with |z| >= 2: {(np.abs(z) >= 2).sum()} of {len(z)} (expected by chance {0.0455 * len(z):.1f}); max z {z.max():+.2f}, min z {z.min():+.2f}")
# generous ceilings
for sg in ("H=0", "H 1-9", "uid0"):
    m = SG[sg]; gap = e[NEAR].mean() - e[m].mean(); d, se = cdiff(e, m, NEAR)
    print(f"generous ceiling {sg}: NDCG {e[m].mean():.4f} vs NEAR {e[NEAR].mean():.4f} (D {d:+.4f}+-{se:.4f}); if lifted to NEAR: val share x gap = {m.mean() * max(gap, 0):.5f}, test share x gap = {tshare[sg] * max(gap, 0):.5f}; skill {((e - np.load(OUT / 'rand_ndcg.npy')) / (1 - np.load(OUT / 'rand_ndcg.npy')))[m].mean():.4f}")
