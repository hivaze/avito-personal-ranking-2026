"""A2: the same H-bucket table for other rankers (difficulty proxies). Per-feed NDCG / AUC saved to out/rk_*.npz."""
from lib import *
S8 = range(951, 959); S4 = range(951, 955)
FD16 = [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8]; FDP = [f"abl_fd10_pop1kd05w192_s{s}" for s in S4]
ld = lambda n: np.load(P / n)
pool = lambda names, suf: np.mean([Z(ld(f"nn_{m}_{suf}.npy")) for m in names], 0)
R = {}
R["FINAL"] = np.load(OUT / "final_score.npy")
b_ft, b_to = pool(FD16, "n2cpv3s3_val_oof"), pool(FD16, "trainonly_val"); p_ft, p_to = pool(FDP, "n2cpv3s3_val_oof"), pool(FDP, "trainonly_val")
np.save(OUT / "pool_b_ft.npy", b_ft); np.save(OUT / "pool_b_to.npy", b_to); np.save(OUT / "pool_p_ft.npy", p_ft); np.save(OUT / "pool_p_to.npy", p_to)
chk = fin(0.8 * Z(b_ft) + 0.2 * Z(p_ft), 0.8 * Z(b_to) + 0.2 * Z(p_to)); print("rebuild from members vs stored FINAL: max abs diff", float(np.abs(chk - R["FINAL"]).max()), "ndcg", ev(chk).mean())
R["FD16 (16 students, no pop-light)"] = fin(b_ft, b_to)
R["FD16 TO branch"] = b_to; R["FD16 FT branch"] = b_ft
R["POPLIGHT pool (4)"] = fin(p_ft, p_to)
T11 = [f"r6_cd_m02_r{r}" for r in range(101, 108)] + [f"w5_e10_base_s{s}" for s in S4]
R["TEACHER 11 hard-label (TO)"] = pool(T11, "trainonly_val")
R["hard-label single w5_e10_base_s951 (TO)"] = Z(ld("nn_w5_e10_base_s951_trainonly_val.npy"))
R["student single fd10_kd05w192_s951 (TO)"] = Z(ld("nn_w5_fd10_kd05w192_s951_trainonly_val.npy"))
R["G3 file (hard labels, FT+TO)"] = fin(ld("ens_G3_ft_val.npy"), ld("ens_G3_to_val.npy"))
f = lambda n: np.nan_to_num(np.load(FC / f"{n}.npy").astype(np.float64), nan=0.0)
R["popularity only (it_hist_all_luo)"] = f("it_hist_all_luo"); R["exposure only (xp_imp_c50)"] = f("xp_imp_c50")
R["co-visitation only (seq_cv_invd20)"] = f("seq_cv_invd20")
R["pre-trained encoder score (pre2_fe)"] = f("pre2_fe")
smw = np.load(OUT / "cos_smaxw.npy"); R["max cosine to history (all 256)"] = smw[:, 5].astype(np.float64); R["cosine to user embedding"] = np.load(OUT / "cos_u.npy").astype(np.float64)
for k in (1, 3):
    x = np.load(FLAT / f"val/cand_f{k}.npy").astype(np.float64); a = np.nanmean(feed_auc(x))
    R[f"f{k} only ({'+' if a >= 0.5 else '-'} sign, chosen by overall AUC)"] = x if a >= 0.5 else -x
E0 = rand_ndcg(); ND, AU = {}, {}
for k, s in R.items():
    ND[k] = ev(s); AU[k] = feed_auc(s)
np.savez(OUT / "rk_ndcg.npz", **{k: v for k, v in ND.items()}); np.savez(OUT / "rk_auc.npz", **{k: v for k, v in AU.items()}); np.save(OUT / "rand_ndcg.npy", E0)
bn = [n for n, _, _ in BUCK]
for title, D in (("NDCG", ND), ("per-feed AUC", AU), ("skill = (NDCG - E_random)/(1 - E_random)", {k: (v - E0) / (1 - E0) for k, v in ND.items()})):
    rows = []
    for k, x in D.items():
        r = [k, f"{np.nanmean(x):.4f}"] + [f"{cse(x, bmask[n])[0]:.4f}" for n in bn]
        d1, s1 = cdiff(x, CAP, NEAR); d2, s2 = cdiff(x, CAP, REST)
        rows.append(r + [f"{d1:+.4f}+-{s1:.4f}", f"{d2:+.4f}+-{s2:.4f}", f"{cse(x, CAP)[0] / cse(x, NEAR)[0]:.3f}"])
    print(f"\n### {title} by H bucket\n" + tab(rows, ["ranker", "ALL"] + bn + ["D(256-NEAR)", "D(256-REST)", "ratio 256/NEAR"]))
rows = [["random order (exact expectation)", f"{E0.mean():.4f}"] + [f"{E0[bmask[n]].mean():.4f}" for n in bn] + ["%+.4f+-%.4f" % cdiff(E0, CAP, NEAR), "%+.4f+-%.4f" % cdiff(E0, CAP, REST), ""]]
print(tab(rows, ["ranker", "ALL"] + bn + ["D(256-NEAR)", "D(256-REST)", ""]))
# rule (c): lift of FINAL over the history-free rankers, paired per feed, H=256 vs NEAR / REST
print("\n### lift of each model over history-free rankers (paired per feed): value at H=256, at NEAR, contrast (256 - NEAR), contrast (256 - REST)")
rows = []
for base in ("popularity only (it_hist_all_luo)", "exposure only (xp_imp_c50)", "cosine to user embedding"):
    for mdl in ("FINAL", "TEACHER 11 hard-label (TO)", "POPLIGHT pool (4)", "max cosine to history (all 256)", "pre-trained encoder score (pre2_fe)"):
        for nm, D in (("NDCG", ND), ("AUC", AU)):
            l = D[mdl] - D[base]
            rows.append([mdl, base, nm, "%.4f" % cse(l, CAP)[0], "%.4f" % cse(l, NEAR)[0], "%+.4f+-%.4f" % cdiff(l, CAP, NEAR), "%+.4f+-%.4f" % cdiff(l, CAP, REST)])
print(tab(rows, ["model", "minus", "metric", "lift@256", "lift@NEAR", "contrast 256-NEAR", "contrast 256-REST"]))
