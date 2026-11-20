"""G1-G3 (DECLARATION block 3)."""
from lib import *
HD = ROOT / "nn_x/tricks/HDROP-ALL/preds/val"; bn = [n for n, _, _ in BUCK]
# G1
rows = []; D = {}
for rate in ("d10", "d20"):
    ds = []
    for mem in ("w5_e10_base_s951", "w5_e10_base_s952"):
        ref = ev(np.load(HD / f"{mem}~ref.npy")); ds.append(np.mean([ev(np.load(HD / f"{mem}~{rate}_k{k}.npy")) for k in range(4)], 0) - ref)
    d = np.mean(ds, 0); D[rate] = d
    rows.append([f"single pass, {rate[1:]}% of events dropped", "%+.5f+-%.5f" % cse(d)[:2]] + ["%+.4f+-%.4f" % cse(d, bmask[n])[:2] for n in bn[1:]] + ["%+.5f+-%.5f" % cdiff(d, CAP, NEAR)])
    # 4-mask TTA (mean of raw logits) as well
    ds = []
    for mem in ("w5_e10_base_s951", "w5_e10_base_s952"):
        ref = ev(np.load(HD / f"{mem}~ref.npy")); ds.append(ev(np.mean([np.load(HD / f"{mem}~{rate}_k{k}.npy") for k in range(4)], 0)) - ref)
    d = np.mean(ds, 0)
    rows.append([f"4-mask average, {rate[1:]}%", "%+.5f+-%.5f" % cse(d)[:2]] + ["%+.4f+-%.4f" % cse(d, bmask[n])[:2] for n in bn[1:]] + ["%+.5f+-%.5f" % cdiff(d, CAP, NEAR)])
print("### G1 cost of dropping history events at inference (hard-label members s951/952), by H bucket\n" + tab(rows, ["mode", "ALL"] + bn[1:] + ["contrast H=256 minus NEAR"]))
ev_drop = {n: (0.2 * np.minimum(Hv[bmask[n]], 256)).mean() for n in bn[1:]}
print("mean number of events removed at 20%:", {k: round(v, 1) for k, v in ev_drop.items()})
print("cost per removed event at 20% (NDCG x 1e-4 per event):", {n: round(1e4 * cse(D["d20"], bmask[n])[0] / ev_drop[n], 3) for n in bn[1:]})
# G2 calibration of the teacher's own top-1 probability
T11 = [ROOT / "preds" / f"nn_r6_cd_m02_r{r}_trainonly_val.npy" for r in range(101, 108)] + [ROOT / "preds" / f"nn_w5_e10_base_s{s}_trainonly_val.npy" for s in range(951, 955)]
lg = np.mean([np.load(f).astype(np.float64) for f in T11], 0)
mx = np.maximum.reduceat(lg, ov); ex = np.exp(lg - np.repeat(mx, Mv)); p = ex / np.repeat(np.add.reduceat(ex, ov), Mv)
ptop = np.maximum.reduceat(p, ov); ent = -np.add.reduceat(p * np.log(np.maximum(p, 1e-300)), ov) / np.log(Mv)
order = np.lexsort((yv, -lg, fid)); first = order[ov]; ytop = yv[first] / NPOS
pmass = np.add.reduceat(p * yv, ov)   # predicted mass on the positives
rows = []
for n in bn + ["REST"]:
    m = SEGS[n]; rows.append([n, "%.4f" % ptop[m].mean(), "%.4f+-%.4f" % cse(ytop, m)[:2], "%+.4f+-%.4f" % cse(ytop - ptop, m)[:2], "%.4f" % ent[m].mean(), "%.4f" % pmass[m].mean(), "%.4f" % (NPOS[m] / Mv[m]).mean()])
print("\n### G2 teacher (11 hard-label members, mean logit) calibration by H bucket\n" + tab(rows, ["H", "predicted p(top-1)", "realised y_top1/n_pos", "realised - predicted", "normalised entropy", "softmax mass on positives", "base rate n_pos/M"]))
print("contrast H=256 minus NEAR: predicted p(top-1) %+.4f+-%.4f | realised %+.4f+-%.4f | (realised - predicted) %+.4f+-%.4f | entropy %+.4f+-%.4f" % (*cdiff(ptop, CAP, NEAR), *cdiff(ytop, CAP, NEAR), *cdiff(ytop - ptop, CAP, NEAR), *cdiff(ent, CAP, NEAR)))
# G3 H=0 split
cov = pl.read_parquet(OUT / "feed_cov.parquet"); ue = cov["ue_norm"].to_numpy() > 0
ND = dict(np.load(OUT / "rk_ndcg.npz")); AU = dict(np.load(OUT / "rk_auc.npz")); rows = []
for nm, m in (("H=0, user embedding present", (Hv == 0) & ue), ("H=0, user embedding all-zero", (Hv == 0) & ~ue), ("H=0, uid0", (Hv == 0) & (uid == 0)), ("H=0, known user id", (Hv == 0) & (uid != 0)),
              ("H 1-9, uid0", (Hv >= 1) & (Hv <= 9) & (uid == 0)), ("H 1-9, known user id", (Hv >= 1) & (Hv <= 9) & (uid != 0)), ("uid0, H>=10", (uid == 0) & (Hv >= 10)), ("known user, H>=10", (uid != 0) & (Hv >= 10))):
    rows.append([nm, int(m.sum()), f"{Mv[m].mean():.0f}", f"{NPOS[m].mean():.2f}", "%.4f+-%.4f" % cse(ND["FINAL"], m)[:2], "%.4f" % cse(AU["FINAL"], m)[0], "%.4f" % cse(ND["popularity only (it_hist_all_luo)"], m)[0],
                 "%.4f" % cse(AU["popularity only (it_hist_all_luo)"], m)[0], "%.4f" % cse(AU["cosine to user embedding"], m)[0], "%.4f" % cse(ND["POPLIGHT pool (4)"] - ND["FINAL"], m)[0]])
print("\n### G3 cold-start splits\n" + tab(rows, ["segment", "feeds", "M", "n_pos", "FINAL NDCG", "FINAL AUC", "popularity NDCG", "popularity AUC", "cos_u AUC", "pop-light pool minus FINAL"]))
ft_, Mt, ot = feed_meta("test"); uet = np.abs(np.load(FLAT / "test/user_emb.npy")).sum(1) > 0; Ht = ft_["H"].to_numpy()
print("test: H=0 feeds %d (%.2f%%), of them user embedding all-zero %.3f; all feeds with all-zero user embedding %.4f (val %.4f)" % ((Ht == 0).sum(), 100 * (Ht == 0).mean(), (~uet[Ht == 0]).mean(), (~uet).mean(), (~ue).mean()))
