"""S (DECLARATION block 0): segments the study may have missed. Descriptive; no choice from labels."""
from rv_lib import *

e = np.load(OUT / "final_ndcg.npy")
s = np.load(OUT / "final_score.npy").astype(np.float64)
a = auc(s)
er = e_rand()
sk = (e - er) / (1 - er)
C = np.load(OUT / "cos.npz")
pop = auc(feat("it_hist_all_luo"))
seen = np.load(OUT / "seen_train.npy")

print("### S(i) functionally cold feeds: share of candidates with a history event at cosine >= 0.8 (label-free), feeds with H >= 10")
rel = np.bincount(FID, weights=(C["n08"] > 0).astype(float), minlength=NF) / M
np.save(OUT / "rel08.npy", rel)
bins = [("0", rel == 0), ("(0, 0.1]", (rel > 0) & (rel <= 0.1)), ("(0.1, 0.25]", (rel > 0.1) & (rel <= 0.25)), ("(0.25, 0.5]", (rel > 0.25) & (rel <= 0.5)), ("> 0.5", rel > 0.5)]
rows = []
for n, m in bins:
    m = m & (H >= 10)
    mu, se, k = cmean(e, m)
    rows.append([n, k, f"{100 * k / NF:.2f}", f"{mu:.4f}+-{se:.4f}", f"{np.nanmean(a[m]):.4f}", f"{sk[m].mean():.4f}", f"{np.nanmean(pop[m]):.4f}", f"{np.median(H[m]):.0f}", f"{100 * CAP[m].mean():.0f}",
                 f"{M[m].mean():.0f}", f"{NPOS[m].mean():.2f}"])
print(tab(rows, ["share of candidates matched", "feeds", "val %", "FINAL NDCG", "FINAL AUC", "skill", "popularity AUC", "median H", "% at cap", "mean M", "n_pos"]))
lowrel = (rel <= 0.1) & (H >= 10)
print(f"feeds with H >= 10 and at most 10% of candidates matched: {100 * lowrel.mean():.2f}% of val; against H <= 9: {100 * (H <= 9).mean():.2f}%")
# positives: is the positive itself matched?
posm = np.bincount(FID, weights=Y * (C["n08"] > 0), minlength=NF) / np.maximum(NPOS, 1)
for n, m in (("H >= 10, no positive matched at 0.8", (H >= 10) & (posm == 0)), ("H >= 10, every positive matched", (H >= 10) & (posm == 1))):
    mu, se, k = cmean(e, m)
    print(f"  [needs labels, diagnostic only] {n}: {k} feeds ({100 * k / NF:.1f}%), FINAL NDCG {mu:.4f} +- {se:.4f}, AUC {np.nanmean(a[m]):.4f}; share at the cap {100 * CAP[m].mean():.0f}%")
for n, m in (("cap", CAP), ("NEAR", NEAR)):
    print(f"  {n}: share of feeds with no positive matched at 0.8 = {100 * (posm[m] == 0).mean():.1f}%; FINAL NDCG there {e[m & (posm == 0)].mean():.4f}, on the others {e[m & (posm > 0)].mean():.4f}")

print("\n### S(ii) parity of the training distribution of H")
rows = []
for sp in ("train", "val", "test"):
    f = pl.read_parquet(FLAT / sp / "feeds.parquet", columns=["H", "user_id", "M"])
    h, u = f["H"].to_numpy(), f["user_id"].to_numpy()
    rows.append([sp, len(h), f"{100 * (h >= 256).mean():.2f}", f"{100 * ((h >= 100) & (h < 256)).mean():.2f}", f"{100 * (h == 0).mean():.2f}", f"{100 * (h <= 9).mean():.2f}", f"{100 * (u == 0).mean():.2f}",
                 f"{np.median(h):.0f}", f"{h.max()}"])
print(tab(rows, ["split", "feeds", "% H >= 256", "% H 100-255", "% H = 0", "% H <= 9", "% user_id 0", "median H", "max H"]))
tr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["H", "M", "timestamp"])
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"]
h = tr["H"].to_numpy()
rl = np.nan_to_num(pr, nan=0.0) > 0.9
print(f"train, random-like feeds (p_random > 0.9; the domain val / test resemble): {rl.sum()} feeds, cap share {100 * (h[rl] >= 256).mean():.2f}%, H = 0 {100 * (h[rl] == 0).mean():.2f}%; "
      f"production-like feeds: cap share {100 * (h[~rl] >= 256).mean():.2f}%, H = 0 {100 * (h[~rl] == 0).mean():.2f}%")

print("\n### S(iii) FINAL NDCG by fine H bins (is the cap a step or the continuation of a trend?)")
rows = []
fb = [("100-149", 100, 149), ("150-199", 150, 199), ("200-255", 200, 255), ("256 (cap)", 256, 10 ** 9)]
for n, lo, hi in fb:
    m = (H >= lo) & (H <= hi)
    mu, se, k = cmean(e, m)
    rows.append([n, k, f"{mu:.4f}+-{se:.4f}", f"{np.nanmean(a[m]):.4f}", f"{sk[m].mean():.4f}", f"{np.nanmean(pop[m]):.4f}", f"{100 * seen[m].mean():.0f}", f"{NPOS[m].mean():.2f}"])
print(tab(rows, ["H", "feeds", "FINAL NDCG", "AUC", "skill", "popularity AUC", "% seen in train", "n_pos"]))
m1, m2, m3 = (H >= 100) & (H <= 149), (H >= 200) & (H <= 255), CAP
d, se = cdiff(e, m3, m2)
print(f"cap minus H 200-255: {d:+.4f} +- {se:.4f} (z {d / se:+.2f})")
d, se = cdiff(e, m2, m1)
print(f"H 200-255 minus H 100-149: {d:+.4f} +- {se:.4f} (z {d / se:+.2f})")
# inside the cap by seen in train
for n, m in (("cap, seen in train", CAP & seen), ("cap, not seen", CAP & ~seen), ("NEAR, seen in train", NEAR & seen), ("NEAR, not seen", NEAR & ~seen)):
    mu, se, k = cmean(e, m)
    print(f"  {n}: {k} feeds, NDCG {mu:.4f} +- {se:.4f}, AUC {np.nanmean(a[m]):.4f}")
