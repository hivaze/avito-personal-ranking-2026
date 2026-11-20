"""A3: where the positives of heavy users sit (descriptive; nothing is chosen from these numbers)."""
from lib import *
s = np.load(OUT / "final_score.npy"); e = np.load(OUT / "final_ndcg.npy")
top = np.load(OUT / "cos_top64.npy").astype(np.float32); tix = np.load(OUT / "cos_top64_idx.npy"); smw = np.load(OUT / "cos_smaxw.npy").astype(np.float64)
cold = np.load(OUT / "cand_cold.npy"); nge = np.load(OUT / "cos_nge.npy"); y = yv.astype(bool)
Hc = np.repeat(Hv, Mv); Mc = np.repeat(Mv, Mv)
# rank of every candidate under FINAL (0 = best; pessimistic ties like the metric)
order = np.lexsort((yv, -s, fid)); rk = np.empty(len(s), np.int64); rk[order] = np.arange(len(s)) - np.repeat(ov, Mv)
bn = [n for n, _, _ in BUCK]; cb = {n: (Hc >= a) & (Hc <= b) for n, a, b in BUCK}
rows = []
for n in bn:
    p = y & cb[n]; r = rk[p]; pc = r / np.maximum(Mc[p] - 1, 1)
    rows.append([n, int(p.sum()), f"{(r == 0).mean():.3f}", f"{(r < 5).mean():.3f}", f"{(r < 10).mean():.3f}", f"{(r >= 20).mean():.3f}", f"{np.median(r):.0f}", f"{np.median(pc):.3f}", f"{pc.mean():.3f}",
                 f"{cold[p].mean():.3f}", f"{cold[cb[n]].mean():.3f}", f"{cold[p].mean() / cold[cb[n]].mean():.2f}"])
print("### rank of the positives under FINAL, and cold positives, by H bucket\n" + tab(rows, ["H", "positives", "top-1", "top-5", "top-10", "rank>=20", "median rank", "median pct", "mean pct", "cold share of positives", "cold share of candidates", "ratio"]))
# rank of positives split by cold / warm
rows = []
for n in ("100-255", "H=256"):
    for nm, m in (("cold positive", cold), ("warm positive", ~cold)):
        p = y & cb[n] & m; r = rk[p]; rows.append([n, nm, int(p.sum()), f"{(r == 0).mean():.3f}", f"{(r < 5).mean():.3f}", f"{(r < 10).mean():.3f}", f"{(r / np.maximum(Mc[p] - 1, 1)).mean():.3f}"])
print("\n" + tab(rows, ["H", "kind", "n", "top-1", "top-5", "top-10", "mean pct"]))
# similarity of the positive to its most similar history event and the POSITION of that event (newest-first index)
smax = top[:, 0].astype(np.float64); am = tix[:, 0].astype(np.int64); has = smax > -1.0
rows = []
for n in bn[1:]:
    for nm, m in (("positives", y), ("negatives", ~y)):
        q = m & cb[n] & has; a = am[q]; v = smax[q]
        rows.append([n, nm, int(q.sum()), f"{v.mean():.3f}", f"{np.median(v):.3f}", f"{(v >= 0.9).mean():.3f}", f"{(v < 0.7).mean():.3f}", f"{(a < 16).mean():.3f}", f"{((a >= 16) & (a < 64)).mean():.3f}", f"{(a >= 64).mean():.3f}", f"{(a >= 128).mean():.3f}", f"{np.median(a):.0f}"])
print("\n### max cosine to the history and the index of the most similar event (0 = newest)\n" + tab(rows, ["H", "set", "n", "mean smax", "median smax", "share smax>=0.9", "share smax<0.7", "idx<16", "idx 16-63", "idx>=64", "idx>=128", "median idx"]))
# how well FINAL ranks the positive by where its best-matching event sits (H=256 and NEAR)
rows = []
for n in ("100-255", "H=256"):
    for nm, m in (("best match idx < 16", am < 16), ("idx 16-63", (am >= 16) & (am < 64)), ("idx 64-127", (am >= 64) & (am < 128)), ("idx >= 128", am >= 128)):
        p = y & cb[n] & has & m; r = rk[p]
        rows.append([n, nm, int(p.sum()), f"{p.sum() / (y & cb[n] & has).sum():.3f}", f"{smax[p].mean():.3f}", f"{(r == 0).mean():.3f}", f"{(r < 5).mean():.3f}", f"{(r / np.maximum(Mc[p] - 1, 1)).mean():.3f}"])
    for nm, m in (("smax >= 0.9", smax >= 0.9), ("0.8-0.9", (smax >= 0.8) & (smax < 0.9)), ("0.7-0.8", (smax >= 0.7) & (smax < 0.8)), ("< 0.7", smax < 0.7)):
        p = y & cb[n] & has & m; r = rk[p]
        rows.append([n, nm, int(p.sum()), f"{p.sum() / (y & cb[n] & has).sum():.3f}", f"{smax[p].mean():.3f}", f"{(r == 0).mean():.3f}", f"{(r < 5).mean():.3f}", f"{(r / np.maximum(Mc[p] - 1, 1)).mean():.3f}"])
print("\n### FINAL's rank of the positive by position / strength of its best-matching history event\n" + tab(rows, ["H", "positives with", "n", "share", "mean smax", "top-1", "top-5", "mean pct (0 = best)"]))
# trivial rankers: smax restricted to the newest w events; mean of the top-k cosines
WIN = [8, 16, 32, 64, 128, 256]
def tk(k):
    v = np.where(top[:, :k] > -1.0, top[:, :k], np.nan); return np.nan_to_num(np.nanmean(v, 1), nan=-1.5)
import warnings; warnings.filterwarnings("ignore")
TR = {f"smax newest {w}": smw[:, j] for j, w in enumerate(WIN)}
TR.update({f"mean top-{k} cosine": tk(k).astype(np.float64) for k in (1, 3, 8, 16, 32, 64)}); TR["mean cosine (all events)"] = np.load(OUT / "cos_smean.npy").astype(np.float64)
A = {k: feed_auc(v) for k, v in TR.items()}; Nd = {k: ev(v) for k, v in TR.items()}
for title, D in (("within-feed AUC", A), ("NDCG", Nd)):
    rows = [[k] + [f"{cse(x, bmask[n])[0]:.4f}" for n in bn[1:]] for k, x in D.items()]
    print(f"\n### trivial history rankers, {title} by H bucket\n" + tab(rows, ["ranker"] + bn[1:]))
print("\n### declared contrasts at H=256 (paired per feed, cluster SE), with the same contrast at H 20-99 and NEAR")
M2099 = (Hv >= 20) & (Hv <= 99); rows = []
for a_, b_ in (("smax newest 64", "smax newest 256"), ("smax newest 128", "smax newest 256"), ("smax newest 16", "smax newest 256"), ("mean top-32 cosine", "mean top-16 cosine"), ("mean top-64 cosine", "mean top-16 cosine"), ("mean top-16 cosine", "mean top-1 cosine"), ("mean top-3 cosine", "mean top-1 cosine")):
    for nm, D in (("NDCG", Nd), ("AUC", A)):
        d = D[a_] - D[b_]; rows.append([f"{a_} - {b_}", nm, "%+.4f+-%.4f" % cse(d, CAP)[:2], "%+.4f+-%.4f" % cse(d, NEAR)[:2], "%+.4f+-%.4f" % cse(d, M2099)[:2]])
print(tab(rows, ["contrast", "metric", "H=256", "NEAR (100-255)", "H 20-99"]))
# saturation of retrieval depth 16: how many events are 'similar' to a candidate
rows = []
for n in bn[1:]:
    q = cb[n] & has
    rows.append([n] + [f"{(nge[q, j] > 16).mean():.3f}" for j in range(3)] + [f"{(nge[q & y, j] > 16).mean():.3f}" for j in range(3)] + [f"{np.nanmean(np.where(top[q, 15] > -1, top[q, 15], np.nan)):.3f}", f"{np.nanmean(np.where(top[q, 31] > -1, top[q, 31], np.nan)):.3f}", f"{top[q, 0].mean():.3f}"])
print("\n### saturation of retrieval depth 16: share of candidates with MORE than 16 history events at cosine >= 0.9 / 0.8 / 0.7; cosine of the 16th and 32nd most similar event\n" +
      tab(rows, ["H", "cand >16 ev at 0.9", "at 0.8", "at 0.7", "POS >16 ev at 0.9", "at 0.8", "at 0.7", "mean cos of 16th", "mean cos of 32nd", "mean cos of 1st"]))
# does anything beyond the 16 most similar events discriminate once the top-16 mean is known? (within-feed partial read, descriptive)
t16, t1732 = tk(16).astype(np.float64), np.nan_to_num(np.nanmean(np.where(top[:, 16:32] > -1, top[:, 16:32], np.nan), 1), nan=-1.5)
res = np.zeros(len(s))
for f_ in np.where(CAP)[0]:
    o, m = ov[f_], Mv[f_]; a, b = t16[o:o + m], t1732[o:o + m]
    if a.std() > 0: res[o:o + m] = b - (np.cov(a, b)[0, 1] / a.var()) * a
ar = feed_auc(res); print(f"\nwithin-feed AUC of (mean cosine of events ranked 17-32) residualised on (mean top-16 cosine), H=256 feeds: {cse(ar, CAP)[0]:.4f} +- {cse(ar, CAP)[1]:.4f} (0.5 = nothing left beyond depth 16)")
# category novelty of the positive
hcat = np.load(FLAT / "val/hist_category_id.npy"); ccat = np.load(FLAT / "val/cand_category_id.npy"); hoff = fv["hist_off"].to_numpy()
inh = np.zeros(len(s), bool)
for f_ in range(NF):
    if Hv[f_]: inh[ov[f_]:ov[f_] + Mv[f_]] = np.isin(ccat[ov[f_]:ov[f_] + Mv[f_]], hcat[hoff[f_]:hoff[f_] + Hv[f_]])
rows = [[n, f"{inh[y & cb[n]].mean():.3f}", f"{inh[cb[n]].mean():.3f}"] for n in bn[1:]]
print("\n### share of positives / of all candidates whose category occurs in the stored history\n" + tab(rows, ["H", "positives", "candidates"]))
