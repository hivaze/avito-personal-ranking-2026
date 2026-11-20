"""When s_ft and s_to disagree inside the head, which one is right? By segment."""
import hk, numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression

c = hk.Ctx(); df = c.df
K = 10
head = df.filter(pl.col("rank") <= K)
fid = head["fidx"].to_numpy(); y = head["label"].to_numpy(); rk = head["rank"].to_numpy()
a = head["s_ft"].to_numpy().astype(float); b = head["s_to"].to_numpy().astype(float)
starts = np.r_[0, np.nonzero(fid[1:] != fid[:-1])[0] + 1, len(fid)]
I, J = [], []
for s0, s1 in zip(starts[:-1], starts[1:]):
    idx = np.arange(s0, s1)
    if y[idx].sum() in (0, len(idx)):
        continue
    for i in idx:
        for j in idx:
            if rk[i] < rk[j] and y[i] != y[j]:
                I.append(i); J.append(j)
I = np.array(I); J = np.array(J); t = y[I]
dft = a[I] - a[J]; dto = b[I] - b[J]
F = np.column_stack([dft, dto])
m = LogisticRegression(C=1e4, max_iter=1000).fit(F, t)
print("t ~ dft + dto:", m.coef_, m.intercept_)
print("std dft", dft.std(), "std dto", dto.std(), "corr", np.corrcoef(dft, dto)[0, 1])
dis = (dft > 0) != (dto > 0)
print("pairs", len(t), "branch-disagree pairs", dis.sum())
print(" ft right", (((dft > 0) == (t == 1)) & dis).sum(), " to right", (((dto > 0) == (t == 1)) & dis).sum())
# AUC of each branch on discordant pairs
print("pair acc ft", ((dft > 0) == (t == 1)).mean(), "to", ((dto > 0) == (t == 1)).mean(), "score", ((dft + dto > 0) == (t == 1)).mean())

# Top-1 disagreement at the feed level, over the whole feed
fa = df["s_ft"].to_numpy().astype(float); fb = df["s_to"].to_numpy().astype(float); yy = c.y
seg = {}
rows = []
for k, (o, n) in enumerate(zip(c.off, c.sz)):
    ia = o + np.argmax(fa[o:o + n]); ib = o + np.argmax(fb[o:o + n]); isc = o + np.argmax(c.score[o:o + n])
    rows.append((k, ia != ib, yy[ia], yy[ib], yy[isc]))
R = np.array(rows)
d = R[:, 1] == 1
print("feeds", len(R), "top1 disagree", d.sum(), "ft-top1 pos", R[d, 2].mean(), "to-top1 pos", R[d, 3].mean(), "score-top1 pos", R[d, 4].mean())
fe = df.group_by("fidx", maintain_order=True).agg(pl.all().first())
for col, cuts in [("M", [0, 30, 100, 200, 400]), ("H", [-1, 10, 50, 150, 256, 300]), ("guest", None), ("newest_src", None), ("newest_type", None)]:
    v = fe[col].to_numpy()
    if cuts is None:
        groups = [(str(u), v == u) for u in np.unique(v)]
    else:
        groups = [(f"{lo}-{hi}", (v > lo) & (v <= hi)) for lo, hi in zip(cuts[:-1], cuts[1:])]
    for gname, g in groups:
        gd = g & d
        if gd.sum() < 30:
            continue
        print(f"  {col}={gname}: feeds={g.sum()} disagree={gd.sum()} ft-top1={R[gd,2].mean():.3f} to-top1={R[gd,3].mean():.3f}")
