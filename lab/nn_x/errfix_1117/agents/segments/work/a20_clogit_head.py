"""Clogit restricted to the head (rank<=K): does being a category follower lower the probability given the score?"""
import fx, kit, numpy as np, clogit
df = kit.load(); c = fx.Ctx(df); s = c.base; y = c.y; feed = c.feed
cr = df["cat_rank_in_feed"].to_numpy().astype(float)
rk = df["rank"].to_numpy()
M = df["M"].to_numpy()
for K in (5, 10, 20, 50, 400):
    for lab, mm in [("all", M > 0), ("M30-100", (M >= 30) & (M < 100)), ("M100+", M >= 100), ("M<30", M < 30)]:
        head = (rk <= K) & mm
        # keep feeds with a positive in head
        hp = np.bincount(feed[head], weights=y[head], minlength=c.nf) > 0
        rm = head & hp[feed]
        X = np.c_[(cr >= 2).astype(float), (cr - 1) ** 0.3]
        out = []
        for j in range(X.shape[1]):
            a0, _, ll0 = clogit.fit(s, np.zeros((len(y), 0)), y, feed, c.nf, rowmask=rm, l2=0.0, a0=1.0)
            a, w, ll = clogit.fit(s, X[:, j:j + 1], y, feed, c.nf, rowmask=rm, l2=0.0, a0=1.0)
            out.append(f"{['cr>=2','(cr-1)^.3'][j]}: w/a={w[0]/a:+.3f} LR={2*(ll-ll0):.1f}")
        print(f"K={K:3d} {lab:8s} feeds={hp[c.off].sum() if False else int(hp.sum())} a={a0:.2f} ", " | ".join(out), flush=True)
