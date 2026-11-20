"""Cross-fitted NDCG-direct test of a one-parameter rule new = score + w * g, w on a grid chosen on 4 folds."""
import numpy as np
import prep
from prep import kit

W = np.array([-0.3, -0.2, -0.15, -0.1, -0.07, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3])


class CV:
    def __init__(self, df):
        self.df = df
        self.y = df["label"].to_numpy()
        self.s = df["score"].to_numpy().astype(np.float64)
        self.off, self.sz = kit.feed_layout(df)
        self.F = prep.FastNDCG(self.y, self.off, self.sz)
        self.ff = kit.folds(df)[self.off]
        self.b = self.F(self.s)

    def run(self, g, name="", W=W, verbose=True):
        nd = np.stack([self.F(self.s + w * g + 1e-7 * self.s) - self.b for w in W])
        oof = np.zeros(len(self.sz)); picks = []
        for k in range(5):
            tr = self.ff != k
            j = int(np.argmax(nd[:, tr].mean(1))); picks.append(float(W[j])); oof[~tr] = nd[j, ~tr]
        d, se = oof.mean(), oof.std(ddof=1) / np.sqrt(len(oof))
        jin = int(np.argmax(nd.mean(1)))
        if verbose:
            print(f"{name:40s} oof={d:+.5f} se={se:.5f} z={d / se if se else 0:+.2f} | in-sample best w={W[jin]:+.2f} "
                  f"d={nd[jin].mean():+.5f} | picks={picks}", flush=True)
        return d, se, nd
