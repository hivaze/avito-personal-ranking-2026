"""Helpers for the category lens (analysis only; rules are self-contained)."""
from __future__ import annotations
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
import kit  # noqa: E402

DISC = 1.0 / np.log2(np.arange(2, 400))
CATNAME = {2: "Cars", 3: "Parts", 4: "Water", 5: "Moto", 6: "Repair", 7: "Furniture", 8: "Appliances",
           9: "Rooms", 10: "Apartments", 11: "Houses", 12: "Land", 13: "Clothing", 14: "Watches",
           15: "KidsClothing", 16: "KidsGoods", 17: "DesktopPC", 18: "AudioVideo", 19: "Tickets",
           20: "Bicycles", 21: "Collectibles", 22: "Music", 23: "Sports", 24: "BizEquip", 25: "Commercial",
           26: "Trucks", 27: "Food", 28: "Books", 29: "Phones", 30: "Garages", 31: "Abroad",
           32: "Tableware", 33: "Beauty", 34: "Dogs", 35: "Cats", 36: "Birds", 37: "Aquarium",
           38: "OtherAnimals", 39: "PetSupplies", 40: "Tablets", 41: "Games", 42: "Laptops", 43: "Office",
           44: "PCparts", 45: "Hunting", 46: "Photo", 47: "Plants", 48: "Vacancies", 49: "CVs",
           50: "Services", 51: "ReadyBiz", 52: "Hotels", 53: "ShiftGigs"}
VERT = {}
for c in (2, 3, 4, 5, 26):
    VERT[c] = "Transport"
for c in (9, 10, 11, 12, 25, 30, 31):
    VERT[c] = "RealEstate"
for c in (6, 7, 8, 27, 32, 47):
    VERT[c] = "Home"
for c in (13, 14, 15, 16, 33):
    VERT[c] = "Personal"
for c in (17, 18, 29, 40, 41, 42, 43, 44, 46):
    VERT[c] = "Electronics"
for c in (19, 20, 21, 22, 23, 28, 45):
    VERT[c] = "Hobby"
for c in (24, 51):
    VERT[c] = "Business"
for c in (34, 35, 36, 37, 38, 39):
    VERT[c] = "Animals"
for c in (48, 49, 53):
    VERT[c] = "Jobs"
VERT[50] = "Services"
VERT[52] = "Travel"
VERTS = sorted(set(VERT.values()))
VERT_ID = {v: i for i, v in enumerate(VERTS)}


def load():
    df = kit.load()
    df = df.with_columns(pl.Series("fold", kit.folds(df)))
    return df


def layout(df):
    return kit.feed_layout(df)


def feed_id(df):
    off, sz = layout(df)
    return np.repeat(np.arange(len(sz)), sz), off, sz


def ndcg_fast(scores, labels, fid, off, sz):
    """Vectorised full-list NDCG per feed, ties pessimistic."""
    order = np.lexsort((labels, -scores, fid))
    ys = labels[order].astype(np.float64)
    pos = np.arange(len(order)) - np.repeat(off, sz)
    dcg = np.bincount(fid, weights=ys * DISC[pos], minlength=len(sz))
    npos = np.bincount(fid, weights=labels.astype(np.float64), minlength=len(sz)).astype(int)
    cum = np.r_[0.0, np.cumsum(DISC)]
    idcg = cum[npos]
    out = np.where(npos > 0, dcg / np.maximum(idcg, 1e-12), 1.0)
    return out


def delta_stats(d):
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return m, se, m / se if se > 0 else 0.0


class Ctx:
    """Data + cached baseline NDCG for fast evaluation and cross-fitting."""

    def __init__(self, df=None):
        self.df = load() if df is None else df
        self.fid, self.off, self.sz = feed_id(self.df)
        self.y = self.df["label"].to_numpy()
        self.s = self.df["score"].to_numpy().astype(np.float64)
        self.base = ndcg_fast(self.s, self.y, self.fid, self.off, self.sz)
        self.ffold = self.df["fold"].to_numpy()[self.off]
        self.n_tried = 0

    def col(self, c):
        return self.df[c].to_numpy()

    def feed(self, x):
        """broadcast feed-level array to rows"""
        return np.asarray(x)[self.fid]

    def d(self, new):
        """per-feed NDCG delta"""
        return ndcg_fast(np.asarray(new, dtype=np.float64), self.y, self.fid, self.off, self.sz) - self.base

    def crossfit(self, grid, make, verbose=True, label=""):
        """grid: list of params; make(params)->new scores. Chooses params on 4 folds, evaluates on the 5th."""
        D = np.stack([self.d(make(p)) for p in grid])  # (P, feeds)
        self.n_tried += len(grid)
        oof = np.empty(D.shape[1])
        chosen = []
        for k in range(5):
            tr = self.ffold != k
            j = int(np.argmax(D[:, tr].mean(1)))
            chosen.append(grid[j])
            oof[~tr] = D[j, ~tr]
        m, se, z = delta_stats(oof)
        jall = int(np.argmax(D.mean(1)))
        ins = D[jall].mean()
        if verbose:
            print(f"[{label}] crossfit {m:+.5f} se {se:.5f} z {z:+.2f} | in-sample best {ins:+.5f} at {grid[jall]} | chosen {chosen}")
        return dict(cf=m, se=se, z=z, ins=ins, best=grid[jall], chosen=chosen, D=D, oof=oof)


def lambda_pairs(C, temp=1.0):
    """Pairs (positive i, negative j) in the same feed with LambdaRank weights at the current score.
    Returns (I, J, lam, feed) arrays; dNDCG/dw for score + w*x is approx sum lam*(x_i - x_j)."""
    s, y, fid, off, sz = C.s, C.y, C.fid, C.off, C.sz
    # current positions with pessimistic ties
    order = np.lexsort((y, -s, fid))
    pos = np.empty(len(s), dtype=np.int64)
    pos[order] = np.arange(len(s)) - np.repeat(off, sz)
    disc = DISC[pos]
    npos = np.bincount(fid, weights=y.astype(float), minlength=len(sz)).astype(int)
    cum = np.r_[0.0, np.cumsum(DISC)]
    idcg = cum[npos]
    P = np.nonzero(y == 1)[0]
    fP = fid[P]
    reps = sz[fP]
    I = np.repeat(P, reps)
    starts = off[fP]
    J = np.repeat(starts, reps) + (np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps))
    keep = y[J] == 0
    I, J = I[keep], J[keep]
    f = fid[I]
    dn = np.abs(disc[I] - disc[J]) / idcg[f]
    sig = 1.0 / (1.0 + np.exp(temp * (s[I] - s[J])))
    return I, J, dn * sig, f


def lambda_test(C, pairs, x):
    I, J, lam, f = pairs
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    g = np.bincount(f, weights=lam * (x[I] - x[J]), minlength=len(C.sz))
    se = np.sqrt((g ** 2).sum())
    return g.sum() / se if se > 0 else 0.0, g.mean()


def logit_cluster(X, y, groups, iters=30):
    """Logistic regression (Newton) with feed-clustered sandwich SE. Returns (beta, se)."""
    X = np.asarray(X, dtype=np.float64)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    Hinv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    sc = X * (y - p)[:, None]
    _, gi = np.unique(groups, return_inverse=True)
    S = np.zeros((gi.max() + 1, X.shape[1]))
    np.add.at(S, gi, sc)
    V = Hinv @ (S.T @ S) @ Hinv
    return b, np.sqrt(np.diag(V))
