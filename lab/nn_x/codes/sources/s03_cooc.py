"""s03: which sources live together (same platform / same session) and which look alike.
(1) feed-level co-presence O/E of source-bearing events (types 1/4/7), expected computed within strata of the number of
    source-bearing events (deciles), so heavy users do not inflate every pair.
(2) adjacent-transition lift among source-bearing events (older -> newer, consecutive in the history), different items only.
(3) category-mix JS divergence between sources (type-1 views only) and type-mix.
Splits: val+test (latest feed per user_id>0, every user-0 feed), train sample 200K feeds."""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/sources")
from scommon import load_hist, train_sample, pshow, OUT

S = list(range(0, 17))


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a[a > 0] * np.log2(a[a > 0] / b[a > 0])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def analyse(df, tag):
    sb = df.filter(pl.col("et").is_in([1, 4, 7]))
    print(f"\n===== {tag}: {df['fidx'].n_unique()} feeds, {sb.height} source-bearing events =====")
    # (1) co-presence
    pres = sb.group_by(["fidx", "es"]).len()
    nsb = sb.group_by("fidx").len().rename({"len": "nsb"}).sort("fidx")
    fids = nsb["fidx"].to_numpy(); fpos = {f: i for i, f in enumerate(fids)}
    P = np.zeros((len(fids), 31), bool)
    P[np.searchsorted(fids, pres["fidx"].to_numpy()), pres["es"].to_numpy()] = True
    q = np.quantile(nsb["nsb"].to_numpy(), np.linspace(0, 1, 11)[1:-1])
    strat = np.searchsorted(q, nsb["nsb"].to_numpy(), side="right")
    O = np.zeros((31, 31)); E = np.zeros((31, 31))
    for st in np.unique(strat):
        X = P[strat == st].astype(np.float64)
        n = X.shape[0]
        O += X.T @ X
        p = X.mean(0)
        E += n * np.outer(p, p)
    L = np.where(E > 0, O / np.maximum(E, 1e-9), np.nan)
    Ls = pl.DataFrame({"src": S, **{str(b): [round(L[a, b], 2) for a in S] for b in S}})
    print("(1) co-presence O/E within activity strata (diagonal = 1/P-ish, ignore)")
    pshow(Ls)
    Ls.write_csv(f"{OUT}/s03_copresence_{tag}.csv")
    print("presence share:", {s: round(float(P[:, s].mean()), 4) for s in S})
    # (2) adjacent transitions, different items
    sbs = sb.sort(["fidx", "hpos"], descending=[False, True])
    f = sbs["fidx"].to_numpy(); s = sbs["es"].to_numpy(); it = sbs["item"].to_numpy()
    ok = (f[1:] == f[:-1]) & (it[1:] != it[:-1])
    a = s[:-1][ok].astype(np.int64); b = s[1:][ok].astype(np.int64)
    C = np.zeros((31, 31)); np.add.at(C, (a, b), 1)
    rowp = C / np.maximum(C.sum(1, keepdims=True), 1)
    colp = C.sum(0) / C.sum()
    T = rowp / np.maximum(colp, 1e-12)
    Ts = pl.DataFrame({"older": S, **{str(j): [round(T[i, j], 2) for i in S] for j in S}})
    print("(2) adjacent transition lift P(newer=col | older=row)/P(newer=col), different items")
    pshow(Ts)
    Ts.write_csv(f"{OUT}/s03_translift_{tag}.csv")
    Rp = pl.DataFrame({"older": S, **{str(j): [round(rowp[i, j], 3) for i in S] for j in S}})
    print("(2b) raw P(newer=col | older=row)")
    pshow(Rp)
    # (3) category JS (type-1 views) and nearest neighbour source
    v = df.filter(pl.col("et") == 1).group_by(["es", "cat"]).len()
    Pc = np.zeros((31, 54))
    Pc[v["es"].to_numpy(), v["cat"].to_numpy()] = v["len"].to_numpy()
    tot = Pc.sum(1)
    rows = []
    for x in S:
        if tot[x] < 300:
            continue
        px = Pc[x] / tot[x]
        d = {y: js(px, Pc[y] / tot[y]) for y in S if y != x and tot[y] >= 300}
        nn = sorted(d.items(), key=lambda t: t[1])[:3]
        rows.append(dict(src=x, n_views=int(tot[x]), nearest=",".join(f"s{y}:{v:.3f}" for y, v in nn), js_s1=d.get(1, 0.0), js_s2=d.get(2, 0.0)))
    pshow(pl.DataFrame(rows))


def latest_per_user(split):
    df, fs = load_hist(split)
    keep = fs.sort("timestamp").with_row_index("r").filter((pl.col("user_id") == 0) | (pl.col("r") == pl.col("r").max().over("user_id")))["fidx"]
    return df.filter(pl.col("fidx").is_in(keep.implode()))


if __name__ == "__main__":
    v = latest_per_user("val"); t = latest_per_user("test").with_columns(pl.col("fidx") + 10_000_000)
    analyse(pl.concat([v, t]), "valtest")
    df, _ = load_hist("train", train_sample(200_000, seed=2))
    analyse(df, "train")
