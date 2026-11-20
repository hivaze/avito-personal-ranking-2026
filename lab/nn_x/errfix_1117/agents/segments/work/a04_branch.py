"""Branch re-weighting (s_ft vs s_to), disagreement penalties, rank fusion; by segment."""
import fx, kit, numpy as np, polars as pl
df = kit.load()
c = fx.Ctx(df)
F = pl.read_parquet("feeds.parquet")
sft = df["s_ft"].to_numpy().astype(np.float64)
sto = df["s_to"].to_numpy().astype(np.float64)
M = F["M"].to_numpy(); H = F["H"].to_numpy()
segs = {
    "all": np.ones(c.nf, bool),
    "M<20": M < 20, "M20-50": (M >= 20) & (M < 50), "M50-100": (M >= 50) & (M < 100),
    "M100-200": (M >= 100) & (M < 200), "M200+": M >= 200,
    "H<20": H < 20, "H20-100": (H >= 20) & (H < 100), "H100-255": (H >= 100) & (H < 256), "H256": H >= 256,
    "guest": F["guest"].to_numpy(), "uvm": F["uvec_missing"].to_numpy(),
    "corr<q30": F["corr_ft_to"].to_numpy() < np.quantile(F["corr_ft_to"].to_numpy(), .3),
    "corr>q70": F["corr_ft_to"].to_numpy() > np.quantile(F["corr_ft_to"].to_numpy(), .7),
    "nt=1": F["newest_type"].to_numpy() == 1, "nt!=1": F["newest_type"].to_numpy() != 1,
    "src1": F["newest_src"].to_numpy() == 1, "src0": F["newest_src"].to_numpy() == 0, "src2": F["newest_src"].to_numpy() == 2,
}
def show(title, grid, mk):
    D = np.stack([c.ndcg(mk(p)) - c.base_nd for p in grid])
    print("=====", title, "grid", grid)
    for sn, m in segs.items():
        dm = D[:, m].mean(1); se = D[:, m].std(1, ddof=1) / np.sqrt(m.sum())
        print(f"{sn:10s} n={m.sum():5d} " + " ".join(f"{d*1e4:+6.1f}" for d in dm) + f"   (se@best {se[np.argmax(dm)]*1e4:.1f})")
    return D

# 1. w_ft
gw = [0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8]
show("w_ft (score=w*sft+(1-w)*sto) delta x1e4", gw, lambda w: w * sft + (1 - w) * sto)
# 2. disagreement penalty
gl = [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.5]
show("score - lam*|sft-sto|", gl, lambda l: c.base - l * np.abs(sft - sto))
# 3. rank fusion: z of within-feed rank
def frank(x):
    order = np.lexsort((x, c.feed))
    r = np.empty(len(x)); r[order] = np.arange(len(x)) - c.off[c.feed[order]]
    return r / c.sz[c.feed]
rft, rto = frank(sft), frank(sto)
show("score + a*(rank-fusion pct)", [0.1, 0.3, 1.0, 3.0], lambda a: c.base + a * (rft + rto))
# 4. power mean / soft max of branches
show("softmax-ish: logsumexp(t*sft,t*sto)/t", [-3, -1, 1, 3], lambda t: np.logaddexp(t * sft, t * sto) / t)
