"""Segment-specific blends: new = score + w_seg * g(x); w_seg chosen per segment on 4 folds, evaluated on the 5th."""
import fx, kit, numpy as np, polars as pl, sys
df = kit.load()
c = fx.Ctx(df)
s = c.base
F = pl.read_parquet("feeds.parquet")
Mf = F["M"].to_numpy(); Hf = F["H"].to_numpy()
segM = np.digitize(Mf, [30, 100])            # 0: <30, 1: 30-100, 2: 100+
segH = np.digitize(Hf, [50, 200])

def nz(x, fill):
    x = np.asarray(x, float); return np.where(np.isfinite(x), x, fill)

feats = {
    "log_ndup": np.log1p(df["n_cand_dup95"].to_numpy().astype(float)),
    "cand_maxcos": nz(df["cand_maxcos"].to_numpy(), 0.85),
    "h1": nz(df["h1"].to_numpy(), 0.45),
    "hmax": nz(df["hmax"].to_numpy(), 0.8),
    "hmean5": nz(df["hmean5"].to_numpy(), 0.5),
    "f4": df["f4"].to_numpy().astype(float),
    "ucos": nz(df["ucos"].to_numpy(), 0.48),
    "f1": df["f1"].to_numpy().astype(float),
    "f3": df["f3"].to_numpy().astype(float),
    "seq_cv": df["seq_cv_invd20"].to_numpy().astype(float),
    "pre2_fe_z": df["pre2_fe_z"].to_numpy().astype(float),
    "log_cn": np.log(df["cat_n_in_feed"].to_numpy().astype(float)),
    "sft_minus_sto": (df["s_ft"].to_numpy() - df["s_to"].to_numpy()).astype(float),
}
grid = [-0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4]

def crossfit_seg(x, seg_feed, grid):
    xs = (x - x.mean()) / x.std()
    D = np.stack([c.ndcg(s + w * xs) - c.base_nd for w in grid])   # (G, nf)
    oof = np.zeros(c.nf); chosen = {}
    for k in range(5):
        tr = c.fold != k; te = ~tr
        for g in np.unique(seg_feed):
            m = seg_feed == g
            b = int(np.argmax(D[:, tr & m].mean(1)))
            chosen.setdefault(int(g), []).append(grid[b])
            oof[te & m] = D[b, te & m]
    se = oof.std(ddof=1) / np.sqrt(c.nf)
    insample = {int(g): [round(d * 1e4, 1) for d in D[:, seg_feed == g].mean(1)] for g in np.unique(seg_feed)}
    return oof.mean(), se, chosen, insample

for nm, x in feats.items():
    for sname, sf in [("M3", segM), ("H3", segH), ("all", np.zeros(c.nf, int))]:
        mean, se, ch, ins = crossfit_seg(x, sf, grid)
        print(f"{nm:14s} {sname:4s} CF {mean*1e4:+6.2f}e-4 se {se*1e4:.2f} z {mean/se:+.2f}  chosen {ch}")
        if sname == "M3":
            for g, v in ins.items():
                print(f"      seg{g}: " + " ".join(f"{d:+6.1f}" for d in v))
