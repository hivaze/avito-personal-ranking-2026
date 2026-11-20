"""Hypothesis: the model extrapolates too far for fresh-cold listings (no engagement data) -> shrink their high scores."""
from util import *
df = load(); c = Ctx(df); s = c.base
def col(n): return df[n].to_numpy().astype(np.float64)
xp, f4 = col("xp_imp_c50"), col("f4")
E0 = np.load("E_calib.npy")
# O/E of fresh-cold vs others by score level (per-feed normalised E)
d = df.with_columns(pl.Series("E0", E0)).with_columns((pl.col("E0") * pl.col("label").sum().over("fidx") / pl.col("E0").sum().over("fidx")).alias("E"))
d = d.with_columns(((pl.col("xp_imp_c50") == 0) & (pl.col("f4") < 0.25)).alias("fc"), pl.col("score").cut([0, 1, 1.5, 2, 2.5]).cast(pl.Utf8).alias("sb"))
t = d.group_by(["sb", "fc"]).agg(pl.len(), pl.col("label").sum().alias("O"), pl.col("E").sum().round(1).alias("E")).with_columns((pl.col("O") / pl.col("E")).round(3).alias("OE")).sort(["sb", "fc"])
for r in t.iter_rows(): print("  ", r)
fc = ((xp == 0) & (f4 < 0.25)).astype(float)
cold = (xp == 0).astype(float)
res = []
for gname, g in [("fresh_cold", fc), ("cold", cold)]:
    for t0 in [1.0, 1.5, 2.0]:
        x = g * np.maximum(s - t0, 0)
        r = c.crossfit(lambda w: s - w * x, [0.05, 0.1, 0.2, 0.3, 0.5])
        print(f"{gname} shrink above {t0}: cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins={r['ins_delta']:+.5f}")
