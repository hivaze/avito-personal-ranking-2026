"""M5 (CPU): (a) M-split form of 'pure pop-light in the FT branch': pop-light part of the FT branch = fine-tuned predictions on
M<50 feeds, train-only predictions on M>=50 feeds (threshold fixed at the team's standard 50; chosen AFTER seeing M4, so this
is a description of the target cell, not evidence). (b) Is the full-view pool over-confident on popular items in large feeds?
O/E of a within-feed softmax by within-feed popularity tercile (it_hist_all_luo rank inside the feed), M>=50 vs M<50,
full TO pool vs pop-light TO pool vs FT pools (FD16P20 members)."""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/training")
import numpy as np, polars as pl
from scipy.optimize import minimize_scalar
from m3_branch_views import pool, Z, ev, line, Mv, ov, yv, SETS, sm, K, yK
MROW = np.repeat(Mv, Mv)
def _main():
  for nm, (base, pop) in SETS.items():
    Fb, Fp, Tb, Tp = pool(base, "ft"), pool(pop, "ft"), pool(base, "to"), pool(pop, "to")
    Fp_mix = np.where(MROW >= 50, Z(Tp), Z(Fp))
    for rule, w in (("w0.2", np.float32(0.2)), ("m0.4_0.2", np.where(MROW >= 50, 0.2, 0.4).astype(np.float32))):
        to = (1 - w) * Z(Tb) + w * Z(Tp)
        cur = ev(0.5 * Z((1 - w) * Z(Fb) + w * Z(Fp)) + 0.5 * Z(to))
        v = ev(0.5 * Z((1 - w) * Z(Fb) + w * Fp_mix) + 0.5 * Z(to))
        print(f"{nm} {rule}: M-split pop-light FT part minus stored: {line(v - cur)}")
tab = pl.read_parquet("<repo>/lab/nn_x/w12/fixmiss/systematic/data/val_table.parquet", columns=["it_hist_all_luo", "xp_imp_c50"])
pop_ = tab["it_hist_all_luo"].to_numpy().astype(np.float64)
# within-feed rank percentile of popularity
fid = np.repeat(np.arange(len(Mv)), Mv)
order = np.lexsort((pop_, fid)); r = np.empty(len(pop_)); r[order] = np.arange(len(pop_)) - np.repeat(ov, Mv)
pct = r / np.maximum(MROW - 1, 1); terc = np.digitize(pct, [1 / 3, 2 / 3])
big = MROW >= 50
def oe_t(x):
    x = x.astype(np.float64)
    t = minimize_scalar(lambda a: -(yK * np.log(sm(a * x) + 1e-30)).sum(), bounds=(0.1, 5), method="bounded").x
    E = sm(t * x) * np.repeat(K, Mv); out = []
    for bm, bn in ((big, "M>=50"), (~big, "M<50")):
        out.append(bn + " " + " ".join(f"{yv[bm & (terc == k)].sum() / E[bm & (terc == k)].sum():.3f}" for k in range(3)))
    return t, " | ".join(out)
if __name__ == "__main__":
    _main()
    base, pop = SETS["FD16P20"]
    for nm, x in (("TO full", pool(base, "to")), ("TO pop", pool(pop, "to")), ("FT full", pool(base, "ft")), ("FT pop", pool(pop, "ft"))):
        t, s = oe_t(Z(x)); print(f"O/E by within-feed popularity tercile low/mid/high, {nm} (t {t:.3f}): {s}")
