"""M3 (CPU, stored predictions only): does the fine-tune erase the popularity-light view?

The pop-light students (FDP4 = abl_fd10_pop1kd05w192_s951-954) are fine-tuned with the same 42 extra inputs as the full-view
students, and about 20 of those extras are popularity-type counts (it_hist_*_luo, seq_cv_*, pre_* id scores). If the fine-tune
re-introduces popularity, the pop-light pool should be closer to the full pool in the FT branch than in the TO branch,
and the admixture should be worth less inside the FT branch.
Reads on FD16P20 members (all-data) and on W16 + PD4w (held-out):
  - mean within-feed Pearson correlation of the pop-light pool with the full pool, per branch;
  - admixture worth per branch: branch(0.8 full + 0.2 pop) - branch(full), NDCG of the branch alone and of the final
    file when only one branch carries the admixture (paired per feed; val, pmix-public, pmix-private, M<50, M>=50);
  - per-member FT gain (FT minus TO NDCG) for full-view vs pop-light students;
  - O/E of the young / cold classes per pool and branch (within-feed softmax of the pool z with a fitted temperature).
"""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix

P = Path("<repo>/preds")
fv, Mv, ov = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy")
W = pmix.weights(); big = Mv >= 50
Z = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
def mz(m, br):
    suf = "n2cpv3s3_val_oof" if br == "ft" else "trainonly_val"
    return Z(np.load(P / f"nn_{m}_{suf}.npy")).astype(np.float32)
pool = lambda ms, br: np.mean([mz(m, br) for m in ms], 0)
def fmt(d, w=None):
    if w is None:
        return f"{d.mean():+.5f}+-{d.std(ddof=1) / np.sqrt(len(d)):.5f}"
    m, se = pmix.wmean_se(d, w); return f"{m:+.5f}+-{se:.5f}"
line = lambda d: f"val {fmt(d)} | pub {fmt(d, W['public'])} | priv {fmt(d, W['private'])} | M<50 {fmt(d[~big])} | M>=50 {fmt(d[big])}"
fid = np.repeat(np.arange(len(Mv)), Mv)
def corr_within(a, b):
    am = a - np.repeat(np.add.reduceat(a, ov) / Mv, Mv); bm = b - np.repeat(np.add.reduceat(b, ov) / Mv, Mv)
    num = np.add.reduceat(am * bm, ov); den = np.sqrt(np.add.reduceat(am * am, ov) * np.add.reduceat(bm * bm, ov))
    c = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan); return np.nanmean(c)

# item classes from the flat val inputs (f4 = listing age, HIGH = OLD; cold from xp_imp_c50 via the sibling table)
import polars as pl
tab = pl.read_parquet("<repo>/lab/nn_x/w12/fixmiss/systematic/data/val_table.parquet", columns=["f4", "xp_imp_c50", "fidx"])
assert (tab["fidx"].to_numpy() == fid).all()
f4 = tab["f4"].to_numpy(); xp = tab["xp_imp_c50"].to_numpy()
young = f4 <= np.quantile(f4, 0.1); cold = xp == 0
K = np.add.reduceat(yv.astype(np.float64), ov); yK = yv / np.repeat(np.maximum(K, 1), Mv)
def sm(x):
    mx = np.maximum.reduceat(x, ov); e = np.exp(x - np.repeat(mx, Mv)); return e / np.repeat(np.add.reduceat(e, ov), Mv)
def oe(z):
    z = z.astype(np.float64)
    t = minimize_scalar(lambda a: -(yK * np.log(sm(a * z) + 1e-30)).sum(), bounds=(0.1, 5), method="bounded").x
    E = sm(t * z) * np.repeat(K, Mv)
    return t, {nm: yv[m].sum() / E[m].sum() for nm, m in (("young10%", young), ("cold", cold), ("young&cold", young & cold), ("warm-old", (~cold) & (~young)))}

S4, S8 = range(951, 955), range(951, 959)
SETS = {
    "FD16P20": ([f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8], [f"abl_fd10_pop1kd05w192_s{s}" for s in S4]),
    "W16+PD4w": ([f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8], [f"abl_e10_pop1kd05w192_s{s}" for s in S4]),
}
def _main():
  for nm, (base, pop) in SETS.items():
    print(f"\n=== {nm}")
    gains = {}
    for grp, ms in (("full", base), ("pop", pop)):
        g = [ev(mz(m, "ft")).mean() - ev(mz(m, "to")).mean() for m in ms]
        gains[grp] = g
        print(f"  per-member FT - TO NDCG, {grp}: mean {np.mean(g):+.5f} (sd {np.std(g):.5f}, n {len(g)})")
    B = {br: (pool(base, br), pool(pop, br)) for br in ("ft", "to")}
    for br in ("to", "ft"):
        b, p = B[br]
        print(f"  [{br}] corr(pop pool, full pool) {corr_within(p, b):.4f} | full {ev(b).mean():.5f} pop {ev(p).mean():.5f} "
              f"| branch(0.8 full + 0.2 pop) - branch(full): {line(ev(0.8 * Z(b) + 0.2 * Z(p)) - ev(b))}")
        for gn, x in (("full", b), ("pop", p), ("mix", 0.8 * Z(b) + 0.2 * Z(p))):
            t, o = oe(Z(x)); print(f"      O/E {br} {gn:4s} (t {t:.3f}): " + " ".join(f"{k} {v:.3f}" for k, v in o.items()))
    fin = lambda ftw, tow: 0.5 * Z((1 - ftw) * Z(B["ft"][0]) + ftw * Z(B["ft"][1])) + 0.5 * Z((1 - tow) * Z(B["to"][0]) + tow * Z(B["to"][1]))
    e00 = ev(fin(0, 0)); e22 = ev(fin(0.2, 0.2)); e02 = ev(fin(0, 0.2)); e20 = ev(fin(0.2, 0)); e04 = ev(fin(0.0, 0.4)); e03 = ev(fin(0.1, 0.3))
    print(f"  final, admixture in both branches  : {line(e22 - e00)}")
    print(f"  final, admixture only in TO branch : {line(e02 - e00)}")
    print(f"  final, admixture only in FT branch : {line(e20 - e00)}")
    print(f"  final, TO-only 0.4 vs both 0.2     : {line(e04 - e22)}")
    print(f"  final, FT 0.1 / TO 0.3 vs both 0.2 : {line(e03 - e22)}")
    print(f"  corr(FT full, TO full) {corr_within(B['ft'][0], B['to'][0]):.4f} | corr(FT pop, TO pop) {corr_within(B['ft'][1], B['to'][1]):.4f}")


if __name__ == "__main__":
    _main()
