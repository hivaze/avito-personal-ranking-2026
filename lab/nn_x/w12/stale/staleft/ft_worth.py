"""Information from STORED predictions (CPU): where on val is the fine-tune half worth something? W8 / W16 final (0.5 z(FT) + 0.5 z(TO)) against the
train-only pool, by feed cold share and M, and under the pmix weights. Tests the status-file explanation of 'val under-reads the FT half'."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft")
import numpy as np, polars as pl
import pilot_read as R
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
d = pl.read_parquet("<repo>/eda/errors/segments/feed_table.parquet")
cs = d["cold_share"].to_numpy()
for name, seeds in (("W8", range(951, 955)), ("W16", range(951, 959))):
    mem = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in seeds]
    ft = np.mean([R.zv(np.load(R.W5 / f"nn_{m}_n2cpv3s3_val_oof.npy")) for m in mem], 0)
    to = np.mean([R.zv(np.load(R.W5 / f"nn_{m}_trainonly_val.npy")) for m in mem], 0)
    nf, nt, nft = R.nd(0.5 * R.zv(ft) + 0.5 * R.zv(to)), R.nd(to), R.nd(ft)
    dd = nf - nt
    P(f"== {name} ({len(mem)} members): final {nf.mean():.5f}, TO pool {nt.mean():.5f}, FT pool {nft.mean():.5f}")
    P(f"FT half worth (final - TO), 14,423 feeds: all {R.f(dd)} | pmix-public {R.f(dd, R.wpub)} | pmix-private {R.f(dd, R.wpriv)}")
    P(f"FT pool - TO pool: all {R.f(nft - nt)} | pmix-public {R.f(nft - nt, R.wpub)} | pmix-private {R.f(nft - nt, R.wpriv)}")
    for lab, m in (("cold share < 15%", cs < 0.15), ("15-30%", (cs >= 0.15) & (cs < 0.30)), ("30-45%", (cs >= 0.30) & (cs < 0.45)), (">= 45%", cs >= 0.45)):
        P(f"  feeds with {lab:17s} n {int(m.sum()):5d} (pmix weight mass: public {R.wpub[m].sum() / R.wpub.sum():.3f} private {R.wpriv[m].sum() / R.wpriv.sum():.3f} val {m.mean():.3f}): final - TO {R.f(dd[m])}")
    for lab, m in (("M < 50", R.Mv < 50), ("M >= 50", R.Mv >= 50)):
        P(f"  feeds with {lab:17s} n {int(m.sum()):5d} (public {R.wpub[m].sum() / R.wpub.sum():.3f} private {R.wpriv[m].sum() / R.wpriv.sum():.3f} val {m.mean():.3f}): final - TO {R.f(dd[m])}")
open(R.HERE / "out/ft_worth.txt", "w").write("\n".join(out) + "\n")
