"""Drift check for sparse (mostly-zero) features: share of nonzero rows and mean of nonzero values in
val / test-public / test-private, plus for test-private split by day (Sun pm / Mon / Tue)."""
import sys
sys.path.insert(0, "<repo>/src")
from common import *
PUB_END = 1790497920
ft = pl.read_parquet(FLAT / "test/feeds.parquet"); tfi = np.load(FLAT / "test/cand_fidx.npy")
tts = ft["timestamp"].to_numpy()[tfi]
day = (tts + 3 * 3600) // 86400
groups = {"pub": tts < PUB_END}
for d, nm in zip(np.unique(day), ["Sat", "Sun", "Mon", "Tue"]):
    if nm in ("Sun",):
        groups["prv_Sun"] = (day == d) & (tts >= PUB_END)
    elif nm in ("Mon", "Tue"):
        groups[f"prv_{nm}"] = day == d
groups["prv_all"] = tts >= PUB_END
pref = sys.argv[1]
names = sorted(p.stem for p in (FEATS / "val").glob(f"{pref}*.npy") if (FEATS / "test" / p.name).exists())
print(f"{'feature':24s} {'val nz':>7s} " + " ".join(f"{g:>8s}" for g in groups) + "   | mean of nonzero: val " + " ".join(groups))
for n in names:
    v = np.load(FEATS / "val" / f"{n}.npy"); t = np.load(FEATS / "test" / f"{n}.npy")
    nzv = np.nanmean(v != 0)
    line = f"{n:24s} {nzv:7.3f} " + " ".join(f"{np.nanmean(t[m] != 0):8.3f}" for m in groups.values())
    mv = np.nanmean(v[(v != 0) & np.isfinite(v)]) if ((v != 0) & np.isfinite(v)).any() else np.nan
    line += f"   | {mv:.3g} " + " ".join(f"{np.nanmean(t[m][(t[m] != 0) & np.isfinite(t[m])]):.3g}" for m in groups.values())
    print(line)
