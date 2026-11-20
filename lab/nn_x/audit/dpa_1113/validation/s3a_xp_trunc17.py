"""S3a: stale counterfactual of the pop1 input xp_imp_c50 on val, matching the stale-val scenario 'both' (review_shift s08:
train Sep 17-20 = MSK days >= 20713 removed from the popularity counts). xp_imp_c50' = log1p(min(imp_{train days < 20713}, 50)).
Check first: the full-train count reproduces data/feats/val/tte_it_imp.npy exactly. Writes out/cf/xp_imp_c50_trunc17_val.npy.
"""
from lib import *  # noqa: F401,F403

ftr, Mtr, offtr = meta("train")
dtr = day(ftr["timestamp"].to_numpy())
it = np.load(FLAT / "train/cand_item_id.npy").astype(np.int64)
dcand = np.repeat(dtr, Mtr)
nmax = int(it.max()) + 1
vi = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64)
full = np.bincount(it, minlength=nmax)
early = np.bincount(it[dcand < 20713], minlength=nmax)
ref = np.load(FEATS / "val/tte_it_imp.npy")
vfull = full[vi] if vi.max() < nmax else np.where(vi < nmax, full[np.minimum(vi, nmax - 1)], 0)
print("check full-train count == tte_it_imp(val): max|d|", np.abs(vfull - ref).max(), "corr", np.corrcoef(vfull, ref)[0, 1])
vearly = np.where(vi < nmax, early[np.minimum(vi, nmax - 1)], 0)
x = np.log1p(np.minimum(vearly, 50)).astype(np.float32)
live = np.load(ROOT / "nn/feat_cache/val/xp_imp_c50.npy")
print("live xp_imp_c50 == log1p(min(tte,50)):", np.abs(live - np.log1p(np.minimum(ref, 50)).astype(np.float32)).max())
print(f"val zero share: live {(live == 0).mean():.4f} -> trunc17 {(x == 0).mean():.4f}; mean {live.mean():.4f} -> {x.mean():.4f}")
(OUT / "cf").mkdir(exist_ok=True)
np.save(OUT / "cf/xp_imp_c50_trunc17_val.npy", x)
# test zero share for reference (private-like target of the scenario)
tt = np.load(FEATS / "test/tte_it_imp.npy")
ft_, Mt, _ = meta("test"); pub = np.repeat(public_mask(), Mt)
print(f"test tte zero share: public {(tt[pub] == 0).mean():.4f} private {(tt[~pub] == 0).mean():.4f}")
