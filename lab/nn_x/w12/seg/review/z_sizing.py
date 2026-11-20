"""Z1 / Z2 (DECLARATION block 3): label-free size of the earlier-snapshot extension; fine-tune gain by H bucket."""
from rv_lib import *
tr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["user_id", "timestamp", "H", "hist_off"])
tu, tts, tH, toff = tr["user_id"].to_numpy().astype(np.int64), tr["timestamp"].to_numpy().astype(np.int64), tr["H"].to_numpy().astype(np.int64), tr["hist_off"].to_numpy().astype(np.int64)
hp = np.load(FLAT / "val/hist_hpos.npy", mmap_mode="r")
print("hist_hpos (val) first 12:", np.asarray(hp[:12]), "dtype", hp.dtype, "| first feed H", H[0])
order = np.argsort(tu, kind="stable")
su = tu[order]
vts = fv["timestamp"].to_numpy().astype(np.int64)
t_item = np.load(FLAT / "train/hist_item_id.npy", mmap_mode="r")
v_item = np.load(FLAT / "val/hist_item_id.npy", mmap_mode="r")
seen = np.load(OUT / "seen_train.npy")
rows = []
for nm, g in (("val cap, user seen in train", CAP & seen), ("val NEAR, user seen in train", NEAR & seen)):
    ids = np.where(g)[0]
    rng = np.random.default_rng(1)
    ids = np.sort(rng.choice(ids, min(len(ids), 1200), replace=False))
    n_old, n_tr, n_old_latest = [], [], []
    for f in ids:
        lo, hi = np.searchsorted(su, UID[f]), np.searchsorted(su, UID[f], side="right")
        fe = order[lo:hi]
        fe = fe[tts[fe] < vts[f]]
        n_tr.append(len(fe))
        cur = set(np.asarray(v_item[HOFF[f]:HOFF[f] + H[f]]).tolist())
        old = set()
        for j in fe:
            old.update(np.asarray(t_item[toff[j]:toff[j] + tH[j]]).tolist())
        n_old.append(len(old - cur))
    n_old, n_tr = np.array(n_old), np.array(n_tr)
    rows.append([nm, int(g.sum()), f"{100 * g.mean():.1f}", len(ids), f"{np.median(n_tr):.0f}", f"{np.median(n_old):.0f}", f"{np.mean(n_old):.0f}", f"{100 * (n_old >= 64).mean():.0f}", f"{100 * (n_old >= 128).mean():.0f}",
                 f"{100 * (n_old == 0).mean():.0f}"])
print(tab(rows, ["segment", "feeds", "% of val", "sampled", "median earlier train feeds", "median old-only distinct items", "mean", "% with >= 64", "% with >= 128", "% with 0"]))
# test side (label-free): share of capped test feeds whose user has an earlier train or val feed
ft = pl.read_parquet(FLAT / "test/feeds.parquet", columns=["user_id", "timestamp", "H"])
ut, Ht, tst = ft["user_id"].to_numpy().astype(np.int64), ft["H"].to_numpy(), ft["timestamp"].to_numpy()
BOUND = 1790497920
in_tr = np.isin(ut, np.unique(tu)) & (ut != 0)
in_va = np.isin(ut, np.unique(UID)) & (ut != 0)
capT = Ht >= 256
for nm, m in (("test all", np.ones(len(ut), bool)), ("test public", tst < BOUND), ("test private", tst >= BOUND)):
    print(f"{nm}: cap share {100 * capT[m].mean():.2f}% | of capped feeds: user has train feeds {100 * in_tr[m & capT].mean():.1f}%, train or val feeds {100 * (in_tr | in_va)[m & capT].mean():.1f}% "
          f"| capped AND seen = {100 * (capT & (in_tr | in_va))[m].mean():.2f}% of feeds | seen share of all feeds {100 * (in_tr | in_va)[m].mean():.1f}%")
print(f"val: capped AND seen in train = {100 * (CAP & seen).mean():.2f}% of feeds; seen share of all {100 * seen.mean():.1f}%")

print("\n### Z2: fine-tune gain (FT OOF branch minus train-only branch, FD16 base pool) by H bucket")
ft_b, to_b = ndcg(pool(FD16, "n2cpv3s3_val_oof")), ndcg(pool(FD16, "trainonly_val"))
g = ft_b - to_b
rows = []
for n, m in BUCK + [("H<256", REST), ("ALL", np.ones(NF, bool))]:
    mu, se, k = cmean(g, m)
    rows.append([n, k, f"{mu:+.5f}+-{se:.5f}"])
print(tab(rows, ["H", "feeds", "FT - TO"]))
d, se = cdiff(g, REST, CAP)
print(f"contrast (H < 256) minus cap: {d:+.5f} +- {se:.5f} (z {d / se:+.2f})")
