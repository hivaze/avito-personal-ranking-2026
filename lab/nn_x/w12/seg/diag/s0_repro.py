"""D0: rebuild FINAL exactly as build_r4.py and reproduce the lead's H-bucket table."""
from lib import *
s = load_final(); e = ev(s)
np.save(OUT / "final_score.npy", s.astype(np.float32)); np.save(OUT / "final_ndcg.npy", e)
print(f"FINAL val {e.mean():.5f} (target 0.40271) | feeds {NF} | feeds without positive {(NPOS == 0).sum()} | users {len(np.unique(uid))} | clusters {NCL}")
ft_, Mt, ot = feed_meta("test"); Ht = ft_["H"].to_numpy()
rows = []
for n, a, b in BUCK:
    m = bmask[n]; mu, se, k = cse(e, m); mt = ((Ht >= a) & (Ht <= b)).mean()
    rows.append([n, k, f"{100 * m.mean():.2f}", f"{100 * mt:.2f}", f"{mu:.4f}", f"{se:.4f}", f"{Mv[m].mean():.1f}", f"{NPOS[m].mean():.2f}"])
for n in ("REST", "uid0"):
    m = SEGS[n]; mu, se, k = cse(e, m); rows.append([n, k, f"{100 * m.mean():.2f}", "", f"{mu:.4f}", f"{se:.4f}", f"{Mv[m].mean():.1f}", f"{NPOS[m].mean():.2f}"])
# users with train feeds
tr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["user_id"])["user_id"].to_numpy()
seen = np.isin(uid, np.unique(tr)) & (uid != 0)
for n, m in (("user seen in train (uid != 0)", seen), ("user never in train (uid != 0)", ~seen & (uid != 0)), ("never in train incl. uid0", ~seen)):
    mu, se, k = cse(e, m); rows.append([n, k, f"{100 * m.mean():.2f}", "", f"{mu:.4f}", f"{se:.4f}", f"{Mv[m].mean():.1f}", f"{NPOS[m].mean():.2f}"])
np.save(OUT / "seen_train.npy", seen)
print(tab(rows, ["segment", "feeds", "val %", "test %", "NDCG", "SE(cluster)", "mean M", "mean n_pos"]))
d, se = cdiff(e, CAP, REST); print(f"D_raw(REST) = {d:+.5f} +- {se:.5f}  -> x share {CAP.mean():.4f} = {-d * CAP.mean():+.5f} (lead: +0.005)")
d, se = cdiff(e, CAP, NEAR); print(f"D_raw(NEAR) = {d:+.5f} +- {se:.5f}")
# test side, label-free: public / private share of H buckets
ts = ft_["timestamp"].to_numpy()
import pmix
print("test H=256 share public / private:", f"{(Ht[ts < pmix.BOUND] >= 256).mean():.4f} / {(Ht[ts >= pmix.BOUND] >= 256).mean():.4f}", "| H=0:", f"{(Ht[ts < pmix.BOUND] == 0).mean():.4f} / {(Ht[ts >= pmix.BOUND] == 0).mean():.4f}",
      "| H<=9:", f"{(Ht[ts < pmix.BOUND] <= 9).mean():.4f} / {(Ht[ts >= pmix.BOUND] <= 9).mean():.4f}", "| uid0:", f"{(ft_['user_id'].to_numpy()[ts < pmix.BOUND] == 0).mean():.4f} / {(ft_['user_id'].to_numpy()[ts >= pmix.BOUND] == 0).mean():.4f}")
print("val: H<=9", f"{(Hv <= 9).mean():.4f}", "uid0", f"{(uid == 0).mean():.4f}")
