"""R0: rebuild FINAL from its 20 member files with own code; segment table; cross-check of own NDCG against src/metrics."""
import sys
from rv_lib import *
from rv_lib import _DISC, _CD

s = final_from(FD16, FDP, 0.2)
e = ndcg(s)
np.save(OUT / "final_score.npy", s.astype(np.float32))
np.save(OUT / "final_ndcg.npy", e)
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed
e2 = ndcg_per_feed(s.astype(np.float32), Y, OFF, M)
print(f"FINAL val (own NDCG, own z, from 20 member files) {e.mean():.5f} | src/metrics on the same score {e2.mean():.5f} | max per-feed |diff| {np.abs(e - e2).max():.2e}")
st = 0.5 * z(np.load(P / "ens_FD16P20kd_ft_val.npy")) + 0.5 * z(np.load(P / "ens_FD16P20kd_to_val.npy"))
print(f"vs stored ens_FD16P20kd branches: max |score diff| {np.abs(st - s).max():.2e}; NDCG of stored {ndcg(st).mean():.5f}")
print(f"feeds {NF} users {len(np.unique(UID))} clusters {NCL} feeds without a positive {(NPOS == 0).sum()}")
# E_rand check by simulation on 300 feeds
rng = np.random.default_rng(3)
ids = rng.choice(NF, 300, replace=False)
er = e_rand()
sim = []
for f in ids:
    y = Y[OFF[f]:OFF[f] + M[f]]
    v = []
    for _ in range(400):
        p = rng.permutation(M[f])
        v.append((y[p] * _DISC[:M[f]]).sum() / _CD[int(y.sum())])
    sim.append(np.mean(v))
print(f"E_rand closed form vs simulation on 300 feeds: mean {er[ids].mean():.4f} vs {np.mean(sim):.4f}; max |diff| {np.abs(er[ids] - np.array(sim)).max():.4f}")

ft = pl.read_parquet(FLAT / "test/feeds.parquet")
Ht = ft["H"].to_numpy()
ut = ft["user_id"].to_numpy()
ts = ft["timestamp"].to_numpy()
BOUND = 1790497920
a = auc(s)
sk = (e - er) / (1 - er)
rows = []
lo_hi = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 255), (256, 10 ** 9)]
for (n, m), (lo, hi) in zip(BUCK, lo_hi):
    mu, se, k = cmean(e, m)
    rows.append([n, k, f"{100 * m.mean():.2f}", f"{100 * ((Ht >= lo) & (Ht <= hi)).mean():.2f}", f"{mu:.4f}", f"{se:.4f}", f"{np.nanmean(a[m]):.4f}",
                 f"{sk[m].mean():.4f}", f"{M[m].mean():.1f}", f"{NPOS[m].mean():.2f}", f"{er[m].mean():.4f}"])
tr_u = np.unique(pl.read_parquet(FLAT / "train/feeds.parquet", columns=["user_id"])["user_id"].to_numpy())
seen = np.isin(UID, tr_u) & (UID != 0)
np.save(OUT / "seen_train.npy", seen)
for n, m, tsh in (("REST", REST, (Ht < 256).mean()), ("uid0", UID == 0, (ut == 0).mean()), ("seen in train (uid != 0)", seen, np.nan),
                  ("never in train (uid != 0)", ~seen & (UID != 0), np.nan), ("never in train incl. uid0", ~seen, np.nan), ("H<=9", H <= 9, (Ht <= 9).mean()),
                  ("H 1-9", (H >= 1) & (H <= 9), ((Ht >= 1) & (Ht <= 9)).mean()), ("ALL", np.ones(NF, bool), 1.0)):
    mu, se, k = cmean(e, m)
    rows.append([n, k, f"{100 * m.mean():.2f}", f"{100 * tsh:.2f}", f"{mu:.4f}", f"{se:.4f}", f"{np.nanmean(a[m]):.4f}", f"{sk[m].mean():.4f}", f"{M[m].mean():.1f}",
                 f"{NPOS[m].mean():.2f}", f"{er[m].mean():.4f}"])
print(tab(rows, ["segment", "feeds", "val %", "test %", "NDCG", "SE(cluster)", "AUC", "skill", "mean M", "n_pos", "E_rand"]))
d, se = cdiff(e, CAP, REST)
print(f"D(cap - REST) {d:+.5f} +- {se:.5f} | share x gap {-d * CAP.mean():+.5f}")
d, se = cdiff(e, CAP, NEAR)
print(f"D(cap - NEAR) {d:+.5f} +- {se:.5f}")
pub = ts < BOUND
print("test public / private shares: H=256 %.4f / %.4f | H=0 %.4f / %.4f | H<=9 %.4f / %.4f | uid0 %.4f / %.4f | n test %d (public %d)" % (
    (Ht[pub] >= 256).mean(), (Ht[~pub] >= 256).mean(), (Ht[pub] == 0).mean(), (Ht[~pub] == 0).mean(), (Ht[pub] <= 9).mean(), (Ht[~pub] <= 9).mean(),
    (ut[pub] == 0).mean(), (ut[~pub] == 0).mean(), len(Ht), pub.sum()))
# feeds per user in val (how much clustering there is) and in the cap
cnt = np.bincount(CLU)
print(f"val users with >= 2 feeds: {(cnt >= 2).sum()} ; feeds in such clusters {cnt[cnt >= 2].sum()}")
# H = 0 splits
ue = np.load(FLAT / "val/user_emb.npy")
zero = (np.abs(ue).sum(1) == 0)
uet = np.load(FLAT / "test/user_emb.npy")
zt = (np.abs(uet).sum(1) == 0)
for n, m in (("H=0, user emb present", (H == 0) & ~zero), ("H=0, all-zero emb", (H == 0) & zero), ("H=0, uid0", (H == 0) & (UID == 0)), ("H=0, known uid", (H == 0) & (UID != 0)),
             ("all-zero emb, any H", zero)):
    mu, se, k = cmean(e, m)
    print(f"{n}: {k} feeds NDCG {mu:.4f} +- {se:.4f} AUC {np.nanmean(a[m]):.4f}")
print(f"test: H=0 {int((Ht == 0).sum())} feeds; of them all-zero emb {((Ht == 0) & zt).sum()} = {100 * ((Ht == 0) & zt).mean():.2f}% of test; all-zero emb share test {100 * zt.mean():.2f}% val {100 * zero.mean():.2f}%")
