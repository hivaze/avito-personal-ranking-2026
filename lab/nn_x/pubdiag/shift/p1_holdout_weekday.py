"""Part 1: holdout (train-period exploration-like feeds, labelled) paired deltas split by MSK weekday/weekend, day, hour.
Pools on the holdout = z-mean of members' train-only holdout predictions."""
from shiftlib import *  # noqa: F401,F403

hold, fh, M, off = holdout_index()
y = np.load(P / "holdout/holdout_y.npy")
ts = fh["timestamp"].to_numpy().astype(np.int64)
uid = fh["user_id"].to_numpy()
day, hr, wd = msk_day(ts), msk_hour(ts), weekday(ts)
wkend = wd >= 5
print("holdout feeds", len(M), "weekend share", wkend.mean().round(4), "user0 share", (uid == 0).mean().round(4))
import datetime as dt
dnames = {d: dt.date(1970, 1, 1) + dt.timedelta(days=int(d)) for d in np.unique(day)}
for d in np.unique(day):
    print(f"  {dnames[d]} {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][(d + 3) % 7]} n={int((day == d).sum())}")

nd = {}
for name, mem in POOLS.items():
    s = holdout_pool(mem, M)
    nd[name] = ndcg_per_feed(s.astype(np.float32), y, off, M)
    print(f"{name:6s} holdout {nd[name].mean():.5f} (n members {len(mem)})")
np.savez(OUT / "p1_holdout_perfeed.npz", ts=ts, uid=uid, M=M, **nd)

CMP = [("M2", "PF7x"), ("G3", "PF7x"), ("F7x", "F7b"), ("F7x", "F7"), ("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"),
       ("F7x70", "F7b")]
rows = []
print("\n== weekday vs weekend (paired per-feed deltas, mean ± SE) ==")
for a, b in CMP:
    d = nd[a] - nd[b]
    m_all, m_wd, m_we = mse(d), mse(d[~wkend]), mse(d[wkend])
    diff = m_we[0] - m_wd[0]; sed = np.hypot(m_we[1], m_wd[1])
    print(f"{a:5s}-{b:5s} all {fmt(*m_all)} | weekday {fmt(*m_wd)} | weekend {fmt(*m_we)} | we-wd {fmt(diff, sed)}")
    rows.append(dict(cmp=f"{a}-{b}", all=m_all[0], all_se=m_all[1], wd=m_wd[0], wd_se=m_wd[1], we=m_we[0],
                     we_se=m_we[1], we_minus_wd=diff, we_minus_wd_se=sed))
pl.DataFrame(rows).write_csv(OUT / "p1_weekday.csv")

print("\n== by MSK day ==")
hdr = "day            n    " + " ".join(f"{a + '-' + b:>18s}" for a, b in CMP[:5])
print(hdr)
rows = []
for d in np.unique(day):
    m = day == d
    if m.sum() < 200:
        continue
    cells = []
    for a, b in CMP:
        mm, se = mse((nd[a] - nd[b])[m]); cells.append((mm, se))
        rows.append(dict(day=str(dnames[d]), wd=int((d + 3) % 7), n=int(m.sum()), cmp=f"{a}-{b}", d=mm, se=se))
    print(f"{dnames[d]} {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][(d + 3) % 7]} {int(m.sum()):5d} " +
          " ".join(f"{c[0]:+.4f}±{c[1]:.4f}" for c in cells[:5]))
pl.DataFrame(rows).write_csv(OUT / "p1_byday.csv")

print("\n== by MSK hour bin (all days) and weekend x hour ==")
hb = np.digitize(hr, [6, 10, 14, 18])  # 0:00-05 1:06-09 2:10-13 3:14-17 4:18-23
hbn = ["h00-05", "h06-09", "h10-13", "h14-17", "h18-23"]
rows = []
for k in range(5):
    for lab, msk in [("all", np.ones_like(wkend)), ("wd", ~wkend), ("we", wkend)]:
        m = (hb == k) & msk
        out = []
        for a, b in CMP[:4]:
            mm, se = mse((nd[a] - nd[b])[m]); out.append(f"{a}-{b} {mm:+.4f}±{se:.4f}")
            rows.append(dict(hour=hbn[k], part=lab, n=int(m.sum()), cmp=f"{a}-{b}", d=mm, se=se))
        print(f"{hbn[k]} {lab:3s} n={int(m.sum()):5d} " + " | ".join(out))
pl.DataFrame(rows).write_csv(OUT / "p1_byhour.csv")

# Saturday + Sunday morning (< 11:32 MSK) = the public-like slice; Sunday afternoon + Mon/Tue = private-like
pub_like = (wd == 5) | ((wd == 6) & (hr * 60 < 11 * 60 + 32))
priv_like = ((wd == 6) & (hr * 60 >= 11 * 60 + 32)) | (wd == 0) | (wd == 1)
print("\n== public-like (Sat + Sun<11:32) vs private-like (Sun>=11:32, Mon, Tue) vs val-like (Tue, Thu) ==")
rows = []
for a, b in CMP:
    d = nd[a] - nd[b]
    r = [mse(d[pub_like]), mse(d[priv_like]), mse(d[(wd == 1) | (wd == 3)])]
    print(f"{a:5s}-{b:5s} pub-like n={pub_like.sum()} {fmt(*r[0])} | priv-like n={priv_like.sum()} {fmt(*r[1])} | "
          f"Tue+Thu {fmt(*r[2])}")
    rows.append(dict(cmp=f"{a}-{b}", pub_like=r[0][0], pub_like_se=r[0][1], priv_like=r[1][0], priv_like_se=r[1][1],
                     tuethu=r[2][0], tuethu_se=r[2][1]))
pl.DataFrame(rows).write_csv(OUT / "p1_publike.csv")
