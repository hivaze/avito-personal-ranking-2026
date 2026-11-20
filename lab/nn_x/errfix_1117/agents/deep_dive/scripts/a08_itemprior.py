"""Data idea: item-level prior from discovery labels (leave-feed-out / out-of-fold click rate of the same item_id)."""
from util import *

df = load()
c = Ctx(df)
item = df["item_id"].to_numpy()
y = c.y
fold = c.folds_row
fid = c.fid
pl.Config.set_tbl_rows(50)

# coverage
u, inv, cnt = np.unique(item, return_inverse=True, return_counts=True)
pos_cnt = np.bincount(inv, weights=y)
print("rows", len(item), "distinct items", len(u))
print("rows whose item appears in another discovery feed: %.3f" % (cnt[inv] > 1).mean())
# per-feed duplicates of item within a feed?
dup_in_feed = df.group_by(["fidx", "item_id"]).len().filter(pl.col("len") > 1).height
print("item repeated inside a feed:", dup_in_feed)
lfo_pos = pos_cnt[inv] - y
lfo_imp = cnt[inv] - 1
print("rows with lfo impressions>0: %.4f; with lfo positives>0: %.4f" % ((lfo_imp > 0).mean(), (lfo_pos > 0).mean()))
# label rate by lfo_pos
for k in [0, 1, 2]:
    m = (lfo_pos == k) if k < 2 else (lfo_pos >= 2)
    print(f"  lfo_pos {k}{'+' if k == 2 else ''}: rows {m.sum()}, label rate {y[m].mean():.4f}, mean rank {df['rank'].to_numpy()[m].mean():.1f}")
m = (lfo_imp > 0) & (lfo_pos == 0)
print(f"  seen elsewhere, never clicked: rows {m.sum()} label rate {y[m].mean():.4f}")
m = lfo_imp == 0
print(f"  not seen elsewhere: rows {m.sum()} label rate {y[m].mean():.4f}")

# out-of-fold prior: counts from the other 4 folds only (as on a new sample of users)
oof_pos = np.zeros(len(y)); oof_imp = np.zeros(len(y))
for f in range(5):
    tr = fold != f
    te = fold == f
    ip = np.bincount(inv[tr], weights=y[tr], minlength=len(u))
    ii = np.bincount(inv[tr], minlength=len(u))
    oof_pos[te] = ip[inv[te]]
    oof_imp[te] = ii[inv[te]]
print("OOF: rows with prior impressions %.4f, with prior positives %.4f" % ((oof_imp > 0).mean(), (oof_pos > 0).mean()))
for k in [0, 1, 2]:
    m = (oof_pos == k) if k < 2 else (oof_pos >= 2)
    print(f"  oof_pos {k}{'+' if k == 2 else ''}: rows {m.sum()}, label rate {y[m].mean():.4f}, mean rank {df['rank'].to_numpy()[m].mean():.1f}, mean E-rate by rank?")
# calibrated comparison: positives vs expected from rank cell
rb = np.digitize(df["rank"].to_numpy(), [2, 3, 4, 6, 11, 21, 31, 51, 101, 151])
mb = np.digitize(df["M"].to_numpy(), [30, 60, 120, 200])
cell = rb * 10 + mb
cs = np.bincount(cell, weights=y, minlength=200) / np.maximum(np.bincount(cell, minlength=200), 1)
E = cs[cell]
for name, m in [("oof_pos>=1", oof_pos >= 1), ("oof_imp>=1&pos=0", (oof_imp >= 1) & (oof_pos == 0)), ("oof_imp=0", oof_imp == 0),
                ("lfo_pos>=1", lfo_pos >= 1), ("lfo_imp>=1&pos=0", (lfo_imp >= 1) & (lfo_pos == 0))]:
    print(f"  {name:20s} rows {m.sum():7d} O {y[m].sum():6.0f} E {E[m].sum():8.1f} O/E {y[m].sum() / E[m].sum():.3f}")

# rules: score + w * (smoothed rate - base) using OOF counts (honest: prior from other users' feeds)
base_rate = y.mean()
a = 20.0
for kind, P, I in [("oof", oof_pos, oof_imp), ("lfo", lfo_pos, lfo_imp)]:
    rate = (P + a * base_rate) / (I + a)
    g1 = np.log(rate / base_rate)
    g2 = (P >= 1).astype(float)
    for gname, g in [("logratio", g1), ("anypos", g2)]:
        r = c.crossfit(lambda w: c.base + w * g, [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5])
        print(f"{kind} {gname}: cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins_best={r['ins_best']} ins={r['ins_delta']:+.5f}")
