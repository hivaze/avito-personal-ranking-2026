"""(4) ID / index artefacts in val/test: emb_idx == -1, emb_idx order, item_id order, feed_id order vs labels / npos,
timestamp patterns, user_id 0 / user_id magnitude. Single-score NDCG (random tie-break) vs random, and nested blends on G3."""
from lib import *
from scipy.stats import spearmanr

V = Split("val"); T = Split("test")
base = g3(V); pf0 = V.ndcg(base); log(f"G3 val {pf0.mean():.5f}")
y = V.y; tb = tiebreak(V)
cei = np.asarray(arr("val", "cand_emb_idx")).astype(np.int64)
hei = np.asarray(arr("val", "hist_emb_idx")).astype(np.int64)
cat = np.asarray(arr("val", "cand_category_id"))
rnd = V.ndcg(tb)
print(f"random order NDCG {rnd.mean():.4f}")

print("\n=== How is emb_idx assigned? ===")
ok = cei >= 0
print(f"  candidate emb_idx == -1: {np.mean(~ok):.5f} of rows ({(~ok).sum()} rows, {len(np.unique(V.fid[~ok]))} feeds); pos rate when -1: {y[~ok].mean():.4f} vs {y[ok].mean():.4f}")
print(f"  hist emb_idx == -1: {np.mean(hei < 0):.4f}")
r = spearmanr(cei[ok][:500000], V.item[ok][:500000]).correlation
print(f"  spearman(emb_idx, item_id) on 500k candidate rows: {r:.4f}")
# is emb_idx monotone in item_id (i.e. sorted bijection)?
u_item, first = np.unique(V.item[ok], return_index=True)
u_idx = cei[ok][first]
print(f"  unique candidate items {len(u_item)}; emb_idx strictly increasing with item_id: {np.all(np.diff(u_idx) > 0)}; fraction of adjacent pairs increasing {np.mean(np.diff(u_idx) > 0):.4f}")
# emb_idx vs first appearance order (row order in the flat file = feed order)
first_row = first
print(f"  spearman(emb_idx, first row of appearance among candidates): {spearmanr(u_idx[:300000], first_row[:300000]).correlation:.4f}")
# candidates that also appear in some history of the split: do they have systematically different idx?
in_hist = np.isin(V.item, np.unique(V.hitem))
print(f"  candidate rows whose item is in some val history: {in_hist.mean():.4f}; mean emb_idx {cei[ok & in_hist].mean():.0f} vs {cei[ok & ~in_hist].mean():.0f} (table size 2516711)")
print(f"  pos rate: in-hist {y[in_hist].mean():.4f} vs not {y[~in_hist].mean():.4f}")

print("\n=== Single-score NDCG with random tie-break (val), vs random %.4f ===" % rnd.mean())
pos_in_feed = np.arange(V.N) - np.repeat(V.off, V.M)
scores = {"+emb_idx": cei.astype(np.float64), "-emb_idx": -cei.astype(np.float64), "+item_id": V.item.astype(np.float64), "-item_id": -V.item.astype(np.float64),
          "emb_idx==-1 first": (~ok).astype(np.float64), "emb_idx==-1 last": ok.astype(np.float64),
          "+array pos": pos_in_feed.astype(np.float64), "-array pos": -pos_in_feed.astype(np.float64),
          "+category_id": cat.astype(np.float64), "-category_id": -cat.astype(np.float64),
          "item in val history first": in_hist.astype(np.float64)}
for nm, s in scores.items():
    pf = V.ndcg(s + tb)
    d, se, z, n = paired(pf, rnd)
    print(f"  {nm:26s} {pf.mean():.4f}  vs random {d:+.4f} ± {se:.4f} (z {z:+.1f})")
# within-feed: relative emb_idx / item_id (rank within feed) blended into G3
print("\n=== Nested blends on G3 (z within feed) ===")
for nm, s in (("emb_idx", cei.astype(np.float64)), ("-emb_idx", -cei.astype(np.float64)), ("item_id", V.item.astype(np.float64)), ("-item_id", -V.item.astype(np.float64)),
              ("emb_missing", (~ok).astype(np.float64)), ("array pos", pos_in_feed.astype(np.float64))):
    nested_blend(V, base, s, pf0, label=nm)
# item_id as a proxy of listing age vs f4 (f4 is the exact listing age, feat_life_.md)
f4 = np.asarray(arr("val", "cand_f4")).astype(np.float64)
print(f"  spearman(item_id, f4) on 500k rows: {spearmanr(V.item[:500000].astype(np.float64), f4[:500000]).correlation:.4f} (f4 = listing age; item_id is a sequential listing id?)")
print(f"  spearman(item_id, -f4) within feed (mean over 2000 feeds with M>=30): ", end="")
rs = []
for i in np.flatnonzero(V.M >= 30)[:2000]:
    sl = slice(V.off[i], V.off[i] + V.M[i]); rs.append(spearmanr(V.item[sl], -f4[sl]).correlation)
print(f"{np.nanmean(rs):.3f}")

print("\n=== feed_id / timestamp / user_id (feed level) ===")
npos = V.npos()
for nm, S in (("val", V), ("test", T)):
    fid_ = S.feed_id.astype(np.int64)
    print(f"  {nm}: feed_id range [{fid_.min()}, {fid_.max()}], unique {len(np.unique(fid_)) == S.n}; spearman(feed_id, ts) {spearmanr(fid_, S.ts).correlation:+.4f}; "
          f"spearman(feed_id, user_id) {spearmanr(fid_, S.user).correlation:+.4f}; spearman(feed_id, M) {spearmanr(fid_, S.M).correlation:+.4f}; flat order sorted by ts: {np.all(np.diff(S.ts) >= 0)}")
print(f"  val: spearman(feed_id, npos) {spearmanr(V.feed_id, npos).correlation:+.4f}; spearman(fidx order within same ts, npos) ", end="")
# within-tick order: rank of fidx within its timestamp group
order_in_tick = np.zeros(V.n);
for t in np.unique(V.ts):
    m = np.flatnonzero(V.ts == t); order_in_tick[m] = np.arange(len(m)) / max(len(m) - 1, 1)
print(f"{spearmanr(order_in_tick, npos).correlation:+.4f}; spearman(order_in_tick, G3 ndcg) {spearmanr(order_in_tick, pf0).correlation:+.4f}; spearman(feed_id, G3 ndcg) {spearmanr(V.feed_id, pf0).correlation:+.4f}")
tr = feeds("train")
print(f"  train feed_id range [{tr['feed_id'].min()}, {tr['feed_id'].max()}]; spearman(feed_id, ts) in train {spearmanr(tr['feed_id'].to_numpy()[:2000000], tr['timestamp'].to_numpy()[:2000000]).correlation:+.4f}")
# timestamp grid
for nm, S in (("val", V), ("test", T)):
    g = np.gcd.reduce(np.unique(S.ts - S.ts.min()))
    print(f"  {nm}: timestamp grid gcd {g} s; unique ticks {len(np.unique(S.ts))}; feeds per tick mean {S.n / len(np.unique(S.ts)):.1f}")
# ts within feed is constant; does the (feed-level) hour affect npos? (composition only, not a ranking signal)
hour = ((V.ts + 3 * 3600) // 3600) % 24
print("  val mean npos by hour bucket: " + " ".join(f"{h0:02d}-{h0 + 6:02d}:{npos[(hour >= h0) & (hour < h0 + 6)].mean():.2f}" for h0 in (0, 6, 12, 18)))
# user id magnitude vs anything
print(f"  val: spearman(user_id, npos) {spearmanr(V.user, npos).correlation:+.4f}; spearman(user_id, H) {spearmanr(V.user, V.H).correlation:+.4f}; "
      f"user 0 feeds {int((V.user == 0).sum())} G3 ndcg {pf0[V.user == 0].mean():.4f} vs others {pf0[V.user > 0].mean():.4f}; random there {rnd[V.user == 0].mean():.4f} vs {rnd[V.user > 0].mean():.4f}")
# user_id range by split: are user ids assigned by first appearance (time)?
for nm, S in (("train", None), ("val", V), ("test", T)):
    u = tr["user_id"].to_numpy() if S is None else S.user
    u = u[u > 0]
    print(f"  {nm}: user_id range [{u.min()}, {u.max()}], q50 {np.median(u):.0f}")
# user_id vs user-in-train: new users could have higher ids if ids were assigned sequentially
tru = np.unique(tr["user_id"].to_numpy())
in_tr = np.isin(V.user, tru) & (V.user > 0)
print(f"  val users in train: mean user_id {V.user[in_tr].mean():.0f} vs not in train {V.user[(~in_tr) & (V.user > 0)].mean():.0f}")
log("done 04")
