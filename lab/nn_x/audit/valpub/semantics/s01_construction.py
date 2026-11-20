"""Part 1: do val and test feeds have identical construction? + positives-per-feed vs hour/day on val and the implied
NDCG level. Writes out/s01_construction.log."""
from lib import *  # noqa: F401,F403
import io, contextlib

buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


dv, dt = descriptors("val"), descriptors("test")
fv, Mv, offv = feed_meta("val"); ft, Mt, offt = feed_meta("test")
y = np.load(FLAT / "val/cand_target.npy")
DAYNAME = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def groups(d, split):
    di = d["dayidx"].to_numpy(); dow = d["dow"].to_numpy()
    g = {}
    for u in np.unique(di):
        m = di == u
        if m.sum() < 100:
            continue
        g[f"{split}:{DAYNAME[int(dow[m][0])]}{int(u)}"] = m
    return g


G = {}
G.update(groups(dv, "val")); G.update(groups(dt, "test"))
G["val:all"] = np.ones(len(dv), bool); G["test:all"] = np.ones(len(dt), bool)
pub = dt["public"].to_numpy() == 1
G["test:public"] = pub; G["test:private"] = ~pub
ORDER = [k for k in G if k.startswith("val:") and k != "val:all"] + ["val:all"] + \
        [k for k in G if k.startswith("test:") and k not in ("test:all", "test:public", "test:private")] + ["test:all", "test:public", "test:private"]


def tab(name, fn, d=3):
    say(f"  {name:28s}" + "".join(f"{k.replace('test:', 't:').replace('val:', 'v:'):>14s}" for k in ORDER))
    row = []
    for k in ORDER:
        src = dv if k.startswith("val") else dt
        row.append(fn(src, G[k]))
    say(f"  {'':28s}" + "".join(f"{v:14.{d}f}" for v in row))
    return row


say("=" * 120)
say("1A. CONSTRUCTION: per-day label-free descriptors (MSK days). val Tue/Thu; test Sat/Sun/Mon/Tue; public = ts < BOUND")
say("=" * 120)
say(f"  feeds per group: " + ", ".join(f"{k} {G[k].sum()}" for k in ORDER))
say(f"  min M val {Mv.min()} test {Mt.min()}; val feeds with npos>=1: {(dv['npos'].to_numpy() >= 1).mean():.4f} (organisers: >=5 cands, >=1 positive)")
say("  -- feed size M")
tab("M mean", lambda d, m: d["M"].to_numpy()[m].mean(), 1)
tab("M median", lambda d, m: np.median(d["M"].to_numpy()[m]), 0)
tab("M q10", lambda d, m: np.quantile(d["M"].to_numpy()[m], .1), 0)
tab("M q90", lambda d, m: np.quantile(d["M"].to_numpy()[m], .9), 0)
tab("share M<50", lambda d, m: (d["M"].to_numpy()[m] < 50).mean())
tab("share M>=200", lambda d, m: (d["M"].to_numpy()[m] >= 200).mean())
say("  -- history / user")
tab("H mean", lambda d, m: d["H"].to_numpy()[m].mean(), 1)
tab("share H=0", lambda d, m: d["H0"].to_numpy()[m].mean())
tab("share H=256", lambda d, m: d["H256"].to_numpy()[m].mean())
tab("share uid0", lambda d, m: d["uid0"].to_numpy()[m].mean())
tab("share train user", lambda d, m: d["train_user"].to_numpy()[m].mean())
tab("n_train_feeds|train user", lambda d, m: d["n_train_feeds"].to_numpy()[m][d["train_user"].to_numpy()[m] == 1].mean(), 2)
say("  -- candidate set")
tab("n_cat mean", lambda d, m: d["n_cat"].to_numpy()[m].mean(), 2)
tab("max_cat_share", lambda d, m: d["max_cat_share"].to_numpy()[m].mean())
tab("cand_cat_in_hist", lambda d, m: d["cand_cat_in_hist_share"].to_numpy()[m].mean())
tab("f1_mean", lambda d, m: d["f1_mean"].to_numpy()[m].mean())
tab("f1_sd (within feed)", lambda d, m: d["f1_sd"].to_numpy()[m].mean())
tab("f1_atmin share", lambda d, m: d["f1_atmin"].to_numpy()[m].mean())
tab("f3_mean", lambda d, m: d["f3_mean"].to_numpy()[m].mean())
tab("f4_mean", lambda d, m: d["f4_mean"].to_numpy()[m].mean())
tab("age median (h, feed med)", lambda d, m: np.median(d["age_median_h"].to_numpy()[m]), 0)
tab("age<1d share", lambda d, m: d["age_lt1d_share"].to_numpy()[m].mean())
tab("age<7d share", lambda d, m: d["age_lt7d_share"].to_numpy()[m].mean())
tab("age capped share", lambda d, m: d["age_capped_share"].to_numpy()[m].mean())
say("  -- coverage / staleness (train-derived)")
tab("cold_share (bitmap)", lambda d, m: d["cold_share"].to_numpy()[m].mean())
tab("luo_cold_share", lambda d, m: d["luo_cold_share"].to_numpy()[m].mean())
tab("share feeds cold>.3", lambda d, m: (d["cold_share"].to_numpy()[m] > .3).mean())
tab("share feeds cold>.45", lambda d, m: (d["cold_share"].to_numpy()[m] > .45).mean())
tab("luo_logmean", lambda d, m: d["luo_logmean"].to_numpy()[m].mean())
tab("ccnt_zero_share", lambda d, m: d["ccnt_zero_share"].to_numpy()[m].mean())
say("  -- hour of day (MSK)")
tab("hour mean", lambda d, m: d["hour"].to_numpy()[m].mean(), 2)
tab("share h00-05", lambda d, m: (d["hourb"].to_numpy()[m] == 0).mean())
tab("share h06-09", lambda d, m: (d["hourb"].to_numpy()[m] == 1).mean())
tab("share h10-13", lambda d, m: (d["hourb"].to_numpy()[m] == 2).mean())
tab("share h14-17", lambda d, m: (d["hourb"].to_numpy()[m] == 3).mean())
tab("share h18-23", lambda d, m: (d["hourb"].to_numpy()[m] == 4).mean())
say("  -- history composition")
tab("het1 share", lambda d, m: d["het1"].to_numpy()[m].mean())
tab("hes0 share", lambda d, m: d["hes0"].to_numpy()[m].mean())
tab("hist_top_cat_share", lambda d, m: d["hist_top_cat_share"].to_numpy()[m].mean())

say("")
say("1B. JS divergences (bits) of marginals between groups; yardstick = val Tue vs val Thu")
cat_v = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64); cat_t = np.load(FLAT / "test/cand_category_id.npy").astype(np.int64)
fidv = np.repeat(np.arange(len(Mv)), Mv); fidt = np.repeat(np.arange(len(Mt)), Mt)


def cat_hist(cat, fid, mask):
    rows = mask[fid]
    return np.bincount(cat[rows], minlength=64)


def m_hist(M, mask):
    return np.bincount(np.minimum(M[mask].astype(int) // 10, 30), minlength=31)


def h_hist(hour, mask):
    return np.bincount(hour[mask], minlength=24)


def hb_hist(hb, mask):
    return np.bincount(hb[mask], minlength=5)


pairs = [("val:Tue20718", "val:Thu20720"), ("val:all", "test:all"), ("val:all", "test:public"), ("val:all", "test:private"),
         ("test:public", "test:private"), ("test:Sat20722", "test:Sun20723"), ("test:Mon20724", "test:Tue20725"),
         ("test:Sun20723", "test:Mon20724")]
say(f"  {'pair':32s} {'JS(cat)':>9s} {'JS(M/10)':>9s} {'JS(hour24)':>10s} {'JS(hourbin5)':>12s} {'JS(topcat)':>10s} {'JS(dow-free M x cold)':>22s}")
for a, b in pairs:
    if a not in G or b not in G:
        continue
    da, db = (dv if a.startswith("val") else dt), (dv if b.startswith("val") else dt)
    ca = cat_hist(cat_v if a.startswith("val") else cat_t, fidv if a.startswith("val") else fidt, G[a])
    cb = cat_hist(cat_v if b.startswith("val") else cat_t, fidv if b.startswith("val") else fidt, G[b])
    ma = m_hist(da["M"].to_numpy(), G[a]); mb = m_hist(db["M"].to_numpy(), G[b])
    ha = h_hist(da["hour"].to_numpy().astype(int), G[a]); hb_ = h_hist(db["hour"].to_numpy().astype(int), G[b])
    hba = hb_hist(da["hourb"].to_numpy(), G[a]); hbb = hb_hist(db["hourb"].to_numpy(), G[b])
    ta = np.bincount(da["top_cat"].to_numpy().astype(int)[G[a]], minlength=64); tb = np.bincount(db["top_cat"].to_numpy().astype(int)[G[b]], minlength=64)
    ka = (np.minimum(da["M"].to_numpy().astype(int) // 50, 5) * 4 + np.minimum((da["cold_share"].to_numpy() // .15).astype(int), 3))[G[a]]
    kb = (np.minimum(db["M"].to_numpy().astype(int) // 50, 5) * 4 + np.minimum((db["cold_share"].to_numpy() // .15).astype(int), 3))[G[b]]
    mc = js_div(np.bincount(ka, minlength=24), np.bincount(kb, minlength=24))
    say(f"  {a + ' vs ' + b:32s} {js_div(ca, cb):9.5f} {js_div(ma, mb):9.5f} {js_div(ha, hb_):10.5f} {js_div(hba, hbb):12.5f} {js_div(ta, tb):10.5f} {mc:22.5f}")

say("")
say("1C. Users: overlap and repeats")
uv = fv["user_id"].to_numpy(); ut = ft["user_id"].to_numpy()
sv = set(uv[uv > 0].tolist()); st = set(ut[ut > 0].tolist())
say(f"  val users (excl 0) {len(sv)}, test users {len(st)}, in both {len(sv & st)} ({len(sv & st) / len(st):.4f} of test users)")
say(f"  feeds/user excl 0: val {np.sum(uv > 0) / len(sv):.4f} test {np.sum(ut > 0) / len(st):.4f}")
_, cv = np.unique(uv[uv > 0], return_counts=True); _, ct = np.unique(ut[ut > 0], return_counts=True)
say(f"  users with >=2 feeds: val {(cv >= 2).sum()} test {(ct >= 2).sum()}; public-only users with >=2 feeds: {np.sum(np.unique(ut[(ut > 0) & pub], return_counts=True)[1] >= 2)}")
ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
utr = ftr["user_id"].to_numpy(); str_ = set(utr[utr > 0].tolist())
for k in ORDER:
    src = fv if k.startswith("val") else ft
    u = src["user_id"].to_numpy()[G[k]]
    say(f"  {k:16s} share of feeds whose user has train feeds {np.isin(u[u > 0], list(str_)).mean():.4f} (excl uid0)")

say("")
say("=" * 120)
say("1D. VAL ONLY: positives per feed vs hour / day, and the implied NDCG level (G3 and the random-ranking baseline)")
say("=" * 120)
npos = dv["npos"].to_numpy(); hour = dv["hour"].to_numpy().astype(int); hb = dv["hourb"].to_numpy(); day = dv["day"].to_numpy()
rnd = random_ndcg(Mv, npos)
g3 = per_feed("G3")
HB = ["h00-05", "h06-09", "h10-13", "h14-17", "h18-23"]
say(f"  {'slice':14s} {'n':>6s} {'npos mean':>10s} {'P(npos>=2)':>11s} {'posrate%':>9s} {'M mean':>7s} {'rand NDCG':>10s} {'G3 NDCG':>9s} {'G3-rand':>9s} {'G3 SE':>8s}")


def line(name, m):
    pr = 100 * npos[m].sum() / Mv[m].sum()
    say(f"  {name:14s} {m.sum():6d} {npos[m].mean():10.3f} {(npos[m] >= 2).mean():11.3f} {pr:9.3f} {Mv[m].mean():7.1f} {rnd[m].mean():10.4f} {g3[m].mean():9.4f} {(g3 - rnd)[m].mean():+9.4f} {g3[m].std() / np.sqrt(m.sum()):8.4f}")


for i, nm in enumerate(HB):
    line(nm, hb == i)
for dname, dval in [("Tue", 0), ("Thu", 2)]:
    line(dname, day == dval)
    for i, nm in enumerate(HB):
        line(f"  {dname} {nm}", (day == dval) & (hb == i))
line("all", np.ones(len(Mv), bool))

say("")
say("  Poisson-style regression of npos on label-free descriptors (OLS on log(npos) is crude; use npos>=2 logistic via OLS):")
import numpy.linalg as la
X = np.column_stack([np.ones(len(Mv)), np.log(Mv), np.log1p(dv["H"].to_numpy()), dv["uid0"].to_numpy().astype(float),
                     dv["train_user"].to_numpy().astype(float), dv["cold_share"].to_numpy(), day == 2] +
                    [(hb == i).astype(float) for i in range(1, 5)])
names = ["const", "logM", "log1pH", "uid0", "train_user", "cold_share", "Thu", "h06-09", "h10-13", "h14-17", "h18-23"]
for tgt_name, tgt in [("npos", npos.astype(float)), ("npos>=2", (npos >= 2).astype(float)), ("G3 NDCG", g3), ("rand NDCG", rnd), ("G3-rand", g3 - rnd)]:
    b, *_ = la.lstsq(X, tgt, rcond=None)
    r = tgt - X @ b
    cov = la.inv(X.T @ X) * (r @ r) / (len(tgt) - X.shape[1])
    se = np.sqrt(np.diag(cov))
    say(f"  target {tgt_name:10s}: " + "  ".join(f"{n} {b[i]:+.4f}±{se[i]:.4f}" for i, n in enumerate(names) if n != "const"))

say("")
say("  Implied LEVEL shift from the hour mix alone (val E[G3 | hour bucket] applied to each group's hour mix):")
eh = np.array([g3[hb == i].mean() for i in range(5)]); er = np.array([rnd[hb == i].mean() for i in range(5)])
for k in ORDER:
    src = dv if k.startswith("val") else dt
    mix = np.bincount(src["hourb"].to_numpy()[G[k]], minlength=5) / G[k].sum()
    say(f"  {k:16s} hour mix {np.round(mix, 3)}  -> G3 level {mix @ eh:.5f} (rand {mix @ er:.5f}); vs val-mix {mix @ eh - g3.mean():+.5f}")
say("  Same with M x hour cells (6 M bins x 5 hour bins), E[G3|cell] from val:")
Mb_v = np.searchsorted(np.array([10, 25, 50, 100, 200]), Mv, side="left"); Mb_t = np.searchsorted(np.array([10, 25, 50, 100, 200]), Mt, side="left")
cell_v = Mb_v * 5 + hb; cell_t = Mb_t * 5 + dt["hourb"].to_numpy()
ec = np.array([g3[cell_v == c].mean() if (cell_v == c).any() else np.nan for c in range(30)])
for k in ORDER:
    cells = (cell_v if k.startswith("val") else cell_t)[G[k]]
    mix = np.bincount(cells, minlength=30) / len(cells)
    ok = ~np.isnan(ec)
    say(f"  {k:16s} -> G3 level {np.nansum(mix[ok] * ec[ok]) / mix[ok].sum():.5f} (M-only: {np.nansum(mix.reshape(6, 5).sum(1) * np.array([g3[Mb_v == b].mean() for b in range(6)])):.5f})")

say("")
say("=" * 120)
say("1E. CROSS-FEED HISTORY CHANNEL (label-free construction check): for users with >=2 feeds in a split, do the EARLIER feed's")
say("    candidates appear in the LATER feed's history? On val the organisers' pipeline leaves the earlier feed's positives in the")
say("    later history (feed_table 'leak'); the G3_later overlay (+0.0045 val) scored public = G3 exactly, so test must differ.")
say("=" * 120)


def cross_feed(split, f, M, off, tag):
    u = f["user_id"].to_numpy(); ts = f["timestamp"].to_numpy()
    hfid = np.load(FLAT / f"{split}/hist_fidx.npy").astype(np.int64)
    hitem = np.load(FLAT / f"{split}/hist_item_id.npy").astype(np.int64)
    citem = np.load(FLAT / f"{split}/cand_item_id.npy").astype(np.int64)
    hoff = f["hist_off"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
    uu, cnt = np.unique(u[u > 0], return_counts=True)
    rep = uu[cnt >= 2]
    n_pairs = n_hit_any = n_cand_hit = n_cand_tot = 0; hits_pos = tot_pos = 0; hit_feeds = []
    ytv = np.load(FLAT / "val/cand_target.npy") if split == "val" else None
    for user in rep:
        idx = np.where(u == user)[0]; idx = idx[np.argsort(ts[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            n_pairs += 1
            hist_b = set(hitem[hoff[b]:hoff[b] + H[b]].tolist())
            ca = citem[off[a]:off[a] + M[a]]
            hit = np.array([c in hist_b for c in ca])
            n_cand_tot += M[a]; n_cand_hit += hit.sum(); n_hit_any += int(hit.any())
            if hit.any():
                hit_feeds.append(a)
            if ytv is not None:
                ya = ytv[off[a]:off[a] + M[a]]
                tot_pos += int(ya.sum()); hits_pos += int((hit & (ya == 1)).sum())
    say(f"  {tag:6s} repeat users {len(rep)}, consecutive pairs {n_pairs}, pairs with >=1 earlier candidate in later history {n_hit_any} "
        f"({n_hit_any / max(n_pairs, 1):.3f}); earlier-candidate rows found in later history {n_cand_hit}/{n_cand_tot} ({n_cand_hit / max(n_cand_tot, 1):.4f})"
        + (f"; earlier POSITIVES found in later history {hits_pos}/{tot_pos} ({hits_pos / max(tot_pos, 1):.3f})" if ytv is not None else ""))
    return np.array(hit_feeds, dtype=np.int64)


hv = cross_feed("val", fv, Mv, offv, "val")
ht = cross_feed("test", ft, Mt, offt, "test")
if len(ht):
    say(f"  test feeds with the channel: {len(ht)}; of which public {int(pub[ht].sum())}, private {int((~pub[ht]).sum())}")
# also: do a user's OWN later-feed candidates appear in his earlier feed's history? (sanity: should be ~0 by 'already viewed removed')
say(f"  val feeds flagged leak in feed_table: {int((dv['leak'].to_numpy() == 1).sum())}")

(OUT / "s01_construction.log").write_text(buf.getvalue())
print("written", OUT / "s01_construction.log")
