"""Part 3 supplement: stability of production-order information over the train period (holdout, random-like feeds, labelled):
f1 / f3 NDCG and G3-TO NDCG by day and weekday/weekend on the 30K holdout; and on val, the FT-TO contrast on the 117
cross-feed-leak feeds vs the rest (does the val fine-tune's gain sit on the leak feeds the public neutralises?).
Writes out/s06_holdout_days.log."""
from lib import *  # noqa: F401,F403
import io
sys.path.insert(0, str(SHIFT))
from shiftlib import holdout_index, holdout_pool, FAM  # noqa: E402

buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


hold, fh, Mh, offh = holdout_index(); yh = np.load(P / "holdout/holdout_y.npy")
ftr = pl.read_parquet(FLAT / "train/feeds.parquet"); otr = ftr["cand_off"].to_numpy().astype(np.int64)
rows = np.concatenate([np.arange(otr[i], otr[i] + m) for i, m in zip(hold, Mh)])
f1 = np.asarray(np.load(FLAT / "train/cand_f1.npy", mmap_mode="r")[rows], np.float64)
f3 = np.asarray(np.load(FLAT / "train/cand_f3.npy", mmap_mode="r")[rows], np.float64)
rng = np.random.default_rng(0)
nd_f1 = ndcg_per_feed(f1 + 1e-7 * rng.standard_normal(len(f1)), yh, offh, Mh)
nd_f3 = ndcg_per_feed(f3 + 1e-7 * rng.standard_normal(len(f3)), yh, offh, Mh)
nd_rand = ndcg_per_feed(rng.standard_normal(len(f1)), yh, offh, Mh)
g3 = ndcg_per_feed(holdout_pool(FAM["F10"] + FAM["F7x"] + FAM["F7x70"], Mh), yh, offh, Mh)
tsh = fh["timestamp"].to_numpy().astype(np.int64); dayi = (tsh + MSK) // 86400; dow = (dayi + 3) % 7
DN = "Mon Tue Wed Thu Fri Sat Sun".split()
say("6A. HOLDOUT (train period, random-like M>=50 feeds): NDCG of f1, f3, random and G3-TO by MSK day. If the production ranker's")
say("    information were drifting over a 2-week window, f1's lift over random would trend; it does not.")
say(f"  {'day':14s} {'n':>5s} {'f1':>8s} {'f1-rand':>8s} {'f3-rand':>8s} {'G3-TO':>8s} {'G3-rand':>8s} {'SE(G3)':>7s}")
for u in np.unique(dayi):
    m = dayi == u
    if m.sum() < 200:
        continue
    say(f"  {int(u)} {DN[int(dow[m][0])]:9s} {m.sum():5d} {nd_f1[m].mean():8.4f} {(nd_f1 - nd_rand)[m].mean():+8.4f} {(nd_f3 - nd_rand)[m].mean():+8.4f} {g3[m].mean():8.4f} {(g3 - nd_rand)[m].mean():+8.4f} {g3[m].std() / np.sqrt(m.sum()):7.4f}")
for nm, m in [("weekday", dow < 5), ("weekend", dow >= 5)]:
    say(f"  {nm:14s} {m.sum():5d} {nd_f1[m].mean():8.4f} {(nd_f1 - nd_rand)[m].mean():+8.4f} {(nd_f3 - nd_rand)[m].mean():+8.4f} {g3[m].mean():8.4f} {(g3 - nd_rand)[m].mean():+8.4f} {g3[m].std() / np.sqrt(m.sum()):7.4f}")
x = dayi - dayi.min()
for nm, v in [("f1-rand", nd_f1 - nd_rand), ("G3-rand", g3 - nd_rand), ("G3-f1", g3 - nd_f1)]:
    b = np.polyfit(x, v, 1)[0]; se = np.sqrt(v.var() / ((x - x.mean()) ** 2).sum())
    say(f"  trend of {nm} per day: {b:+.5f} ± {se:.5f}")

say("")
say("6B. VAL: FT-TO contrast (G3 halves) on the 117 cross-feed-leak feeds vs the rest; and G3 itself on those feeds. If the FT")
say("    half had learned the leak, its gain would concentrate there (public neutralises those feeds).")
dv = descriptors("val"); fv, Mv, offv = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
g3ft = ndcg_per_feed(np.load(P / "ens_G3_ft_val.npy"), y, offv, Mv); g3to = ndcg_per_feed(np.load(P / "ens_G3_to_val.npy"), y, offv, Mv)
leak = dv["leak"].to_numpy() == 1
d = g3ft - g3to
say(f"  leak feeds {leak.sum()}: FT-TO {wmean_se(d[leak])[0]:+.5f}±{wmean_se(d[leak])[1]:.5f} | other feeds FT-TO {wmean_se(d[~leak])[0]:+.5f}±{wmean_se(d[~leak])[1]:.5f}")
say(f"  G3 NDCG on leak feeds {per_feed('G3')[leak].mean():.4f} vs others {per_feed('G3')[~leak].mean():.4f}; contribution of leak feeds to the val mean: {per_feed('G3')[leak].sum() / len(Mv):.5f}")
for p in ["M2", "Pall", "Cd", "S2", "TOonly"]:
    dd = per_feed(p) - per_feed("G3")
    say(f"  {p:6s}-G3 on leak feeds {wmean_se(dd[leak])[0]:+.5f}±{wmean_se(dd[leak])[1]:.5f} (val-mean contribution {dd[leak].sum() / len(Mv):+.6f}) | others {wmean_se(dd[~leak])[0]:+.5f}")

say("")
say("6C. SE ARITHMETIC for the rule (val, n = 14,423; public n = 8,562; private n = 20,033; holdout n = 30,000, M>=50):")
for p in ["M2", "S2", "TOonly", "PF7x", "Pall", "Cd"]:
    dd = per_feed(p) - per_feed("G3"); sd = dd.std(ddof=1)
    say(f"  {p:6s}-G3: per-feed sd {sd:.4f} -> SE val {sd / np.sqrt(14423):.5f}, Tue-half {sd / np.sqrt(7542):.5f}, public {sd / np.sqrt(8562):.5f}, private {sd / np.sqrt(20033):.5f}, val+public transfer {np.sqrt(sd ** 2 / 14423 + sd ** 2 / 8562):.5f}")

(OUT / "s06_holdout_days.log").write_text(buf.getvalue())
print("written", OUT / "s06_holdout_days.log")
