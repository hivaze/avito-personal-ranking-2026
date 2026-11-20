"""a06: (1) pagination / slot structure: M histogram spikes (30/31?), positive rate and feature means by production pos in
random-like vs production-like train feeds (a non-flat feature profile by pos in random-like feeds would mean fixed slots
that leak through features); M=30/31 feeds: where is the positive. (2) listing age (f4 -> hours) tables on val and test
public/private; positive rate by age; 30-day re-post cycle. (3) per-event-type same-category lift (recent events) to
identify negative-feedback types."""
from common_av import *

tr = meta("train")
hold, p = holdout_index()
rl = np.where((p > 0.9) & (tr["M"] >= 50))[0]
pr = np.where((p < 0.05) & (tr["M"] >= 50))[0]
rng = np.random.default_rng(3)
rl = np.sort(rng.choice(rl, 40000, replace=False))
pr = np.sort(rng.choice(pr, 40000, replace=False))


def rows_of(d, feeds):
    return np.concatenate([np.arange(o, o + m) for o, m in zip(d["off"][feeds], d["M"][feeds])])


log("== 1a. M histogram around page boundaries ==")
for name, M in [("train all", tr["M"]), ("val", meta("val")["M"]), ("test", meta("test")["M"])]:
    h = np.bincount(M, minlength=301) / len(M)
    log(f"{name:9s} M=28..34: {np.round(h[28:35] * 1000, 2)} per mille; 58..64: {np.round(h[58:65] * 1000, 2)}; 88..94: {np.round(h[88:95] * 1000, 2)}; 298..300: {np.round(h[298:301] * 1000, 2)}")
    log(f"{name:9s} smoothness ratio h[31]/mean(h[29],h[33]) = {h[31] / np.mean([h[29], h[33]]):.2f}; h[30]/mean(h[28],h[32]) = {h[30] / np.mean([h[28], h[32]]):.2f}; h[61]/mean(h[59],h[63]) = {h[61] / np.mean([h[59], h[63]]):.2f}")

log("== 1b. train feeds with M in {30,31}: pos structure ==")
for Mv in [30, 31, 60, 61]:
    feeds = np.where(tr["M"] == Mv)[0][:60000]
    rows = rows_of(tr, feeds)
    pos = np.asarray(cand("train", "pos")[rows]).astype(np.int64)
    y = np.asarray(cand("train", "target")[rows]).astype(np.float64)
    mxpos = np.bincount(np.repeat(np.arange(len(feeds)), Mv), weights=pos, minlength=len(feeds)) if False else None
    maxpos = np.maximum.reduceat(pos, np.arange(0, len(rows), Mv))
    log(f"M={Mv}: feeds {len(feeds)}, max pos distribution top: {sorted(zip(*np.unique(maxpos, return_counts=True)), key=lambda t: -t[1])[:5]}; share with gaps {(maxpos != Mv - 1).mean():.3f}")
    rate_by_pos = np.bincount(pos, weights=y, minlength=70)[:Mv + 5] / np.maximum(np.bincount(pos, minlength=70)[:Mv + 5], 1)
    log(f"   positive rate by pos 0..{Mv + 1}: {np.round(rate_by_pos[:Mv + 2] * 100, 1)}")
    log(f"   share of positives at pos >= {Mv - 2}: {y[pos >= Mv - 2].sum() / y.sum():.3f}")

log("== 1c. positive rate and feature means by pos (0..59) in random-like vs production-like train feeds (M>=50) ==")
for name, feeds in [("random-like", rl), ("production-like", pr)]:
    rows = rows_of(tr, feeds)
    pos = np.asarray(cand("train", "pos")[rows]).astype(np.int64)
    y = np.asarray(cand("train", "target")[rows]).astype(np.float64)
    f4 = np.asarray(cand("train", "f4")[rows]).astype(np.float64)
    f5 = np.asarray(cand("train", "f5")[rows]).astype(np.float64)
    f1 = np.asarray(cand("train", "f1")[rows]).astype(np.float64)
    cat = np.asarray(cand("train", "category_id")[rows]).astype(np.int64)
    ok = pos < 60
    cnt = np.bincount(pos[ok], minlength=60).astype(np.float64)
    r = np.bincount(pos[ok], weights=y[ok], minlength=60) / cnt
    m4 = np.bincount(pos[ok], weights=f4[ok], minlength=60) / cnt
    m5 = np.bincount(pos[ok], weights=f5[ok], minlength=60) / cnt
    m1 = np.bincount(pos[ok], weights=f1[ok], minlength=60) / cnt
    # category entropy by pos
    ent = []
    for q in range(60):
        c = np.bincount(cat[ok & (pos == q)], minlength=64) / max(cnt[q], 1)
        ent.append(-(c[c > 0] * np.log(c[c > 0])).sum())
    log(f"{name}: pos rate %: {np.round(r * 100, 2)[:40]}")
    log(f"{name}: mean f4 (age code): {np.round(m4, 3)[:40]}")
    log(f"{name}: mean f5: {np.round(m5, 3)[:40]}")
    log(f"{name}: mean f1: {np.round(m1, 3)[:40]}")
    log(f"{name}: cat entropy: {np.round(ent, 3)[:40]}")
    # periodicity check: autocorrelation of de-meaned feature profile at lags 1..12 (pos 1..59)
    for nm, prof in [("f4", m4), ("f5", m5), ("f1", m1), ("rate", r)]:
        x = prof[1:] - prof[1:].mean()
        ac = [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, 13)]
        log(f"   {name} {nm} profile sd {x.std():.4f}; autocorr lags 1..12: {np.round(ac, 2)}")

log("== 2. listing age (f4 -> hours) on val / public / private ==")
knots = np.load(FEATS / "life_f4_age_knots.npy")
K = 64
CAP = 63 / 64 - 1e-6


def age_hours(x):
    x = x.astype(np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    a = knots[k] * (1 - w) + knots[k + 1] * w
    return np.where(x < CAP, a, np.nan)


bins = [0, 1, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1440, 2160, 4320, 8760, 1e9]
labels = ["<1h", "1-3h", "3-6h", "6-12h", "12-24h", "1-2d", "2-3d", "3-7d", "1-2w", "2-4w", "1-2m", "2-3m", "3-6m", "6-12m", ">1y"]
v = meta("val")
yv = np.asarray(cand("val", "target"), dtype=np.float64)
av = age_hours(np.asarray(cand("val", "f4")))
t = meta("test")
at = age_hours(np.asarray(cand("test", "f4")))
pub_rows = np.repeat(t["ts"] < PUBLIC_TS, t["M"])
bv = np.digitize(np.nan_to_num(av, nan=1e8), bins) - 1
bt = np.digitize(np.nan_to_num(at, nan=1e8), bins) - 1
base = yv.mean()
log("age bin: val share | val pos-rate lift | public share | private share")
for i, lab in enumerate(labels):
    mv = bv == i
    log(f"{lab:7s} {mv.mean():.4f} | {yv[mv].mean() / base if mv.sum() else float('nan'):.2f} | {(bt == i)[pub_rows].mean():.4f} | {(bt == i)[~pub_rows].mean():.4f}")
# 30-day cycle: fine histogram of age in days 25..35
d = av / 24
log(f"val age-days histogram 26..34 (per mille): {np.round(np.histogram(d[~np.isnan(d)], bins=np.arange(26, 35))[0] / (~np.isnan(d)).sum() * 1000, 2)}")
log(f"val age-days histogram 56..64: {np.round(np.histogram(d[~np.isnan(d)], bins=np.arange(56, 65))[0] / (~np.isnan(d)).sum() * 1000, 2)}")
# G3 calibration by age bin (act/exp with softmax-normalised G3 within feed)
s = g3("val")
fid = np.asarray(cand("val", "fidx")).astype(np.int64)
z = s - np.repeat(np.maximum.reduceat(s, v["off"]), v["M"])
e = np.exp(z)
pz = e / np.repeat(np.add.reduceat(e, v["off"]), v["M"])
npos = np.bincount(fid, weights=yv, minlength=v["n"])
exp_y = pz * npos[fid]
log("G3 act/exp by age bin (softmax of pool score x npos): " + ", ".join(f"{labels[i]} {yv[bv == i].sum() / max(exp_y[bv == i].sum(), 1e-9):.3f}" for i in range(len(labels)) if (bv == i).sum() > 2000))

log("== 3. per-event-type same-category lift (recent events, hpos<32), val ==")
hf = np.asarray(hist("val", "fidx")).astype(np.int64)
ht = np.asarray(hist("val", "event_type")).astype(np.int64)
hc = np.asarray(hist("val", "category_id")).astype(np.int64)
hp = np.asarray(hist("val", "hpos")).astype(np.int64)
catv = np.asarray(cand("val", "category_id")).astype(np.int64)
NC = 64
kall = np.bincount(hf[hp < 32] * NC + hc[hp < 32], minlength=v["n"] * NC)
same_any = kall[fid * NC + catv] > 0
for et in range(1, 11):
    m = (ht == et) & (hp < 32)
    kc = np.bincount(hf[m] * NC + hc[m], minlength=v["n"] * NC)
    same = kc[fid * NC + catv] > 0
    ff = np.bincount(hf[m], minlength=v["n"]) > 0
    rows = np.repeat(ff, v["M"])
    if rows.sum() == 0:
        continue
    b0 = yv[rows].mean()
    log(f"type {et:2d}: feeds with recent type {ff.mean():.3f}; same-cat-as-type lift {yv[rows & same].mean() / b0:.2f} (share {same[rows].mean():.3f}); same-cat-as-other-recent-only {yv[rows & same_any & ~same].mean() / b0:.2f}; not-in-recent {yv[rows & ~same_any].mean() / b0:.2f}")
log("done")
