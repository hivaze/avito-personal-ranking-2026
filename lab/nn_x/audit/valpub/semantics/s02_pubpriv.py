"""Part 2: the public/private boundary; public vs private differences beyond cold (effect sizes); which differences move the
NDCG LEVEL vs the within-feed RANKING (pool deltas); weekend / public-like vs private-like contrasts on the train-period
holdout (train-only pools, selection-free). Writes out/s02_pubpriv.log."""
from lib import *  # noqa: F401,F403
import io
import numpy.linalg as la
sys.path.insert(0, str(SHIFT))
from shiftlib import holdout_index, holdout_pool, FAM  # noqa: E402

buf = io.StringIO()


def say(*a):
    print(*a); print(*a, file=buf)


dv, dt = descriptors("val"), descriptors("test")
fv, Mv, offv = feed_meta("val"); ft, Mt, offt = feed_meta("test")
ts_t = ft["timestamp"].to_numpy().astype(np.int64)
pub = ts_t < BOUND

say("=" * 120)
say("2A. BOUNDARY. Organisers (research_competition_avito.md / review_metric.md:74): public = earliest ~30% of test feeds by time,")
say("    same-timestamp feeds kept together, closest to 30% by count.")
say("=" * 120)
n = len(ts_t); st = np.sort(ts_t)
q = st[int(np.floor(0.3 * n))]
say(f"  n_test {n}; 0.30 n = {0.3 * n:.1f}; timestamp at rank floor(0.3n) = {q} ({'==' if q == BOUND else '!='} BOUND {BOUND})")
for rule, m in [("ts <  BOUND", ts_t < BOUND), ("ts <= BOUND", ts_t <= BOUND)]:
    say(f"  {rule}: {m.sum()} feeds = {m.mean():.5f}; |count - 0.3n| = {abs(m.sum() - 0.3 * n):.1f}")
say(f"  feeds exactly at BOUND: {(ts_t == BOUND).sum()}; next tick {st[st > BOUND][0]} ({(ts_t <= st[st > BOUND][0]).sum()} feeds)")
say("  -> the 'closest to 30% by count with ties kept together' rule gives ts <= BOUND (8,580 feeds); the project's calib/pmix use ts < BOUND")
say("     (8,562). The 18 boundary feeds change any reweighted prediction by < 0.00003 (REVIEW_SYNTHESIS METRIC-4); immaterial. We keep")
say("     '< BOUND' for comparability with every number on record.")
say("  Caveat on record: the 59 public-time feeds re-ranked by the later-history overlay did not move the public score at all")
say("  (CURRENT_STATUS 21:52), so feeds whose user has a later test feed are either not scored or have no positives; 0.7% of public.")

say("")
say("=" * 120)
say("2B. PUBLIC vs PRIVATE: standardised mean differences (SMD = (private - public) / pooled sd) of label-free feed descriptors,")
say("    with three yardsticks: val Tue vs Thu (2-day gap, same pipeline), test Sat vs Sun (both weekend), test Mon vs Tue (both weekday)")
say("=" * 120)
DESC = ["logM", "H", "H256", "H0", "uid0", "train_user", "n_train_feeds", "cold_share", "luo_cold_share", "luo_logmean", "luo_logmax",
        "ccnt_zero_share", "et1_logmean", "age_logmean_h", "age_lt1d_share", "age_lt7d_share", "age_capped_share", "f1_mean", "f1_sd",
        "f1_atmin", "f2_mean", "f3_mean", "f4_mean", "f5_mean", "n_cat", "max_cat_share", "cand_cat_in_hist_share", "hist_n_cat",
        "hist_top_cat_share", "het1", "het2", "het4", "hes0", "hes1", "hour"]
di_v = dv["dayidx"].to_numpy(); di_t = dt["dayidx"].to_numpy()
vt, vh = di_v == 20718, di_v == 20720
sat, sun, mon, tue = di_t == 20722, di_t == 20723, di_t == 20724, di_t == 20725
say(f"  {'descriptor':24s} {'pub mean':>10s} {'priv mean':>10s} {'SMD pub->priv':>14s} | {'val Tue->Thu':>12s} {'Sat->Sun':>9s} {'Mon->Tue':>9s} | {'val->pub':>9s} {'val->priv':>9s}")
rows = []
for c in DESC:
    xv = dv[c].to_numpy().astype(float); xt = dt[c].to_numpy().astype(float)
    r = (c, xt[pub].mean(), xt[~pub].mean(), smd(xt[pub], xt[~pub]), smd(xv[vt], xv[vh]), smd(xt[sat], xt[sun]), smd(xt[mon], xt[tue]), smd(xv, xt[pub]), smd(xv, xt[~pub]))
    rows.append(r)
for r in sorted(rows, key=lambda r: -abs(r[3])):
    say(f"  {r[0]:24s} {r[1]:10.4f} {r[2]:10.4f} {r[3]:+14.3f} | {r[4]:+12.3f} {r[5]:+9.3f} {r[6]:+9.3f} | {r[7]:+9.3f} {r[8]:+9.3f}")
hb_t = dt["hourb"].to_numpy(); hb_v = dv["hourb"].to_numpy()
say(f"  hour bucket shares (h00-05 h06-09 h10-13 h14-17 h18-23): public {np.round(np.bincount(hb_t[pub], minlength=5) / pub.sum(), 3)} "
    f"private {np.round(np.bincount(hb_t[~pub], minlength=5) / (~pub).sum(), 3)} val {np.round(np.bincount(hb_v, minlength=5) / len(hb_v), 3)}")
say(f"  weekend share: public {dt['weekend'].to_numpy()[pub].mean():.3f} private {dt['weekend'].to_numpy()[~pub].mean():.3f} val 0")

say("")
say("2C. HOUR PROFILE WITH OUTAGE HOURS REMOVED. Feeds per (day, hour); an outage hour = count < 25% of that day's median hourly count.")
say("    Val lost Tue 20h and Thu 20-21h, so val's evening share is depressed by construction, not behaviour.")
HR_v = dv["hour"].to_numpy().astype(int); HR_t = dt["hour"].to_numpy().astype(int)


def day_table(hours, dayidx, tag):
    out = {}
    for u in np.unique(dayidx):
        m = dayidx == u
        if m.sum() < 100:
            continue
        cnt = np.bincount(hours[m], minlength=24)
        med = np.median(cnt[cnt > 0])
        outage = np.where(cnt < 0.25 * med)[0]
        out[u] = (cnt, outage)
        say(f"  {tag} day {u}: outage hours {outage.tolist()} (median hourly count {med:.0f}); counts {cnt.tolist()}")
    return out


tv = day_table(HR_v, di_v, "val"); tt = day_table(HR_t, di_t, "test")
# corrected shares: rescale each day's counts by filling outage hours with the day's mean of its non-outage neighbours (hour-1, hour+1)


def corrected_share(table, mask_fn):
    tot = np.zeros(24)
    for u, (cnt, outage) in table.items():
        c = cnt.astype(float).copy()
        for h in outage:
            nb = [cnt[(h - 1) % 24], cnt[(h + 1) % 24]]
            nb = [x for x in nb if x >= 0.25 * np.median(cnt[cnt > 0])]
            c[h] = np.mean(nb) if nb else c[h]
        if mask_fn(u):
            tot += c
    return tot / tot.sum()


def share_14_24(p):
    return p[14:].sum(), p[18:].sum()


for tag, table, fn in [("val", tv, lambda u: True), ("test Sat", tt, lambda u: u == 20722), ("test Sun", tt, lambda u: u == 20723),
                       ("test Mon", tt, lambda u: u == 20724), ("test Tue", tt, lambda u: u == 20725), ("test all", tt, lambda u: True)]:
    p = corrected_share(table, fn)
    raw_mask = (np.ones(len(HR_v), bool) if tag == "val" else np.array([fn(u) for u in di_t]))
    raw = np.bincount((HR_v if tag == "val" else HR_t)[raw_mask], minlength=24) / raw_mask.sum()
    say(f"  {tag:9s} share 14-24h raw {share_14_24(raw)[0]:.3f} corrected {share_14_24(p)[0]:.3f} | 18-24h raw {share_14_24(raw)[1]:.3f} corrected {share_14_24(p)[1]:.3f}")
say("  (public = Sat + Sun<11:32 by construction has one evening in two days; private = 2.5 evenings in 2.5 days: the public->private")
say("   hour gap is the boundary's construction, and the val->private gap is mostly val's two evening outages.)")

say("")
say("=" * 120)
say("2D. LEVEL vs RANKING. OLS on val feeds: per-feed NDCG(G3) [level] and per-feed delta (pool - G3) [ranking] on standardised")
say("    label-free descriptors; projected shift = beta . (mean_target - mean_source) with SE from the OLS covariance (HC0).")
say("    A descriptor matters for POOL CHOICE only if it has a non-zero coefficient in a delta model AND shifts between public and private.")
say("=" * 120)
REG = ["logM", "H", "uid0", "train_user", "cold_share", "luo_logmean", "age_lt7d_share", "f1_sd", "n_cat", "cand_cat_in_hist_share", "hist_top_cat_share"]
Xv = np.column_stack([dv[c].to_numpy().astype(float) for c in REG] + [(hb_v == i).astype(float) for i in range(1, 5)] + [(dv["day"].to_numpy() == 2).astype(float)])
Xt = np.column_stack([dt[c].to_numpy().astype(float) for c in REG] + [(hb_t == i).astype(float) for i in range(1, 5)] + [np.zeros(len(Mt))])
names = REG + ["h06-09", "h10-13", "h14-17", "h18-23", "Thu"]
mu, sd = Xv.mean(0), Xv.std(0) + 1e-12
Zv = (Xv - mu) / sd; Zt = (Xt - mu) / sd
Zv1 = np.column_stack([np.ones(len(Zv)), Zv])
shift_pp = Zt[~pub].mean(0) - Zt[pub].mean(0)      # public -> private
shift_vp = Zt[pub].mean(0) - Zv.mean(0)             # val -> public
shift_vq = Zt[~pub].mean(0) - Zv.mean(0)            # val -> private
shift_vq[-1] = 0 - Zv[:, -1].mean()                 # 'Thu' has no test analogue: project at the val mean (0 shift) instead
shift_vp[-1] = 0 - Zv[:, -1].mean()
shift_pp[-1] = 0
say("  standardised descriptor shifts (in val sd units): " + ", ".join(f"{n} pub->priv {a:+.2f} / val->priv {b:+.2f}" for n, a, b in zip(names, shift_pp, shift_vq)))


def ols(yv):
    b, *_ = la.lstsq(Zv1, yv, rcond=None)
    r = yv - Zv1 @ b
    XtX_inv = la.inv(Zv1.T @ Zv1)
    meat = (Zv1 * (r ** 2)[:, None]).T @ Zv1
    cov = XtX_inv @ meat @ XtX_inv
    return b, cov


targets = {"LEVEL G3": per_feed("G3")}
for p in ["Cd", "Pall", "PF7x", "M2", "S2", "TOonly"]:
    targets[f"d {p}-G3"] = per_feed(p) - per_feed("G3")
y = np.load(FLAT / "val/cand_target.npy")
g3ft = ndcg_per_feed(np.load(P / "ens_G3_ft_val.npy"), y, offv, Mv); g3to = ndcg_per_feed(np.load(P / "ens_G3_to_val.npy"), y, offv, Mv)
targets["d FT-TO (G3 halves)"] = g3ft - g3to
say(f"  {'target':22s} {'val mean':>9s} | {'proj pub->priv':>15s} {'proj val->pub':>15s} {'proj val->priv':>15s} | coefficients with |t| >= 2 (per val sd of the descriptor)")
coef_rows = {}
for nm, yv in targets.items():
    b, cov = ols(yv)
    bb = b[1:]; cc = cov[1:, 1:]
    proj = [(float(bb @ s), float(np.sqrt(s @ cc @ s))) for s in (shift_pp, shift_vp, shift_vq)]
    tt_ = bb / np.sqrt(np.diag(cc))
    sig = [f"{n} {bb[i]:+.5f}({tt_[i]:+.1f})" for i, n in enumerate(names) if abs(tt_[i]) >= 2]
    coef_rows[nm] = (bb, np.sqrt(np.diag(cc)))
    say(f"  {nm:22s} {yv.mean():+9.5f} | {proj[0][0]:+.5f}±{proj[0][1]:.5f} {proj[1][0]:+.5f}±{proj[1][1]:.5f} {proj[2][0]:+.5f}±{proj[2][1]:.5f} | " + "; ".join(sig))
say("  Joint reading per descriptor (delta models only): max |t| over the 7 delta targets, and the public->private shift:")
for i, nm in enumerate(names):
    ts_ = [coef_rows[k][0][i] / coef_rows[k][1][i] for k in coef_rows if k != "LEVEL G3"]
    say(f"    {nm:22s} max|t| {max(abs(t) for t in ts_):.1f} (signs {''.join('+' if t > 0 else '-' for t in ts_)}) | pub->priv shift {shift_pp[i]:+.2f} sd | level t {coef_rows['LEVEL G3'][0][i] / coef_rows['LEVEL G3'][1][i]:+.1f}")
say("  (15 descriptors x 7 delta targets = 105 tests; expect ~5 with |t| >= 2 by chance.)")

say("")
say("=" * 120)
say("2E. WEEKEND / PUBLIC-LIKE vs PRIVATE-LIKE on the 30K TRAIN-PERIOD HOLDOUT (train-only member pools; the only labelled weekend")
say("    sample). public-like = Sat all day + Sun < 11:32 MSK; private-like = Sun >= 11:32 + Mon + Tue; val-like = Tue + Thu.")
say("=" * 120)
hold, fh, Mh, offh = holdout_index()
yh = np.load(P / "holdout/holdout_y.npy")
tsh = fh["timestamp"].to_numpy().astype(np.int64)
dowh = (((tsh + MSK) // 86400) + 3) % 7; hh = ((tsh + MSK) % 86400) // 3600
mins = (tsh + MSK) % 86400 // 60
publike = (dowh == 5) | ((dowh == 6) & (mins < 11 * 60 + 32))
privlike = ((dowh == 6) & (mins >= 11 * 60 + 32)) | (dowh == 0) | (dowh == 1)
valike = (dowh == 1) | (dowh == 3)
wkend = dowh >= 5
hbh = np.searchsorted(np.array([5, 9, 13, 17]), hh, side="right")
say(f"  holdout feeds {len(Mh)}: weekend {wkend.sum()}, public-like {publike.sum()}, private-like {privlike.sum()}, val-like {valike.sum()}; M>=50 only (mean M {Mh.mean():.1f})")
HP = {"Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"], "Pall": FAM["F5cd"] + FAM["F6"] + FAM["F7"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
      "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"], "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
      "M2": FAM["F7x"] + FAM["F7x70"], "F7x": FAM["F7x"], "F10": FAM["F10"]}
PFH = {k: ndcg_per_feed(holdout_pool(v, Mh), yh, offh, Mh) for k, v in HP.items()}
say(f"  holdout NDCG: " + "  ".join(f"{k} {v.mean():.5f}" for k, v in PFH.items()))
say(f"  {'contrast':12s} {'all':>18s} {'weekday':>18s} {'weekend':>18s} {'wkend-wkday':>18s} {'public-like':>18s} {'private-like':>18s} {'priv-pub like':>18s} {'val-like':>18s}")
for a, b in [("Pall", "G3"), ("Cd", "G3"), ("PF7x", "G3"), ("M2", "G3"), ("F7x", "F10"), ("M2", "PF7x")]:
    d = PFH[a] - PFH[b]
    cells = []
    for m in [np.ones(len(d), bool), ~wkend, wkend]:
        cells.append(wmean_se(d[m]))
    dw = (cells[2][0] - cells[1][0], np.sqrt(cells[2][1] ** 2 + cells[1][1] ** 2))
    cp, cq = wmean_se(d[publike]), wmean_se(d[privlike])
    dpq = (cq[0] - cp[0], np.sqrt(cp[1] ** 2 + cq[1] ** 2))
    cv_ = wmean_se(d[valike])
    say(f"  {a + '-' + b:12s} " + " ".join(f"{x[0]:+.5f}±{x[1]:.5f}" for x in [cells[0], cells[1], cells[2], dw, cp, cq, dpq, cv_]))
say("  by hour bucket x weekend (M2-PF7x and Pall-G3; the private-like set is evening-heavy):")
for a, b in [("M2", "PF7x"), ("Pall", "G3"), ("PF7x", "G3")]:
    d = PFH[a] - PFH[b]
    say(f"  {a + '-' + b:12s} " + " | ".join(f"{['h00-05', 'h06-09', 'h10-13', 'h14-17', 'h18-23'][i]} wd {wmean_se(d[(hbh == i) & ~wkend])[0]:+.5f} we {wmean_se(d[(hbh == i) & wkend])[0]:+.5f}" for i in range(5)))
say("  holdout val-like vs public-like vs private-like LEVEL of G3 (M>=50 production-random feeds): "
    f"{PFH['G3'][valike].mean():.5f} / {PFH['G3'][publike].mean():.5f} / {PFH['G3'][privlike].mean():.5f}; weekday {PFH['G3'][~wkend].mean():.5f} weekend {PFH['G3'][wkend].mean():.5f}")

(OUT / "s02_pubpriv.log").write_text(buf.getvalue())
print("written", OUT / "s02_pubpriv.log")
