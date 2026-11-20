"""web_shift probe s2 (label-free build): lag-equalised ("de-truncated") copies of the 5 item-occurrence inputs for VAL rows.

Idea under test (reports/TRICKS_1115/web_shift.md, proposal P1): the count of a listing is a function of its exposure length
E before the dense cutoff (s1_growth.log: the curve is the same on val, test and the in-period holdout). A train row of a
listing of age a sees E = a + u, u = time from the row to the end of the dense data; a val / test row sees E = a - lag.
The unchanged ranker therefore reads a deflated count on warm val / test rows of listings younger than about lag + 20 d.
Correction, warm rows only (count > 0), zero rows untouched:
    log1p(c') = log1p(c) + kappa * clip(m_ref(a) - m(E), 0, 2.5)
    m(E)     = mean log1p(count | count > 0) at exposure E, measured on val + test rows pooled (label-free, monotone)
    m_ref(a) = mean over train feeds of m(a + u)
Arms: A kappa = 1.0, B kappa = 0.5. Capped f4 (age unknown, old) -> no change. No label is read here.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "<repo>/feat_new")
import hz_common as H

OUT = Path("<repo>/reports/TRICKS_1115/web_shift_probe/cf"); OUT.mkdir(exist_ok=True)
EDG = np.array([0, .5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 30, 40, 60, 90, 1e4]) * 24.0  # hours
CAP = 9e4

def load(split):
    f = H.feeds(split); ts = f["timestamp"].to_numpy().astype(np.int64)
    E = H.expo_hours(split, np.full(len(ts), H.CUT_VT, np.int64), ts)
    fi = np.asarray(np.load(H.FLAT / split / "cand_fidx.npy", mmap_mode="r"))
    c = np.asarray(np.load(H.FEATS / split / "it_hist_all_luo.npy", mmap_mode="r")).astype(np.float64)
    lagh = (ts[fi] - H.CUT_VT) / 3600.0
    return E.astype(np.float64), c, lagh, fi, ts

Ev, cv, lv, fiv, tsv = load("val"); Et, ct, lt, fit, tst = load("test")
E = np.concatenate([Ev, Et]); c = np.concatenate([cv, ct])
warm = c > 0; lc = np.log1p(np.maximum(c, 0))
# m(E): bin 0 = E<0, bins 1.. = EDG intervals, last = capped
b = np.where(E >= CAP, len(EDG), np.where(E < 0, 0, np.digitize(E, EDG)))
nb = len(EDG) + 1
m = np.array([lc[warm & (b == k)].mean() if (warm & (b == k)).sum() > 200 else np.nan for k in range(nb)])
cnt = np.array([(warm & (b == k)).sum() for k in range(nb)])
m = np.maximum.accumulate(np.nan_to_num(m, nan=0.0))  # monotone
ctr = np.concatenate([[-24.0], (EDG[:-1] + EDG[1:]) / 2.0]); ctr[-1] = EDG[-2] * 1.2
print("m(E) [bin centre h : mean log1p(count | warm) : warm rows]")
for k in range(nb - 1):
    print(f"  {ctr[k]:8.0f} h  {m[k]:.3f}  {cnt[k]:,}")
print(f"  capped       {m[-1]:.3f}  {cnt[-1]:,}")
mtop = max(m[-2], m[-1])
def m_of(Eh):  # piecewise-linear in E (hours); E<0 -> m[0]
    return np.where(Eh < 0, m[0], np.interp(Eh, ctr[1:], np.minimum(m[1:-1], mtop), right=mtop))
# u = hours from a train feed to the end of the dense data
ftr = H.feeds("train"); ttr = ftr["timestamp"].to_numpy().astype(np.int64)
u = np.clip((H.CUT_VT - ttr) / 3600.0, 0, None)
uq = np.quantile(u, (np.arange(200) + 0.5) / 200)  # 200-point quadrature of the train row-day distribution
print(f"u (h to the end of the dense data): mean {u.mean():.1f}, q10 {np.quantile(u,.1):.0f}, q50 {np.quantile(u,.5):.0f}, q90 {np.quantile(u,.9):.0f}")
def m_ref(ah):
    out = np.zeros(len(ah))
    for s in range(0, len(ah), 200000):
        out[s:s + 200000] = m_of(ah[s:s + 200000, None] + uq[None, :]).mean(1)
    return out
# label-free check: predicted reference by age vs the measured mean on the in-period holdout rows (random-like train rows)
hold = np.zeros(len(ftr), bool); hold[H.holdout_feeds()] = True
fi_tr = np.load(H.FLAT / "train" / "cand_fidx.npy", mmap_mode="r")
kn = H.knots()
idx = np.where(hold[np.asarray(fi_tr)])[0]
f4h = np.asarray(np.load(H.FLAT / "train" / "cand_f4.npy", mmap_mode="r"))[idx]
ch = np.asarray(np.load(H.FEATS / "train" / "it_hist_all_luo.npy", mmap_mode="r"))[idx].astype(np.float64)
ah = H.age_hours(f4h, kn)
AB = np.array([0, 1, 2, 4, 7, 10, 14, 20, 30, 60, 1e4]) * 24.0
print("\ncheck on holdout rows (train semantics): age bucket | measured mean log1p(count|warm) | predicted m_ref(age) | val rows measured | val rows after arm A")
f4v = np.asarray(np.load(H.FLAT / "val" / "cand_f4.npy", mmap_mode="r")); av = H.age_hours(f4v, kn)
known = ~np.isnan(av)
dv = np.zeros(len(av)); wv = cv > 0
sel = known & wv
dv[sel] = np.clip(m_ref(av[sel]) - m_of(Ev[sel]), 0, 2.5)
lcv = np.log1p(np.maximum(cv, 0))
for k in range(len(AB) - 1):
    mh = (~np.isnan(ah)) & (ah >= AB[k]) & (ah < AB[k + 1]) & (ch > 0)
    mv = sel & (av >= AB[k]) & (av < AB[k + 1])
    if mh.sum() < 100 or mv.sum() < 100:
        continue
    pred = m_ref(ah[mh]).mean()
    print(f"  {AB[k]/24:5.0f}-{AB[k+1]/24:<6.0f} d | {np.log1p(ch[mh]).mean():.3f} | {pred:.3f} | {lcv[mv].mean():.3f} | {(lcv[mv] + dv[mv]).mean():.3f} | n_val {mv.sum():,}")
print(f"\nval rows: warm {wv.mean():.3f}; warm rows changed (delta > 0.05) {(dv[wv] > 0.05).mean():.3f}; mean delta on warm rows {dv[wv].mean():.3f}; "
      f"on changed rows {dv[dv > 0.05].mean():.3f}; share of ALL rows changed {(dv > 0.05).mean():.3f}")
day = H.day_of(tsv)[fiv]
for d in np.unique(day):
    s = wv & (day == d)
    if s.sum() > 1000:
        print(f"  val day {d}: lag {lv[s].mean()/24:.2f} d, warm rows {s.sum():,}, mean delta {dv[s].mean():.3f}")
np.save(OUT / "delta_val.npy", dv.astype(np.float32))
for kappa, arm in ((1.0, "A"), (0.5, "B")):
    for n in ("it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"):
        x = np.asarray(np.load(H.FEATS / "val" / f"{n}.npy")).astype(np.float64)
        y = np.where(x > 0, np.expm1(np.log1p(np.maximum(x, 0)) + kappa * dv), x)
        np.save(OUT / f"{arm}_{n}.npy", y.astype(np.float32))
print("written:", sorted(p.name for p in OUT.iterdir()))
