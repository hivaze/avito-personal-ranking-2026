"""web_shift probe s10 (label-free): arm-B lag-equalised copies of the 5 item-occurrence inputs for TEST rows (same curve and rule as
s2_build.py: log1p(c') = log1p(c) + 0.5 * clip(m_ref(age) - m(E), 0, 2.5) on warm rows; zero rows and capped-age rows unchanged).
Written for a possible re-prediction of train-only members on test; nothing here is a submission."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "<repo>/feat_new")
import hz_common as H
OUT = Path("<repo>/reports/TRICKS_1115/web_shift_probe/cf_test"); OUT.mkdir(exist_ok=True)
EDG = np.array([0, .5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 30, 40, 60, 90, 1e4]) * 24.0
CAP = 9e4
def load(split):
    f = H.feeds(split); ts = f["timestamp"].to_numpy().astype(np.int64)
    E = H.expo_hours(split, np.full(len(ts), H.CUT_VT, np.int64), ts).astype(np.float64)
    c = np.asarray(np.load(H.FEATS / split / "it_hist_all_luo.npy", mmap_mode="r")).astype(np.float64)
    a = H.age_hours(np.asarray(np.load(H.FLAT / split / "cand_f4.npy", mmap_mode="r")), H.knots())
    return E, c, a
Ev, cv, av = load("val"); Et, ct, at = load("test")
E = np.concatenate([Ev, Et]); c = np.concatenate([cv, ct]); warm = c > 0; lc = np.log1p(np.maximum(c, 0))
b = np.where(E >= CAP, len(EDG), np.where(E < 0, 0, np.digitize(E, EDG))); nb = len(EDG) + 1
m = np.array([lc[warm & (b == k)].mean() if (warm & (b == k)).sum() > 200 else np.nan for k in range(nb)])
m = np.maximum.accumulate(np.nan_to_num(m, nan=0.0))
ctr = np.concatenate([[-24.0], (EDG[:-1] + EDG[1:]) / 2.0]); ctr[-1] = EDG[-2] * 1.2
mtop = max(m[-2], m[-1])
m_of = lambda Eh: np.where(Eh < 0, m[0], np.interp(Eh, ctr[1:], np.minimum(m[1:-1], mtop), right=mtop))
ttr = H.feeds("train")["timestamp"].to_numpy().astype(np.int64)
uq = np.quantile(np.clip((H.CUT_VT - ttr) / 3600.0, 0, None), (np.arange(200) + 0.5) / 200)
def m_ref(ah):
    out = np.zeros(len(ah))
    for s in range(0, len(ah), 200000):
        out[s:s + 200000] = m_of(ah[s:s + 200000, None] + uq[None, :]).mean(1)
    return out
def delta(E_, c_, a_):
    d = np.zeros(len(c_)); sel = (c_ > 0) & ~np.isnan(a_); d[sel] = np.clip(m_ref(a_[sel]) - m_of(E_[sel]), 0, 2.5); return d
dv = delta(Ev, cv, av); ref = np.load(OUT.parent / "cf" / "delta_val.npy")
print("val delta reproduces s2:", float(np.abs(dv - ref).max()))
dt = delta(Et, ct, at); np.save(OUT / "delta_test.npy", dt.astype(np.float32))
for n in ("it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"):
    x = np.asarray(np.load(H.FEATS / "test" / f"{n}.npy")).astype(np.float64)
    y = np.where(x > 0, np.expm1(np.log1p(np.maximum(x, 0)) + 0.5 * dt), x)
    np.save(OUT / f"B_{n}.npy", y.astype(np.float32))
    print(n, "rows", len(x), "changed", float((np.abs(y - x) > 1e-6).mean()), "mean log1p before/after", float(np.log1p(np.maximum(x,0)).mean()), float(np.log1p(y).mean()))
