"""web_shift probe s7 (label-free): size of the lag deflation delta (s2 definition) on TEST rows by day, next to val."""
import sys
import numpy as np
sys.path.insert(0, "<repo>/feat_new")
import hz_common as H
EDG = np.array([0, .5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 30, 40, 60, 90, 1e4]) * 24.0
CAP = 9e4
def load(split):
    f = H.feeds(split); ts = f["timestamp"].to_numpy().astype(np.int64)
    E = H.expo_hours(split, np.full(len(ts), H.CUT_VT, np.int64), ts).astype(np.float64)
    fi = np.asarray(np.load(H.FLAT / split / "cand_fidx.npy", mmap_mode="r"))
    c = np.asarray(np.load(H.FEATS / split / "it_hist_all_luo.npy", mmap_mode="r")).astype(np.float64)
    a = H.age_hours(np.asarray(np.load(H.FLAT / split / "cand_f4.npy", mmap_mode="r")), H.knots())
    return E, c, a, fi, ts
D = {s: load(s) for s in ("val", "test")}
E = np.concatenate([D["val"][0], D["test"][0]]); c = np.concatenate([D["val"][1], D["test"][1]])
warm = c > 0; lc = np.log1p(np.maximum(c, 0))
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
print(f"{'split / day':24s} {'lag d':>6s} {'rows':>10s} {'cold':>6s} {'warm changed':>12s} {'all changed':>11s} {'mean delta|chg':>14s} {'delta>0.8 (all)':>15s} {'E<0 & warm (all)':>16s}")
for s in ("val", "test"):
    E_, c_, a_, fi, ts = D[s]
    w = c_ > 0; sel = w & ~np.isnan(a_)
    d = np.zeros(len(c_)); d[sel] = np.clip(m_ref(a_[sel]) - m_of(E_[sel]), 0, 2.5)
    day = H.day_of(ts)[fi]; pub = (ts < 1790497920)[fi]
    groups = [(f"{s} day {dd}", day == dd) for dd in np.unique(day) if (day == dd).sum() > 50000]
    if s == "test":
        groups += [("test public", pub), ("test private", ~pub)]
    else:
        groups += [("val all", np.ones(len(c_), bool))]
    for name, g in groups:
        lag = ((ts[fi][g] - H.CUT_VT) / 86400).mean()
        ch = d > 0.05
        print(f"{name:24s} {lag:6.2f} {g.sum():10,d} {1 - w[g].mean():6.3f} {ch[g & w].mean():12.3f} {ch[g].mean():11.3f} {d[g & ch].mean():14.3f} {(d[g] > 0.8).mean():15.3f} {((E_ < 0) & w)[g].mean():16.3f}")
