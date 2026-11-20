"""Part 3: slice decomposition of the val pool deltas (M2-PF7x etc.) and slice shares in val / public / private.
coldpos/warmpos use cold = it_hist_all_luo == 0 per candidate (label-based slice; its test share is PREDICTED from
the val relation P(slice | luo cold-share bin x M bin) applied to the test descriptors)."""
from shiftlib import *  # noqa: F401,F403
sys.path.insert(0, str(ROOT / "eda/errors/segments"))
from segdefs import cut  # noqa: E402

dv = pl.read_parquet(OUT / "desc_val.parquet"); dt = pl.read_parquet(OUT / "desc_test.parquet")
ftab = pl.read_parquet(ROOT / "eda/errors/segments/feed_table.parquet")
pub = dt["timestamp"].to_numpy() < BOUND
R = dict(np.load(OUT / "p2_val_perfeed.npz"))
fv, Mv, ov = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
n = len(Mv); fid = np.repeat(np.arange(n), Mv)
cold = np.load(OUT / "val_cand_cold_luo.npy")
npos = np.bincount(fid, y, n); ncpos = np.bincount(fid, y * cold, n)
posk = np.where(npos == 0, "nopos", np.where(ncpos == npos, "coldpos", np.where(ncpos == 0, "warmpos", "mixedpos")))

CMP = [("M2", "PF7x"), ("G3", "PF7x"), ("M2", "Cd"), ("PF7x", "Cd"), ("F7x", "F7b"), ("F7x", "F7"), ("F7b", "F7")]


def slicers(d, test):
    S = {}
    S["M"] = cut(d["M"], [10, 25, 50, 100, 200], ["M5-10", "M11-25", "M26-50", "M51-100", "M101-200", "M201-300"])
    S["H"] = cut(d["H"], [0, 19, 63, 191, 255], ["H0", "H1-19", "H20-63", "H64-191", "H192-255", "H256"])
    S["user"] = np.where(d["uid0"].to_numpy() > 0, "uid0",
                         cut(d["n_train_feeds"], [0, 1, 4], ["tr0", "tr1", "tr2-4", "tr5+"]))
    S["cold_share(bitmap)"] = cut(d["cold_share"], [0.15, 0.3, 0.45], ["c<15%", "c15-30%", "c30-45%", "c>=45%"])
    S["luo_cold_share"] = cut(d["luo_cold_share"], [0.2, 0.35, 0.5, 0.65], ["l<20%", "l20-35%", "l35-50%", "l50-65%", "l>=65%"])
    tc = d["top_cat"].to_numpy().astype(int)
    TOP = [2, 50, 3, 13, 6, 7, 10, 11]
    S["top_cat"] = np.where(np.isin(tc, TOP), np.char.add("cat", tc.astype(str)), "cat_other")
    hr = d["minute_of_day"].to_numpy() // 60
    S["hour"] = cut(hr, [5, 9, 13, 17], ["h00-05", "h06-09", "h10-13", "h14-17", "h18-23"])
    return S


Sv, St = slicers(dv, False), slicers(dt, True)
Sv["day"] = np.where(msk_day(dv["timestamp"].to_numpy()) < 20720, "Tue", "Thu")
Sv["pos_cold(luo)"] = posk
Sv["has_later(leak)"] = np.where(ftab["leak"].to_numpy(), "has_later", "no_later")

# predicted test share of the label-based posk slices: P(posk | luo cold-share bin x M bin) from val
cellv = np.char.add(Sv["luo_cold_share"].astype(str), Sv["M"].astype(str))
cellt = np.char.add(St["luo_cold_share"].astype(str), St["M"].astype(str))
pred_share = {}
for k in np.unique(posk):
    pk = {c: (posk[cellv == c] == k).mean() for c in np.unique(cellv)}
    pt = np.array([pk.get(c, (posk == k).mean()) for c in cellt])
    pred_share[k] = (pt[pub].mean(), pt[~pub].mean())

lines = []


def P(s=""):
    print(s); lines.append(s)


rows = []
for var in ["M", "H", "user", "cold_share(bitmap)", "luo_cold_share", "top_cat", "hour", "day", "pos_cold(luo)",
            "has_later(leak)"]:
    P(f"\n=== {var} ===")
    P(f"{'slice':12s} {'n':>5s} {'val%':>6s} {'pub%':>6s} {'priv%':>6s} | " + " | ".join(f"{a + '-' + b:>17s}" for a, b in CMP[:4]))
    for s in sorted(np.unique(Sv[var])):
        m = Sv[var] == s
        sv = m.mean()
        if var in St:
            sp, sq = (St[var][pub] == s).mean(), (St[var][~pub] == s).mean()
        elif var == "pos_cold(luo)":
            sp, sq = pred_share[s]
        else:
            sp = sq = np.nan
        cells = []
        for a, b in CMP:
            d = (R[a] - R[b])[m]; mm, se = mse(d)
            rows.append(dict(var=var, slice=s, n=int(m.sum()), val_share=sv, pub_share=sp, priv_share=sq, cmp=f"{a}-{b}",
                             d=mm, se=se, contrib=mm * sv))
            cells.append(f"{mm:+.4f}±{se:.4f}")
        P(f"{s:12s} {int(m.sum()):5d} {100 * sv:6.1f} {100 * sp:6.1f} {100 * sq:6.1f} | " + " | ".join(cells[:4]))
df = pl.DataFrame(rows)
df.write_csv(OUT / "p3_slices.csv")

P("\n=== F7x-F7b / F7x-F7 / F7b-F7 by pos_cold(luo) and day (member-family pools) ===")
for var in ["pos_cold(luo)", "day"]:
    for s in sorted(np.unique(Sv[var])):
        m = Sv[var] == s
        P(f"{var}={s:9s} n={int(m.sum()):5d} " + " | ".join(f"{a}-{b} {fmt(*mse((R[a] - R[b])[m]))}" for a, b in CMP[4:]))

# post-stratified (1-D) public / private estimates of the val deltas, per slicing variable
P("\n=== post-stratified val deltas (val slice deltas x test slice shares), per slicing variable ===")
P(f"{'variable':20s} " + " | ".join(f"{a + '-' + b:>34s}" for a, b in CMP[:4]))
ps_rows = []
for var in ["M", "H", "user", "cold_share(bitmap)", "luo_cold_share", "top_cat", "hour", "pos_cold(luo)"]:
    out = []
    for a, b in CMP:
        d = R[a] - R[b]; res = []
        for tgt in (0, 1):
            w = np.zeros(n)
            for s in np.unique(Sv[var]):
                m = Sv[var] == s
                if var in St:
                    ts_ = (St[var][pub] == s).mean() if tgt == 0 else (St[var][~pub] == s).mean()
                else:
                    ts_ = pred_share[s][tgt]
                w[m] = ts_ / m.mean()
            res.append(mse(d, w))
        ps_rows.append(dict(var=var, cmp=f"{a}-{b}", pub=res[0][0], pub_se=res[0][1], priv=res[1][0], priv_se=res[1][1]))
        out.append(f"pub {res[0][0]:+.5f} priv {res[1][0]:+.5f}±{res[1][1]:.5f}")
    P(f"{var:20s} " + " | ".join(out[:4]))
P(f"{'(unweighted val)':20s} " + " | ".join(f"val {(R[a] - R[b]).mean():+.5f}{'':22s}" for a, b in CMP[:4]))
pl.DataFrame(ps_rows).write_csv(OUT / "p3_poststrat.csv")
open(OUT / "p3_slices.log.txt", "w").write("\n".join(lines))
