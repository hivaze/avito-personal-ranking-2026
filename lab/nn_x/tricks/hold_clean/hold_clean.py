"""HOLD-CLEAN label-free format readout (TRICKS_SWEEP_1113 §2 residual Step 2). CPU only, read-only on project data.
Rule: nn_x/tricks/hold_clean/PREDECLARED.txt. Outputs: out/stats.json, out/members.json, RESULT.txt (written by hand from log)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[v] = "12"
import sys, json, time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT, FEATS, PREDS = ROOT / "data/flat", ROOT / "data/feats", ROOT / "preds"
OUT = ROOT / "nn_x/tricks/hold_clean/out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402

T0 = time.time()
LOG = open(OUT / "run.log", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time() - T0:7.1f}s] {s}", flush=True)
    LOG.write(s + "\n"); LOG.flush()


CUT_EL = 1788901200      # 2026-09-09 00:00 MSK = 2026-09-08 21:00 UTC
PUBLIC_TS = 1790497920   # test: ts < this = public
MSK = 10800

# ---------- age proxy exactly as nn_x/audit/ageprod/ap01_lifts.py ----------
KNOTS = np.load(FEATS / "life_f4_age_knots.npy")
K = 64
CAP = 63 / 64 - 1e-6
EDGES = [0, 3, 12, 24, 72, 336, 2160, 8760, 1e9]
AGE_LABELS = ["<3h", "3-12h", "12-24h", "1-3d", "3-14d", "2w-3m", "3m-1y", ">1y"]


def age_hours(x):
    x = np.asarray(x, dtype=np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    a = KNOTS[k] * (1 - w) + KNOTS[k + 1] * w
    return np.where(x < CAP, a, np.nan)


def age_bin(f4):
    return np.digitize(np.nan_to_num(age_hours(f4), nan=1e8), EDGES) - 1


# ---------- holdout index (nn_x/r6/train.py rule) ----------
ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
Mtr = ftr["M"].to_numpy().astype(np.int64)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(len(Mtr))).all()
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
p[Mtr < 50] = 0.0
cand = np.where((p > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
hm = np.load(PREDS / "holdout/holdout_meta.npy")
hy = np.load(PREDS / "holdout/holdout_y.npy")
assert len(hold) == 30000 and (Mtr[hold] == hm[0]).all(), "holdout M mismatch"
HM, HO = hm[0].astype(np.int64), hm[1].astype(np.int64)
toff = ftr["cand_off"].to_numpy().astype(np.int64)
hrows = np.concatenate([np.arange(o, o + m) for o, m in zip(toff[hold], Mtr[hold])])
ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
assert np.array_equal(np.asarray(ytr[hrows]).astype(np.float64), hy.astype(np.float64)), "holdout_y mismatch"
log(f"holdout: {len(hold)} feeds, {len(hrows)} rows; candidate pool {len(cand)}; M and labels match holdout_meta/y")


def split_data(split, feeds_sel=None):
    """Per-feed arrays for the selected feeds of a split (feed order preserved)."""
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    n_all = f.height
    sel = np.arange(n_all) if feeds_sel is None else feeds_sel
    M = f["M"].to_numpy().astype(np.int64)[sel]
    H = f["H"].to_numpy().astype(np.int64)[sel]
    off = f["cand_off"].to_numpy().astype(np.int64)[sel]
    hoff = f["hist_off"].to_numpy().astype(np.int64)[sel]
    ts = f["timestamp"].to_numpy().astype(np.int64)[sel]
    uid = f["user_id"].to_numpy().astype(np.int64)[sel]
    n = len(sel)
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off, M)]) if feeds_sel is not None else np.arange(M.sum())
    hrws = np.concatenate([np.arange(o, o + h) for o, h in zip(hoff, H)]) if feeds_sel is not None else np.arange(H.sum())
    fid = np.repeat(np.arange(n), M)
    hfid = np.repeat(np.arange(n), H)
    ld = lambda base, name, r: np.asarray(np.load(base / split / f"{name}.npy", mmap_mode="r")[r])
    # sanity: flat cand_fidx consistent with our row mapping (for full splits)
    cf = ld(FLAT, "cand_fidx", rows).astype(np.int64)
    assert np.array_equal(cf, (sel[fid] if feeds_sel is not None else fid)), f"{split}: cand_fidx mismatch"
    d = dict(n=n, M=M, H=H, ts=ts, uid=uid, fid=fid)
    # S1/S2/S3
    d["logM"] = np.log(M.astype(np.float64))
    d["log1pH"] = np.log1p(H.astype(np.float64))
    d["uid0"] = (uid == 0).astype(np.float64)
    # S4 candidate-category entropy per feed
    cat = ld(FLAT, "cand_category_id", rows).astype(np.int64)
    key = fid * 256 + (cat + 128)
    u, c = np.unique(key, return_counts=True)
    uf = u // 256
    pp = c / M[uf]
    d["cat_ent"] = np.bincount(uf, weights=-pp * np.log(pp), minlength=n)
    d["cat_distinct_frac"] = np.bincount(uf, minlength=n) / M
    # row-level numerators (per-feed sums)
    fs = [ld(FLAT, f"cand_f{i}", rows) for i in range(1, 6)]
    any_mz = np.zeros(len(rows), bool)
    for i, x in enumerate(fs, 1):
        nan_i, zero_i = np.isnan(x), (x == 0)
        d[f"f{i}_nan"] = np.bincount(fid, weights=nan_i, minlength=n)
        d[f"f{i}_zero"] = np.bincount(fid, weights=zero_i, minlength=n)
        vals, cnts = np.unique(x[~np.isnan(x)], return_counts=True)
        d[f"f{i}_modal"] = (float(vals[np.argmax(cnts)]), float(cnts.max() / len(x)), int(len(vals)))
        any_mz |= nan_i | zero_i
    d["any_f_mz"] = np.bincount(fid, weights=any_mz, minlength=n)
    # S6 seen in the same feed's history
    ci = ld(FLAT, "cand_item_id", rows).astype(np.int64)
    hi = ld(FLAT, "hist_item_id", hrws).astype(np.int64)
    d["cand_id_invalid"] = int((ci <= 0).sum()); d["hist_id_invalid"] = int((hi <= 0).sum())
    hk = np.unique(hfid[hi > 0] * (1 << 31) + hi[hi > 0])
    ck = fid * (1 << 31) + ci
    seen = np.isin(ck, hk) & (ci > 0)
    d["seen"] = np.bincount(fid, weights=seen, minlength=n)
    # S7 listing age
    f4 = ld(FEATS, "f4", rows)
    ah = age_hours(f4)
    ab = age_bin(f4)
    d["age_lt24"] = np.bincount(fid, weights=(ab <= 2), minlength=n)
    d["age_bins"] = np.bincount(ab, minlength=8)
    d["age_median_h"] = float(np.nanmedian(ah))
    d["age_nan_share"] = float(np.isnan(ah).mean())
    # S8-S10 pre2 / popularity
    pre2 = ld(FEATS, "pre2_fe", rows)
    d["pre2_nan"] = np.bincount(fid, weights=np.isnan(pre2), minlength=n)
    luo = ld(FEATS, "it_hist_all_luo", rows).astype(np.float64)
    d["luo_nan_share"] = float(np.isnan(luo).mean())
    d["luo_zero"] = np.bincount(fid, weights=(luo == 0), minlength=n)
    d["luo_log1p"] = np.bincount(fid, weights=np.log1p(np.nan_to_num(luo, nan=0.0)), minlength=n)
    return d


def sub(d, m):
    """restrict per-feed arrays to feeds m (bool)."""
    o = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == d["n"] and k != "fid":
            o[k] = v[m]
    o["n"] = int(m.sum())
    return o


def feed_mean(x):
    x = np.asarray(x, np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def ratio(a, m):
    a = np.asarray(a, np.float64); m = np.asarray(m, np.float64)
    r = a.sum() / m.sum()
    n = len(a)
    se = np.sqrt(((a - r * m) ** 2).sum() * n / (n - 1)) / m.sum()
    return float(r), float(se)


STATS = [("S1_mean_logM", "feed", "logM"), ("S2_mean_log1pH", "feed", "log1pH"), ("S3_uid0_share", "feed", "uid0"),
         ("S4_cat_entropy", "feed", "cat_ent"), ("S5_any_f1to5_nan_or_zero", "row", "any_f_mz"),
         ("S6_seen_in_history", "row", "seen"), ("S7_age_lt24h", "row", "age_lt24"), ("S8_pre2_nan", "row", "pre2_nan"),
         ("S9_luo_zero", "row", "luo_zero"), ("S10_mean_log1p_luo", "row", "luo_log1p")]


def stat_of(d, kind, key):
    return feed_mean(d[key]) if kind == "feed" else ratio(d[key], d["M"])


def info_of(d):
    M, H = d["M"], d["H"]
    q = [10, 25, 50, 75, 90]
    o = dict(n_feeds=int(d["n"]), n_rows=int(M.sum()), mean_M=float(M.mean()),
             M_q=dict(zip(map(str, q), np.percentile(M, q).tolist())), share_M_lt50=float((M < 50).mean()),
             mean_H=float(H.mean()), H_q=dict(zip(map(str, q), np.percentile(H, q).tolist())),
             share_H256=float((H >= 256).mean()), share_H_le10=float((H <= 10).mean()),
             cat_distinct_frac=float(d["cat_distinct_frac"].mean()))
    for i in range(1, 6):
        o[f"f{i}_nan_share"] = float(d[f"f{i}_nan"].sum() / M.sum())
        o[f"f{i}_zero_share"] = float(d[f"f{i}_zero"].sum() / M.sum())
    return o


# ---------- build parts ----------
log("loading holdout (train) feeds ...")
HD = split_data("train", hold)
assert np.array_equal(HD["M"], HM)
log("loading val ...")
VD = split_data("val")
log("loading test ...")
TD = split_data("test")

E = HD["ts"] < CUT_EL
L = ~E
day_h = (HD["ts"] + MSK) // 86400
log(f"E feeds {E.sum()} (MSK days {sorted(set(day_h[E].tolist()))}), L feeds {L.sum()} (MSK days {sorted(set(day_h[L].tolist()))}); "
    f"L feeds stamped Sep 20 MSK: {(day_h == 20716).sum()}")
log(f"E ts range UTC {time.strftime('%Y-%m-%d %H:%M', time.gmtime(HD['ts'][E].min()))} .. {time.strftime('%Y-%m-%d %H:%M', time.gmtime(HD['ts'][E].max()))}; "
    f"L {time.strftime('%Y-%m-%d %H:%M', time.gmtime(HD['ts'][L].min()))} .. {time.strftime('%Y-%m-%d %H:%M', time.gmtime(HD['ts'][L].max()))}")
pub = TD["ts"] < PUBLIC_TS
parts = {"E": sub(HD, E), "L": sub(HD, L), "hold_all": sub(HD, np.ones(HD["n"], bool)),
         "val50": sub(VD, VD["M"] >= 50), "test50": sub(TD, TD["M"] >= 50),
         "pub50": sub(TD, pub & (TD["M"] >= 50)), "priv50": sub(TD, ~pub & (TD["M"] >= 50)),
         "val_all": sub(VD, np.ones(VD["n"], bool)), "pub_all": sub(TD, pub), "priv_all": sub(TD, ~pub)}
# age bins / modal values per part need row-level arrays: recompute from per-feed sums where possible
res = {"parts": {}, "stats": {}}
for pn, d in parts.items():
    res["parts"][pn] = info_of(d)
    for sname, kind, key in STATS:
        res["stats"].setdefault(sname, {})[pn] = stat_of(d, kind, key)
# split-level info (not per-part): age bins, modal f values, id sentinels
res["split_info"] = {}
for nm, D in (("holdout", HD), ("val", VD), ("test", TD)):
    res["split_info"][nm] = dict(age_bins=dict(zip(AGE_LABELS, (D["age_bins"] / D["age_bins"].sum()).round(5).tolist())),
                                 age_median_h=D["age_median_h"], age_nan_share=D["age_nan_share"],
                                 luo_nan_share=D["luo_nan_share"], cand_id_invalid=D["cand_id_invalid"],
                                 hist_id_invalid=D["hist_id_invalid"],
                                 f_modal={f"f{i}": D[f"f{i}_modal"] for i in range(1, 6)})

# age bins E vs L (row-level, holdout)
f4h = np.asarray(np.load(FEATS / "train/f4.npy", mmap_mode="r")[hrows])
abh = age_bin(f4h)
rowE = np.repeat(E, HM)
res["age_bins_E"] = dict(zip(AGE_LABELS, (np.bincount(abh[rowE], minlength=8) / rowE.sum()).round(5).tolist()))
res["age_bins_L"] = dict(zip(AGE_LABELS, (np.bincount(abh[~rowE], minlength=8) / (~rowE).sum()).round(5).tolist()))
res["age_median_E_h"] = float(np.nanmedian(age_hours(f4h[rowE])))
res["age_median_L_h"] = float(np.nanmedian(age_hours(f4h[~rowE])))
for nm, D, msk in (("val50", VD, VD["M"] >= 50), ("test50", TD, TD["M"] >= 50)):
    pass  # split-level age bins reported above (all-M); M>=50 bins below
for nm, split, D, msk in (("val50", "val", VD, VD["M"] >= 50), ("test50", "test", TD, TD["M"] >= 50)):
    f4s = np.asarray(np.load(FEATS / f"{split}/f4.npy", mmap_mode="r"))
    rm = np.repeat(msk, D["M"])
    res[f"age_bins_{nm}"] = dict(zip(AGE_LABELS, (np.bincount(age_bin(f4s[rm]), minlength=8) / rm.sum()).round(5).tolist()))
    res[f"age_median_{nm}_h"] = float(np.nanmedian(age_hours(f4s[rm])))

# p_random E vs L (info)
ph = p[hold]
q = [1, 5, 10, 25, 50, 75]
res["p_random"] = {pn: dict(mean=float(ph[m].mean()), q=dict(zip(map(str, q), np.percentile(ph[m], q).round(5).tolist())),
                            share_gt099=float((ph[m] > 0.99).mean()), share_gt0999=float((ph[m] > 0.999).mean()))
                   for pn, m in (("E", E), ("L", L))}

# ---------- pre-declared count ----------
cnt, cnt_noguard, cnt_pub, cnt_priv = 0, 0, 0, 0
rows_out = []
for sname, kind, key in STATS:
    s = res["stats"][sname]
    e, se_e = s["E"]; l, se_l = s["L"]
    v = s["val50"][0]; t = s["test50"][0]; pu = s["pub50"][0]; pv = s["priv50"][0]
    se_el = float(np.hypot(se_e, se_l))
    c1 = abs(e - v) > abs(l - v); c2 = abs(e - t) > abs(l - t); g = abs(e - l) > 2 * se_el
    out = bool(c1 and c2 and g)
    cnt += out; cnt_noguard += bool(c1 and c2)
    cnt_pub += bool(abs(e - v) > abs(l - v) and abs(e - pu) > abs(l - pu) and g)
    cnt_priv += bool(abs(e - v) > abs(l - v) and abs(e - pv) > abs(l - pv) and g)
    rows_out.append(dict(stat=sname, E=e, E_se=se_e, L=l, L_se=se_l, E_minus_L=e - l, E_minus_L_se=se_el,
                         val50=v, test50=t, pub50=pu, priv50=pv, val_all=s["val_all"][0], pub_all=s["pub_all"][0],
                         priv_all=s["priv_all"][0], absE_val=abs(e - v), absL_val=abs(l - v), absE_test=abs(e - t),
                         absL_test=abs(l - t), guard_2se=bool(g), E_out=out))
    log(f"{sname:28s} E {e:.5f}±{se_e:.5f}  L {l:.5f}±{se_l:.5f}  E-L {e - l:+.5f}±{se_el:.5f} | val50 {v:.5f} test50 {t:.5f} "
        f"pub50 {pu:.5f} priv50 {pv:.5f} | |E-val| {abs(e - v):.5f} vs |L-val| {abs(l - v):.5f}; |E-test| {abs(e - t):.5f} vs "
        f"|L-test| {abs(l - t):.5f} | E_out {out}")
res["count"] = dict(primary=int(cnt), no_guard=int(cnt_noguard), vs_public50=int(cnt_pub), vs_private50=int(cnt_priv),
                    n_stats=len(STATS), verdict="E is out-of-format" if cnt >= 3 else "E is not out-of-format by the pre-declared rule")
res["table"] = rows_out
log(f"COUNT primary {cnt}/10 (no guard {cnt_noguard}, vs pub50 {cnt_pub}, vs priv50 {cnt_priv}) -> {res['count']['verdict']}")

# ---------- G3 + members on E vs L ----------
names = [x.strip() for x in open(ROOT / "nn_x/pubdiag/build/g3_members.txt").read().replace("\n", ",").split(",") if x.strip()]
fidxh = np.repeat(np.arange(len(HM)), HM)


def feed_z(x):
    x = np.asarray(x, dtype=np.float64)
    c = np.bincount(fidxh, minlength=len(HM)).astype(np.float64)
    mu = np.bincount(fidxh, weights=x, minlength=len(HM)) / c
    v = np.bincount(fidxh, weights=(x - mu[fidxh]) ** 2, minlength=len(HM)) / c
    sd = np.sqrt(v); sd[sd < 1e-9] = 1.0
    return (x - mu[fidxh]) / sd[fidxh]


acc = None
mem = {}
for nm in names:
    s = np.load(PREDS / "holdout" / f"nn_{nm}_holdout.npy")
    nd = ndcg_per_feed(s, hy, HO, HM)
    mem[nm] = nd
    z = feed_z(s)
    acc = z if acc is None else acc + z
g3 = acc / len(names)
nd3 = ndcg_per_feed(g3, hy, HO, HM)
log(f"G3 holdout (20-member z-mean): {nd3.mean():.5f} (expected 0.32379)")
assert abs(nd3.mean() - 0.32379) < 6e-6, nd3.mean()


def two(x):
    a, sa = feed_mean(x[E]); b, sb = feed_mean(x[L])
    return dict(E=a, E_se=sa, L=b, L_se=sb, E_minus_L=a - b, E_minus_L_se=float(np.hypot(sa, sb)), all=float(x.mean()))


g = dict(G3=two(nd3))
npos = np.bincount(fidxh, weights=hy, minlength=len(HM))
g["post_declaration_info_npos_per_feed"] = dict(E=float(npos[E].mean()), L=float(npos[L].mean()),
                                                 share_nopos_E=float((npos[E] == 0).mean()), share_nopos_L=float((npos[L] == 0).mean()))
log(f"G3 E {g['G3']['E']:.5f}±{g['G3']['E_se']:.5f}  L {g['G3']['L']:.5f}±{g['G3']['L_se']:.5f}  E-L {g['G3']['E_minus_L']:+.5f}±{g['G3']['E_minus_L_se']:.5f}")
mE = np.array([mem[n][E].mean() for n in names]); mL = np.array([mem[n][L].mean() for n in names])
mA = np.array([mem[n].mean() for n in names])
gapE = np.array([feed_mean(mem[n][E] - nd3[E]) for n in names]); gapL = np.array([feed_mean(mem[n][L] - nd3[L]) for n in names])
fam = np.array(["F10" if n.startswith("v23") else ("F7x70" if "p70k" in n else "F7x") for n in names])
g["members"] = {n: dict(E=float(mE[i]), L=float(mL[i]), all=float(mA[i]), E_minus_L=float(mE[i] - mL[i]),
                        gap_to_G3_E=gapE[i].tolist(), gap_to_G3_L=gapL[i].tolist()) for i, n in enumerate(names)}
dE, dL = mE - mE.mean(), mL - mL.mean()
g["member_spread"] = dict(sd_E=float(mE.std(ddof=1)), sd_L=float(mL.std(ddof=1)), sd_all=float(mA.std(ddof=1)),
                          sd_E_minus_L=float((mE - mL).std(ddof=1)), range_E=float(np.ptp(mE)), range_L=float(np.ptp(mL)),
                          pearson_dev_E_vs_L=float(np.corrcoef(dE, dL)[0, 1]),
                          mean_member_E=float(mE.mean()), mean_member_L=float(mL.mean()),
                          mean_gap_to_G3_E=float(gapE[:, 0].mean()), mean_gap_to_G3_L=float(gapL[:, 0].mean()))
# within-family (family mean removed) seed-level correlation of E vs L
wf_E, wf_L = [], []
g["families"] = {}
for fm in ("F10", "F7x", "F7x70"):
    s = fam == fm
    g["families"][fm] = dict(n=int(s.sum()), E=float(mE[s].mean()), L=float(mL[s].mean()), E_minus_L=float((mE[s] - mL[s]).mean()),
                             sd_E=float(mE[s].std(ddof=1)), sd_L=float(mL[s].std(ddof=1)))
    wf_E += list(mE[s] - mE[s].mean()); wf_L += list(mL[s] - mL[s].mean())
g["member_spread"]["pearson_within_family_dev_E_vs_L"] = float(np.corrcoef(wf_E, wf_L)[0, 1])
log("member spread:", json.dumps(g["member_spread"]))
log("families:", json.dumps(g["families"]))
for n in names:
    m = g["members"][n]
    log(f"  {n:26s} E {m['E']:.5f} L {m['L']:.5f} all {m['all']:.5f} E-L {m['E_minus_L']:+.5f} gapG3 E {m['gap_to_G3_E'][0]:+.5f}±{m['gap_to_G3_E'][1]:.5f} L {m['gap_to_G3_L'][0]:+.5f}±{m['gap_to_G3_L'][1]:.5f}")
res["ndcg"] = g
json.dump(res, open(OUT / "stats.json", "w"), indent=1, default=float)
log("parts info:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('n_feeds', 'mean_M', 'share_M_lt50', 'mean_H', 'share_H256', 'share_H_le10')} for k, v in res["parts"].items()}))
log("done")
