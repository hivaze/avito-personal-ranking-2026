"""hzcv CPU screen: train-fit LightGBM gate. Copy of the review harness
scratchpad/review/leakage/lgb_causal.py (unchanged: train_sample_feeds.npy = 50K random-like non-holdout train feeds,
PARAMS, 350 rounds, 2 seeds, clean BASE without it_cand_cnt_* / hour), threads 16.
Arms (pre-declared):
  R0  = BASE + LUO + it_hist_all_luo_{rk,zz}                     (lgb_causal "luo(as built)", val 0.37226 there)
  R1  = R0 with TRAIN-row seq_cv_{invd20,pop,norm} -> seq_cvlag_* (same column slots) + hzcv_expo_slog,
        hzcv_censored, hzcv_stale_d (appended)                    <- the verdict arm
Diagnostics only (not used for the verdict):
  D1  = R1 without the 3 offsets (lagged co-vis only)
  D2  = R0 + the 3 offsets built at L = 0 (src_expo_slog, src_censored, stale_d at the as-built cutoff)
  R2  = R1 but train-sample rows use the capped lag of hz_lagcap.py (dense part >= 3 days), if its npz exists
Evaluated on val (paired per feed; Tue/Thu, M<50/M>=50 splits), the 30K train holdout (all L = 0) incl. its
Sep 17-19 slice, and stale-val (val with seq_cv from review/verify_SHIFT-3/out/cv_sep18_*, offsets with
c = 2026-09-18 00:00 MSK) as a diagnostic.
Usage: CUDA_VISIBLE_DEVICES="" python feat_new/hz_gate.py
"""
import sys, time, json
from pathlib import Path
import numpy as np, numba as nb, lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hz_common as hz  # noqa: E402
SCRR = "/tmp/scratchpad"
sys.path.insert(0, SCRR + "/review/leakage")
from common_rows import load, rows_of, holdout_feeds, feeds, FLAT, SCR  # noqa: E402
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

OUT = Path(SCRR) / "featrev/hzcv"
STALE = Path(SCRR) / "review/verify_SHIFT-3/out"
T0 = time.time()
TH = 16
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1000], learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              lambdarank_truncation_level=1000, verbose=-1, num_threads=TH)
BASE = ["f1", "f2", "f3", "f4", "f5", "f1_rk", "f2_rk", "f3_rk", "f4_rk", "f5_rk", "f1_z", "f2_z", "f3_z", "f4_z", "f5_z",
        "cos_u", "cos_u_rk", "cos_u_z", "maxsim32", "maxsim32_rk", "maxsim32_z", "attn32", "cos_h10", "cos_hall",
        "meansim32", "cos_het1", "cat_cnt_all", "cat_cnt_all_rk", "cat_cnt_et1", "cat_share_all", "cat_min_hpos",
        "feed_cat_share", "centrality", "H", "M", "item_norm", "dot_u", "category_id", "ctx_hs_k128", "ctx_hn_soft05",
        "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_top5h", "seq_cv_invd20",
        "seq_cv_pop", "seq_cv_norm", "u_het1_share"]
LUO = ["it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]
CV = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm"]
LAG = {"seq_cv_invd20": "seq_cvlag_invd20", "seq_cv_pop": "seq_cvlag_pop", "seq_cv_norm": "seq_cvlag_norm"}
OFFS = ["hzcv_expo_slog", "hzcv_censored", "hzcv_stale_d"]
OFF0 = ["off0_expo_slog", "off0_censored", "off0_stale_d"]
CUT_STALE = hz.CUT_VT - 2 * hz.DAY  # 2026-09-18 00:00 MSK


@nb.njit(parallel=True)
def rank_z(x, off, M, out):
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        v = x[o:o + m]
        mu = v.mean(); sd = v.std()
        for i in range(m):
            lo = 0; eq = 0
            for j in range(m):
                if v[j] < v[i]:
                    lo += 1
                elif v[j] == v[i]:
                    eq += 1
            out[o + i, 0] = (lo + 0.5 * (eq - 1)) / max(m - 1, 1)
            out[o + i, 1] = (v[i] - mu) / (sd + 1e-6)


def newf(sp, name, rows):
    return np.asarray(np.load(hz.FEATS_NEW / sp / f"{name}.npy", mmap_mode="r")[rows], np.float32)


def block(sp, fids):
    rows, off, M = rows_of(sp, fids)
    X = {n: load(sp, n, rows).astype(np.float32) for n in BASE + LUO}
    o = np.zeros((len(rows), 2), np.float64)
    rank_z(X["it_hist_all_luo"].astype(np.float64), off, M, o)
    X["it_hist_all_luo_rk"] = o[:, 0].astype(np.float32); X["it_hist_all_luo_zz"] = o[:, 1].astype(np.float32)
    for n in list(LAG.values()) + OFFS:
        X[n] = newf(sp, n, rows)
    # offsets at the as-built cutoff (L = 0) for D2
    ts, c0, _ = hz.cutoff_per_feed(sp, lag=False)
    fr = np.repeat(np.asarray(fids), M)
    X["off0_expo_slog"] = newf(sp, "src_expo_slog", rows)
    X["off0_censored"] = newf(sp, "src_censored", rows)
    X["off0_stale_d"] = hz.stale_d(ts, c0)[fr]
    y = np.asarray(np.load(FLAT / sp / "cand_target.npy", mmap_mode="r")[rows], np.float32)
    return X, y, off, M, rows, fr


def stale_val(Xv, rows_v, fr_v):
    """val copy with the train source cut at Sep 18 (review SHIFT-3 files) and offsets with c = Sep 18 00:00 MSK."""
    S = dict(Xv)
    for n in CV:
        x = np.asarray(np.load(STALE / f"cv_sep18_{n}.npy", mmap_mode="r")[rows_v], np.float32)
        S[n] = x; S[LAG[n]] = x
    ts, _, _ = hz.cutoff_per_feed("val")
    cut = np.full(len(ts), CUT_STALE, np.int64)
    E = hz.expo_hours("val", cut, ts)[rows_v]
    S["hzcv_expo_slog"] = S["off0_expo_slog"] = hz.slog_days(E)
    S["hzcv_censored"] = S["off0_censored"] = (E <= 0).astype(np.float32)
    S["hzcv_stale_d"] = S["off0_stale_d"] = hz.stale_d(ts, cut)[fr_v]
    return S


def main():
    hold = holdout_feeds()[0]
    trs = np.load(SCR / "train_sample_feeds.npy")
    D = {"tr": block("train", trs), "ho": block("train", hold), "va": block("val", np.arange(len(feeds("val"))))}
    D["st"] = (stale_val(D["va"][0], D["va"][4], D["va"][5]),) + D["va"][1:]
    print("loaded", {k: len(v[1]) for k, v in D.items()}, f"{time.time()-T0:.0f}s", flush=True)
    # sanity: val lagged co-vis == as-built co-vis; holdout lag == 0
    for a, b in LAG.items():
        assert np.array_equal(D["va"][0][a], D["va"][0][b])
    assert np.all(newf("train", "hz_lag", D["ho"][4]) == 0)
    R0 = BASE + LUO + ["it_hist_all_luo_rk", "it_hist_all_luo_zz"]
    BL = [LAG.get(n, n) for n in BASE]
    ARMS = {"R0": R0, "R1": BL + LUO + ["it_hist_all_luo_rk", "it_hist_all_luo_zz"] + OFFS,
            "D1": BL + LUO + ["it_hist_all_luo_rk", "it_hist_all_luo_zz"],
            "D2": R0 + OFF0}
    capf = OUT / "lagcap_trs.npz"
    if capf.exists():
        c = np.load(capf)
        assert np.array_equal(c["rows"], D["tr"][4])
        Xr2 = dict(D["tr"][0])
        for n in list(LAG.values()) + OFFS:
            Xr2[n] = c[n].astype(np.float32)
        ARMS["R2"] = ARMS["R1"]
        print("R2: capped-lag rows changed share", float(c["changed"].mean()), flush=True)
    fv = feeds("val")
    dv = hz.day_of(fv["timestamp"].to_numpy())
    Mv = fv["M"].to_numpy()
    fh = feeds("train")
    dh = hz.day_of(fh["timestamp"].to_numpy())[hold]
    splits = {"va": {"tue": dv <= dv.min() + 1, "thu": dv >= dv.min() + 2, "M<50": Mv < 50, "M>=50": Mv >= 50},
              "ho": {"sep17-19": dh >= hz.train_days()[-3]}, "st": {"tue": dv <= dv.min() + 1, "thu": dv >= dv.min() + 2}}
    res = {}
    for arm, names in ARMS.items():
        Xsrc = Xr2 if arm == "R2" else D["tr"][0]
        Xtr = np.stack([Xsrc[n] for n in names], 1)
        acc = {k: [] for k in ("va", "ho", "st")}
        for s in range(2):
            ds = lgb.Dataset(Xtr, D["tr"][1], group=D["tr"][3], feature_name=names, categorical_feature=["category_id"])
            b = lgb.train(dict(PARAMS, seed=s), ds, 350)
            for key in acc:
                X, y, off, M = D[key][:4]
                acc[key].append(ndcg_per_feed(b.predict(np.stack([X[n] for n in names], 1), num_threads=TH), y, off, M))
            if s == 0:
                imp = dict(zip(names, b.feature_importance("gain")))
                tot = sum(imp.values())
                print(f"  {arm} gain share:", {n: round(imp[n] / tot, 4) for n in names
                                               if n in OFFS + OFF0 + list(LAG.values()) + CV}, flush=True)
        res[arm] = {k: np.mean(v, 0) for k, v in acc.items()}
        print(f"{arm:3s} val {res[arm]['va'].mean():.5f} holdout {res[arm]['ho'].mean():.5f} "
              f"stale-val {res[arm]['st'].mean():.5f}  {time.time()-T0:.0f}s", flush=True)
    summ = {}
    for arm in ARMS:
        if arm == "R0":
            continue
        summ[arm] = {}
        for key, lab in (("va", "VAL"), ("ho", "HOLDOUT"), ("st", "STALE-VAL")):
            d = res[arm][key] - res["R0"][key]
            se = d.std() / np.sqrt(len(d))
            line = f"{arm} - R0 {lab:9s} {d.mean():+.5f} ({se:.5f}) z {d.mean()/se:+.2f}"
            summ[arm][key] = dict(delta=float(d.mean()), se=float(se))
            for sn, m in splits.get(key, {}).items():
                dm = d[m]
                sse = dm.std() / np.sqrt(len(dm))
                line += f" | {sn} {dm.mean():+.5f} ({sse:.5f})"
                summ[arm][f"{key}:{sn}"] = dict(delta=float(dm.mean()), se=float(sse))
            print(line, flush=True)
    (OUT / "gate_summary.json").write_text(json.dumps(summ, indent=1))
    np.savez(OUT / "gate_res.npz", **{f"{a}_{k}": v for a, r in res.items() for k, v in r.items()})


if __name__ == "__main__":
    main()
