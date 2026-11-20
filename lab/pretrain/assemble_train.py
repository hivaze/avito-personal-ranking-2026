"""Assemble train-stage pre_* features from the time-windowed encoders (see build_corpus_window.py) and run the
mandatory leak checks.

For window k and config c in {v3, v4}: raw features of model w<k>_<c> on the window's train feeds are moment-matched
(label-free, on a 60K-feed sample of the window) to the full-corpus twin c, exactly like the val side was matched to
its test-side twin; then the 9 features are built as in assemble.py and averaged over configs.
Writes data/feats/train/pre_<feat>.npy (float32, aligned with data/flat/train/cand_*, NaN outside the windows) and
preds/pre_final_train.npy.
Usage: python pretrain/assemble_train.py [--windows 1,2,3] [--check_only]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, polars as pl

sys.path.insert(0, "<repo>/src")
sys.path.insert(0, "<repo>/pretrain")
from stack_val import feed_rank_z
from metrics import ndcg_per_feed
from assemble import FEATS, RAW

ROOT = Path("<repo>")
FLAT, F, P, PT = ROOT / "data/flat", ROOT / "data/feats", ROOT / "preds", ROOT / "data/pretrain"
WS = PT / "winscores"
CONFIGS = ["v3", "v4"]
WTAG = "w{k}_{c}"      # window model tag pattern (8c: w{k}r_{c})
RTAG = "{c}"           # full-corpus reference tag pattern for the 60K-feed alignment samples (8c: {c}t2)
PREFIX, SCORE = "pre_", "pre_final"
warnings.filterwarnings("ignore", category=RuntimeWarning)


def moments(x):
    m = np.isfinite(x) & (x > -2)
    return x[m].astype(np.float64).mean(), x[m].astype(np.float64).std()


def window_feats(k, M_all, coff_all):
    feeds = np.load(PT / f"win{k}/window_feeds.npy")
    rows = np.load(PT / f"win{k}/cand_rows.npy")
    s_rows = np.load(PT / f"win{k}/cand_rows_s60000.npy")
    pos = np.searchsorted(rows, s_rows)  # sample rows inside the window arrays (rows are sorted)
    assert (rows[pos] == s_rows).all()
    M = M_all[feeds]
    off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    acc = {kk: [] for kk in FEATS + ["score"]}
    calib = {}
    for c in CONFIGS:
        L0 = lambda key: np.load(WS / f"pre_{WTAG.format(k=k, c=c)}_{key}_win{k}.npy").astype(np.float32)
        fns = {}
        for key in RAW:
            x = L0(key)
            mx, sx = moments(x[pos])
            mr, sr = moments(np.load(WS / f"pre_{RTAG.format(c=c)}_{key}_win{k}_s60000.npy"))
            aa = sr / sx; bb = mr - aa * mx
            fns[key] = (aa, bb)
            calib[f"{c}_{key}"] = (round(aa, 4), round(bb, 4))
        L = lambda key: np.where(L0(key) > -2, fns[key][0] * L0(key) + fns[key][1], L0(key)).astype(np.float32) if key in fns else L0(key)
        fe, new, noid = L("fe"), L("new"), L("fe_noid")
        zf, zn = feed_rank_z(off, M, fe)[:, 1], feed_rank_z(off, M, new)[:, 1]
        idm = L("idmax32")
        # parity fix (review P5): raw difference times the single fe slope -> exactly 0 for rows without an item ID
        idc = (fns["fe"][0] * (L0("fe").astype(np.float64) - L0("fe_noid").astype(np.float64))).astype(np.float32)
        vals = dict(fe=fe, fe_z=zf, new=new, new_z=zn, hmax32=L("hmax32"), hmax8=L("hmax8"), fe_noid=noid,
                    id_contrib=idc, idmax32=np.where(idm > -2, idm, np.nan), score=0.5 * (zf + zn))
        for kk, v in vals.items():
            acc[kk].append(v)
    out = {}
    for kk, v in acc.items():
        out[kk] = np.nanmean(np.stack(v), axis=0) if kk == "idmax32" else np.mean(v, axis=0)
    return rows, out, calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="1,2,3")
    ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--wtag", default="w{k}_{c}"); ap.add_argument("--rtag", default="{c}")
    ap.add_argument("--prefix", default="pre_"); ap.add_argument("--score_name", default="pre_final")
    ap.add_argument("--configs", default="v3,v4")
    ap.add_argument("--no_checks", action="store_true")
    a = ap.parse_args()
    global WTAG, RTAG, PREFIX, SCORE, CONFIGS
    CONFIGS = a.configs.split(",")
    WTAG, RTAG, PREFIX, SCORE = a.wtag, a.rtag, a.prefix, a.score_name
    f = pl.read_parquet(FLAT / "train/feeds.parquet")
    M_all = f["M"].to_numpy().astype(np.int64); coff_all = f["cand_off"].to_numpy().astype(np.int64)
    N = len(np.load(FLAT / "train/cand_fidx.npy", mmap_mode="r"))
    wins = [int(x) for x in a.windows.split(",")]
    if not a.check_only:
        full = {kk: np.full(N, np.nan, np.float32) for kk in FEATS + ["score"]}
        for k in wins:
            rows, out, calib = window_feats(k, M_all, coff_all)
            for kk, v in out.items():
                full[kk][rows] = v
            print(f"window {k}: {len(rows)} rows; calib {calib}", flush=True)
        for kk in FEATS:
            np.save(F / "train" / f"{PREFIX}{kk}.npy", full[kk])
        np.save(P / f"{SCORE}_train.npy", full["score"])
    if not a.no_checks:
        leak_checks(f, wins)


def leak_checks(f, wins):
    y = np.load(FLAT / "train/cand_target.npy").astype(np.float64)
    score = np.load(P / f"{SCORE}_train.npy")
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"]
    ts = f["timestamp"].to_numpy().astype(np.int64); users = f["user_id"].to_numpy()
    M_all = f["M"].to_numpy().astype(np.int64); coff = f["cand_off"].to_numpy().astype(np.int64)
    res = {}
    for k in wins:
        info = json.loads((PT / f"win{k}/info.json").read_text())
        feeds = np.load(PT / f"win{k}/window_feeds.npy")
        # user has another feed later in the same window?
        wf = pl.DataFrame({"fidx": feeds, "u": users[feeds], "t": ts[feeds]})
        last = wf.filter(pl.col("u") != 0).group_by("u").agg(pl.col("t").max().alias("tmax"))
        wf = wf.join(last, on="u", how="left").sort("fidx")
        later = ((wf["u"] != 0) & (wf["tmax"].fill_null(0) > wf["t"])).to_numpy()
        rnd = pr[feeds] > 0.9
        M = M_all[feeds]
        rows = np.concatenate([np.arange(o, o + m) for o, m in zip(coff[feeds], M)])
        off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
        sc, yy = score[rows], y[rows]
        nan_rate = float(np.isnan(sc).mean())
        pf = ndcg_per_feed(np.nan_to_num(sc, nan=-1e9), yy, off, M)
        refs = {}
        for rname in ["maxsim32", "cos_u", "f1"]:
            x = np.load(F / "train" / f"{rname}.npy", mmap_mode="r")[rows].astype(np.float64)
            refs[rname] = float(ndcg_per_feed(x, yy, off, M)[rnd].mean())
        sm = lambda m: (int(m.sum()), round(float(pf[m].mean()), 4), round(float(pf[m].std() / np.sqrt(max(m.sum(), 1))), 4))
        r = dict(corpus=info, nan_rate_score=nan_rate,
                 nan_rate_idmax32=float(np.isnan(np.load(F / f"train/{PREFIX}idmax32.npy", mmap_mode="r")[rows]).mean()),
                 ndcg_all=sm(np.ones(len(feeds), bool)), ndcg_random=sm(rnd),
                 ndcg_random_user_has_later_feed_in_window=sm(rnd & later), ndcg_random_no_later=sm(rnd & ~later),
                 ndcg_all_has_later=sm(later), ndcg_all_no_later=sm(~later), ref_single_feature_ndcg_on_random=refs)
        # contrast: full-corpus models (they saw later same-user snapshots -> leaky on train rows) on the 60K-feed sample
        s_rows = np.load(PT / f"win{k}/cand_rows_s60000.npy")
        s_feeds = np.unique(np.load(FLAT / "train/cand_fidx.npy", mmap_mode="r")[s_rows])
        Ms = M_all[s_feeds]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64)
        zsum = 0
        for c in CONFIGS:
            for key in ["fe", "new"]:
                zsum = zsum + feed_rank_z(offs, Ms, np.load(WS / f"pre_{RTAG.format(c=c)}_{key}_win{k}_s60000.npy"))[:, 1] / 4
        rs = pr[s_feeds] > 0.9
        p_leaky = ndcg_per_feed(zsum, y[s_rows], offs, Ms)
        p_win = ndcg_per_feed(score[s_rows], y[s_rows], offs, Ms)
        r["contrast_sample_random_feeds"] = dict(n=int(rs.sum()), windowed=round(float(p_win[rs].mean()), 4),
                                                 full_corpus_leaky=round(float(p_leaky[rs].mean()), 4))
        res[k] = r
        print(json.dumps({"window": k, **{kk: v for kk, v in r.items() if kk != "corpus"}}), flush=True)
    N = len(score)
    print(f"overall train coverage: {np.isfinite(score).mean():.4f} of {N} candidate rows", flush=True)
    (ROOT / f"logs/{PREFIX}train_windows_checks.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
