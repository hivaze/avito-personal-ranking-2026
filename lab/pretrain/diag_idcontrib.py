"""Label vs pre_id_contrib on train (in-window random-like feeds) vs val, conditional on the co-visitation /
history-popularity count of the candidate (it_hist_all_luo, seq_cv_pop).
Per bin: positive rate, and the lift = positive rate in the top within-feed quintile of id_contrib / bottom quintile.
Usage: python pretrain/diag_idcontrib.py"""
import sys, json
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from stack_val import feed_rank_z
from metrics import ndcg_per_feed

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"; PT = ROOT / "data/pretrain"


def get(split, feeds=None, nsample=None):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    if feeds is None:
        rows = None; Mf = M
    else:
        Mf = M[feeds]
        rows = np.repeat(off[feeds] - np.concatenate([[0], np.cumsum(Mf)[:-1]]), Mf) + np.arange(Mf.sum())
    L = lambda n: np.asarray(np.load(FEATS / split / f"{n}.npy", mmap_mode="r")[rows] if rows is not None else np.load(FEATS / split / f"{n}.npy"), np.float32)
    y = np.asarray(np.load(FLAT / split / "cand_target.npy", mmap_mode="r")[rows] if rows is not None else np.load(FLAT / split / "cand_target.npy"), np.float64)
    o = np.concatenate([[0], np.cumsum(Mf)[:-1]]).astype(np.int64)
    d = dict(y=y, M=Mf, off=o, idc=L("pre_id_contrib"), idm=L("pre_idmax32"), fe=L("pre_fe"), noid=L("pre_fe_noid"),
             ith=L("it_hist_all_luo"), cvp=L("seq_cv_pop"))
    d["idc_rk"] = feed_rank_z(o, Mf, d["idc"])[:, 0]
    return d


def table(d, key, edges):
    out = []
    b = np.digitize(d[key], edges)
    for i in range(len(edges) + 1):
        m = b == i
        if m.sum() < 1000:
            continue
        top = m & (d["idc_rk"] >= 0.8); bot = m & (d["idc_rk"] <= 0.2)
        out.append(dict(bin=f"{key} bin{i}", share=round(float(m.mean()), 3), pos=round(float(d["y"][m].mean()), 4),
                        pos_top20=round(float(d["y"][top].mean()), 4), pos_bot20=round(float(d["y"][bot].mean()), 4),
                        lift=round(float(d["y"][top].mean() / max(d["y"][bot].mean(), 1e-9)), 2)))
    return out


def main():
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"]
    rng = np.random.default_rng(1)
    res = {}
    sets = {"val": get("val")}
    for k in [1, 3]:
        wf = np.load(PT / f"win{k}/window_feeds.npy"); wf = np.sort(rng.permutation(wf[pr[wf] > 0.9])[:40000])
        sets[f"train_W{k}_random"] = get("train", wf)
    for name, d in sets.items():
        q = np.quantile(d["ith"][d["ith"] > 0], [0.5, 0.9]) if (d["ith"] > 0).any() else [1, 2]
        r = dict(ndcg_id_contrib=round(float(ndcg_per_feed(d["idc"], d["y"], d["off"], d["M"]).mean()), 4),
                 ndcg_fe=round(float(ndcg_per_feed(d["fe"], d["y"], d["off"], d["M"]).mean()), 4),
                 ndcg_fe_noid=round(float(ndcg_per_feed(d["noid"], d["y"], d["off"], d["M"]).mean()), 4),
                 ndcg_it_hist_all_luo=round(float(ndcg_per_feed(d["ith"], d["y"], d["off"], d["M"]).mean()), 4),
                 share_ith_pos=round(float((d["ith"] > 0).mean()), 3), share_has_id=round(float(np.isfinite(d["idm"]).mean()), 3),
                 corr_idc_ith_withinfeed=round(float(np.corrcoef(feed_rank_z(d["off"], d["M"], d["idc"])[:, 1],
                                                                 feed_rank_z(d["off"], d["M"], np.log1p(np.maximum(d["ith"], 0)))[:, 1])[0, 1]), 3),
                 by_ith=table(d, "ith", [1e-9, 10.0, 100.0]), by_cvp=table(d, "cvp", [1e-9] + list(np.quantile(d["cvp"][d["cvp"] > 0], [0.5, 0.9])) if (d["cvp"] > 0).any() else [1e-9]))
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if not k.startswith("by_")}), flush=True)
        for row in r["by_ith"] + r["by_cvp"]:
            print("   ", row, flush=True)
    (ROOT / "logs/pretrain_diag_idcontrib.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
