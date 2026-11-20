"""Paired evaluation of the sx GPU arms (same logic as nn/pcmp.py, plus per-seed slices, z values, final-step readouts
and the ensemble-diversity readout). CPU only.

  CUDA_VISIBLE_DEVICES="" python nn_x/solvers/sx_eval.py base tau30 pl20 [w1 ...]
Baseline = first name. Seeds 23/24. Reads nn_x/solvers/preds/nn_sx_<arm>_s<seed>_trainonly_val.npy and
nn_x/solvers/preds/holdout/nn_sx_<arm>_s<seed>_holdout.npy, logs nn_x/solvers/logs/sx_*.json.
Writes nn_x/solvers/logs/sx_eval_<arms>.json.
"""
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = "<repo>"
S = f"{ROOT}/nn_x/solvers"
SEEDS = tuple(int(x) for x in __import__("os").environ.get("SX_SEEDS", "23,24").split(","))


def fz(s, fid, M):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def setup(mode):
    if mode == "val":
        f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
        M = f["M"].to_numpy().astype(np.int64)
        off = f["cand_off"].to_numpy().astype(np.int64)
        y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
        day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
        day = np.where(day <= day.min() + 1, 0, 1)
        path = lambda n: f"{S}/preds/nn_{n}_trainonly_val.npy"  # noqa: E731
        slices = {"all": np.ones(len(M), bool), "M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1}
    else:
        D = f"{S}/preds/holdout"
        M, off = np.load(f"{D}/holdout_meta.npy")
        y = np.load(f"{D}/holdout_y.npy").astype(np.float64)
        tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
        pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32),
                           nan=0.0)
        trM = tr["M"].to_numpy()
        pr[trM < 50] = 0.0
        cand = np.where((pr > 0.9) & (trM >= 50))[0]
        hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
        assert (trM[hold] == M).all()
        ts = tr["timestamp"].to_numpy()[hold]
        path = lambda n: f"{D}/nn_{n}_holdout.npy"  # noqa: E731
        slices = {"all": np.ones(len(M), bool), "late20%": ts >= np.quantile(ts, 0.8)}
    fid = np.repeat(np.arange(len(M)), M)
    return dict(M=M, off=off, y=y, fid=fid, path=path, slices=slices)


def per_feed(ctx, spec):
    """spec = 'a+b+...' (equal-weight per-feed z ensemble of sx variant names)."""
    sc = np.mean([fz(np.load(ctx["path"](n)), ctx["fid"], ctx["M"]) for n in spec.split("+")], 0)
    return ndcg_per_feed(sc, ctx["y"], ctx["off"], ctx["M"])


def stats(d, slices):
    return {k: dict(delta=float(d[m].mean()), se=float(d[m].std() / np.sqrt(m.sum())),
                    z=float(d[m].mean() / (d[m].std() / np.sqrt(m.sum()) + 1e-12))) for k, m in slices.items()}


def fmt(st):
    return "  ".join(f"{k} {v['delta']:+.5f}+-{v['se']:.5f} (z {v['z']:+.2f})" for k, v in st.items())


def compare(ctx, pairs):
    """pairs: list of (A, B). Returns per-pair stats and pooled (per-feed mean of the pair deltas) stats."""
    ds, out = [], {"pairs": []}
    for a, b in pairs:
        pa, pb = per_feed(ctx, a), per_feed(ctx, b)
        d = pb - pa
        ds.append(d)
        out["pairs"].append(dict(A=a, B=b, A_mean=float(pa.mean()), B_mean=float(pb.mean()),
                                 slices=stats(d, ctx["slices"])))
    out["pooled"] = stats(np.mean(ds, 0), ctx["slices"])
    return out


def log_readout(v):
    j = json.load(open(f"{S}/logs/{v}.json"))
    h = j["history"]
    best = max(h, key=lambda r: r["holdout_ndcg"])
    return dict(best_step=j["best_step"], total=h[-1]["step"], sel_val=best["val_ndcg"], sel_hold=best["holdout_ndcg"],
                final_val=h[-1]["val_ndcg"], final_hold=h[-1]["holdout_ndcg"],
                max_val=max(r["val_ndcg"] for r in h), minutes=j["train_minutes"])


def main(arms):
    base, res = arms[0], {}
    vctx, hctx = setup("val"), setup("hold")
    v = lambda arm, s: f"sx_{arm}_s{s}"  # noqa: E731
    print("== single runs (log readouts: selected step / final step)")
    res["runs"] = {}
    for arm in arms:
        for s in SEEDS:
            try:
                r = log_readout(v(arm, s))
            except FileNotFoundError:
                continue
            res["runs"][v(arm, s)] = r
            print(f"  {v(arm, s):22s} best_step {r['best_step']:>6}/{r['total']}  val {r['sel_val']:.5f} hold "
                  f"{r['sel_hold']:.5f} | final val {r['final_val']:.5f} hold {r['final_hold']:.5f} | "
                  f"max val {r['max_val']:.5f} | {r['minutes']:.0f} min")
    for arm in arms[1:]:
        pairs = [(v(base, s), v(arm, s)) for s in SEEDS]
        print(f"\n== {arm} vs {base} (B = treatment)")
        rv, rh = compare(vctx, pairs), compare(hctx, pairs)
        for p in rv["pairs"]:
            print(f"  VAL  {p['B']} {p['B_mean']:.5f} vs {p['A']} {p['A_mean']:.5f}: {fmt(p['slices'])}")
        print(f"  VAL  pooled: {fmt(rv['pooled'])}")
        for p in rh["pairs"]:
            print(f"  HOLD {p['B']} {p['B_mean']:.5f} vs {p['A']} {p['A_mean']:.5f}: {fmt(p['slices'])}")
        print(f"  HOLD pooled: {fmt(rh['pooled'])}")
        # final-step readout (information only): deltas of the log's last-eval numbers
        fr = [(res["runs"][v(arm, s)]["final_val"] - res["runs"][v(base, s)]["final_val"],
               res["runs"][v(arm, s)]["final_hold"] - res["runs"][v(base, s)]["final_hold"]) for s in SEEDS]
        print(f"  final-step (log, info): val deltas {[round(x[0], 5) for x in fr]}, hold deltas "
              f"{[round(x[1], 5) for x in fr]}")
        # ensemble diversity readout (information only): eq(base_s23, X_s24), eq(base_s24, X_s23) vs eq(base_s23, base_s24)
        A = f"{v(base, 23)}+{v(base, 24)}"
        ens_pairs = [(A, f"{v(base, 23)}+{v(arm, 24)}"), (A, f"{v(base, 24)}+{v(arm, 23)}")]
        ev, eh = compare(vctx, ens_pairs), compare(hctx, ens_pairs)
        print(f"  ENS VAL  2-member mixed vs 2x base: {fmt(ev['pooled'])}  (base pair {ev['pairs'][0]['A_mean']:.5f})")
        print(f"  ENS HOLD 2-member mixed vs 2x base: {fmt(eh['pooled'])}")
        # 4-member: base23+base24+X23+X24 vs base pair (more members; info only)
        e4 = compare(vctx, [(A, f"{A}+{v(arm, 23)}+{v(arm, 24)}")])
        print(f"  ENS VAL  4-member (2 base + 2 arm) vs 2x base: {fmt(e4['pooled'])}  (info; unequal member count)")
        res[arm] = dict(val=rv, hold=rh, final_step=fr, ens_val=ev, ens_hold=eh, ens4_val=e4)
    json.dump(res, open(f"{S}/logs/sx_eval_{'_'.join(arms)}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
