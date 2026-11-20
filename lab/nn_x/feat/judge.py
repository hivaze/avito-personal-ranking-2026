"""Paired A/B judge for nn_x/feat (design doc 1.6) + one results.tsv line per comparison.

  $PY judge.py --exp B1 --arm ple --treat feat_ple_s{seed} [--base_dir <repo>/preds_x/optim --base base_s{seed}]
Per seed: per-feed paired NDCG differences on val (14,423 feeds) and on the fixed 30K random-like train holdout.
Pooled over seeds: mean of the per-seed per-feed differences, SE over feeds.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

FL = "<repo>/data/flat"
HERE = Path(__file__).resolve().parent
PREDS = Path("<repo>/preds_x/feat")


def log_info(path):
    """best step, minutes, params, val at the eval closest to step 10000 and at the last step."""
    if not Path(path).exists():
        return {}
    txt = Path(path).read_text().split("==== ")[-1]
    out = {}
    m = re.search(r"BEST step (\d+)", txt)
    out["best"] = int(m.group(1)) if m else -1
    m = re.search(r"train time ([\d.]+) min", txt)
    out["minutes"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"params: ([\d.]+)M", txt)
    out["params"] = float(m.group(1)) if m else float("nan")
    steps = [(int(a), float(b), float(c)) for a, b, c in
             re.findall(r"step (\d+)/\d+ loss \S+ val_ndcg ([\d.]+) holdout_ndcg ([\d.]+)", txt)]
    out["curve"] = steps
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--treat", required=True, help="variant pattern with {seed}")
    p.add_argument("--treat_dir", default=str(PREDS))
    p.add_argument("--base", default="base_s{seed}")
    p.add_argument("--base_dir", default="<repo>/preds_x/optim")
    p.add_argument("--base_logs", default="<repo>/nnx/optim/logs")
    p.add_argument("--seeds", default="23,24")
    p.add_argument("--val_suffix", default="_trainonly_val.npy")
    p.add_argument("--base_map", default="", help="per-seed base prefix overrides, e.g. 25=/path/nn_feat_base_s25")
    p.add_argument("--note", default="")
    p.add_argument("--no_write", action="store_true")
    a = p.parse_args()
    f = pl.read_parquet(f"{FL}/val/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    y = np.load(f"{FL}/val/cand_target.npy").astype(np.float64)
    seeds = a.seeds.split(",")
    D = {"val": [], "hold": []}
    lev = {"val": [[], []], "hold": [[], []]}
    per_seed, best_steps, minutes, params = [], [], [], []
    bmap = dict(kv.split("=") for kv in a.base_map.split(",") if kv)
    for s in seeds:
        bn, tn = a.base.format(seed=s), a.treat.format(seed=s)
        b, t = f"{a.base_dir}/nn_{bn}", f"{a.treat_dir}/nn_{tn}"
        if s in bmap:
            b = bmap[s]
            bn = Path(b).name.removeprefix("nn_")
        vb = ndcg_per_feed(np.load(b + a.val_suffix), y, off, M)
        vt = ndcg_per_feed(np.load(t + a.val_suffix), y, off, M)
        D["val"].append(vt - vb)
        lev["val"][0].append(vb.mean())
        lev["val"][1].append(vt.mean())
        msg = (f"seed {s}: val {vb.mean():.5f} -> {vt.mean():.5f} d={vt.mean() - vb.mean():+.5f} "
               f"se={(vt - vb).std(ddof=1) / np.sqrt(len(vb)):.5f}")
        sd = {"seed": s, "d_val": vt.mean() - vb.mean()}
        if Path(b + "_holdout.npz").exists() and Path(t + "_holdout.npz").exists():
            hb, ht = np.load(b + "_holdout.npz"), np.load(t + "_holdout.npz")
            assert (hb["M"] == ht["M"]).all()
            Mh = hb["M"].astype(np.int64)
            oh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
            xb = ndcg_per_feed(hb["scores"], hb["y"], oh, Mh)
            xt = ndcg_per_feed(ht["scores"], ht["y"], oh, Mh)
            D["hold"].append(xt - xb)
            lev["hold"][0].append(xb.mean())
            lev["hold"][1].append(xt.mean())
            sd["d_hold"] = xt.mean() - xb.mean()
            msg += (f" | hold {xb.mean():.5f} -> {xt.mean():.5f} d={xt.mean() - xb.mean():+.5f} "
                    f"se={(xt - xb).std(ddof=1) / np.sqrt(len(xb)):.5f}")
        bl = Path(a.base_logs) / f"{bn}.log"
        li, lb = log_info(HERE / "logs" / f"{tn}.log"), log_info(bl if bl.exists() else HERE / "logs" / f"{bn}.log")
        if li:
            best_steps.append(li["best"])
            minutes.append(li["minutes"])
            params.append(li["params"])
            cv = {st: v for st, v, _ in li["curve"]}
            cb = {st: v for st, v, _ in lb.get("curve", [])}
            if 10000 in cv and cb.get(10000):
                last = max(cv)
                msg += (f" | val@10k {cv[10000]:.5f} (base {cb[10000]:.5f}) val@last {cv[last]:.5f} "
                        f"(base {cb.get(last, float('nan')):.5f}) best_step {li['best']} (base {lb.get('best')})")
        per_seed.append(sd)
        print(msg)
    res = {}
    for k, v in D.items():
        if v:
            d = np.mean(v, 0)
            se = d.std(ddof=1) / np.sqrt(len(d))
            res[k] = (d.mean(), se, np.mean(lev[k][0]), np.mean(lev[k][1]))
            print(f"POOLED {k}: d={d.mean():+.5f} se={se:.5f} z={d.mean() / se:+.2f} (n_seeds={len(v)})")
    dv, sv = res["val"][0], res["val"][1]
    dh, sh = res.get("hold", (float("nan"),) * 4)[:2]
    zv, zh = dv / sv, (dh / sh if "hold" in res else float("nan"))
    worst = min(x["d_val"] for x in per_seed)
    if dv > 0 and dh > 0 and max(zv, zh) >= 2 and worst >= -0.0010:
        verdict = "WIN"
    elif dv + dh > 0 and dv > -0.0003 and dh > -0.0003 and 1 <= max(zv, zh) < 2:
        verdict = "LIKELY"
    else:
        verdict = "NO"
    print(f"VERDICT {verdict} (zv {zv:+.2f}, zh {zh:+.2f}, worst seed d_val {worst:+.5f})")
    if not a.no_write:
        tsv = HERE / "results.tsv"
        if not tsv.exists():
            tsv.write_text("exp\tarm\tseeds\tval_base\tval_treat\td_val\tse_val\thold_base\thold_treat\td_hold\tse_hold"
                           "\tbest_steps\tminutes\tparams\tverdict\tnote\n")
        hb_, ht_ = (res["hold"][2], res["hold"][3]) if "hold" in res else (float("nan"), float("nan"))
        per = ";".join(f"s{x['seed']}:v{x['d_val']:+.5f}/h{x.get('d_hold', float('nan')):+.5f}" for x in per_seed)
        line = [a.exp, a.arm, a.seeds, f"{res['val'][2]:.5f}", f"{res['val'][3]:.5f}", f"{dv:+.5f}", f"{sv:.5f}",
                f"{hb_:.5f}", f"{ht_:.5f}", f"{dh:+.5f}", f"{sh:.5f}", ",".join(map(str, best_steps)),
                ",".join(f"{m:.1f}" for m in minutes), ",".join(f"{q:.2f}M" for q in params), verdict,
                (per + " " + a.note).strip()]
        with open(tsv, "a") as fh:
            fh.write("\t".join(line) + "\n")


if __name__ == "__main__":
    main()
