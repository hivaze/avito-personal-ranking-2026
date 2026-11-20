# Read-out of the optimiser group: per-feed NDCG difference between a treatment and its same-seed baseline on
# val and on the 30,000-feed holdout, averaged over seeds 23 and 24, with the per-feed standard error.
# Run on CPU from lab/archive/nnx/optim, as given in the script's own first lines:
#   python paired.py --base .../preds/nn_base_s{seed} --treat .../preds/nn_optim_muon_s{seed}
# With --tsv it appends one line to results.tsv (copied to ../results/optim_results.tsv).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/archive/nnx/optim/paired.py
#
# paired.py: paired A/B on val (+ 30K random-like train holdout). {seed} in prefixes is substituted.
#   $PY paired.py --base <repo>/nnx/optim/preds/nn_base_s{seed} --treat <dir>/preds/nn_<variant>_s{seed}
#   fine-tune OOF: --val_suffix _val_oof.npy (no holdout files needed)
#   --tsv EXP,ARM,NOTE appends one line to results.tsv (design doc 1.5 format; best_steps/minutes/params left blank)
import argparse, os, sys
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
FL = "<repo>/data/flat"
p = argparse.ArgumentParser()
p.add_argument("--base"); p.add_argument("--treat"); p.add_argument("--seeds", default="23,24")
p.add_argument("--val_suffix", default="_trainonly_val.npy")
p.add_argument("--treat_suffix", default=None, help="val suffix of the treatment (default: --val_suffix)")
p.add_argument("--tsv", default="")
a = p.parse_args()
f = pl.read_parquet(f"{FL}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{FL}/val/cand_target.npy").astype(np.float64)
def pf(s, y, off, M): return ndcg_per_feed(s, y, off, M)
D = {"val": [], "hold": []}
B = {"val": [], "hold": []}
T = {"val": [], "hold": []}
for s in a.seeds.split(","):
    b, t = a.base.format(seed=s), a.treat.format(seed=s)
    vb, vt = pf(np.load(b + a.val_suffix), y, off, M), pf(np.load(t + (a.treat_suffix or a.val_suffix)), y, off, M)
    D["val"].append(vt - vb); B["val"].append(vb.mean()); T["val"].append(vt.mean())
    msg = f"seed {s}: val {vb.mean():.5f} -> {vt.mean():.5f} d={vt.mean()-vb.mean():+.5f} se={(vt-vb).std(ddof=1)/np.sqrt(len(vb)):.5f}"
    if os.path.exists(b + "_holdout.npz") and os.path.exists(t + "_holdout.npz"):
        hb, ht = np.load(b + "_holdout.npz"), np.load(t + "_holdout.npz")
        assert (hb["M"] == ht["M"]).all()
        Mh = hb["M"].astype(np.int64); oh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
        xb, xt = pf(hb["scores"], hb["y"], oh, Mh), pf(ht["scores"], ht["y"], oh, Mh)
        D["hold"].append(xt - xb); B["hold"].append(xb.mean()); T["hold"].append(xt.mean())
        msg += f" | hold {xb.mean():.5f} -> {xt.mean():.5f} d={xt.mean()-xb.mean():+.5f} se={(xt-xb).std(ddof=1)/np.sqrt(len(xb)):.5f}"
    print(msg)
R = {}
for k, v in D.items():
    if v:
        d = np.mean(v, 0); se = d.std(ddof=1) / np.sqrt(len(d))
        R[k] = (np.mean(B[k]), np.mean(T[k]), d.mean(), se)
        print(f"POOLED {k}: d={d.mean():+.5f} se={se:.5f} z={d.mean()/se:+.2f} (n_seeds={len(v)})")
if a.tsv:
    exp, arm, note = (a.tsv.split(",", 2) + ["", ""])[:3]
    v = R.get("val", (np.nan,) * 4); h = R.get("hold", (np.nan,) * 4)
    zv = v[2] / v[3] if v[3] == v[3] else np.nan; zh = h[2] / h[3] if h[3] == h[3] else np.nan
    seed_d = [float(np.mean(x)) for x in D["val"]]
    if h[2] != h[2]:  # no holdout (fine-tune OOF): WIN needs every seed positive and pooled z >= 2
        verdict = "WIN" if v[2] > 0 and zv >= 2 and min(seed_d) > 0 else ("LIKELY" if v[2] > 0 and zv >= 1 else "NO")
    elif v[2] > 0 and h[2] > 0 and max(zv, zh) >= 2 and min(seed_d) >= -0.001:
        verdict = "WIN"
    elif (v[2] + (h[2] if h[2] == h[2] else 0)) > 0 and v[2] > -0.0003 and (h[2] != h[2] or h[2] > -0.0003) \
            and 1 <= np.nanmax([zv, zh]) < 2:
        verdict = "LIKELY"
    else:
        verdict = "NO"
    tsv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.tsv")
    new = not os.path.exists(tsv)
    with open(tsv, "a") as fh:
        if new:
            fh.write("exp\tarm\tseeds\tval_base\tval_treat\td_val\tse_val\thold_base\thold_treat\td_hold\tse_hold\t"
                     "best_steps\tminutes\tparams\tverdict\tnote\n")
        fh.write(f"{exp}\t{arm}\t{a.seeds}\t{v[0]:.5f}\t{v[1]:.5f}\t{v[2]:+.5f}\t{v[3]:.5f}\t{h[0]:.5f}\t{h[1]:.5f}\t"
                 f"{h[2]:+.5f}\t{h[3]:.5f}\t\t\t\t{verdict}\t{note} seed_d_val={['%+.5f' % x for x in seed_d]}\n")
    print("verdict (auto, decision rule 1.6):", verdict)
