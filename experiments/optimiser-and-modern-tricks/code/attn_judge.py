# Read-out and decision rule of the attention group: paired per-feed NDCG differences against the shared
# baseline on val and on the holdout over seeds 23 and 24, a WIN / LIKELY / NO verdict, and one line appended
# to results.tsv. Run on CPU from lab/nn_x/attn, as given in the script's docstring:
#   python judge.py --exp A1 --arm gate --treat attn_gate
# Produced the rows of ../results/attn_results.tsv.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/attn/judge.py
#
"""Paired A/B judgement (nnx_design.md sec 1.6) + one results.tsv line.
  python judge.py --exp A1 --arm gate --treat attn_gate [--base_prefix .../nn_base_s{seed}] [--seeds 23,24]"""
import argparse, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
FL = "<repo>/data/flat"
P = "<repo>/preds_x/attn"
p = argparse.ArgumentParser()
p.add_argument("--exp"); p.add_argument("--arm"); p.add_argument("--treat")
p.add_argument("--base_prefix", default="<repo>/preds_x/optim/nn_base_s{seed}")
p.add_argument("--seeds", default="23,24"); p.add_argument("--note", default="")
p.add_argument("--val_suffix", default="_trainonly_val.npy")
p.add_argument("--no_write", action="store_true")
a = p.parse_args()
f = pl.read_parquet(f"{FL}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{FL}/val/cand_target.npy").astype(np.float64)
seeds = a.seeds.split(",")
dv, dh, rows = [], [], []
vb_l, vt_l, hb_l, ht_l, steps, mins, params = [], [], [], [], [], [], None
for s in seeds:
    b, t = a.base_prefix.format(seed=s), f"{P}/nn_{a.treat}_s{s}"
    vb, vt = ndcg_per_feed(np.load(b + a.val_suffix), y, off, M), ndcg_per_feed(np.load(t + a.val_suffix), y, off, M)
    dv.append(vt - vb); vb_l.append(vb.mean()); vt_l.append(vt.mean())
    line = f"seed {s}: val {vb.mean():.5f} -> {vt.mean():.5f} d={vt.mean()-vb.mean():+.5f} se={(vt-vb).std(ddof=1)/np.sqrt(len(vb)):.5f}"
    if os.path.exists(b + "_holdout.npz") and os.path.exists(t + "_holdout.npz"):
        hb, ht = np.load(b + "_holdout.npz"), np.load(t + "_holdout.npz")
        assert (hb["M"] == ht["M"]).all()
        Mh = hb["M"].astype(np.int64); oh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
        xb, xt = ndcg_per_feed(hb["scores"], hb["y"], oh, Mh), ndcg_per_feed(ht["scores"], ht["y"], oh, Mh)
        dh.append(xt - xb); hb_l.append(xb.mean()); ht_l.append(xt.mean())
        line += f" | hold {xb.mean():.5f} -> {xt.mean():.5f} d={xt.mean()-xb.mean():+.5f} se={(xt-xb).std(ddof=1)/np.sqrt(len(xb)):.5f}"
    js = f"<repo>/nn_x/attn/logs/{a.treat}_s{s}.json"
    if os.path.exists(js):
        j = json.load(open(js)); steps.append(j["best_step"]); mins.append(round(j["train_minutes"], 1)); params = j.get("params")
    print(line)
def pool(v):
    d = np.mean(v, 0); se = d.std(ddof=1) / np.sqrt(len(d)); return d.mean(), se
Dv, SEv = pool(dv)
print(f"POOLED val: d={Dv:+.5f} se={SEv:.5f} z={Dv/SEv:+.2f}")
if dh:
    Dh, SEh = pool(dh)
    print(f"POOLED hold: d={Dh:+.5f} se={SEh:.5f} z={Dh/SEh:+.2f}")
else:
    Dh, SEh = float("nan"), float("nan")
per_seed_v = [x.mean() for x in dv]
zmax = max(Dv / SEv, Dh / SEh if dh else -9)
if not dh:  # fine-tune OOF only (design C3 rule): WIN needs every seed positive and pooled z >= 2
    verdict = "WIN" if (min(per_seed_v) > 0 and Dv / SEv >= 2) else ("LIKELY" if Dv > 0 and Dv / SEv >= 1 else "NO")
elif Dv > 0 and Dh > 0 and zmax >= 2 and min(per_seed_v) >= -0.0010:
    verdict = "WIN"
elif Dv + Dh > 0 and Dv > -0.0003 and Dh > -0.0003 and 1 <= zmax < 2:
    verdict = "LIKELY"
else:
    verdict = "NO"
print("VERDICT", verdict, "per-seed val d:", [round(x, 5) for x in per_seed_v])
if not a.no_write:
    hdr = "exp\tarm\tseeds\tval_base\tval_treat\td_val\tse_val\thold_base\thold_treat\td_hold\tse_hold\tbest_steps\tminutes\tparams\tverdict\tnote\n"
    fn = "<repo>/nn_x/attn/results.tsv"
    new = not os.path.exists(fn)
    with open(fn, "a") as fo:
        if new:
            fo.write(hdr)
        fo.write("\t".join(map(str, [a.exp, a.arm, a.seeds, f"{np.mean(vb_l):.5f}", f"{np.mean(vt_l):.5f}", f"{Dv:+.5f}", f"{SEv:.5f}",
                                     f"{np.mean(hb_l):.5f}" if hb_l else "", f"{np.mean(ht_l):.5f}" if ht_l else "",
                                     f"{Dh:+.5f}", f"{SEh:.5f}", "/".join(map(str, steps)), "/".join(map(str, mins)),
                                     params, verdict, a.note])) + "\n")
