"""WIDTH-GATE Step 0: generalisation-GAP probe (reports/SOLVER_POWER_1113.md 6.3 item 1; rule nn_x/w5/logs/B_widthgate_predeclared.txt).
Inference only, read-only on every checkpoint / feature file; one GPU.

  cd <repo>/nn_x/w11/width && CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 ../../../.venv/bin/python gap_probe.py
      -> nn_x/w11/width/logs/gap_probe.{txt,json}

Snapshots @77500 (plateau end, before the cooldown-pool replay at step 78,680): v29 d192 m02 s711-715, v30 d128 m02 s901-906,
w5_e10_base d128 s951-956. Main-loss listwise CE in eval mode (model.eval(), bf16 autocast, use_pos False, the train domain token as
in training, NO aux): per feed -sum_i (y_i / sum y) log softmax(logits)_i over the valid candidates, mean over feeds with >= 1 positive.
Feed sets of the TRAIN split (M >= 50): H = the fixed 30K holdout (train.py selection, never trained on); R = 30K in-sample random-like
feeds (p_random > 0.9, not in H; rng 777); P = 30K in-sample production feeds (p_random < 0.1; rng 778). GAP = CE(H) - CE(R).
"""
import json
import os
import time

import numpy as np
import torch

from common import ROOT, W5, ndcg
from data import SplitData
from model import Ranker

FAM = {"d192 v29 m02": [(f"{ROOT}/nn/ckpt/v29_d192_m02_s{s}@77500.pt", f"v29_s{s}") for s in range(711, 716)],
       "d128 v30 m02": [(f"{ROOT}/nn/ckpt/v30_f7_m02_s{s}@77500.pt", f"v30_s{s}") for s in range(901, 907)],
       "d128 w5_e10_base": [(f"{ROOT}/nn_x/w5/ckpt/w5_e10_base_s{s}@77500.pt", f"e10_s{s}") for s in range(951, 957)]}
OUT_TXT, OUT_JSON = W5 / "logs" / "gap_probe.txt", W5 / "logs" / "gap_probe.json"
LINES = []


def P(s):
    print(s, flush=True)
    LINES.append(s)


@torch.no_grad()
def per_feed_ce(model, tr, feeds):
    """(per-feed listwise CE, per-feed NDCG, #positives) of `model` on `feeds` of the train split; eval mode, bf16, use_pos False."""
    model.eval()
    feeds = np.sort(feeds)
    ce = np.full(len(feeds), np.nan)
    pos = np.zeros(len(feeds))
    out = torch.zeros(tr.n_cand, dtype=torch.float32, device=tr.device)
    idx = {int(f): i for i, f in enumerate(feeds)}
    for fb in tr.batches(feeds, 65536, 1024, shuffle=False):
        b = tr.gather(fb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b, use_pos=False)
        tr.scatter(out, b, lg)
        lgm = lg.float().masked_fill(~b["cmask"], -1e4)
        logp = torch.log_softmax(lgm, 1)
        y = b["y"]
        ys = y.sum(1)
        l = -(y * logp).sum(1) / ys.clamp_min(1)
        for f, li, yi in zip(fb.tolist(), l.cpu().numpy(), ys.cpu().numpy()):
            ce[idx[f]] = li
            pos[idx[f]] = yi
    M, off = tr.M_np[feeds], tr.cand_off_np[feeds]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off, M)])
    sc = out[torch.from_numpy(rows).to(tr.device)].cpu().numpy()
    offs = np.concatenate([[0], np.cumsum(M)[:-1]])
    from metrics import ndcg_per_feed
    nd = ndcg_per_feed(sc, tr.target_np[rows], offs, M)
    return ce, nd, pos


def main():
    t0 = time.time()
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    tr = SplitData("train", dev)
    pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    pr[tr.M_np < 50] = 0.0
    cand = np.where((pr > 0.9) & (tr.M_np >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))  # == train.py --holdout 30000
    rest = np.setdiff1d(cand, hold)
    rand_in = np.sort(np.random.default_rng(777).choice(rest, 30000, replace=False))
    prod_pool = np.where((pr < 0.1) & (tr.M_np >= 50))[0]
    prod_in = np.sort(np.random.default_rng(778).choice(prod_pool, 30000, replace=False))
    assert not np.isin(rand_in, hold).any() and not np.isin(prod_in, hold).any()
    sets = {"H": hold, "R": rand_in, "P": prod_in}
    P(f"== WIDTH-GATE Step 0 GAP probe {time.strftime('%FT%TZ', time.gmtime())}: train split, M >= 50; H = 30K holdout (never trained on), "
      f"R = 30K in-sample random-like (p_random > 0.9, of {len(rest)} non-holdout), P = 30K in-sample production (p_random < 0.1, of {len(prod_pool)}); "
      f"listwise CE in eval mode (no aux, bf16, use_pos False, train domain token); GAP = CE(H) - CE(R)")
    for k, f in sets.items():
        npos = np.array([tr.target_np[o:o + m].sum() for o, m in zip(tr.cand_off_np[f], tr.M_np[f])])
        P(f"   {k}: {len(f)} feeds, mean M {tr.M_np[f].mean():.1f}, mean p_random {pr[f].mean():.3f}, feeds with >= 1 positive {int((npos > 0).sum())}")
    prt = torch.from_numpy(pr).to(dev)
    res = {}
    xt_loaded = None
    for fam, cks in FAM.items():
        for path, tag in cks:
            ck = torch.load(path, map_location=dev, weights_only=False)
            cfg = ck["cfg"]
            if cfg.get("xt_names") != xt_loaded:
                tr.xt = None
                tr.attach_xt(cfg["xt_names"])
                xt_loaded = cfg["xt_names"]
            tr.dom = torch.stack([1 - prt, prt, torch.zeros_like(prt)], 1) if cfg.get("domain") else None
            m = Ranker(**cfg).to(dev)
            m.load_state_dict(ck["state_dict"])
            r = dict(family=fam, d=cfg["d"], step=ck.get("step"), train_val=ck.get("val_ndcg"), train_hold=ck.get("holdout_ndcg"))
            for k, f in sets.items():
                ce, nd, pos = per_feed_ce(m, tr, f)
                ok = pos > 0
                r[f"CE_{k}"] = float(ce[ok].mean())
                r[f"CEse_{k}"] = float(ce[ok].std(ddof=1) / np.sqrt(ok.sum()))
                r[f"NDCG_{k}"] = float(nd.mean())
                r[f"n_{k}"] = int(ok.sum())
            r["GAP"] = r["CE_H"] - r["CE_R"]
            r["GAP_P"] = r["CE_H"] - r["CE_P"]
            res[tag] = r
            P(f"  {tag:9s} {fam:17s} d{r['d']} @{r['step']}: CE H {r['CE_H']:.5f} R {r['CE_R']:.5f} P {r['CE_P']:.5f} | GAP H-R {r['GAP']:+.5f} "
              f"(feed SE ~{np.hypot(r['CEse_H'], r['CEse_R']):.5f}) H-P {r['GAP_P']:+.5f} | NDCG H {r['NDCG_H']:.5f} R {r['NDCG_R']:.5f} P {r['NDCG_P']:.5f} "
              f"| in-training @77500 val {r['train_val']:.5f} hold {r['train_hold']:.5f} | {time.time() - t0:.0f}s")
            del m
            torch.cuda.empty_cache()
    P("\n== family means +- SE over seeds")
    keys = ["CE_H", "CE_R", "CE_P", "GAP", "GAP_P", "NDCG_H", "NDCG_R", "NDCG_P"]
    fam_stats = {}
    for fam in FAM:
        rows = [r for r in res.values() if r["family"] == fam]
        st = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(len(rows)))) for k in keys}
        fam_stats[fam] = dict(n=len(rows), **{k: dict(mean=v[0], se=v[1]) for k, v in st.items()})
        P(f"  {fam:17s} n {len(rows)}: " + "  ".join(f"{k} {v[0]:.5f}+-{v[1]:.5f}" for k, v in st.items()))
    P("\n== d192 - d128 (Welch SE over seeds)")
    d192 = [r for r in res.values() if r["family"] == "d192 v29 m02"]
    diffs = {}
    for ref_name, ref_rows in (("d128 v30 m02 (6)", [r for r in res.values() if r["family"] == "d128 v30 m02"]),
                               ("d128 all 12", [r for r in res.values() if r["d"] == 128])):
        d = {}
        for k in keys:
            x = np.array([r[k] for r in d192]); y = np.array([r[k] for r in ref_rows])
            se = float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))
            d[k] = dict(delta=float(x.mean() - y.mean()), se=se, z=float((x.mean() - y.mean()) / se) if se > 0 else float("nan"))
        diffs[ref_name] = d
        P(f"  vs {ref_name:17s}: " + "  ".join(f"{k} {v['delta']:+.5f}+-{v['se']:.5f} (z {v['z']:+.1f})" for k, v in d.items()))
    g = diffs["d128 all 12"]
    reading = ("DATA-limited reading (d192 memorises more: GAP larger by > 2 SE)" if g["GAP"]["z"] > 2 else
               "INFO-limited reading (no larger GAP at d192; the same fit on unseen and seen random-like feeds)" if abs(g["GAP"]["z"]) <= 2 else
               "d192 has the SMALLER gap (z < -2): width does not buy memorisation here")
    P(f"\n== READING (information, no gate): {reading}; CE(R) d192 - d128 {g['CE_R']['delta']:+.5f} (z {g['CE_R']['z']:+.1f}), "
      f"CE(H) {g['CE_H']['delta']:+.5f} (z {g['CE_H']['z']:+.1f}); NDCG(H) {g['NDCG_H']['delta']:+.5f} (z {g['NDCG_H']['z']:+.1f}); {time.time() - t0:.0f}s total")
    OUT_TXT.write_text("\n".join(LINES) + "\n")
    json.dump(dict(per_ckpt=res, family=fam_stats, diffs=diffs, reading=reading, sets={k: int(len(v)) for k, v in sets.items()}),
              open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
