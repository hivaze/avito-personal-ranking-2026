"""Critique item 5: fine-tune a train-only checkpoint on ONE val day (no folds, one model) and score all val feeds.

Thin wrapper around nn_x/w5/finetune.py (imported, not modified): same recipe as the production n2cpv3 fine-tune
(see plan.txt), but the training feed list is fixed (Tue-only / Thu-only / a size-matched random user-grouped mix).
Outputs go to nn_x/audit/ftforward/{preds,logs}; nothing is written into preds/ or nn*/ckpt.

Usage (one GPU process):
  CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 python ft_day.py --variant r6_cd_m02_r101 --sets tue,thu,mix_tue,mix_thu --seeds 0,1,2
"""
import argparse
import ast
import copy
import json
import math
import os
import sys
import time

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(k, "12")
W5 = "<repo>/nn_x/w5"
sys.path.insert(0, W5)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import Logger, eval_ndcg, predict  # noqa: E402  (w5 copy: CKPT/PREDS point to nn_x/w5, unused here)
from data import SplitData, feat_exists  # noqa: E402
from blend import feed_z  # noqa: E402
from model import listwise_loss  # noqa: E402
from finetune import build_model  # noqa: E402

HERE = "<repo>/nn_x/audit/ftforward"
R6_CKPT = "<repo>/nn_x/r6/ckpt"
R6_LOGS = "<repo>/nn_x/r6/logs"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--sets", default="tue,thu", help="comma list of tue | thu | mix_tue | mix_thu")
    p.add_argument("--seeds", default="0")
    # production n2cpv3 values (nn_x/r6/logs/<v>_n2cpv3_ft.json)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--cand_budget", type=int, default=8192)
    p.add_argument("--max_feeds", type=int, default=128)
    p.add_argument("--loss_norm", default="batch")
    p.add_argument("--extra_lr_mult", type=float, default=10.0)
    p.add_argument("--decay", default="cos")
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--ema_start", type=float, default=0.5)
    return p.parse_args()


def production_extra_names(variant):
    """The exact extra-feature list the production fine-tune used (parsed from its log)."""
    for line in open(f"{R6_LOGS}/{variant}_n2cpv3_ft.log"):
        if line.startswith("extra features ("):
            return ast.literal_eval(line[line.index("["):].strip())
    raise RuntimeError("no 'extra features' line in the production log")


def train_sets(va, rng_seed):
    ts = va.feeds["timestamp"].to_numpy().astype(np.int64)
    tue = (ts + 10800) // 86400 < 20720
    uid = va.user_id.astype(np.int64)
    groups = np.where(uid == 0, -1 - np.arange(va.n), uid)  # u0 feeds as singleton groups (production --u0_singleton)
    out = {"tue": tue, "thu": ~tue}
    for name, n_train in (("mix_tue", int(tue.sum())), ("mix_thu", int((~tue).sum()))):
        rng = np.random.default_rng(12345 + rng_seed)
        ug, inv = np.unique(groups, return_inverse=True)
        order = rng.permutation(len(ug))
        cnt = np.bincount(inv, minlength=len(ug))[order]
        k = int(np.searchsorted(np.cumsum(cnt), n_train)) + 1  # first k groups reach n_train feeds
        chosen = np.zeros(len(ug), bool)
        chosen[order[:k]] = True
        out[name] = chosen[inv]
    return out, tue


def main():
    a = parse()
    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    log = Logger(f"{HERE}/logs/{a.variant}_ftday.log")
    log(f"==== {time.ctime()} args: {vars(a)}")
    ck = torch.load(f"{R6_CKPT}/{a.variant}.pt", map_location=dev, weights_only=False)
    va = SplitData("val", dev, max_hist=ck["args"]["max_hist"])
    if ck["cfg"].get("xt_names"):
        va.attach_xt(ck["cfg"]["xt_names"])
    if ck["cfg"].get("id_buckets"):
        va.load_item_ids()
    names = production_extra_names(a.variant)
    miss = [n for n in names if not feat_exists("val", n)]
    assert not miss, f"production extra features missing on disk: {miss}"
    va.attach_extra(names)  # stats on all val feeds, as in production
    log(f"extra features ({len(names)}) = production list; first/last {names[0]} / {names[-1]}")
    to = np.load(f"<repo>/preds/nn_{a.variant}_trainonly_val.npy").astype(np.float32)
    E = a.epochs
    res = {}
    for s in [int(x) for x in a.seeds.split(",")]:
        sets, tue = train_sets(va, s)
        for sname in a.sets.split(","):
            tr_mask = sets[sname]
            target = (~tue) if sname in ("tue", "mix_tue") else tue
            ho_mask = target & ~tr_mask
            tr_f, ho_f = np.where(tr_mask)[0], np.where(ho_mask)[0]
            t0 = time.time()
            torch.manual_seed(s)
            rng = np.random.default_rng(s)
            model = build_model(ck, dev, None, False, len(names))
            preds = np.zeros((E + 1, va.n_cand), np.float32)
            preds[0] = predict(model, va)
            c0 = np.corrcoef(feed_z(preds[0], va.M_np), feed_z(to, va.M_np))[0, 1]
            log(f"[{sname} seed {s}] train {len(tr_f)} feeds (tue {int(tr_mask[tue].sum())}, thu {int(tr_mask[~tue].sum())}), "
                f"held-out target-day {len(ho_f)}; epoch-0 vs TO per-feed-z corr {c0:.5f}")
            assert c0 > 0.999, "epoch-0 predictions do not reproduce the train-only predictions"
            ex = [p for n, p in model.named_parameters() if n.startswith("extra.")]
            base = [p for n, p in model.named_parameters() if not n.startswith("extra.")]
            groups = [{"params": base, "mult": 1.0}] + ([{"params": ex, "mult": a.extra_lr_mult}] if ex else [])
            groups = [dict(g, params=[p for p in g["params"] if p.requires_grad]) for g in groups]
            opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
            step = 0
            S_total, ema, ema_params, ema_step0 = None, None, None, None
            params = list(model.parameters())
            curve_ho, curve_tr = [eval_ndcg(preds[0], va, ho_f)], [eval_ndcg(preds[0], va, tr_f)]
            for ep in range(1, E + 1):
                model.train()
                jobs = [(va, fb) for fb in va.batches(tr_f, a.cand_budget, a.max_feeds, shuffle=True, rng=rng)]
                jobs = [jobs[i] for i in rng.permutation(len(jobs))]
                if S_total is None:
                    S_total = len(jobs) * E
                    ema_step0 = int(a.ema_start * len(jobs))
                    log(f"  {len(jobs)} batches/epoch, {S_total} steps, EMA from step {ema_step0}")
                for src, fb in jobs:
                    lr_now = a.lr * min(1.0, (step + 1) / a.warmup)
                    if a.decay == "cos":
                        lr_now *= 0.5 * (1 + math.cos(math.pi * min(step, S_total) / S_total))
                    for g in opt.param_groups:
                        g["lr"] = lr_now * g["mult"]
                    b = src.gather(fb)
                    X = model.dense(b)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = model(b, X, use_pos=False)
                    m = b["cmask"]
                    loss = listwise_loss(lg, b["y"], m, norm=a.loss_norm, denom=a.max_feeds)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if a.ema > 0 and step >= ema_step0:
                        with torch.no_grad():
                            if ema is None:
                                for attr in ("aux_out", "intent_logp"):
                                    if hasattr(model, attr):
                                        delattr(model, attr)
                                ema = copy.deepcopy(model).float()
                                ema_params = list(ema.parameters())
                            else:
                                torch._foreach_lerp_(ema_params, params, 1.0 - a.ema)
                    step += 1
                preds[ep] = predict(model, va)
                curve_ho.append(eval_ndcg(preds[ep], va, ho_f))
                curve_tr.append(eval_ndcg(preds[ep], va, tr_f))
            fin = predict(ema, va)
            nd_ho, nd_tr = eval_ndcg(fin, va, ho_f), eval_ndcg(fin, va, tr_f)
            log(f"  held-out target-day ndcg by epoch {np.round(curve_ho, 5).tolist()} | final EMA {nd_ho:.5f}")
            log(f"  in-sample (train feeds) ndcg by epoch {np.round(curve_tr, 5).tolist()} | final EMA {nd_tr:.5f}  "
                f"({(time.time() - t0) / 60:.1f} min)")
            tag = f"{a.variant}_ft{sname}_s{s}"
            np.save(f"{HERE}/preds/nn_{tag}_val.npy", fin.astype(np.float32))
            np.save(f"{HERE}/preds/nn_{tag}_val_epochs.npy", preds.astype(np.float16))
            np.save(f"{HERE}/preds/nn_{tag}_trainmask.npy", tr_mask)
            res[tag] = dict(train=int(len(tr_f)), heldout=int(len(ho_f)), curve_heldout=curve_ho, curve_train=curve_tr,
                            ema_heldout=nd_ho, ema_train=nd_tr, steps=S_total, ema_step0=ema_step0,
                            minutes=(time.time() - t0) / 60)
            del model, ema, opt
            torch.cuda.empty_cache()
    json.dump(dict(args=vars(a), extra=names, runs=res), open(f"{HERE}/{a.variant}_ftday.json", "w"), indent=2)
    log(f"done {time.ctime()}")


if __name__ == "__main__":
    main()
