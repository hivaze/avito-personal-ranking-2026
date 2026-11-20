"""Parse every training / fine-tune run log into one table (CPU, read-only).

Sources: nn/logs/*.log, nn_x/{attn,feat,solvers}/logs/*.log, nnx/optim/logs/*.log (+aborted), nn_x/verify/*/logs/*.log.
(nn_x/optim is a symlink to nnx/optim -> skipped to avoid duplicates.)
Each log may hold several runs (Logger appends); every '==== ... args: {...}' starts a segment.
Output: runs.csv (train runs), ft_runs.csv (fine-tunes), curves.jsonl (per-step val/holdout curves).
"""
import ast
import csv
import glob
import json
import os
import re
import sys

ROOT = "<repo>"
OUT = os.path.dirname(os.path.abspath(__file__))  # nn_x/r6/recap
SRC = (sorted(glob.glob(f"{ROOT}/nn/logs/*.log")) + sorted(glob.glob(f"{ROOT}/nn_x/attn/logs/*.log"))
       + sorted(glob.glob(f"{ROOT}/nn_x/feat/logs/*.log")) + sorted(glob.glob(f"{ROOT}/nn_x/solvers/logs/*.log"))
       + sorted(glob.glob(f"{ROOT}/nnx/optim/logs/*.log")) + sorted(glob.glob(f"{ROOT}/nnx/optim/logs/aborted/*.log"))
       + sorted(glob.glob(f"{ROOT}/nn_x/verify/*/logs/*.log")))

STEP = re.compile(r"^step (\d+)/(\d+) loss ([\d.na]+) val_ndcg ([\d.na]+) holdout_ndcg ([\d.na]+) lr ([\d.e+-]+) ([\d.]+) min"
                  r"(?: cd_loss_reg ([\d.na]+) cd_loss_pool ([\d.na]+))?")
BEST = re.compile(r"^BEST step (\d+) val_ndcg ([\d.]+)")
PARAMS = re.compile(r"^params: ([\d.]+)M")
STEPS = re.compile(r"steps/epoch (\d+), total steps (\d+)")
OOF = re.compile(r"OOF ndcg by epoch: \[([^\]]*)\]")
SNAP = re.compile(r"snapshot mean of epochs .*: OOF ndcg ([\d.]+)")
EMA = re.compile(r"final EMA weights: OOF ndcg ([\d.]+)")


def fl(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def segments(path):
    cur = None
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("==== ") and " args: " in line:
                if cur:
                    yield cur
                ts, a = line[5:].split(" args: ", 1)
                try:
                    args = ast.literal_eval(a)
                except Exception:
                    args = {}
                cur = dict(path=path, start=ts.strip(), args=args, lines=[])
            elif cur is not None:
                cur["lines"].append(line)
    if cur:
        yield cur


train_rows, ft_rows, curves = [], [], []
for p in SRC:
    segs = list(segments(p))
    for si, s in enumerate(segs):
        a = s["args"]
        src = os.path.relpath(p, ROOT)
        if "d" in a and "epochs" in a and "out" not in a:  # train.py run
            st = [m.groups() for m in (STEP.match(l) for l in s["lines"]) if m]
            best = [m.groups() for m in (BEST.match(l) for l in s["lines"]) if m]
            params = [m.group(1) for m in (PARAMS.match(l) for l in s["lines"]) if m]
            stp = [m.groups() for m in (STEPS.search(l) for l in s["lines"]) if m]
            steps = [int(x[0]) for x in st]
            val = [fl(x[3]) for x in st]
            hol = [fl(x[4]) for x in st]
            total = int(st[0][1]) if st else (int(stp[0][1]) if stp else -1)
            row = dict(src=src, seg=si, nseg=len(segs), variant=a.get("variant"), start=s["start"],
                       epochs=a.get("epochs"), lr=a.get("lr"), muon=a.get("muon", False), sched=a.get("sched", "cos"),
                       cooldown_frac=a.get("cooldown_frac"), cooldown_mix=a.get("cooldown_mix"),
                       aux_anneal=a.get("aux_anneal"), aux_pos=a.get("aux_pos"), aux_mode=a.get("aux_mode"),
                       aux_tau=a.get("aux_tau"), retr_m=a.get("retr_m", 0), attn_gate=a.get("attn_gate", False),
                       arch=a.get("arch", "tf"), d=a.get("d"), heads=a.get("heads"), blocks=a.get("blocks"),
                       hist_layers=a.get("hist_layers"), ff=a.get("ff"), drop=a.get("drop"), wd=a.get("wd"),
                       warmup=a.get("warmup"), cand_budget=a.get("cand_budget"), max_feeds=a.get("max_feeds"),
                       max_hist=a.get("max_hist"), loss=a.get("loss"), loss_norm=a.get("loss_norm"),
                       xt=a.get("xt", ""), xt_drop=a.get("xt_drop", 0.0), sim_bias=a.get("sim_bias"),
                       no_pos=a.get("no_pos"), domain=a.get("domain"), user0=a.get("user0"),
                       holdout=a.get("holdout", 0), select_on=a.get("select_on"), keep=a.get("keep"),
                       init=a.get("init", ""), only_random=a.get("only_random", -1), prand_base=a.get("prand_base", -1),
                       seed=a.get("seed"), id_buckets=a.get("id_buckets", 0), intent=a.get("intent", 0),
                       params_M=fl(params[0]) if params else float("nan"), total_steps=total,
                       n_evals=len(st), last_step=steps[-1] if steps else -1,
                       complete=bool(best) or (bool(steps) and steps[-1] == total),
                       best_step=int(best[-1][0]) if best else -1, best_val=fl(best[-1][1]) if best else float("nan"),
                       extra_keys=";".join(sorted(k for k in a if k not in (
                           "variant", "d", "heads", "blocks", "hist_layers", "drop", "no_hist", "no_set", "no_pos",
                           "loss", "loss_norm", "max_hist", "train_feeds", "epochs", "lr", "wd", "warmup",
                           "cand_budget", "max_feeds", "eval_every", "seed", "no_test", "aux_pos", "aux_mode",
                           "aux_tau", "ff", "sim_bias", "prand_base", "only_random", "keep", "intent", "xt", "arch",
                           "attn_gate", "retr_m", "muon", "sched", "cooldown_frac", "cooldown_mix", "aux_anneal",
                           "id_buckets", "id_drop", "id_hist_only", "xt_drop", "user0", "domain", "holdout",
                           "select_on", "save_steps", "init", "pos_mode", "time_limit")
                           and a[k] not in (False, 0, 0.0, "", None, -1, -1.0))),
                       nondefault=";".join(f"{k}={a[k]}" for k in sorted(a) if k in (
                           "aux_pl_k", "fox", "qk_norm", "ssmax", "moe", "ple", "tabm", "sid", "ema", "kd",
                           "split_kv", "age_drop", "agedrop") and a[k] not in (False, 0, 0.0, "", None)))
            if steps:
                i = max(range(len(val)), key=lambda k: val[k] if val[k] == val[k] else -1)
                row.update(max_val=val[i], max_val_step=steps[i], last_val=val[-1], last_hold=hol[-1])
                hv = [h if h == h else -1 for h in hol]
                j = max(range(len(hv)), key=lambda k: hv[k])
                row.update(max_hold=hol[j], max_hold_step=steps[j])
                if row["best_step"] in steps:
                    k = steps.index(row["best_step"])
                    row.update(sel_val=val[k], sel_hold=hol[k])
            train_rows.append(row)
            curves.append(dict(src=src, seg=si, variant=a.get("variant"), total=total,
                               steps=steps, val=val, hold=hol, loss=[fl(x[2]) for x in st],
                               lr=[fl(x[5]) for x in st], cd_reg=[fl(x[7]) for x in st], cd_pool=[fl(x[8]) for x in st]))
        elif "out" in a or "extra" in a:  # finetune.py run
            oof = [m.group(1) for m in (OOF.search(l) for l in s["lines"]) if m]
            snap = [m.group(1) for m in (SNAP.search(l) for l in s["lines"]) if m]
            ema = [m.group(1) for m in (EMA.search(l) for l in s["lines"]) if m]
            ep = [fl(x) for x in oof[-1].split(",")] if oof else []
            ft_rows.append(dict(src=src, seg=si, variant=a.get("variant"), out=a.get("out"), start=s["start"],
                                epochs=a.get("epochs"), lr=a.get("lr"), wd=a.get("wd"), warmup=a.get("warmup"),
                                cand_budget=a.get("cand_budget"), max_feeds=a.get("max_feeds"),
                                loss_norm=a.get("loss_norm"), drop=a.get("drop"), extra=a.get("extra"),
                                extra_lr_mult=a.get("extra_lr_mult"), decay=a.get("decay"), ema=a.get("ema"),
                                ema_start=a.get("ema_start"), select=a.get("select"), seeds=a.get("seeds"),
                                folds=a.get("folds"), u0=a.get("u0_singleton"), pseudo=a.get("pseudo"),
                                add_random=a.get("add_random"), from_scratch=a.get("from_scratch"),
                                oof_by_epoch=";".join(f"{x:.5f}" for x in ep),
                                ep0=ep[0] if ep else float("nan"), ep_best=max(ep) if ep else float("nan"),
                                ep_best_i=ep.index(max(ep)) if ep else -1,
                                snap=fl(snap[-1]) if snap else float("nan"), ema_oof=fl(ema[-1]) if ema else float("nan")))

keys = []
for r in train_rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(f"{OUT}/runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(train_rows)
fk = []
for r in ft_rows:
    for k in r:
        if k not in fk:
            fk.append(k)
with open(f"{OUT}/ft_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fk)
    w.writeheader()
    w.writerows(ft_rows)
with open(f"{OUT}/curves.jsonl", "w") as f:
    for c in curves:
        f.write(json.dumps(c) + "\n")
print(f"train segments {len(train_rows)} (complete {sum(r['complete'] for r in train_rows)}), "
      f"ft segments {len(ft_rows)}, files {len(SRC)}")
