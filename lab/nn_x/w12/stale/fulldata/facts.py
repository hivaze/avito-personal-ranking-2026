"""Facts about the current finals, measured from logs / checkpoints / prediction files (item "facts to verify"), the null
band s16 of the acceptance check (PREDECLARED.txt R4) and a data-value estimate. CPU only, read-only, about 3 minutes.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/fulldata/facts.py
Writes logs/facts.txt / facts.json in this directory.
"""
import ast
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/src")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

R = "<repo>"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(f"{HERE}/logs/facts.txt", "w")
RES = {}


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


def ms(d):
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def fmt(t):
    return f"{t[0]:+.5f} +- {t[1]:.5f}"


W16 = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)]
POP = [f"abl_e10_pop1kd05t25_s{s}" for s in range(951, 955)] + [f"abl_e10_pop1kd05w192_s{s}" for s in range(951, 955)]


def base(v):
    return f"{R}/nn_x/w5" if v.startswith("w5_") else f"{R}/nn_x/abl"


# ------------------------------------------------------------------ 1. what each member trained on (logs)
P("== 1. Training data of every final member (from each member's own log)")
rows = []
for v in W16 + POP:
    txt = open(f"{base(v)}/logs/{v}.log").read()
    a = ast.literal_eval(re.search(r"args: (\{.*\})", txt).group(1))
    g = lambda pat: re.search(pat, txt)  # noqa: E731
    rows.append(dict(v=v, holdout=a["holdout"], select_on=a["select_on"], keep=a["keep"], train_feeds_arg=a["train_feeds"],
                     val_mix=a["val_mix"], kd=os.path.basename(a["kd_file"]), kd_w=a["kd_w"], d=a["d"], epochs=a["epochs"],
                     training_feeds=int(g(r"training feeds (\d+)").group(1)), steps_epoch=int(g(r"steps/epoch (\d+)").group(1)),
                     total=int(g(r"total steps (\d+)").group(1)), pool=int(g(r"exploration pool (\d+) feeds").group(1)),
                     t_cool=int(g(r"from step (\d+), exploration pool").group(1)), best_step=int(g(r"BEST step (\d+)").group(1)),
                     val_sel=float(g(r"BEST step \d+ val_ndcg ([\d.]+)").group(1)),
                     val_last=float(g(r"LAST checkpoint .*? val_ndcg ([\d.]+)").group(1)),
                     hold_last=float(g(r"LAST checkpoint .*? holdout_ndcg ([\d.]+)").group(1)),
                     hold_sel=float(g(r"holdout NDCG ([\d.]+)\)").group(1)),
                     excluded="holdout: 30000 random-like train feeds excluded from training" in txt))
RES["members"] = rows
same = lambda k: sorted({r[k] for r in rows})  # noqa: E731
P(f"   {len(rows)} members (16 W16 + 8 popularity-light): --holdout {same('holdout')}, --select_on {same('select_on')}, --keep {same('keep')}, "
  f"--train_feeds {same('train_feeds_arg')}, --val_mix {same('val_mix')}, log line 'holdout: 30000 ... excluded' in all: {all(r['excluded'] for r in rows)}")
P(f"   training feeds {same('training_feeds')} of 3,151,208; steps/epoch {same('steps_epoch')}; total steps {same('total')}; cooldown from step {same('t_cool')}; "
  f"cooldown pool {same('pool')} feeds; teacher files {same('kd')}; kd_w {same('kd_w')}; d {same('d')}; epochs {same('epochs')}")
P(f"   kept (holdout-selected) step: {sorted(r['best_step'] for r in rows)}")
for nm, grp in (("W16 (16)", rows[:16]), ("pop-light (8)", rows[16:])):
    dv = np.array([r["val_sel"] - r["val_last"] for r in grp])
    dh = np.array([r["hold_sel"] - r["hold_last"] for r in grp])
    P(f"   {nm}: selected - @last per member (log values): holdout mean {dh.mean():+.5f} (min {dh.min():+.5f}, max {dh.max():+.5f}); val mean {dv.mean():+.5f} "
      f"(min {dv.min():+.5f}, max {dv.max():+.5f}); members whose selected step is the last step: {sum(r['best_step'] == r['total'] for r in grp)}")
    RES[f"sel_minus_last_{nm}"] = dict(holdout=float(dh.mean()), val=float(dv.mean()))

# ------------------------------------------------------------------ 2. pool level selected vs @last, null band s16
f = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet")
Mv = f["M"].to_numpy().astype(np.int64)
ov = f["cand_off"].to_numpy().astype(np.int64)
yv = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)
Mh, oh = (x.astype(np.int64) for x in np.load(f"{R}/nn_x/w5/preds/holdout/holdout_meta.npy"))
yh = np.load(f"{R}/nn_x/w5/preds/holdout/holdout_y.npy").astype(np.float64)


def zf(s, M):
    s = np.asarray(s, np.float64)
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-9
    return ((s - mu[fid]) / sd[fid]).astype(np.float32)


def load(v, suf, what):
    b = base(v)
    return np.load(f"{b}/preds/nn_{v}{suf}_trainonly_val.npy" if what == "val" else f"{b}/preds/holdout/nn_{v}{suf}_holdout.npy")


P("== 2. Pool level (train-only, per-feed z-mean): selected vs @last")
Z = {}
for suf in ("", "@last"):
    Z["val", suf] = np.stack([zf(load(v, suf, "val"), Mv) for v in W16])
    Z["hold", suf] = np.stack([zf(load(v, suf, "hold"), Mh) for v in W16])
nv = {s: ndcg_per_feed(Z["val", s].mean(0).astype(np.float64), yv, ov, Mv) for s in ("", "@last")}
nh = {s: ndcg_per_feed(Z["hold", s].mean(0).astype(np.float64), yh, oh, Mh) for s in ("", "@last")}
RES["w16_pool"] = dict(val_sel=float(nv[""].mean()), val_last=float(nv["@last"].mean()), hold_sel=float(nh[""].mean()), hold_last=float(nh["@last"].mean()),
                       d_val=ms(nv[""] - nv["@last"]), d_hold=ms(nh[""] - nh["@last"]))
P(f"   W16 train-only 16-pool: val selected {nv[''].mean():.5f} / @last {nv['@last'].mean():.5f} (selected - @last {fmt(ms(nv[''] - nv['@last']))}, n {len(Mv)}); "
  f"holdout selected {nh[''].mean():.5f} / @last {nh['@last'].mean():.5f} ({fmt(ms(nh[''] - nh['@last']))}, n {len(Mh)})")
mem_val = np.array([ndcg_per_feed(Z["val", "@last"][i].astype(np.float64), yv, ov, Mv).mean() for i in range(16)])
P(f"   member val NDCG @last: mean {mem_val.mean():.5f}, SD across the 16 members {mem_val.std(ddof=1):.5f} (kd05w192 {mem_val[:8].mean():.5f}, kd05t25w192 {mem_val[8:].mean():.5f})")
RES["member_val_last"] = dict(mean=float(mem_val.mean()), sd=float(mem_val.std(ddof=1)), values=mem_val.round(5).tolist())
P("== 3. Null band s16 (PREDECLARED R4): random splits of the 16 members (@last) into two disjoint 8-pools, 4 + 4 per arm")
rng = np.random.default_rng(20261115)
d8, se8 = [], []
for _ in range(60):
    a_ = np.concatenate([rng.permutation(8)[:4], 8 + rng.permutation(8)[:4]])
    b_ = np.setdiff1d(np.arange(16), a_)
    da = ndcg_per_feed(Z["val", "@last"][a_].mean(0).astype(np.float64), yv, ov, Mv) - ndcg_per_feed(Z["val", "@last"][b_].mean(0).astype(np.float64), yv, ov, Mv)
    d8.append(da.mean())
    se8.append(da.std(ddof=1) / np.sqrt(len(da)))
d8 = np.array(d8)
s8 = float(np.sqrt((d8 ** 2).mean()))
s16 = s8 / np.sqrt(2)
RES["null"] = dict(n_splits=60, s8=s8, s16=s16, band3=3 * s16, paired_feed_se_8v8=float(np.mean(se8)), max_abs_8v8=float(np.abs(d8).max()))
P(f"   60 splits: RMS of the 8-pool vs 8-pool val difference s8 = {s8:.5f} (max |d| {np.abs(d8).max():.5f}; mean paired per-feed SE of one split {np.mean(se8):.5f}); "
  f"s16 = s8 / sqrt(2) = {s16:.5f}; acceptance band 3 s16 = {3 * s16:.5f}")
P("   (the splits share members pairwise, so s8 is a slight under-estimate of the fully independent case; the per-feed SE of a split is the larger, safer number)")

# ------------------------------------------------------------------ 4. value of data: 80% of the feeds (xfold) vs 100%
P("== 4. Value of training data: hard-label base (all 3,121,208 feeds) vs the cross-fit members (--xfold k: one user fold of 5 dropped, 80% of feeds and steps)")
try:
    bs = [f"w5_e10_base_s{s}" for s in (951, 952, 953)]
    xs = [f"w5_e10_xf{k}_s{s}" for k in range(5) for s in (951, 952, 953)]
    nb_ = np.array([ndcg_per_feed(zf(load(v, "@last", "val"), Mv).astype(np.float64), yv, ov, Mv).mean() for v in bs])
    nx_ = np.array([ndcg_per_feed(zf(load(v, "@last", "val"), Mv).astype(np.float64), yv, ov, Mv).mean() for v in xs])
    pb = ndcg_per_feed(np.mean([zf(load(v, "@last", "val"), Mv) for v in bs], 0).astype(np.float64), yv, ov, Mv)
    px = [ndcg_per_feed(np.mean([zf(load(f"w5_e10_xf{k}_s{s}", "@last", "val"), Mv) for s in (951, 952, 953)], 0).astype(np.float64), yv, ov, Mv) for k in range(5)]
    dpool = np.mean([pb - x for x in px], 0)
    d20 = float(nb_.mean() - nx_.mean())
    RES["data_value"] = dict(member_base=float(nb_.mean()), member_xf=float(nx_.mean()), member_delta_20pct=d20, pool3_delta_20pct=ms(dpool),
                             per_1pct_log_law=float(d20 * np.log(1 / 0.99048) / np.log(1 / 0.8)), per_1pct_linear=float(d20 * 0.00952 / 0.2))
    P(f"   member val NDCG @last: base (3 seeds) {nb_.mean():.5f} vs xf (15 members) {nx_.mean():.5f}: 20% of the feeds (and steps) = {d20:+.5f} per member; "
      f"3-pool base vs 3-pool xf_k (mean over k): {fmt(ms(dpool))}")
    P(f"   scaled to the holdout's 0.952% of feeds: log law {RES['data_value']['per_1pct_log_law']:+.6f}, linear {RES['data_value']['per_1pct_linear']:+.6f} "
      f"(val; caveat: xfold drops whole users and 20% of the steps, the holdout feeds are exploration-like and 1.88% of candidates)")
except Exception as e:  # noqa: BLE001
    P(f"   not computed: {type(e).__name__}: {e}")

# ------------------------------------------------------------------ 5. the fine-tune half
P("== 5. Fine-tune half (n2cpv3s3) of a W16 member, from its own log")
v = W16[0]
txt = open(f"{R}/nn_x/w5/logs/{v}_n2cpv3s3_ft.log").read()
fa = ast.literal_eval(re.search(r"args: (\{.*\})", txt).group(1))
RES["ft_args"] = fa
P(f"   args: folds {fa['folds']}, seeds {fa['seeds']}, lr {fa['lr']}, epochs {fa['epochs']}, decay {fa['decay']}, ema {fa['ema']} (start {fa['ema_start']} epochs), "
  f"wd {fa['wd']}, warmup {fa['warmup']}, extra {fa['extra']}, u0_singleton {fa['u0_singleton']}, add_random {fa['add_random']}, pseudo '{fa['pseudo']}', select {fa['select']}")
for l in txt.splitlines():
    if l.startswith(("user-0", "OOF ndcg", "best epoch", "final EMA", "snapshot mean", "extra features (")):
        P("   " + l[:230])
from sklearn.model_selection import GroupKFold  # noqa: E402

uid = f["user_id"].to_numpy().astype(np.int64)
grp = np.where(uid == 0, -1 - np.arange(len(uid)), uid)
sizes = [len(tr) for tr, _ in GroupKFold(n_splits=5).split(np.arange(len(uid)), groups=grp)]
RES["ft_fold_train_sizes"] = sizes
P(f"   GroupKFold(5) by user (user-0 feeds singletons): fit sizes {sizes} of {len(uid)} val feeds ({np.mean(sizes) / len(uid):.1%} each); every val feed is in the fit set of "
  f"4 of the 5 folds -> its label enters 12 of the 15 models whose mean is the test prediction; no model sees all of val")
json.dump(RES, open(f"{HERE}/logs/facts.json", "w"), indent=1, default=str)
P("done")
