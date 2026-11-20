"""veto (resume run): independent re-check of a SAMPLE of the first run's numbers before they are trusted.
Independent path: no vlib import. z by np.add.reduceat in float64, NDCG by src/metrics.ndcg_per_feed (the organisers' evaluator in
pure numpy, not the numba copy), pmix by src/pmix.weights() / wmean_se. Compared with out/read{1,2,4}.json of the first run.
  cd <repo> && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 ... .venv/bin/python nn_x/w12/stale/veto/recheck.py
Output: logs/recheck.txt (stdout)
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path("<repo>"); HERE = ROOT / "nn_x/w12/stale/veto"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eda/errors/segments"))
from metrics import ndcg_per_feed
import pmix

fv = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy().astype(np.int64)
yv = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
Mh, oh = np.load(ROOT / "nn_x/w5/preds/holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(ROOT / "nn_x/w5/preds/holdout/holdout_y.npy").astype(np.float64)
W = pmix.weights()
luo = np.nan_to_num(np.asarray(np.load(ROOT / "nn/feat_cache/val/it_hist_all_luo.npy"), np.float64), nan=0.0)
npos = np.add.reduceat(yv, ov); ncold = np.add.reduceat(yv * (luo == 0), ov)
coldpos = (npos > 0) & (ncold == npos)
DIRS = [ROOT / "nn_x/w5/preds", ROOT / "nn_x/abl/preds", ROOT / "preds", ROOT / "nn_x/r6/preds"]


def path(v, kind):
    rel = {"val": f"nn_{v}_trainonly_val.npy", "hold": f"holdout/nn_{v}_holdout.npy", "stale": f"stale/nn_{v}_staleval.npy"}[kind]
    for d in DIRS:
        if (d / rel).exists():
            return d / rel
    raise FileNotFoundError(rel)


def z(x, off, M):
    x = np.asarray(x, np.float64)
    mu = np.add.reduceat(x, off) / M
    var = np.add.reduceat(x * x, off) / M - mu ** 2
    sd = np.sqrt(np.maximum(var, 0)) + 1e-9
    return (x - np.repeat(mu, M)) / np.repeat(sd, M)


def pool(spec, kind):
    off, M = (oh, Mh) if kind == "hold" else (ov, Mv)
    if isinstance(spec[0], tuple):
        return sum(w * z(np.mean([z(np.load(path(v, kind)), off, M) for v in ns], 0), off, M) for w, ns in spec)
    return np.mean([z(np.load(path(v, kind)), off, M) for v in spec], 0)


def nd(s, kind):
    # scores are rounded to float32 first, as the stored pipeline does (ties are defined on float32 values)
    s = np.asarray(s, np.float32)
    return ndcg_per_feed(s, yh, oh, Mh) if kind == "hold" else ndcg_per_feed(s, yv, ov, Mv)


def ms(d, w=None):
    m, se = pmix.wmean_se(d, w)
    return float(m), float(se)


rng = lambda p, a, b: [f"{p}{i}" for i in range(a, b + 1)]
F10 = rng("v23_gru_pre2s_s", 501, 506); F7x = rng("r6_cd_m02_r", 101, 107); F7x70 = rng("r6_cd_m02_p70k_r", 101, 107); G3 = F10 + F7x + F7x70
POP1 = rng("abl_e10_pop1_s", 951, 958)
W192 = rng("w5_e10_kd05w192_s", 951, 958); T192 = rng("w5_e10_kd05t25w192_s", 951, 958); W8 = W192[:4] + T192[:4]; W16 = W192 + T192
P1KD = rng("abl_e10_pop1kd05t25_s", 951, 954)
N8 = rng("w5_e10_kd05_s", 951, 954) + rng("w5_e10_kd05t25_s", 951, 954)
HZ = rng("w5_cd_hzcd_r", 101, 104); REF = rng("w5_cd_m02ref_r", 101, 104)
last = lambda ns: [v + "@last" for v in ns]

OLD = {}
for k in (1, 2, 4):
    for r in json.load(open(HERE / f"out/read{k}.json")):
        if isinstance(r, dict) and "label" in r:
            OLD.setdefault(r["label"], r)


def old(sub, key):
    for lab, r in OLD.items():
        if sub in lab and key in r and r[key] is not None:
            return r[key], lab
    return None, None


worst = 0.0


def cmp(label, sub, A, B, kinds):
    global worst
    print(f"== {label}")
    for kind, key in kinds:
        base = {"val": "val", "pmix_priv": "val", "coldpos": "val", "stale": "stale", "hold": "hold"}[key]
        d = nd(pool(A, base), base) - nd(pool(B, base), base)
        if key == "pmix_priv": new = ms(d, W["private"])
        elif key == "coldpos": new = ms(d[coldpos])
        else: new = ms(d)
        o, lab = old(sub, key)
        if o is None:
            print(f"   {key:9s} recheck {new[0]:+.5f} +- {new[1]:.5f} | first run: not found under '{sub}'"); continue
        dm, ds = abs(new[0] - o[0]), abs(new[1] - o[1]); worst = max(worst, dm, ds)
        print(f"   {key:9s} recheck {new[0]:+.6f} +- {new[1]:.6f} | first run {o[0]:+.6f} +- {o[1]:.6f} | |diff| {dm:.1e} / {ds:.1e}  [{lab[:60]}]")
    sys.stdout.flush()


K = lambda *ks: [(k, k) for k in ks]
cmp("1. hzcd x4 vs m02ref x4 [selected]", "hzcd x4 vs m02ref x4 [selected]", HZ, REF, K("val", "pmix_priv", "coldpos", "stale", "hold"))
cmp("2. Q2a8-TO G3 + pop1 x8 vs G3", "Q2a8-TO", G3 + POP1, G3, K("val", "pmix_priv", "hold"))
cmp("3. 0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8", "0.8 z(W8) + 0.2 z(pop1kd d128 x4) vs W8", [(0.8, W8), (0.2, P1KD)], W8, K("val", "pmix_priv", "coldpos", "hold"))
cmp("4. W16 selected vs W16 @last (pool)", "W16: S - L (16 wide students)", W16, last(W16), K("val", "pmix_priv", "hold"))
cmp("5. teacher_mean25 students x12 vs teacher_mean11 twins x12", "POOLED: all t25 students", rng("w5_e10_kd05t25_s", 951, 954) + T192,
    rng("w5_e10_kd05_s", 951, 954) + W192, K("val", "pmix_priv", "coldpos", "hold"))
cmp("6. W8 (8 wide) vs 8 narrow students", "narrow", W8, N8, K("val", "pmix_priv", "coldpos", "hold"))
print(f"\nlargest |difference| between the re-check and the first run over all compared means and SEs: {worst:.2e}")
print("labels of the first run that were matched are printed in brackets; a 'not found' line is a re-check without a stored counterpart.")
