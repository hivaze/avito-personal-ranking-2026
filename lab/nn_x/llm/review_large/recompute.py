"""Independent recompute of A1/A2 (reports/llm_large.md section 3) from the saved prediction files.
Own loading of labels/meta, own holdout feed selection (nn/train.py logic), src/metrics.ndcg_per_feed, own z / eq / rho.
CPU-only light analysis."""
import json
import sys

import numpy as np
import polars as pl

ROOT = "<repo>"
sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed  # noqa: E402

LLM = f"{ROOT}/nn_x/llm"

# ---------------- val context
fv = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64)
offv = fv["cand_off"].to_numpy().astype(np.int64)
assert (offv == np.concatenate([[0], np.cumsum(Mv)[:-1]])).all()
yv = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
assert len(yv) == Mv.sum()
dayv = (fv["timestamp"].to_numpy() + 3 * 3600) // 86400
dayv = np.where(dayv <= dayv.min() + 1, 0, 1)
fidv = np.repeat(np.arange(len(Mv)), Mv)
import os
luo_p = f"{ROOT}/nn/feat_cache/val/it_hist_all_luo.npy"
if not os.path.exists(luo_p):
    luo_p = f"{ROOT}/data/feats/val/it_hist_all_luo.npy"
luo = np.load(luo_p)
cold = (luo == 0).astype(np.float64)
npos = np.bincount(fidv, yv, len(Mv))
allcold = (npos > 0) & (np.bincount(fidv, cold * yv, len(Mv)) == npos)
VS = {"all": np.ones(len(Mv), bool), "M<50": Mv < 50, "M>=50": Mv >= 50, "Tue": dayv == 0, "Thu": dayv == 1,
      "allcold": allcold}

# ---------------- holdout context (independent: nn/train.py logic)
ft = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["M", "cand_off", "timestamp"])
trM = ft["M"].to_numpy().astype(np.int64)
trOff = ft["cand_off"].to_numpy().astype(np.int64)
pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[trM < 50] = 0.0
cand = np.where((pr > 0.9) & (trM >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
Mh = trM[hold]
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
ytr = np.load(f"{ROOT}/data/flat/train/cand_target.npy", mmap_mode="r")
rows = np.concatenate([np.arange(trOff[f], trOff[f] + trM[f]) for f in hold])
yh = np.asarray(ytr[rows]).astype(np.float64)
for p in (f"{LLM}/preds/holdout/holdout_y.npy", f"{LLM}/baseline/preds/holdout/holdout_y.npy",
          f"{ROOT}/preds/holdout/holdout_y.npy"):
    if os.path.exists(p):
        print("holdout_y identical to", p, ":", np.array_equal(np.load(p).astype(np.float64), yh))
for p in (f"{LLM}/preds/holdout/holdout_meta.npy", f"{LLM}/baseline/preds/holdout/holdout_meta.npy"):
    mm = np.load(p)
    print("holdout_meta identical", p, ":", np.array_equal(mm[0], Mh) and np.array_equal(mm[1], offh))
tsh = ft["timestamp"].to_numpy()[hold]
fidh = np.repeat(np.arange(len(Mh)), Mh)
HS = {"all": np.ones(len(Mh), bool), "late20%": tsh >= np.quantile(tsh, 0.8)}

# ---------------- subsets (nn/train.py logic) -- check counts and holdout exclusion
for s in (23, 24):
    rng = np.random.default_rng(s)
    feeds = np.sort(rng.choice(len(trM), 150000, replace=False))
    feeds = np.setdiff1d(feeds, hold)
    print(f"subset seed {s}: {len(feeds)} train feeds; overlap with holdout {len(np.intersect1d(feeds, hold))}")

CTX = {"val": (yv, offv, Mv, fidv, VS), "hold": (yh, offh, Mh, fidh, HS)}


def path(name, split):
    if name.startswith("llmcmp_base"):
        return (f"{LLM}/baseline/preds/nn_{name}_trainonly_val.npy" if split == "val"
                else f"{LLM}/baseline/preds/holdout/nn_{name}_holdout.npy")
    return f"{LLM}/preds/{name}_trainonly_val.npy" if split == "val" else f"{LLM}/preds/holdout/{name}_holdout.npy"


cache = {}


def scores(name, split):
    k = (name, split)
    if k not in cache:
        s = np.load(path(name, split)).astype(np.float64)
        y, off, M, fid, _ = CTX[split]
        assert len(s) == len(y), (name, split, len(s), len(y))
        assert np.isfinite(s).all(), (name, split)
        cache[k] = s
    return cache[k]


def zf(s, fid, M):
    mu = np.bincount(fid, s, len(M)) / M
    var = np.bincount(fid, (s - mu[fid]) ** 2, len(M)) / M
    return (s - mu[fid]) / (np.sqrt(var)[fid] + 1e-9)


def nd(name, split):
    y, off, M, fid, _ = CTX[split]
    return ndcg_per_feed(scores(name, split), y, off, M)


def eq(names, split):
    y, off, M, fid, _ = CTX[split]
    z = np.mean([zf(scores(n, split), fid, M) for n in names], 0)
    return ndcg_per_feed(z, y, off, M)


def corr(a, b, split):
    y, off, M, fid, _ = CTX[split]
    za, zb = zf(scores(a, split), fid, M), zf(scores(b, split), fid, M)
    return float((np.bincount(fid, za * zb, len(M)) / M).mean())


def mse(d, m=None):
    d = d if m is None else d[m]
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


B = ["llmcmp_base_s23", "llmcmp_base_s24"]
SETS = {"0.6B-4ep(E1, judged)": ["llm06_L1_e4_s23", "llm06_L1_e4_s24"],
        "0.6B-2ep(info)": ["llm06_L1_s23", "llm06_L1_s24"],
        "4B": ["llmq4_L1_s23", "llmq4_L1_s24"],
        "9B": ["llmq9_L1_s23", "llmq9_L1_s24"]}
R = "llm06rand_L1_s23"
out = {}
for n in B + [R] + [x for v in SETS.values() for x in v]:
    print(f"single {n}: val {nd(n, 'val').mean():.5f} | hold {nd(n, 'hold').mean():.5f}")
eqbb = {sp: eq(B, sp) for sp in ("val", "hold")}
print(f"rho(B23,B24) val {corr(B[0], B[1], 'val'):.4f}")
for lab, L in SETS.items():
    r = {}
    for sp in ("val", "hold"):
        d = np.mean([nd(l, sp) - nd(b, sp) for l, b in zip(L, B)], 0)
        r[f"paired_{sp}"] = {k: mse(d, m) for k, m in CTX[sp][4].items()}
        r[f"per_seed_{sp}"] = [mse(nd(l, sp) - nd(b, sp)) for l, b in zip(L, B)]
        e = np.mean([eq([l, b], sp) for l, b in zip(L, B)], 0) - eqbb[sp]
        r[f"eq_{sp}"] = mse(e)
    pv, pse = r["paired_val"]["all"]
    bad = [f"{sp}:{k}" for sp in ("val", "hold") for k, (m, se) in r[f"paired_{sp}"].items() if m < -2 * se]
    A1 = pv > 2 * pse and all(m > 0 for m, _ in r["per_seed_val"]) and r["paired_hold"]["all"][0] >= 0 and not bad
    gv = np.mean([nd(l, "val").mean() - nd(b, "val").mean() for l, b in zip(L, B)])
    gh = np.mean([nd(l, "hold").mean() - nd(b, "hold").mean() for l, b in zip(L, B)])
    A2i = gv >= -0.002 and gh >= -0.002
    em, ese = r["eq_val"]
    A2ii = em > 0 and em / ese >= 1.5 and r["eq_hold"][0] >= 0
    rhos = [corr(l, b, "val") for l, b in zip(L, B)]
    law = 0.0186 * (1 - np.mean(rhos)) - 0.378 * abs(gv)
    ctrl = {sp: mse(nd(L[0], sp) - nd(R, sp)) for sp in ("val", "hold")}
    print(f"\n== {lab}: {L}")
    for sp in ("val", "hold"):
        print(f"  paired L-B {sp}: " + "  ".join(f"{k} {m:+.5f}+-{se:.5f}" for k, (m, se) in r[f"paired_{sp}"].items()))
        print(f"    per seed: " + "  ".join(f"{m:+.5f}+-{se:.5f}" for m, se in r[f"per_seed_{sp}"]))
    print(f"  A1 {A1} (val z {pv / pse:+.2f}; slices < -2SE {bad})")
    print(f"  A2(i) gap val {gv:+.5f} hold {gh:+.5f} -> {A2i}; A2(ii) eq val {em:+.5f}+-{ese:.5f} (z {em / ese:+.2f}) "
          f"hold {r['eq_hold'][0]:+.5f}+-{r['eq_hold'][1]:.5f} -> {A2ii}; A2 {A2i and A2ii}")
    print(f"  rho(L,B) {[round(x, 4) for x in rhos]}; law {law:+.5f}")
    print(f"  control L(s23)-R: val {ctrl['val'][0]:+.5f}+-{ctrl['val'][1]:.5f} hold {ctrl['hold'][0]:+.5f}+-{ctrl['hold'][1]:.5f}")
    out[lab] = dict(A1=bool(A1), A2=bool(A2i and A2ii), val=(pv, pse), eq_val=(em, ese), gap_val=gv, gap_hold=gh,
                    rho=rhos, law=law, ctrl=ctrl, bad=bad)

# ---------------- size curve paired (info)
print("\n== size differences at 2 epochs (paired, pooled over seeds)")
pairs = {"4B-0.6B": (SETS["4B"], SETS["0.6B-2ep(info)"]), "9B-0.6B": (SETS["9B"], SETS["0.6B-2ep(info)"]),
         "9B-4B": (SETS["9B"], SETS["4B"]), "0.6B 4ep-2ep": (SETS["0.6B-4ep(E1, judged)"], SETS["0.6B-2ep(info)"])}
for k, (X, Y) in pairs.items():
    s = []
    for sp in ("val", "hold"):
        d = np.mean([nd(x, sp) - nd(y, sp) for x, y in zip(X, Y)], 0)
        s.append("%s %+.5f+-%.5f" % ((sp,) + mse(d)))
    print(f"  {k}: " + "  ".join(s))
print("\n== correlations among LLM arms (val)")
arms = ["llm06_L1_s23", R, "llmq4_L1_s23", "llmq9_L1_s23", "llm06_L1_e4_s23"]
for i in range(len(arms)):
    for j in range(i + 1, len(arms)):
        print(f"  rho({arms[i]},{arms[j]}) {corr(arms[i], arms[j], 'val'):.4f}")
print("  allcold share", allcold.mean(), allcold.sum())
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1, default=float)
