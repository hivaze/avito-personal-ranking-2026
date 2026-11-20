"""Randomization read of a re-cool arm against the reference re-cools (round T1117, rule in the ledger).

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
      .venv/bin/python nn_x/w12/gap1117/perm_read.py <arm> [ref=ref] [patterns=2000] [feeds=all|ext]

Members: the K (kd05w192) and L (kd05t25w192) re-cools of seeds 951-958 whose arm AND reference runs are
complete; pair i = (arm re-cool, reference re-cool) of the same stage and seed. Pools are the mean of the
members' within-feed rank-z (train-only val predictions, selected holdout predictions), as in arm_read8.py.

Statistic: pooled z = (z_val + z_holdout) / sqrt(2), z = mean per-feed NDCG difference / its per-feed SE.
Null: inside each pair the arm and the reference are exchangeable. A pattern swaps the two members of a
random subset of pairs and recomputes both pools; p = (1 + #{pattern z >= observed z}) / (1 + patterns).
This includes the training noise of the pools, which the per-feed SE leaves out.
feeds=ext restricts both splits to the feeds that the longer-history arm extends (lane B's bucket read).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/recheck/pool_composition")
from pc_common import Meta, ndcg_fast, paths  # noqa: E402

ROOT = Path("<repo>")
arm = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else "ref"
B = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
which = sys.argv[4] if len(sys.argv) > 4 else "all"
m = Meta()
name = lambda a, st, x: f"w5_{st}cd_{a}_s{x}"
done = lambda v: all(p.exists() for p in paths(v)[:2]) and (ROOT / f"nn_x/w5/logs/{v}.json").exists()
pairs = [(name(arm, st, x), name(ref, st, x)) for st in "kl" for x in range(951, 959)]
pairs = [(a, r) for a, r in pairs if done(a) and done(r)]
if not pairs:
    sys.exit(f"no completed ({arm}, {ref}) pair yet")
n = len(pairs)


def zs(v):
    pv, ph = paths(v)[:2]
    return m.z("val", np.load(pv)), m.z("hold", np.load(ph))


A = [zs(a) for a, _ in pairs]
R = [zs(r) for _, r in pairs]
mask = {"val": np.ones(len(m.Mv), bool), "hold": np.ones(len(m.Mh), bool)}
if which == "ext":
    ext = ROOT / "lab/nn_x/w12/gap1117/lifelong/ext_feeds.json"
    e = json.load(open(ext))                       # {"val": [feed indices], "hold": [holdout feed indices]}
    for sp in ("val", "hold"):
        mask[sp] = np.zeros_like(mask[sp]); mask[sp][np.asarray(e[sp], int)] = True


def pooled_z(flip: np.ndarray):
    out = {}
    for k, sp, y, off, M in ((0, "val", m.yv, m.ov, m.Mv), (1, "hold", m.yh, m.oh, m.Mh)):
        pa = np.mean([R[i][k] if flip[i] else A[i][k] for i in range(n)], 0)
        pr = np.mean([A[i][k] if flip[i] else R[i][k] for i in range(n)], 0)
        d = (ndcg_fast(pa, y, off, M) - ndcg_fast(pr, y, off, M))[mask[sp]]
        out[sp] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
    zv, zh = out["val"][0] / out["val"][1], out["hold"][0] / out["hold"][1]
    return out, (zv + zh) / np.sqrt(2)


obs, z_obs = pooled_z(np.zeros(n, bool))
rng = np.random.default_rng(20261117)
null = np.array([pooled_z(rng.random(n) < 0.5)[1] for _ in range(B)])
p = (1 + (null >= z_obs).sum()) / (1 + B)
res = dict(arm=arm, ref=ref, feeds=which, pairs=n, val=obs["val"], holdout=obs["hold"], pooled_z=float(z_obs),
           null_sd=float(null.std()), null_q975=float(np.quantile(null, 0.975)), p_perm=float(p), patterns=B)
print(f"== {arm} vs {ref} ({n} pairs, feeds={which})")
print(f"   val     {obs['val'][0]:+.5f} +- {obs['val'][1]:.5f}")
print(f"   holdout {obs['hold'][0]:+.5f} +- {obs['hold'][1]:.5f}")
print(f"   pooled z {z_obs:+.2f} | null sd {null.std():.2f}, 97.5% {np.quantile(null, 0.975):.2f} | p_perm {p:.4f} ({B} patterns)")
out = ROOT / f"nn_x/w12/gap1117/reads/{arm}_vs_{ref}_{which}.json"
out.parent.mkdir(exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
