"""R4: A/A null of pool-vs-pool contrasts of same-recipe distilled students (train-only val), overall and on the cap bucket.
Gives the paired per-feed SE and the sd of the A/A delta (seed noise) for 4-vs-4 and 8-vs-8 pools -> power of a z >= 2 rule."""
import itertools
from math import erf, sqrt
from rv_lib import *

Zm = {(a, s): z(np.load(P / f"nn_w5_e10_{a}_s{s}_trainonly_val.npy")) for a in ("kd05w192", "kd05t25w192") for s in S8}
rng = np.random.default_rng(5)
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))


def read(A, B):
    d = ndcg(np.mean([Zm[k] for k in A], 0)) - ndcg(np.mean([Zm[k] for k in B], 0))
    mu, se, _ = cmean(d)
    mc, sc, _ = cmean(d, CAP)
    i, si = cdiff(d, CAP, ~CAP)
    return mu, se, mc, sc, i, si


out = {}
# 4 vs 4 inside one recipe (35 distinct splits per recipe)
res = []
for a in ("kd05w192", "kd05t25w192"):
    for c in itertools.combinations(S8, 4):
        if 951 not in c:
            continue
        A = [(a, s) for s in c]
        B = [(a, s) for s in S8 if s not in c]
        res.append(read(A, B))
out["4 vs 4 (same recipe, 70 splits)"] = np.array(res)
# 8 vs 8: 4 + 4 of each recipe on both sides (200 random splits)
res = []
seen = set()
while len(res) < 200:
    c1 = tuple(sorted(rng.choice(S8, 4, replace=False).tolist()))
    c2 = tuple(sorted(rng.choice(S8, 4, replace=False).tolist()))
    if (c1, c2) in seen:
        continue
    seen.add((c1, c2))
    A = [("kd05w192", s) for s in c1] + [("kd05t25w192", s) for s in c2]
    B = [("kd05w192", s) for s in S8 if s not in c1] + [("kd05t25w192", s) for s in S8 if s not in c2]
    res.append(read(A, B))
out["8 vs 8 (4 + 4 of each recipe per side, 200 splits)"] = np.array(res)
for k, r in out.items():
    print(f"\n### A/A {k}")
    print(f"  overall: paired cluster SE of the delta {r[:, 1].mean():.5f}; sd of the A/A delta {r[:, 0].std():.5f}; sd of z {(r[:, 0] / r[:, 1]).std():.2f}; P(|z| >= 2) {np.mean(np.abs(r[:, 0] / r[:, 1]) >= 2):.3f}")
    print(f"  cap bucket (28.95% of feeds): paired cluster SE {r[:, 3].mean():.5f}; sd of the A/A delta {r[:, 2].std():.5f}; sd of z {(r[:, 2] / r[:, 3]).std():.2f}; P(|z| >= 2) {np.mean(np.abs(r[:, 2] / r[:, 3]) >= 2):.3f}")
    print(f"  interaction cap minus rest: SE {r[:, 5].mean():.5f}; sd of the A/A interaction {r[:, 4].std():.5f}; sd of z {(r[:, 4] / r[:, 5]).std():.2f}; P(|z| >= 2) {np.mean(np.abs(r[:, 4] / r[:, 5]) >= 2):.3f}")
    se, sdz = r[:, 1].mean(), (r[:, 0] / r[:, 1]).std()
    sec, sdzc = r[:, 3].mean(), (r[:, 2] / r[:, 3]).std()
    for eff in (0.0005, 0.0010):
        print(f"  power of 'overall z >= 2' at a true pool effect {eff:+.4f}: {1 - Phi((2 - eff / se) / max(sdz, 1e-9)):.2f} (z of the arm ~ N(effect / SE, sd_AA(z)))")
    for effc in (0.0010, 0.0022, 0.0027):
        print(f"  power of 'cap-bucket z >= 2' at a true cap effect {effc:+.4f}: {1 - Phi((2 - effc / sec) / max(sdzc, 1e-9)):.2f}")
print("\nNote: the A/A sd of z is the seed-noise part only (same feeds on both sides); a new arm's z also carries the feed-sampling part (sd 1), so the sd used for the power above is a LOWER bound and the powers are upper bounds.")
# with feed sampling: total sd of z ~ sqrt(1 + sd_AA(z)^2)
for k, r in out.items():
    se, sdz = r[:, 1].mean(), np.sqrt(1 + (r[:, 0] / r[:, 1]).std() ** 2)
    sec, sdzc = r[:, 3].mean(), np.sqrt(1 + (r[:, 2] / r[:, 3]).std() ** 2)
    print(f"{k}: with the feed-sampling part (sd of z = sqrt(1 + sd_AA^2) = {sdz:.2f} overall / {sdzc:.2f} cap): power overall +0.0005 -> {1 - Phi((2 - 0.0005 / se) / sdz):.2f}, +0.0010 -> {1 - Phi((2 - 0.0010 / se) / sdz):.2f}; "
          f"cap +0.0022 -> {1 - Phi((2 - 0.0022 / sec) / sdzc):.2f}, cap +0.0010 -> {1 - Phi((2 - 0.0010 / sec) / sdzc):.2f}; false-pass rate under no effect {1 - Phi(2 / sdz):.3f} / {1 - Phi(2 / sdzc):.3f}")
