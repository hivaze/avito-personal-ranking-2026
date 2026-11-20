"""Critic check 2 (CPU): item 1 by M bucket across member sets, and the expected delta under the TEST feed-size mix.
Reuses c1_recheck's own z / NDCG (imported, no shared project code beyond pmix)."""
import sys
sys.path.insert(0, "<repo>/lab/nn_x/w12/fixmiss/critic_training")
import io, contextlib
import numpy as np, polars as pl




from c1_lib import *  # noqa

ft_ = pl.read_parquet(FL / "test/feeds.parquet"); Mt = ft_["M"].to_numpy(); ts = ft_["timestamp"].to_numpy()
import pmix as _pm
BOUND = _pm.BOUND
pub = ts < BOUND
edges = [2, 30, 50, 75, 100, 150, 300, 10 ** 6]
print(f"test feeds {len(Mt)} (public {pub.sum()}, private {(~pub).sum()}); M mean val {Mv.mean():.1f} public {Mt[pub].mean():.1f} private {Mt[~pub].mean():.1f}")
print("bucket shares  val / public / private:")
for a, b in zip(edges[:-1], edges[1:]):
    print(f"  M [{a},{b}): {((Mv >= a) & (Mv < b)).mean():.3f} / {((Mt[pub] >= a) & (Mt[pub] < b)).mean():.3f} / {((Mt[~pub] >= a) & (Mt[~pub] < b)).mean():.3f}")

sets = {
    "FD16+FDP4 (file)": (FD16, FDP4, FT),
    "W16+PD4w": (W16, PD4w, FT),
    "W16+PC4a(1-seed FT)": (W16, [f"abl_e10_pop1_s{s}" for s in S4], "n2cpv3"),
    "FD16+PC4a(1-seed FT)": (FD16, [f"abl_e10_pop1_s{s}" for s in S4], "n2cpv3"),
    "W16+PD4d": (W16, [f"abl_e10_pop1kd05t25_s{s}" for s in S4], FT),
}
D = {}
for nm, (b, p, ftsuf) in sets.items():
    Fb, Fp, Tb, Tp = pool(b, FT), pool(p, ftsuf), pool(b, TO), pool(p, TO)
    c = ndcg(final(Fb, Fp, Tb, Tp)); d = ndcg(final(Fb, np.where(MROW >= 50, Tp, Fp), Tb, Tp)) - c
    D[nm] = d
    row = []
    for a, bb in zip(edges[2:-1], edges[3:]):
        m = (Mv >= a) & (Mv < bb)
        row.append(f"{d[m].mean():+.5f}")
    # expected delta under the test mix: bucket means weighted by test bucket shares (M>=50 buckets only; M<50 is 0 by construction)
    exp = {}
    for tag, mm in (("public", Mt[pub]), ("private", Mt[~pub])):
        e = 0.0; v = 0.0
        for a, bb in zip(edges[2:-1], edges[3:]):
            m = (Mv >= a) & (Mv < bb)
            if m.sum() == 0: continue
            sh = ((mm >= a) & (mm < bb)).mean(); e += sh * d[m].mean(); v += sh ** 2 * d[m].var(ddof=1) / m.sum()
        exp[tag] = (e, np.sqrt(v))
    print(f"{nm:24s} val {d.mean():+.5f} | M>=50 buckets [50,75) [75,100) [100,150) [150,300): {' '.join(row)} | "
          f"test-mix expectation public {exp['public'][0]:+.5f}+-{exp['public'][1]:.5f} private {exp['private'][0]:+.5f}+-{exp['private'][1]:.5f}")
ks = list(D)
print("per-feed corr of item-1 deltas between member sets (M>=50 feeds):")
for i in range(len(ks)):
    print("  " + ks[i][:22].ljust(22) + " " + " ".join(f"{np.corrcoef(D[ks[i]][big], D[ks[j]][big])[0, 1]:.2f}" for j in range(len(ks))))
