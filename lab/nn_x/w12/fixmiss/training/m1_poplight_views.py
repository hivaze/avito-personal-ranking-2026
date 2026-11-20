"""M1 (CPU, stored predictions only): what kind of popularity-light pool is the best admixture?

Base = W16 (16 held-out wide students, w5_e10_kd05w192 / kd05t25w192 s951-958); FD16 (all-data) as a second base.
Pop-light pools compared at weight 0.2 and with the m0.4_0.2 rule (0.4 on M<50, 0.2 on M>=50):
  PD4w  distilled wide  abl_e10_pop1kd05w192_s951-954  (teacher = full-view teacher_mean11)
  PD4d  distilled d128  abl_e10_pop1kd05t25_s951-954
  PC4a  CLICK-LABEL     abl_e10_pop1_s951-954          (no teacher)
  PC4b  CLICK-LABEL     abl_e10_pop1_s955-958
  PC8   CLICK-LABEL     abl_e10_pop1_s951-958
  PD4w+PC4a mix
Question for the training proposal 'popularity-light teacher for popularity-light students': does a pop-light part that is
LESS tied to the full-view teacher (click-label) add more or less as an admixture than the distilled one at equal count?
Reads: final 0.5 FT + 0.5 TO, paired per feed vs the no-pop base and vs PD4w; val, pmix-public, pmix-private, M<50, M>=50.
Also: mean within-feed correlation of each pop pool with the base pool (TO branch, val), single-member TO NDCG.
"""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np
from pathlib import Path
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix

P = Path("<repo>/preds")
fv, Mv, ov = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy")
W = pmix.weights(); big = Mv >= 50; MROW = np.repeat(Mv, Mv)
Z = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
_cache = {}
def mz(m, br):
    k = (m, br)
    if k not in _cache:
        suf = "n2cpv3s3_val_oof" if br == "ft" else "trainonly_val"
        _cache[k] = Z(np.load(P / f"nn_{m}_{suf}.npy")).astype(np.float32)
    return _cache[k]
pool = lambda ms, br: np.mean([mz(m, br) for m in ms], 0)

def fmt(d, w=None):
    if w is None:
        return f"{d.mean():+.5f}+-{d.std(ddof=1) / np.sqrt(len(d)):.5f}"
    m, se = pmix.wmean_se(d, w); return f"{m:+.5f}+-{se:.5f}"
def line(d):
    return (f"val {fmt(d)} | pub {fmt(d, W['public'])} | priv {fmt(d, W['private'])} | M<50 {fmt(d[~big])} | M>=50 {fmt(d[big])}")

S4a, S4b, S8 = range(951, 955), range(955, 959), range(951, 959)
BASES = {
    "W16": [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8],
    "FD16": [f"w5_fd10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in S8],
}
POPS = {
    "PD4w": [f"abl_e10_pop1kd05w192_s{s}" for s in S4a],
    "PD4d": [f"abl_e10_pop1kd05t25_s{s}" for s in S4a],
    "PC4a": [f"abl_e10_pop1_s{s}" for s in S4a],
    "PC4b": [f"abl_e10_pop1_s{s}" for s in S4b],
    "PC8": [f"abl_e10_pop1_s{s}" for s in S8],
    "PD4w+PC4a": [f"abl_e10_pop1kd05w192_s{s}" for s in S4a] + [f"abl_e10_pop1_s{s}" for s in S4a],
    "FDP4": [f"abl_fd10_pop1kd05w192_s{s}" for s in S4a],
}

def corr_within(a, b):
    # mean over feeds of the within-feed Pearson correlation (feeds with M >= 5)
    cs = np.empty(len(Mv))
    for f in range(len(Mv)):
        o, m = ov[f], Mv[f]
        x, y = a[o:o + m], b[o:o + m]
        sx, sy = x.std(), y.std()
        cs[f] = np.nan if sx == 0 or sy == 0 else ((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)
    return np.nanmean(cs)

# single-member strength (TO branch)
for nm, ms in POPS.items():
    if "+" in nm:
        continue
    s = [ev(mz(m, "to")).mean() for m in ms]
    print(f"member TO NDCG {nm}: mean {np.mean(s):.5f} (min {min(s):.5f} max {max(s):.5f})")
for bn, bm in BASES.items():
    bft, bto = pool(bm, "ft"), pool(bm, "to")
    e0 = ev(0.5 * Z(bft) + 0.5 * Z(bto))
    print(f"\n=== base {bn}: no-pop final val {e0.mean():.5f} | TO pool {ev(bto).mean():.5f} | FT pool {ev(bft).mean():.5f}")
    res = {}
    for pn, pm in POPS.items():
        if bn == "W16" and pn == "FDP4":
            continue
        if bn == "FD16" and pn not in ("FDP4", "PD4w", "PC4a", "PC8"):
            continue
        pft, pto = pool(pm, "ft"), pool(pm, "to")
        for rule, wv in (("w0.2", np.float32(0.2)), ("m0.4_0.2", np.where(MROW >= 50, 0.2, 0.4).astype(np.float32))):
            ft = (1 - wv) * Z(bft) + wv * Z(pft); to = (1 - wv) * Z(bto) + wv * Z(pto)
            e = ev(0.5 * Z(ft) + 0.5 * Z(to)); res[(pn, rule)] = e
            print(f"{bn}+{pn:10s} {rule:8s} final {e.mean():.5f} | vs no-pop: {line(e - e0)}")
        print(f"      corr(pop TO pool, base TO pool) {corr_within(pto, bto):.4f} | pop TO pool alone {ev(pto).mean():.5f}")
    ref = "PD4w" if bn == "W16" else "FDP4"
    for (pn, rule), e in res.items():
        if pn != ref and (ref, rule) in res:
            print(f"  {bn} {pn} minus {ref} [{rule}]: {line(e - res[(ref, rule)])}")
