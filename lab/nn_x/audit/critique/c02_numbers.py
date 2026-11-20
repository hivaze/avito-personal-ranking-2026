"""Critique check 2: independent re-computation of decision-relevant numbers on val (CPU).
 (a) pmix weights: effective sample size of the public/private reweighting (sol_advval: coverage weights with ESS 2,880 were ruled invalid).
 (b) val -> public calibration line over the 9 public-scored submissions; where #1 / #2 map on val.
 (c) G3 branches: blend vs TO vs FT paired deltas (all / M<50 / M>=50 / pmix-weighted), and the public-projected paired SE.
 (d) same-user later-feed overlay on G3 (val+test later feeds; val-only later feeds).
 (e) per-feed NDCG sd of G3 and the SE of a public score difference against a dissimilar model (what the 0.0054 gap to #1 is worth)."""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ.setdefault(k, "12")
sys.path.insert(0, "<repo>/src")
import numpy as np
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
import pmix, later_overlay as LO

P = "<repo>/preds/"
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
nval = len(M); npub = 8562; npriv = 20033
fid = np.repeat(np.arange(nval), M)

def se(d, w=None):
    return pmix.wmean_se(d, w)

# (a) ESS of the pmix weights
W = pmix.weights()
for k in ("public", "private"):
    w = W[k]; ess = w.sum() ** 2 / (w ** 2).sum()
    print(f"pmix {k}: ESS {ess:.0f} of {nval} ({ess/nval:.2f}); max w {w.max():.2f}, share of weight on top 5% feeds {np.sort(w)[::-1][:int(0.05*nval)].sum()/w.sum():.2f}")

# (b) calibration line
pts = {"v3": (0.3816, 0.3739), "v4": (0.3839, 0.3772), "v5": (0.3869, 0.3802), "v10": (0.3940, 0.38786),
       "v13_Cd": (0.39901, 0.39208), "PF7x": (0.39974, 0.392723), "M2": (0.40094, 0.392434), "G3": (0.40032, 0.392819),
       "S2": (0.40088, 0.392473)}
v = np.array([p[0] for p in pts.values()]); pb = np.array([p[1] for p in pts.values()])
A = np.vstack([v, np.ones_like(v)]).T; slope, icpt = np.linalg.lstsq(A, pb, rcond=None)[0]
res = pb - (slope * v + icpt)
print(f"calibration public = {icpt:+.4f} + {slope:.3f} * val over {len(v)} points; residual sd {res.std(ddof=2):.5f}; residuals: " + ", ".join(f"{k} {r:+.4f}" for k, r in zip(pts, res)))
# the last 5 (modern pools) only
v5, p5 = v[-5:], pb[-5:]; A5 = np.vstack([v5, np.ones_like(v5)]).T; s5, i5 = np.linalg.lstsq(A5, p5, rcond=None)[0]
print(f"  modern pools only (5 pts): slope {s5:.3f}; public-val offset mean {np.mean(p5 - v5):+.5f} sd {np.std(p5 - v5, ddof=1):.5f}")
for name, pub in (("#1", 0.398256), ("#2", 0.394695), ("#3", 0.393767)):
    print(f"  {name} public {pub:.6f} -> val-equivalent {(pub - icpt)/slope:.4f} (line) / {pub - np.mean(p5 - v5):.4f} (offset); gap to G3 on val-scale {(pub - icpt)/slope - 0.40032:+.4f}")

# (c) branches
to = np.load(P + "ens_G3_to_val.npy").astype(np.float64); ft = np.load(P + "ens_G3_ft_val.npy").astype(np.float64)
def z(s):
    mu = np.bincount(fid, s, nval) / M; sd = np.sqrt(np.maximum(np.bincount(fid, s * s, nval) / M - mu ** 2, 1e-12))
    return (s - mu[fid]) / sd[fid]
zto, zft = z(to), z(ft)
nd = {}
for w_ft in (0.0, 0.25, 0.5, 0.75, 1.0):
    nd[w_ft] = ndcg_per_feed(((1 - w_ft) * zto + w_ft * zft).astype(np.float32), y, off, M)
    print(f"w_ft {w_ft:.2f}: val {nd[w_ft].mean():.5f}")
try:
    fin = np.load(P + "ens_G3_final_val.npy"); nfin = ndcg_per_feed(fin, y, off, M); print(f"ens_G3_final_val.npy: {nfin.mean():.5f} (corr with blend per-feed {np.corrcoef(nfin, nd[0.5])[0,1]:.4f})")
except Exception as e:
    print("final_val:", e)
small = M < 50
for a, b, lab in ((0.5, 0.0, "blend - TO"), (0.5, 1.0, "blend - FT"), (1.0, 0.0, "FT - TO"), (0.25, 0.5, "w.25 - blend")):
    d = nd[a] - nd[b]
    m, s = se(d); mp, sp = se(d, W["private"]); mq, sq = se(d, W["public"])
    print(f"{lab:12s}: all {m:+.5f} ± {s:.5f} | M<50 {d[small].mean():+.5f} ± {d[small].std()/np.sqrt(small.sum()):.5f} | M>=50 {d[~small].mean():+.5f} ± {d[~small].std()/np.sqrt((~small).sum()):.5f} | pmix-private {mp:+.5f} ± {sp:.5f} | pmix-public {mq:+.5f} ± {sq:.5f} | public-projected paired SE {d.std()/np.sqrt(npub):.5f}")

# (d) overlay on G3 (blend)
base = (0.5 * zto + 0.5 * zft).astype(np.float32)
p0 = nd[0.5]
sets_vt = LO.later_sets("val", ("test",)); new, nf, nfl = LO.overlay("val", base, sets_vt); p1 = ndcg_per_feed(new, y, off, M)
aff = np.zeros(nval, bool); aff[list(sets_vt)] = True; d = p1 - p0
print(f"overlay (val+test later feeds): delta {d.mean():+.5f} ± {d.std()/np.sqrt(nval):.5f}; feeds with later feed {len(sets_vt)}, with flagged cands {nf}, flagged {nfl}; affected {p0[aff].mean():.3f} -> {p1[aff].mean():.3f}; per-affected-feed gain (flagged feeds) {d[d != 0].mean():+.3f} ± {d[d != 0].std()/np.sqrt((d != 0).sum()):.3f} (n {(d != 0).sum()})")
sets_v = LO.later_sets("val"); new2, nf2, nfl2 = LO.overlay("val", base, sets_v); p2 = ndcg_per_feed(new2, y, off, M); d2 = p2 - p0
print(f"overlay (val-only later feeds): delta {d2.mean():+.5f} ± {d2.std()/np.sqrt(nval):.5f}; feeds with flagged cands {nf2}, flagged {nfl2}; per-affected-feed {d2[d2 != 0].mean():+.3f} (n {(d2 != 0).sum()})")
g = d[d != 0]
print(f"public projection with 57 flagged feeds of 8562: {57 * g.mean() / npub:+.5f}, SE from per-feed spread {np.sqrt(57) * g.std() / npub:.5f}; private 46/20033: {46 * g.mean() / npriv:+.5f}")

# (e) per-feed sd and the SE of a public gap vs a dissimilar model
sd_g3 = p0.std()
print(f"G3 per-feed NDCG sd {sd_g3:.4f}; single-score SE on public (unpaired) {sd_g3/np.sqrt(npub):.5f}")
for nm in ("nn_final2_trainonly_val.npy", "lgb_tr300k_none_val.npy"):
    try:
        q = ndcg_per_feed(np.load(P + nm), y, off, M); r = np.corrcoef(q, p0)[0, 1]; dd = q - p0
        print(f"  vs {nm}: mean {q.mean():.5f}, per-feed corr with G3 {r:.3f}, paired sd {dd.std():.4f} -> public paired SE {dd.std()/np.sqrt(npub):.5f}")
    except Exception as e:
        print("  ", nm, e)
for rho in (0.98, 0.95, 0.9, 0.85, 0.8):
    print(f"  at per-feed corr {rho:.2f}: public paired SE of a score difference {sd_g3*np.sqrt(2*(1-rho))/np.sqrt(npub):.5f}")
print("done c02")
