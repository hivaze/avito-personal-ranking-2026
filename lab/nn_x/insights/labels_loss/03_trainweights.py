"""(1b) How the TRAINING loss weights feeds vs how the METRIC weights them.
- Composition: train (production / random-like) vs val vs test (public/private) by M bin and H bin.
- Effective per-feed weights under the live recipe (nn/train.py; M-bucketed batches, B = min(512, 32768 // Mpad)):
  main CE  (loss_norm feed): w_main = 1/512 for every feed, so the batch total is B/512;
  K9 aux (always batch-mean): w_aux_f = aux_w (1-p_f) / sum_batch (1-p)  -> ~aux_w / B per production feed;
  m02 pool batches (p>0.5, M>=50): aux still sums to aux_w over the few production-like rows (REVIEW P8).
- Metric-ideal weight per M bin from 02 (Sm/Sl on val and holdout) x test/train composition, and a second-order
  gain estimate calibrated on the K1 result (batch-mean -> feed: +0.0020 val).
Writes out/03_trainweights.txt"""
import sys, os
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta

OUT = "<repo>/nn_x/insights/labels_loss/out"
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

ft, Mt, _ = feed_meta("train"); fv, Mv, _ = feed_meta("val"); fs, Ms, _ = feed_meta("test")
Ht, Hv, Hs = ft["H"].to_numpy(), fv["H"].to_numpy(), fs["H"].to_numpy()
pub = fs["timestamp"].to_numpy() <= 1790497920
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[Mt < 50] = 0
cand = np.where((p > 0.9) & (Mt >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
tr = np.ones(len(Mt), bool); tr[hold] = False
pool = tr & (p > 0.5)
P(f"train feeds (holdout excluded) {tr.sum()}, pool (p>0.5, M>=50) {pool.sum()}, holdout {len(hold)}")

MB = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 301]
def share(M, m=None):
    m = np.ones(len(M), bool) if m is None else m
    return np.array([((M >= a) & (M < b) & m).sum() for a, b in zip(MB[:-1], MB[1:])]) / m.sum()

# effective weights per feed
Mp = (Mt + 15) // 16 * 16
B = np.minimum(512, 32768 // Mp)
w_main = np.where(tr, 1 / 512, 0.0)
# aux: per M-bucket, mean (1-p) over regular batches -> per-feed aux weight (1-p_f)/(B * mean(1-p))
q = 1 - p
w_aux = np.zeros(len(Mt))
for mp in np.unique(Mp):
    m = tr & (Mp == mp)
    if m.sum() == 0: continue
    w_aux[m] = q[m] / (B[m] * q[m].mean())
# per-epoch totals (aux_w = 1 for shares): steps per epoch = sum over buckets ceil(n/B)
nb = sum(int(np.ceil((tr & (Mp == mp)).sum() / max(1, min(512, 32768 // mp))))
         for mp in np.unique(Mp))
P(f"steps/epoch (approx) {nb}")
P("\nM bin | train% | prod% | randlike% | val% | test% | public% | private% | B (feeds/batch) | main w share% | "
  "aux w share% | aux/main per-feed ratio (rel) | test/train")
st = share(Mt, tr); sp = share(Mt, tr & (p <= 0.5)); sr = share(Mt, pool); sv = share(Mv); ss = share(Ms)
spub = share(Ms, pub); spri = share(Ms, ~pub)
mains, auxs = [], []
for i, (a, b) in enumerate(zip(MB[:-1], MB[1:])):
    m = tr & (Mt >= a) & (Mt < b)
    mains.append(w_main[m].sum()); auxs.append(w_aux[m].sum())
mains = np.array(mains) / sum(mains); auxs = np.array(auxs) / sum(auxs)
ratio_all = auxs.sum() / mains.sum()
for i, (a, b) in enumerate(zip(MB[:-1], MB[1:])):
    m = tr & (Mt >= a) & (Mt < b)
    P(f"[{a},{b}) | {st[i]*100:5.1f} | {sp[i]*100:5.1f} | {sr[i]*100:5.1f} | {sv[i]*100:5.1f} | {ss[i]*100:5.1f} | "
      f"{spub[i]*100:5.1f} | {spri[i]*100:5.1f} | {np.median(B[m]):.0f} | {mains[i]*100:5.1f} | {auxs[i]*100:5.1f} | "
      f"{(auxs[i]/mains[i]):.2f} | {ss[i]/st[i]:.2f}")

HB = [(0, 1), (1, 32), (32, 128), (128, 256), (256, 257)]
P("\nH bin | train% | val% | test%")
for a, b in HB:
    P(f"[{a},{b}) | {((Ht>=a)&(Ht<b)&tr).sum()/tr.sum()*100:5.1f} | {((Hv>=a)&(Hv<b)).mean()*100:5.1f} | "
      f"{((Hs>=a)&(Hs<b)).mean()*100:5.1f}")

# m02 pool batches: aux renormalisation
Mpp = Mp[pool]; qp = q[pool]
P("\nm02 pool batches (cooldown, 2% of steps): per batch the aux still sums to aux_w; production mass sum(1-p) per batch:")
for mp in [64, 96, 128, 192, 256, 304]:
    m = Mpp == mp
    if m.sum() == 0: continue
    Bb = min(512, 32768 // mp)
    reg = tr & (Mp == mp) & (p <= 0.5)
    P(f"  Mpad {mp}: B {Bb}, mean(1-p) pool {qp[m].mean():.3f} vs regular {q[tr & (Mp == mp)].mean():.3f} -> "
      f"aux weight per unit of production mass x{q[tr & (Mp == mp)].mean()/max(qp[m].mean(),1e-6):.1f}")
P(f"  pool mean (1-p) = {qp.mean():.3f}; share of pool feeds with p>0.9: {(p[pool]>0.9).mean():.3f}")

# exploration share of the main loss
P(f"\nexploration-like share of main-loss feed weight: plateau {(w_main[pool]).sum()/w_main.sum():.4f} "
  f"(p-weighted {(w_main*p).sum()/w_main.sum():.4f}); test = 1.0 (all exploration)")
Bpool = np.minimum(512, 32768 // Mpp).mean(); Ball = (len(Mt) / nb)
P(f"  mean feeds per pool batch {Bpool:.0f}, per regular batch {Ball:.0f}; m02 adds ~{0.02*Bpool/Ball:.4f} of feed weight "
  f"in the cooldown; m10 ~{0.1*Bpool/Ball:.4f}")

# metric-ideal weights by M bin from 02 (val + holdout Sm/Sl), combined with composition
sv_ratio = {"[5,10)": 1.33, "[10,20)": 1.25, "[20,30)": 1.19, "[30,50)": 1.13, "[50,75)": 1.02, "[75,100)": 0.98,
            "[100,150)": 0.88, "[150,200)": 0.83, "[200,250)": 0.73, "[250,301)": 0.71}
P("\nmetric-ideal per-feed weight w*(M) = Sm/Sl(val) x test/train share (normalised to train-weighted mean 1); "
  "current main = 1; per-batch-mean (pre-K1) = 512/B")
lab = [f"[{a},{b})" for a, b in zip(MB[:-1], MB[1:])]
smsl = np.array([sv_ratio[l] for l in lab])
comp = ss / st
wstar = smsl * comp
wstar = wstar / (wstar * st).sum()
wsens = smsl / (smsl * st).sum()
wbm = np.array([np.median(512 / B[tr & (Mt >= a) & (Mt < b)]) for a, b in zip(MB[:-1], MB[1:])])
wbm = wbm / (wbm * st).sum()
for i, l in enumerate(lab):
    P(f"{l} | Sm/Sl {smsl[i]:.2f} | test/train {comp[i]:.2f} | w* {wstar[i]:.2f} | w*(sens only) {wsens[i]:.2f} | "
      f"batch-mean {wbm[i]:.2f}")
# second-order calibration: gain ~ c * E_test[(log w - log w*)^2]; K1 moved batch-mean -> feed (+0.0020 val)
def dev(w, ref):
    return float((ss * (np.log(w) - np.log(ref)) ** 2).sum())
d_bm, d_feed = dev(wbm, wstar), dev(np.ones_like(wstar), wstar)
c = 0.0020 / max(d_bm - d_feed, 1e-9)
P(f"\nsquared log-deviation from w*: batch-mean {d_bm:.4f}, feed (current) {d_feed:.4f}; K1 calibration c = {c:.4f} "
  f"-> predicted gain of moving feed -> w*: {c*d_feed:+.5f} val")
d_bm2, d_feed2 = dev(wbm, wsens), dev(np.ones_like(wsens), wsens)
c2 = 0.0020 / max(d_bm2 - d_feed2, 1e-9)
P(f"(sensitivity-only w*, no composition term): batch-mean {d_bm2:.4f}, feed {d_feed2:.4f}, c {c2:.4f}, "
  f"predicted gain {c2*d_feed2:+.5f}")
open(f"{OUT}/03_trainweights.txt", "w").write("\n".join(lines) + "\n")
