"""W25 summary table (CPU): every arm in out/*.json with val / halves / trH / M-slices, paired deltas (A2-A1, A2c-A0, loo_X - A2,
add_X - A1) with SE and z, and the W25 feature importances inside A2. Appends to out/results.txt."""
import json, sys, time
import numpy as np
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import meta, ROOT  # noqa: E402
from cbrule import pse, val_slices  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
import pmix  # noqa: E402

OUT = ROOT / "nn_x/w25/out"
fv, Mv, ov, yv = meta("val"); fh, Mh, oh, yh = meta("trH")
nv = lambda s: ndcg_per_feed(s, yv, ov, Mv); nh = lambda s: ndcg_per_feed(s, yh, oh, Mh)
SL = val_slices(); WP = pmix.weights()["private"]
lines = [f"==== W25 summary {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===="]
arms = {}
for p in sorted(OUT.glob("*.json")):
    if p.stem in ("psi_w25",) or not (OUT / f"{p.stem}_val.npy").exists():
        continue
    j = json.loads(p.read_text())
    arms[p.stem] = dict(j=j, pv=nv(np.load(OUT / f"{p.stem}_val.npy")), ph=nh(np.load(OUT / f"{p.stem}_trH.npy")))
lines.append(f"{'arm':18s} {'feeds':>7s} {'nfeat':>5s} {'iter':>5s} {'min':>4s} | {'val':>8s} {'halfA':>8s} {'halfB':>8s} {'trH':>8s} | {'M<50':>7s} {'M>=50':>7s} {'Tue':>7s} {'Thu':>7s} {'user0':>7s} {'cold_hi':>7s}")
for t, a in arms.items():
    j, pv, ph = a["j"], a["pv"], a["ph"]
    lines.append(f"{t:18s} {j['n_train_feeds']:7d} {len(j['features']):5d} {j['best_iter']:5d} {j['time'] / 60:4.0f} | {pv.mean():.5f} {pv[SL['Tue'] | SL['Thu']].mean() * 0 + j['val_halfA_ES']:.5f} {j['val_halfB']:.5f} {ph.mean():.5f} | "
                 f"{pv[SL['M<50']].mean():.4f} {pv[SL['M>=50']].mean():.4f} {pv[SL['Tue']].mean():.4f} {pv[SL['Thu']].mean():.4f} {pv[SL['user0']].mean():.4f} {pv[SL['cold_hi']].mean():.4f}")


def paired(a, b, label):
    if a not in arms or b not in arms:
        return
    dv = arms[a]["pv"] - arms[b]["pv"]; dh = arms[a]["ph"] - arms[b]["ph"]
    m, s = pse(dv); hm, hs = pse(dh); pm, ps = pmix.wmean_se(dv, WP)
    sl = " ".join(f"{k}:{pse(dv[msk])[0]:+.4f}" for k, msk in SL.items() if k in ("M<50", "M>=50", "Tue", "Thu", "user0", "known"))
    worst = min((pse(dv[msk])[0] / max(pse(dv[msk])[1], 1e-9), k) for k, msk in SL.items())
    lines.append(f"{label:22s} val {m:+.5f} +- {s:.5f} (z {m / s:+.2f}) | trH {hm:+.5f} +- {hs:.5f} (z {hm / hs:+.2f}) | pmix {pm:+.5f} +- {ps:.5f} | {sl} | worst slice z {worst[0]:+.2f} ({worst[1]})")


lines.append("-- paired deltas (first minus second) --")
paired("A1_base_xm", "A0_cb26repro", "A1 - A0 (rows/lr)")
paired("A1_base_xm_s1", "A1_base_xm", "A1 seed1 - seed0 (NOISE)")
paired("A2_all_xm", "A1_base_xm", "A2 - A1 (G1..G5)")
paired("A2c_cb26", "A0_cb26repro", "A2c - A0 (G1..G5, cb26 rows)")
paired("abl_add_all", "A1_base_xm", "add_all - A1")
for g in ("G1", "G2", "G3", "G4", "G5"):
    paired("A2_all_xm", f"abl_loo_{g}", f"A2 - loo_{g} (= {g} on top)")
    paired(f"abl_add_{g}", "A1_base_xm", f"add_{g} - A1 (= {g} alone)")
if "A2_all_xm" in arms:
    j = arms["A2_all_xm"]["j"]
    lines.append("-- A2 W25 feature importances (PredictionValuesChange, top 25) --")
    lines.append("   " + ", ".join(f"{n}:{x}" for n, x in j["imp_w25"][:25]))
    tot = sum(x for _, x in j["top"]) if j["top"] else 0
    lines.append(f"   sum of W25 importances {sum(x for _, x in j['imp_w25']):.1f} of 100; top-5 overall {j['top'][:5]}")
txt = "\n".join(lines)
print(txt)
with open(OUT / "results.txt", "a") as f:
    f.write(txt + "\n\n")
