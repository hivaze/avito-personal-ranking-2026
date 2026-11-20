"""Replication of the val PD statistics on the holdout: LR main ratio (LR/df) per input on val vs holdout, and the val top
interactions (strata available on both: M3 [2 levels on holdout], u0, vert, coldrow [holdout: LUO==0 proxy])."""
import numpy as np
D = "/tmp/scratchpad/dpa/nonlinear"
V = {r["name"]: r for r in np.load(f"{D}/pd_val.npy", allow_pickle=True)}
H = {r["name"]: r for r in np.load(f"{D}/pd_hold.npy", allow_pickle=True)}
common = [n for n in V if n in H]
rv = np.array([V[n]["lr"] / (V[n]["nb"] - 1) for n in common]); rh = np.array([H[n]["lr"] / (H[n]["nb"] - 1) for n in common])
from scipy.stats import spearmanr
print(f"inputs {len(common)}; LR/df val median {np.median(rv):.2f}, holdout median {np.median(rh):.2f}; Spearman(val, hold) {spearmanr(rv, rh)[0]:.2f}")
print(f"{'input':22s} {'val LR/df':>9s} {'hold LR/df':>10s}")
for n in sorted(common, key=lambda n: -V[n]["lr"] / (V[n]["nb"] - 1))[:20]:
    print(f"{n:22s} {V[n]['lr']/(V[n]['nb']-1):9.2f} {H[n]['lr']/(H[n]['nb']-1):10.2f}")
print("\nval top interactions -> holdout ((LR-df)/sqrt(2df)):")
inter = []
for n in common:
    for s in ["M3", "u0", "vert", "coldrow"]:
        lv, dv = V[n][f"I_{s}"], V[n][f"I_{s}_df"]; lh, dh = H[n][f"I_{s}"], H[n][f"I_{s}_df"]
        inter.append((n, s, (lv - dv) / np.sqrt(2 * dv), (lh - dh) / np.sqrt(2 * max(dh, 1)), dv, dh))
iv = np.array([t[2] for t in inter]); ih = np.array([t[3] for t in inter])
print(f"  all {len(inter)} (input x stratum) cells: val mean {iv.mean():+.2f}, hold mean {ih.mean():+.2f}; corr {np.corrcoef(iv, ih)[0,1]:+.2f}; val>3: {np.sum(iv>3)}, hold>3: {np.sum(ih>3)}, both>3: {np.sum((iv>3)&(ih>3))}")
for t in sorted(inter, key=lambda t: -t[2])[:15]:
    print(f"  {t[0]:20s} x {t[1]:8s} val {t[2]:+5.2f} (df {t[4]}) | hold {t[3]:+5.2f} (df {t[5]})")
