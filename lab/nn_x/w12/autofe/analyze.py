import json, glob, numpy as np
from lib import OUT
S = [json.loads(l) for f in sorted(glob.glob(str(OUT / "screen_*.jsonl"))) if "screen_test" not in f for l in open(f)]
print("total", len(S), "const", sum(1 for r in S if r.get("const")))
S = [r for r in S if not r.get("const")]
fams = sorted(set(r["fam"] for r in S))
print("| family | n | lin z>2 | lin z<-2 | lin mean z (sd) | lin pass* | bin z>2 | bin z<-2 | bin mean z (sd) | bin pass* | hold lin <0 / =0 / >0 | corr(val z, hold z) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
def row(nm, R):
    lz = np.array([r["lin_z"] for r in R]); bz = np.array([r["bin_z"] for r in R])
    ld = np.array([r["lin_d"] for r in R]); bd = np.array([r["bin_d"] for r in R])
    lh = np.array([r["lin_h"] for r in R]); bh = np.array([r["bin_h"] for r in R]); lhz = np.array([r["lin_hz"] for r in R])
    lp = ((lz >= 2) & (ld >= 5e-4) & (lh >= 0)).sum(); bp = ((bz >= 2) & (bd >= 5e-4) & (bh >= 0)).sum()
    c = np.corrcoef(lz, lhz)[0, 1] if len(R) > 3 else float("nan")
    print(f"| {nm} | {len(R)} | {(lz > 2).sum()} ({(lz > 2).mean():.3f}) | {(lz < -2).sum()} | {lz.mean():+.2f} ({lz.std():.2f}) | {lp} | {(bz > 2).sum()} ({(bz > 2).mean():.3f}) | {(bz < -2).sum()} | {bz.mean():+.2f} ({bz.std():.2f}) | {bp} | {(lh < 0).sum()} / {(lh == 0).sum()} / {(lh > 0).sum()} | {c:+.2f} |")
for f in fams:
    row(f, [r for r in S if r["fam"] == f])
row("ALL non-ctrl", [r for r in S if r["fam"] != "CTRL"])
R = [r for r in S if r["fam"] != "CTRL"]
for key, hk in (("lin", "lin_h"), ("bin", "bin_h")):
    print(f"\nTop 20 by {key}_z (val nested delta, se, z | holdout delta, z | w)")
    for r in sorted(R, key=lambda r: -r[key + "_z"])[:20]:
        print(f"  {r['name']:42s} {r['fam']:10s} {r[key+'_d']:+.5f} ±{r[key+'_se']:.5f} z {r[key+'_z']:+.2f} | hold {r[hk]:+.5f} z {r[hk+'z']:+.2f} | wF {r['wF']:+.2f}")
    print(f"Formal passes ({key}): ", [(r["name"], round(r[key + "_d"], 5), round(r[key + "_z"], 2), round(r[hk], 5), round(r[hk + "z"], 2)) for r in R if r[key + "_z"] >= 2 and r[key + "_d"] >= 5e-4 and r[hk] >= 0])
print("\nTop 15 by holdout lin z:")
for r in sorted(R, key=lambda r: -r["lin_hz"])[:15]:
    print(f"  {r['name']:42s} {r['fam']:10s} val {r['lin_d']:+.5f} z {r['lin_z']:+.2f} | hold {r['lin_h']:+.5f} z {r['lin_hz']:+.2f} | wF {r['wF']:+.2f}")
for fit in "ABF":
    t = sorted(R, key=lambda r: -max(r[f"cv1_{fit}"], r[f"cv2_{fit}"]))[:12]
    print(f"\ntop-12 inner-CV gain fit {fit}:", [(r["name"], round(max(r[f'cv1_{fit}'], r[f'cv2_{fit}']), 1)) for r in t])
a = set(r["name"] for r in sorted(R, key=lambda r: -max(r["cv1_A"], r["cv2_A"]))[:40]); b = set(r["name"] for r in sorted(R, key=lambda r: -max(r["cv1_B"], r["cv2_B"]))[:40])
print("overlap of top-40 selections between halves:", len(a & b))
cv = np.array([[max(r["cv1_A"], r["cv2_A"]), max(r["cv1_B"], r["cv2_B"])] for r in R])
print("share of features with positive inner-CV gain: A", (cv[:, 0] > 0).mean(), "B", (cv[:, 1] > 0).mean(), "both", ((cv > 0).all(1)).mean(), "corr A,B", np.corrcoef(cv.T)[0, 1])
C = [r for r in S if r["fam"] == "CTRL"]
if C:
    cc = np.array([[max(r["cv1_A"], r["cv2_A"]), max(r["cv1_B"], r["cv2_B"])] for r in C])
    print("CTRL: positive inner-CV gain A", (cc[:, 0] > 0).mean(), "B", (cc[:, 1] > 0).mean(), "max", cc.max(0))
