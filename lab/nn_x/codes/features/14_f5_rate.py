"""Which rate is f5? On val rows with enough item evidence, Spearman of f5 with: train feed CTR (positives/impressions),
history view count per impression (it_hist_et1 / tte_it_imp), contact-type share of the item's history events
(types 2+3 and messaging 5+6+8 over all), favourite-like share (type 4); and the impression count of the top-f5 decile (noisy small-denominator rates?)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *
from scipy.stats import spearmanr

ets = [f"it_hist_et{i}_luo" for i in range(1, 11)]
d = load("val", None, extra_feats=ets + ["it_hist_all_luo", "tte_it_imp", "tte_it_pos"])
f5 = d["f5"]; imp = d["tte_it_imp"]; pos = d["tte_it_pos"]; allh = d["it_hist_all_luo"]
E = {i: d[f"it_hist_et{i}_luo"] for i in range(1, 11)}
m = (imp >= 50) & (allh >= 10)
print(f"rows with >= 50 train impressions and >= 10 history appearances: {m.sum()}")
cands = {
    "train feed CTR pos/imp": pos / np.maximum(imp, 1),
    "history appearances per impression": allh / np.maximum(imp, 1),
    "type-1 (view) per impression": E[1] / np.maximum(imp, 1),
    "contact share (2+3)/all": (E[2] + E[3]) / np.maximum(allh, 1),
    "messaging share (5+6+8)/all": (E[5] + E[6] + E[8]) / np.maximum(allh, 1),
    "type-4 share": E[4] / np.maximum(allh, 1),
    "type-7 share": E[7] / np.maximum(allh, 1),
    "contacts per view (2+3)/(1)": (E[2] + E[3]) / np.maximum(E[1], 1),
    "history appearances (count)": allh,
    "impressions (count)": imp,
}
for k, x in cands.items():
    print(f"   Spearman(f5, {k}) = {spearmanr(f5[m], x[m]).correlation:+.3f}")
# impressions by f5 decile (all rows with imp > 0)
mm = imp > 0
q = np.quantile(f5[mm], np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q[1:-1], f5[mm], side="right"), 0, 9)
print("median train impressions by f5 decile: " + " ".join(f"{np.median(imp[mm][b==i]):.0f}" for i in range(10)))
print("median history appearances by f5 decile: " + " ".join(f"{np.median(allh[mm][b==i]):.0f}" for i in range(10)))
print("target rate (val) by f5 decile among rows with train impressions: " + " ".join(f"{d['target'][mm][b==i].mean()*100:.2f}" for i in range(10)))
# within high-evidence rows the top-decile drop?
hi = imp >= 200
q = np.quantile(f5[hi], np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q[1:-1], f5[hi], side="right"), 0, 9)
print("rows with >= 200 impressions: target rate by f5 decile: " + " ".join(f"{d['target'][hi][b==i].mean()*100:.2f}" for i in range(10)))
