"""S10: pre-registered public predictions (vs G3) for UNSCORED pools from each lens, using the H6 fitted slopes
(holdout k 0.71, stale-TO k 0.45, val k 0.28, 0.5H+0.5S k 0.72; s5/s8 logs). Writes out/s10_predict.log."""
from lib import *  # noqa
LOG = open(OUT / "s10_predict.log", "w")
def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n")
f, M, off = meta("val"); y = val_y(); ST = OUT / "stale"
W5B = [f"w5_e10_base_s{i}" for i in range(951, 955)]
V30 = [f"v30_f7_m02_s{i}" for i in range(901, 907)]
NEW12 = ["v27_gru_m02_s511", "v27_gru_m02_s513", "v27_gru_m02_s515", "v28_lr1_m02_s411", "v28_lr1_m02_s413", "v28_lr1_m02_s414",
         "v28_lr1_m02_s415", "v28_lr1_m02_s417", "v29_d192_m02_s711", "v29_d192_m02_s712", "v30_f7_m02_s901", "v30_f7_m02_s902"]
G3 = MEMBERS["G3"]
CAND = {"F10only": FAM["F10"], "NEW12": NEW12, "U1": G3 + W5B, "U3": G3 + W5B + V30, "N1": G3 + NEW12, "Q2a": MEMBERS["Q2a"]}
def mpath(m, suf):
    for d in (P, ROOT / "nn_x/w5/preds", ROOT / "nn_x/abl/preds"):
        p = d / f"nn_{m}_{suf}.npy"
        if p.exists():
            return p
    return None
def nd(s): return ndcg_per_feed(np.asarray(s, np.float32), y, off, M)
base = {"to": nd(zmean([np.load(mpath(m, "trainonly_val")) for m in G3], M)),
        "st": nd(zmean([np.load(ST / f"nn_{m}_both.npy") for m in G3], M)), "fin": np.load(OUT / "pfv_G3.npy"), "h": np.load(OUT / "pfh_G3.npy")}
holdout_index()
from lib import _HOLD
pr(f"{'pool':8s} {'val':>9s} {'val_TO':>9s} {'holdout':>9s} {'stale_TO':>9s} | pred public: {'0.28*val':>9s} {'0.71*H':>9s} {'0.45*S':>9s} {'0.72*(H+S)/2':>12s}")
for name, ms in CAND.items():
    try:
        to = nd(zmean([np.load(mpath(m, "trainonly_val")) for m in ms], M)); dto = (to - base["to"]).mean()
        st = nd(zmean([np.load(ST / f"nn_{m}_both.npy") for m in ms], M)); dst = (st - base["st"]).mean()
        fts = [mpath(m, "n2cpv3_val_oof") for m in ms]
        if all(p is not None for p in fts):
            ft = zmean([np.load(p) for p in fts], M); fin = nd(0.5 * zf(ft, M) + 0.5 * zf(zmean([np.load(mpath(m, "trainonly_val")) for m in ms], M), M))
            dv = (fin - base["fin"]).mean()
        else:
            dv = np.nan
        try:
            hs = zmean([hold_member(m) for m in ms], _HOLD["M"]); dh = (ndcg_per_feed(hs.astype(np.float32), _HOLD["y"], _HOLD["off"], _HOLD["M"]) - base["h"]).mean()
        except FileNotFoundError:
            dh = np.nan
        pr(f"{name:8s} {dv:+9.5f} {dto:+9.5f} {dh:+9.5f} {dst:+9.5f} | {0.28 * dv:+21.5f} {0.71 * dh:+9.5f} {0.45 * dst:+9.5f} {0.72 * 0.5 * (dh + dst):+12.5f}")
    except Exception as e:  # noqa
        pr(f"{name}: skipped ({e})")
