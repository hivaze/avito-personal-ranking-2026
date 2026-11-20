"""Label-free check on test: do the frozen candidates disagree more on private than on public / val?
Per-feed Spearman between two score vectors (within feed), averaged per segment; plus the within-feed z of cold
candidates (never seen in train) under each model."""
import numba as nb
from common_av import *


@nb.njit(parallel=True)
def feed_spearman(off, M, a, b):
    out = np.empty(len(M))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        ra = np.argsort(np.argsort(a[o:o + m])).astype(np.float64)
        rb = np.argsort(np.argsort(b[o:o + m])).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        out[f] = (ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum() + 1e-12)
    return out


first_seen = np.load(f"{SCR}/item_first_seen_day.npy")
res = {}
for split in ["val", "test"]:
    f, M, off = meta(split)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    wd = (((ts + MSK) // 86400) + 3) % 7
    sfx = "val" if split == "val" else "test"
    oof = "val_oof" if split == "val" else "test"
    z = lambda n: feed_z(off, M, load_pred(n))
    s = {"ft5": z(f"nn_final5_{oof}"), "to5": z(f"nn_final5_trainonly_{sfx}"),
         "ft4": z(f"nn_final4_{oof}"), "to4": z(f"nn_final4_trainonly_{sfx}"), "pre": z(f"pre_final_{sfx}")}
    s["v11"] = 0.5 * s["ft5"] + 0.5 * s["to5"]
    s["v10"] = 0.5 * s["ft4"] + 0.5 * s["to4"] + 0.1 * s["pre"]
    s["cd12"] = 0.5 * z(f"nn_final5cd12_{oof}") + 0.5 * z(f"nn_final5cd12_trainonly_{sfx}")
    item = np.load(f"{FLAT}/{split}/cand_item_id.npy")
    cold = first_seen[item] >= 10 ** 9
    fid = np.repeat(np.arange(len(M)), M)
    segs = {"val_tue": wd == 1, "val_thu": wd == 3} if split == "val" else {"public": ts < BOUND, "private": ts >= BOUND}
    pairs = [("ft5", "to5"), ("v11", "v10"), ("v11", "cd12"), ("ft4", "to4")]
    rho = {p: feed_spearman(off, M, s[p[0]], s[p[1]]) for p in pairs}
    for sg, m in segs.items():
        r = {f"{a}~{b}": float(rho[(a, b)][m].mean()) for a, b in pairs}
        cm = m[fid] & cold
        wm = m[fid] & ~cold
        for k in ["ft5", "to5", "v11", "v10"]:
            r[f"zcold_{k}"] = float(s[k][cm].mean()); r[f"zwarm_{k}"] = float(s[k][wm].mean())
        r["cold_share"] = float(cm.sum() / m[fid].sum())
        res[sg] = r
cols = list(res["public"].keys())
print("| segment | " + " | ".join(cols) + " |")
for sg, r in res.items():
    print(f"| {sg} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " |")
import json
json.dump(res, open(f"{OUT}/agree.json", "w"), indent=1)
