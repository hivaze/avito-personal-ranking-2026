"""HR (DECLARATION block 1): holdout replication of the audit's cap cells. 30,000 in-period holdout feeds, bucket H >= 256
fixed a priori. Also re-computes the same cells on val with my code (R3 i)."""
import sys
from rv_lib import *

DIRS = [ROOT / "nn_x/w5/preds", ROOT / "nn_x/abl/preds", ROOT / "nn_x/r6/preds", ROOT / "nn_x/r7/preds", ROOT / "preds"]


def find(name, kind):
    """kind: 'val' -> nn_<name>_trainonly_val.npy ; 'hold' -> holdout/nn_<name>_holdout.npy"""
    for d in DIRS:
        f = d / (f"nn_{name}_trainonly_val.npy" if kind == "val" else f"holdout/nn_{name}_holdout.npy")
        if f.exists():
            return f
    return None


# ---- holdout frame (definition as nn_x/tricks/HDROP-ALL/hdrop_replay.py; asserted against preds/holdout/holdout_meta.npy)
tr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["user_id", "M", "H"])
Mtr = tr["M"].to_numpy().astype(np.int64)
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
assert (pr["fidx"] == np.arange(len(Mtr))).all()
prand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
prand[Mtr < 50] = 0.0
cand = np.where((prand > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
meta = np.load(P / "holdout/holdout_meta.npy")
assert (meta[0] == Mtr[hold]).all()
hM = Mtr[hold]
hOFF = np.concatenate([[0], np.cumsum(hM)[:-1]])
hY = np.load(P / "holdout/holdout_y.npy").astype(np.float64)
hH = np.minimum(tr["H"].to_numpy()[hold], 256)
assert (hH == np.load(ROOT / "nn_x/tricks/HDROP-ALL/preds/holdout/hold_H.npy")).all()
hU = tr["user_id"].to_numpy()[hold].astype(np.int64)
hN = len(hM)
hFID = np.repeat(np.arange(hN), hM)
hNPOS = np.bincount(hFID, weights=hY, minlength=hN).astype(np.int64)
hkey = np.where(hU == 0, -(np.arange(hN) + 1), hU)
hCLU = np.unique(hkey, return_inverse=True)[1]
hNCL = hCLU.max() + 1
_DISC = 1.0 / np.log2(np.arange(2, 3002))
_CD = np.concatenate([[0.0], np.cumsum(_DISC)])
vu = set(np.unique(UID).tolist())
print(f"holdout: {hN} feeds, {len(np.unique(hU))} users, clusters {hNCL}, cap share {(hH >= 256).mean():.4f}, H=0 {(hH == 0).mean():.4f}, H<=9 {(hH <= 9).mean():.4f}, uid0 {(hU == 0).mean():.4f}, "
      f"mean M {hM.mean():.1f}, mean n_pos {hNPOS.mean():.2f}, feeds of users that also have a val feed {np.mean([u in vu and u != 0 for u in hU]):.4f}")


def hz(x):
    x = np.asarray(x, np.float64)
    s1 = np.bincount(hFID, weights=x, minlength=hN) / hM
    s2 = np.bincount(hFID, weights=x * x, minlength=hN) / hM
    return (x - s1[hFID]) / (np.sqrt(np.maximum(s2 - s1 * s1, 0)) + 1e-9)[hFID]


def hndcg(s):
    order = np.lexsort((hY, -np.asarray(s, np.float64), hFID))
    pos = np.arange(len(hY)) - hOFF[hFID]
    dcg = np.bincount(hFID, weights=hY[order] * _DISC[pos], minlength=hN)
    out = np.ones(hN)
    ok = hNPOS > 0
    out[ok] = dcg[ok] / _CD[hNPOS[ok]]
    return out


def hstat(d, m):
    """mean and cluster SE on mask m; interaction m minus ~m (cluster SE through the influence function)."""
    def one(mm):
        mu = d[mm].mean()
        r = np.bincount(hCLU[mm], weights=d[mm] - mu, minlength=hNCL)
        return mu, np.sqrt((r ** 2).sum()) / mm.sum()
    a, sa = one(m)
    b, sb = one(~m)
    infl = np.where(m, (d - a) / m.sum(), -(d - b) / (~m).sum())
    r = np.bincount(hCLU, weights=infl, minlength=hNCL)
    return a, sa, b, sb, a - b, np.sqrt((r ** 2).sum())


def vstat(d, m):
    a, sa, _ = cmean(d, m)
    b, sb, _ = cmean(d, ~m)
    i, si = cdiff(d, m, ~m)
    return a, sa, b, sb, i, si


e_h = None
CELLS = [
    ("1 GRU under KD (4 seeds)", [f"w5_e10_kd05w192gru_s{s}" for s in S4], [f"w5_e10_kd05w192_s{s}" for s in S4], "kd"),
    ("7 kw_hl2: 2 history layers under KD", [f"w5_e10_kw_hl2_s{s}" for s in (951, 952)], [f"w5_e10_kd05w192_s{s}" for s in (951, 952)], "kd"),
    ("2 a4u 10 ep (6 seeds vs F7)", [f"r6_a4u_e10_s{s}" for s in range(101, 107)], [f"v17_pre2s_s{s}" for s in range(101, 107)], "v17"),
    ("3 a4u m02 re-cools (6 roots)", [f"r6_cd_m02_a4u_r{s}" for s in range(101, 107)], [f"r6_cd_m02_r{s}" for s in range(101, 107)], "cdm (same snapshots as cell 2)"),
    ("4 kv64 2 ep", [f"abl_kv64_s{s}" for s in (23, 24)], [f"abl_base_s{s}" for s in (23, 24)], "abl_base"),
    ("5 retr32 2 ep", [f"abl_retr32_s{s}" for s in (23, 24)], [f"abl_base_s{s}" for s in (23, 24)], "abl_base"),
    ("6a retr32 10 ep", [f"abl_e10_retr32_s{s}" for s in (951, 952)], [f"w5_e10_base_s{s}" for s in (951, 952)], "w5_e10_base"),
    ("6b retr32u 10 ep", [f"abl_e10_retr32u_s{s}" for s in (951, 952)], [f"w5_e10_base_s{s}" for s in (951, 952)], "w5_e10_base"),
    ("8 cset 10 ep (candidate side)", [f"r6_cset_e10_s{s}" for s in range(101, 107)], [f"v17_pre2s_s{s}" for s in range(101, 107)], "v17"),
    ("9 kwh30 history-drop 0.3 under KD (4 seeds)", [f"w5_e10_kwh30_s{s}" for s in S4], [f"w5_e10_kd05w192_s{s}" for s in S4], "kd"),
    ("GRU family v23 (6) vs F7 v17 (6), not seed-paired", [f"v23_gru_pre2s_s{s}" for s in range(501, 507)], [f"v17_pre2s_s{s}" for s in range(101, 107)], "v17"),
]
rows = []
RES = {}
for lab, arm, ctl, grp in CELLS:
    r = [lab, grp]
    # val (my code)
    fa, fc = [find(n, "val") for n in arm], [find(n, "val") for n in ctl]
    if None in fa or None in fc:
        r += ["val files missing", "", ""]
        v = None
    else:
        d = ndcg(np.mean([z(np.load(f)) for f in fa], 0)) - ndcg(np.mean([z(np.load(f)) for f in fc], 0))
        a, sa, b, sb, i, si = vstat(d, CAP)
        mu, se, _ = cmean(d)
        r += [f"{mu:+.5f}+-{se:.5f}", f"{a:+.5f}+-{sa:.5f} (z {a / sa:+.2f})", f"{i:+.5f} (z {i / si:+.2f})"]
        v = (a, sa, i / si)
    for sfx in ("", "@last"):
        fa, fc = [find(n + sfx, "hold") for n in arm], [find(n + sfx, "hold") for n in ctl]
        if None in fa or None in fc:
            r += ["no files", "", ""]
            continue
        d = hndcg(np.mean([hz(np.load(f)) for f in fa], 0)) - hndcg(np.mean([hz(np.load(f)) for f in fc], 0))
        a, sa, b, sb, i, si = hstat(d, hH >= 256)
        r += [f"{d.mean():+.5f}", f"{a:+.5f}+-{sa:.5f} (z {a / sa:+.2f})", f"{i:+.5f} (z {i / si:+.2f})"]
        if sfx == "":
            verdict = "REPLICATED" if (a / sa >= 1.64 and i > 0) else ("REFUTED" if (a <= 0 or (v is not None and a + 1.64 * sa < 0.5 * v[0])) else "UNDECIDED")
            RES[lab] = (a, sa, i / si, verdict, grp)
            r.append(verdict)
    rows.append(r)
print(tab([r + [""] * (12 - len(r)) for r in rows], ["cell", "control group", "VAL all", "VAL cap", "VAL cap - rest", "HOLD all", "HOLD cap", "HOLD cap - rest", "cell rule", "HOLD@last all", "HOLD@last cap", "HOLD@last cap - rest"]))
# family rule: one unit per control group; cells 1+7 (kd), 2 (v17), 4+5 (abl_base), 6a+6b (w5_e10_base)
g = {"kd": ["1 GRU under KD (4 seeds)", "7 kw_hl2: 2 history layers under KD"], "v17": ["2 a4u 10 ep (6 seeds vs F7)"], "abl_base": ["4 kv64 2 ep", "5 retr32 2 ep"],
     "w5_e10_base": ["6a retr32 10 ep", "6b retr32u 10 ep"]}
zs = [np.mean([RES[c][2] for c in cs if c in RES]) for cs in g.values()]
print("family rule: per-control mean holdout interaction z:", {k: round(float(v), 2) for k, v in zip(g, zs)}, f"-> mean {np.mean(zs):+.2f} x sqrt({len(zs)}) = {np.mean(zs) * np.sqrt(len(zs)):+.2f} (SUPPORTED if >= 2)")
