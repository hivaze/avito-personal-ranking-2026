"""Independent rebuild of G3 / M2 / PF7x from raw member preds (no overfit cache), sub-file rank identity,
val + holdout deltas vs PF7x. CPU only, read-only on preds/ subs/."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "12"
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from pathlib import Path
from common import feed_meta, FLAT
from stack_val import feed_rank_z
from metrics import ndcg_per_feed

ROOT = Path("<repo>"); P = ROOT / "preds"; H = P / "holdout"; S = ROOT / "subs"
F = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{s}" for s in range(92, 97)],
    "F7x": [f"r6_cd_m02_r{s}" for s in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{s}" for s in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{s}" for s in range(701, 708)],
}
POOLS = {"PF7x": ["F5cd", "F6", "F7x", "F9", "F10", "F12"], "G3": ["F10", "F7x", "F7x70"], "M2": ["F7x", "F7x70"]}
SUBS = {"PF7x": ["cand_PF7x_memeq_6fam.parquet"], "G3": ["cand_R1_G3_F10F7xF7x70_memeq.parquet", "cand_G_F10F7xF7x70_memeq.parquet"],
        "M2": ["cand_R1_M2_F7xF7x70_memeq.parquet", "cand_M_F7xF7x70_memeq.parquet"]}
ENS = {"PF7x": ["PF7x"], "G3": ["G3", "R1G3"], "M2": ["M2", "R1M2"]}

def zf(o, M, x): return feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]

fv, Mv, ov = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
ft, Mt, ot = feed_meta("test")
Mh, oh = np.load(H / "holdout_meta.npy"); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(H / "holdout_y.npy").astype(np.float64)
print("val feeds", len(Mv), "test feeds", len(Mt), "holdout feeds", len(Mh))

# member file existence + duplicate check (identical arrays)
allm = sorted({m for p in POOLS.values() for f in p for m in F[f]})
sig = {}
for m in allm:
    for suf in ("trainonly_val", "n2cpv3_val_oof", "trainonly_test", "n2cpv3_test"):
        assert (P / f"nn_{m}_{suf}.npy").exists(), (m, suf)
    assert (H / f"nn_{m}_holdout.npy").exists(), m
    x = np.load(P / f"nn_{m}_trainonly_val.npy")[:200000]
    sig.setdefault(x.tobytes()[:4000], []).append(m)
print("duplicate trainonly_val prefixes:", [v for v in sig.values() if len(v) > 1])

res = {}
for pool, fams in POOLS.items():
    ms = [m for f in fams for m in F[f]]
    assert len(ms) == len(set(ms))
    br = {}
    for split, o, M, ftn, ton in [("val", ov, Mv, "n2cpv3_val_oof", "trainonly_val"), ("test", ot, Mt, "n2cpv3_test", "trainonly_test")]:
        br[("ft", split)] = np.mean([zf(o, M, np.load(P / f"nn_{m}_{ftn}.npy")) for m in ms], 0).astype(np.float32)
        br[("to", split)] = np.mean([zf(o, M, np.load(P / f"nn_{m}_{ton}.npy")) for m in ms], 0).astype(np.float32)
    hold = np.mean([zf(oh, Mh, np.load(H / f"nn_{m}_holdout.npy")) for m in ms], 0).astype(np.float32)
    for tag in ENS[pool]:
        for h in ("ft", "to"):
            for s in ("val", "test"):
                ref = np.load(P / f"ens_{tag}_{h}_{s}.npy")
                print(f"{pool}: ens_{tag}_{h}_{s} max|diff| {np.abs(ref - br[(h, s)]).max():.2e}")
    finv = 0.5 * zf(ov, Mv, br[("ft", "val")]) + 0.5 * zf(ov, Mv, br[("to", "val")])
    fint = 0.5 * zf(ot, Mt, br[("ft", "test")]) + 0.5 * zf(ot, Mt, br[("to", "test")])
    res[pool] = dict(val=ndcg_per_feed(finv, yv, ov, Mv), val_to=ndcg_per_feed(br[("to", "val")], yv, ov, Mv),
                     val_ft=ndcg_per_feed(br[("ft", "val")], yv, ov, Mv), hold=ndcg_per_feed(hold, yh, oh, Mh), fint=fint)
    r = res[pool]
    print(f"{pool}: n={len(ms)} val {r['val'].mean():.5f} val_to {r['val_to'].mean():.5f} val_ft {r['val_ft'].mean():.5f} holdout {r['hold'].mean():.5f}")

# --- sub file rank identity ---
fidx = np.load(FLAT / "test/cand_fidx.npy"); item = np.load(FLAT / "test/cand_item_id.npy")
fid = ft["feed_id"].to_numpy()[fidx]
key = pl.DataFrame({"feed_id": fid.astype(np.int32), "item_id": item.astype(np.int32), "pos": np.arange(len(fid))})
fidrep = np.repeat(np.arange(len(Mt)), Mt)
def within_rank(x):
    order = np.lexsort((x, fidrep)); r = np.empty(len(x), np.int64); r[order] = np.arange(len(x)) - np.repeat(ot, Mt); return r
subr = {}
for pool, files in SUBS.items():
    myr = within_rank(res[pool]["fint"].astype(np.float64))
    for fn in files:
        sb = pl.read_parquet(S / fn)
        j = key.join(sb, on=["feed_id", "item_id"], how="left").sort("pos")
        assert j["score"].null_count() == 0 and len(sb) == len(key)
        sr = j["score"].to_numpy().astype(np.int64)
        subr[fn] = sr
        # per-feed Spearman = Pearson of ranks
        d = (myr - sr).astype(np.float64)
        dd = np.bincount(fidrep, d * d, len(Mt)); Mm = Mt.astype(np.float64)
        rho = np.where(Mm > 1, 1 - 6 * dd / np.maximum(Mm * (Mm ** 2 - 1), 1), 1.0)
        nties = 0
        print(f"{pool} vs subs/{fn}: exact rank match {np.mean(myr == sr):.6f}; per-feed Spearman min {rho.min():.6f} mean {rho.mean():.8f}; feeds with any diff {np.mean(dd > 0):.5f}")
for a, b in [("cand_R1_G3_F10F7xF7x70_memeq.parquet", "cand_G_F10F7xF7x70_memeq.parquet"), ("cand_R1_M2_F7xF7x70_memeq.parquet", "cand_M_F7xF7x70_memeq.parquet")]:
    print(f"{a} == {b} ranks: {np.array_equal(subr[a], subr[b])}")

# --- deltas vs PF7x ---
def ug(users):
    g = users.astype(np.int64).copy(); z = np.where(users == 0)[0]; g[z] = -(np.arange(len(z)) + 1)
    return np.unique(g, return_inverse=True)[1]
def clse(d, g):
    s = np.bincount(g, d - d.mean()); return np.sqrt((s ** 2).sum()) / len(d)
gv = ug(fv["user_id"].to_numpy())
sys.path.insert(0, str(ROOT / "nn_x/pubdiag/overfit"))
from lib import holdout_users_ts
hu, hts = holdout_users_ts(Mh); gh = ug(hu)
print("holdout users unique", len(np.unique(hu)), "user0 share", np.mean(hu == 0))
for pool in ("G3", "M2"):
    for k, g in (("val", gv), ("val_to", gv), ("val_ft", gv), ("hold", gh)):
        d = res[pool][k] - res["PF7x"][k]
        print(f"{pool}-PF7x {k:7s}: {d.mean():+.5f}  iid SE {d.std() / np.sqrt(len(d)):.5f}  user-cl SE {clse(d, g):.5f}  z {d.mean() / clse(d, g):+.2f}")
d = res["M2"]["val"] - res["G3"]["val"]; print(f"M2-G3 val {d.mean():+.5f} SE {clse(d, gv):.5f}")
d = res["M2"]["hold"] - res["G3"]["hold"]; print(f"M2-G3 hold {d.mean():+.5f} SE {clse(d, gh):.5f}")
np.savez("<repo>/nn_x/pubdiag/verify_build/v1_perfeed.npz", **{f"{p}_{k}": res[p][k] for p in res for k in ("val", "val_to", "val_ft", "hold")})
