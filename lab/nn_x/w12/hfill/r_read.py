"""hfill Stage A read (CPU), under the rule fixed before any arm number was read (task text, 2026-11-17):

  pre-checks : A0 reproduces the stored train-only member predictions to 1e-5 and the stored pool ens_FD16P20kd_to_val;
               in A1 / A2 every feed with no recovered history event is bit-identical (member files and pool).
  read       : train-only branch 0.8 z(16 full-view pool) + 0.2 z(4 popularity-light pool) (avito_ranker.ensemble.build,
               float32, recipe order), per-feed NDCG paired against ens_FD16P20kd_to_val on the 14,423 val feeds.
               Holm over the 2 arms (A1, A2), step-down at alpha = 0.02275 (one-sided, z = 2).
  PASS iff   : delta >= +0.0002 and the Holm step rejects (z >= 2 after Holm), pmix-private delta >= 0
               (lab/src/pmix.py weights()["private"], wmean_se), and both val days >= 0 (Tue / Thu as lab/src/presub_gate.py).
  KILL iff   : delta < +0.0001.   Otherwise: HOLD (neither).
Writes out/read_val.json and prints the table.
"""
import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, "<repo>")
sys.path.insert(0, "<repo>/lab/src")
from avito_ranker.ensemble.build import feed_zscore  # noqa: E402
from avito_ranker.ensemble.feeds import load_feed_index  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402  (lab/src/metrics.py, as pmix / presub_gate)
from pmix import weights, wmean_se  # noqa: E402

ROOT = Path("<repo>")
HF = Path(__file__).resolve().parent
STORED = ROOT / "artifacts" / "predictions"
cfg = tomllib.loads((ROOT / "configs/ensembles/fd16p20.toml").read_text())
FULL, LIGHT = cfg["members"]["full_view"], cfg["members"]["popularity_light"]
W = cfg["weights"]["popularity_light"]
ALPHA = 0.02275
fi = load_feed_index("val")
M, off = fi.sizes, fi.offsets
y = np.load(ROOT / "data/flat/val/cand_target.npy")
ts = fi.table["timestamp"].to_numpy()
tue = (ts + 10800) // 86400 < 20720
fz = np.load(HF / "out" / "fill_map_val.npz")
touched = np.zeros(len(M), bool)
touched[fz["feeds_rec"]] = True
row_touched = np.repeat(touched, M)
nrec = np.bincount(np.load(ROOT / "data/flat/val/hist_fidx.npy")[fz["ev"]], minlength=len(M))


def pool(folder):
    def p(ms):
        return np.mean([feed_zscore(fi, np.load(folder / f"nn_{m}_trainonly_val.npy")) for m in ms], 0)
    return ((1 - W) * feed_zscore(fi, p(FULL)) + W * feed_zscore(fi, p(LIGHT))).astype(np.float32)


def phi_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


ref_pool = np.load(STORED / "ens_FD16P20kd_to_val.npy")
nd_ref = ndcg_per_feed(ref_pool, y, off, M)
Wm = weights()
res = {"n_feeds": int(len(M)), "ref_val_ndcg": float(nd_ref.mean()), "feeds_touched": int(touched.sum())}
pre_ok = True
for arm in ("A0", "A1", "A2"):
    d = HF / "preds" / arm
    r = {}
    maxd, nbit_untouched = 0.0, 0
    for m in FULL + LIGHT:
        a = np.load(d / f"nn_{m}_trainonly_val.npy")
        s = np.load(STORED / f"nn_{m}_trainonly_val.npy")
        maxd = max(maxd, float(np.abs(a - s).max()))
        nbit_untouched += int((a[~row_touched].view(np.uint32) != s[~row_touched].view(np.uint32)).sum())
    pl_ = pool(d)
    r["member_maxabs_vs_stored"] = maxd
    r["untouched_rows_not_bit_identical_members"] = nbit_untouched
    r["untouched_rows_not_bit_identical_pool"] = int((pl_[~row_touched].view(np.uint32) != ref_pool[~row_touched].view(np.uint32)).sum())
    if arm == "A0":
        r["pool_bit_identical_to_stored"] = bool(np.array_equal(pl_.view(np.uint32), ref_pool.view(np.uint32)))
        ok = maxd <= 1e-5 and r["pool_bit_identical_to_stored"]
    else:
        ok = nbit_untouched == 0 and r["untouched_rows_not_bit_identical_pool"] == 0
    r["precheck_ok"] = bool(ok)
    pre_ok &= ok
    nd = ndcg_per_feed(pl_, y, off, M)
    dd = nd - nd_ref
    mv, sv = wmean_se(dd)
    mp, sp = wmean_se(dd, Wm["private"])
    mpu, spu = wmean_se(dd, Wm["public"])
    r.update(val_ndcg=float(nd.mean()), delta=float(mv), se=float(sv), z=float(mv / sv) if sv > 0 else 0.0,
             pmix_private=float(mp), pmix_private_se=float(sp), pmix_public=float(mpu), pmix_public_se=float(spu),
             tue=float(dd[tue].mean()), tue_se=float(dd[tue].std() / np.sqrt(tue.sum())), n_tue=int(tue.sum()),
             thu=float(dd[~tue].mean()), thu_se=float(dd[~tue].std() / np.sqrt((~tue).sum())), n_thu=int((~tue).sum()),
             touched_delta=float(dd[touched].mean()), touched_se=float(dd[touched].std() / np.sqrt(touched.sum())),
             untouched_delta=float(dd[~touched].mean()) if (~touched).any() else 0.0)
    for lo, hi in ((1, 5), (5, 15), (15, 10 ** 6)):
        mk = (nrec >= lo) & (nrec < hi)
        r[f"delta_nrec_{lo}_{hi}"] = [float(dd[mk].mean()), float(dd[mk].std() / np.sqrt(mk.sum())), int(mk.sum())]
    for nm, mk in (("M<50", M < 50), ("M>=50", M >= 50)):
        r[f"delta_{nm}"] = [float(dd[mk].mean()), float(dd[mk].std() / np.sqrt(mk.sum())), int(mk.sum())]
    res[arm] = r

# Holm over A1, A2 (one-sided p of z)
arms = ["A1", "A2"]
for a in arms:
    res[a]["p_raw"] = phi_sf(res[a]["z"])
order = sorted(arms, key=lambda a: res[a]["p_raw"])
still, running = True, 0.0
for j, a in enumerate(order, start=1):
    k = len(arms) - j + 1
    thr = ALPHA / k
    running = max(running, min(1.0, k * res[a]["p_raw"]))
    still = still and res[a]["p_raw"] <= thr
    res[a].update(holm_rank=j, holm_thr=thr, holm_p=running, holm_reject=bool(still))
for a in arms:
    r = res[a]
    c = dict(delta_ge_0002=r["delta"] >= 0.0002, holm_z2=r["holm_reject"], pmix_priv_ge0=r["pmix_private"] >= 0,
             days_ge0=(r["tue"] >= 0 and r["thu"] >= 0))
    r["criteria"] = {k: bool(v) for k, v in c.items()}
    r["verdict"] = ("PASS" if all(c.values()) and pre_ok else "KILL" if r["delta"] < 0.0001 else "HOLD")
res["prechecks_all_ok"] = bool(pre_ok)
(HF / "out" / "read_val.json").write_text(json.dumps(res, indent=1))

print(f"ref ens_FD16P20kd_to_val: val NDCG {res['ref_val_ndcg']:.5f}; feeds with >= 1 recovered history event {res['feeds_touched']} of {len(M)}")
for arm in ("A0", "A1", "A2"):
    r = res[arm]
    print(f"{arm}: precheck {'OK' if r['precheck_ok'] else 'FAIL'} (member max|d| {r['member_maxabs_vs_stored']:.3g}; untouched rows "
          f"not bit-identical: members {r['untouched_rows_not_bit_identical_members']}, pool {r['untouched_rows_not_bit_identical_pool']}"
          + (f"; pool == stored {r['pool_bit_identical_to_stored']}" if arm == "A0" else "") + ")")
    print(f"    val {r['val_ndcg']:.5f}  delta {r['delta']:+.5f} +- {r['se']:.5f} (z {r['z']:+.2f})  pmix-priv {r['pmix_private']:+.5f} +- "
          f"{r['pmix_private_se']:.5f}  pmix-pub {r['pmix_public']:+.5f} +- {r['pmix_public_se']:.5f}")
    print(f"    Tue {r['tue']:+.5f} +- {r['tue_se']:.5f} (n {r['n_tue']})  Thu {r['thu']:+.5f} +- {r['thu_se']:.5f} (n {r['n_thu']})  "
          f"touched feeds {r['touched_delta']:+.5f} +- {r['touched_se']:.5f}  untouched {r['untouched_delta']:+.6f}")
    print("    by #recovered events " + "  ".join(f"[{k[11:]}] {v[0]:+.5f} +- {v[1]:.5f} (n {v[2]})" for k, v in r.items() if k.startswith("delta_nrec"))
          + "  " + "  ".join(f"{k[6:]} {v[0]:+.5f} +- {v[1]:.5f}" for k, v in r.items() if k.startswith("delta_M")))
    if arm in arms:
        print(f"    Holm: p_raw {r['p_raw']:.4f} rank {r['holm_rank']} thr {r['holm_thr']:.5f} adj p {r['holm_p']:.4f} reject {r['holm_reject']}; "
              f"criteria {r['criteria']} -> {r['verdict']}")
print("prechecks all OK" if pre_ok else "PRECHECK FAILED: verdicts are void")
