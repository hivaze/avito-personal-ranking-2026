"""staleft label-free GATE (DECLARATION.txt, block 1): is the stale val view closer to the test rows than the fresh val view?
No target file is opened. CPU only. Output: logs/gate.txt, logs/gate.json.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... .venv/bin/python nn_x/w12/stale/staleft/gate.py
"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("<repo>")
HERE = ROOT / "nn_x/w12/stale/staleft"
sys.path.insert(0, str(HERE))
from stale_map import STALE_XT, STALE_XE, N2C_PRE_V3, XT_NAMES  # noqa: E402

FC = ROOT / "nn/feat_cache"
FLAT = ROOT / "data/flat"
PUB_TS = 1790497920
K_KNOTS, F4_CAP = 64, 63 / 64
AGE_EDGES_D = np.array([0, 1, 3, 7, 10, 24, 76, np.inf])
AGE_NAMES = ['<1d', '1-3d', '3-7d', '7-10d', '10-24d', '24-76d', '>76d(+cap)']
out_lines = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    out_lines.append(s)


def age_bucket(f4):  # nn_x/w12/next/poplag/pl_common.py (age_hours + age_bucket)
    kn = np.load(ROOT / "data/feats/life_f4_age_knots.npy")
    x = f4.astype(np.float64)
    u = x * K_KNOTS
    k = np.clip(np.floor(u).astype(int), 1, K_KNOTS - 2)
    w = u - k
    a = np.where(x < F4_CAP, kn[k] * (1 - w) + kn[k + 1] * w, np.nan)
    d = np.where(np.isnan(a), 1e9, a / 24.0)
    return np.clip(np.searchsorted(AGE_EDGES_D, d, side='right') - 1, 0, 6)


def xt_transform(name, x):  # nn_x/w5/data.py attach_xt
    x = np.asarray(x).astype(np.float32)
    if name.startswith('has:') or name.endswith('_rk'):
        pass
    elif name.endswith('_z'):
        x = np.clip(x, -5, 5)
    elif name.startswith(('seq_cv', 'it_hist')):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


def xe_transform(fresh_val_raw):  # nn_x/w5/data.py attach_extra with the FRESH val statistics
    v = fresh_val_raw.astype(np.float32)
    logm = bool(np.nanmin(v) >= 0 and np.nanmax(v) > 20)
    f = (lambda x: np.log1p(np.maximum(x, 0))) if logm else (lambda x: x)
    vv = f(v)
    mu, sd = np.nanmean(vv), np.nanstd(vv) + 1e-6
    return lambda x: np.nan_to_num(np.clip((f(x.astype(np.float32)) - mu) / sd, -6, 6), nan=0.0)


def special(name, raw):
    """Indicator used for dZ (None = not defined for this column)."""
    base = name[4:] if name.startswith("has:") else name
    if name.startswith("has:"):
        return np.isnan(raw)
    if base.endswith(("_rk", "_z", "_zsl", "_gap", "_s")) or base in ("pre_fe", "pre2_fe"):
        return None
    if base == "pre_idmax32":
        return np.isnan(raw)
    return raw == 0  # counts, pre_id_contrib_v2, pre_idmax32_has


def stats(x, z, bk):
    n = np.bincount(bk, minlength=7).astype(np.float64)
    m = np.bincount(bk, weights=x, minlength=7) / np.maximum(n, 1)
    zz = None if z is None else np.bincount(bk, weights=z.astype(np.float64), minlength=7) / np.maximum(n, 1)
    return m, zz


def main():
    fv = pl.read_parquet(FLAT / "val/feeds.parquet")
    ft = pl.read_parquet(FLAT / "test/feeds.parquet")
    bkv = age_bucket(np.load(FLAT / "val/cand_f4.npy"))
    bkt = age_bucket(np.load(FLAT / "test/cand_f4.npy"))
    fi = np.load(FLAT / "test/cand_fidx.npy")
    pub = (ft["timestamp"].to_numpy().astype(np.int64) < PUB_TS)[fi]
    parts = {"private": ~pub, "public": pub, "all": np.ones(len(pub), bool)}
    wv = np.bincount(bkv, minlength=7) / len(bkv)
    P("=== staleft GATE (label-free) ===")
    P(f"val rows {len(bkv)}; test rows private {int((~pub).sum())} public {int(pub.sum())}")
    P("age-bucket share of rows: " + " | ".join(
        f"{AGE_NAMES[b]} val {wv[b]:.3f} pub {np.mean(bkt[pub] == b):.3f} priv {np.mean(bkt[~pub] == b):.3f}" for b in range(7)))
    cols = [("xt", n) for n in XT_NAMES if n in STALE_XT or (n.startswith("has:") and n[4:] in STALE_XT)]
    cols += [("xe", n) for n in N2C_PRE_V3 if n in STALE_XE]
    info = [("xe", n) for n in N2C_PRE_V3 if n not in STALE_XE]
    res = {}
    P(f"\n{'kind':3s} {'column':24s} | private: dZ fresh -> stale | dM fresh -> stale || all test: dZ fresh -> stale | dM fresh -> stale || public: dZ fresh -> stale | dM fresh -> stale || verdict")
    for kind, n in cols + info:
        base = n[4:] if n.startswith("has:") else n
        rv = np.load(FC / "val" / f"{base}.npy").astype(np.float32)
        rt = np.load(FC / "test" / f"{base}.npy").astype(np.float32)
        mp = STALE_XT if kind == "xt" else STALE_XE
        rs = np.load(mp[base]).astype(np.float32) if base in mp else None
        if kind == "xt":
            tf = (lambda x: (~np.isnan(x)).astype(np.float32)) if n.startswith("has:") else (lambda x: xt_transform(n, x))
        else:
            tf = xe_transform(rv)
        r = {"switched_candidate": rs is not None}
        Xv, Xt = tf(rv), tf(rt)
        Zv, Zt = special(n, rv), special(n, rt)
        mv, zv = stats(Xv, Zv, bkv)
        if rs is not None:
            ms, zs = stats(tf(rs), special(n, rs), bkv)
        for pn, pm in parts.items():
            mt, zt = stats(Xt[pm], None if Zt is None else Zt[pm], bkt[pm])
            sd = float(Xt[pm].std()) + 1e-9
            r[pn] = {"dM_fresh": float((wv * np.abs(mv - mt)).sum() / sd),
                     "dZ_fresh": None if zv is None else float((wv * np.abs(zv - zt)).sum()),
                     "mean_fresh": float(Xv.mean()), "mean_test": float(Xt[pm].mean()),
                     "special_fresh": None if Zv is None else float(Zv.mean()), "special_test": None if Zt is None else float(Zt[pm].mean())}
            if rs is not None:
                r[pn].update({"dM_stale": float((wv * np.abs(ms - mt)).sum() / sd),
                              "dZ_stale": None if zs is None else float((wv * np.abs(zs - zt)).sum()),
                              "mean_stale": float(tf(rs).mean()), "special_stale": None if zs is None else float(special(n, rs).mean()),
                              "bucket_special": None if zs is None else {"fresh": zv.tolist(), "stale": zs.tolist(), "test": zt.tolist()},
                              "bucket_mean": {"fresh": mv.tolist(), "stale": ms.tolist(), "test": mt.tolist()}})
        if rs is not None:
            ok = True
            for pn in ("private", "all"):
                q = r[pn]
                if q["dZ_fresh"] is not None and not q["dZ_stale"] <= q["dZ_fresh"]:
                    ok = False
                if not q["dM_stale"] <= q["dM_fresh"] + 0.005:
                    ok = False
            r["pass"] = ok
            pubover = (r["public"]["dM_stale"] > r["public"]["dM_fresh"] + 0.005) or (
                r["public"]["dZ_fresh"] is not None and r["public"]["dZ_stale"] > r["public"]["dZ_fresh"])
            r["public_overshoot"] = bool(pubover)
            f = lambda v: "   -  " if v is None else f"{v:.4f}"  # noqa: E731
            P(f"{kind:3s} {n:24s} | {f(r['private']['dZ_fresh'])} -> {f(r['private']['dZ_stale'])} | {r['private']['dM_fresh']:.4f} -> {r['private']['dM_stale']:.4f} || "
              f"{f(r['all']['dZ_fresh'])} -> {f(r['all']['dZ_stale'])} | {r['all']['dM_fresh']:.4f} -> {r['all']['dM_stale']:.4f} || "
              f"{f(r['public']['dZ_fresh'])} -> {f(r['public']['dZ_stale'])} | {r['public']['dM_fresh']:.4f} -> {r['public']['dM_stale']:.4f} || "
              f"{'SWITCH' if ok else 'KEEP FRESH (fails)'}{' [overshoots public]' if pubover else ''}")
        else:
            f = lambda v: "   -  " if v is None else f"{v:.4f}"  # noqa: E731
            P(f"{kind:3s} {n:24s} | {f(r['private']['dZ_fresh'])} ->    n/a | {r['private']['dM_fresh']:.4f} ->    n/a || "
              f"{f(r['all']['dZ_fresh'])} ->    n/a | {r['all']['dM_fresh']:.4f} ->    n/a || {f(r['public']['dZ_fresh'])} ->    n/a | {r['public']['dM_fresh']:.4f} ->    n/a || kept fresh (not in the scenario)")
        res[f"{kind}:{n}"] = r
    P("\npooled special share (zero / NaN) fresh val -> stale val | private test | public test, for the switched columns:")
    for k_, r in res.items():
        if r.get("switched_candidate") and r["private"].get("special_fresh") is not None:
            P(f"  {k_:28s} {r['private']['special_fresh']:.4f} -> {r['private']['special_stale']:.4f} | {r['private']['special_test']:.4f} | {r['public']['special_test']:.4f}")
    sw = [k_ for k_, r in res.items() if r.get("pass")]
    fl = [k_ for k_, r in res.items() if r.get("switched_candidate") and not r.get("pass")]
    P(f"\nSWITCH ({len(sw)}): {sw}\nFAIL -> keep fresh ({len(fl)}): {fl}")
    res["_switch"], res["_fail"] = sw, fl
    json.dump(res, open(HERE / "logs/gate.json", "w"), indent=1)
    (HERE / "logs/gate.txt").write_text("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
