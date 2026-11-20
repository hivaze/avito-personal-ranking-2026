"""Assemble the final pre_* feature family and per-candidate scores.

Information rule ("data up to the end of the feed's own split"): VAL features come from encoders pretrained
WITHOUT any test snapshot (--exclude_test 1: train + val snapshots), TEST features from the same configs/seeds
pretrained on train + val + test snapshots. Neither ever sees a snapshot of the same user taken after the feed.
Each feature is the mean over the listed configs.

Usage: python pretrain/assemble.py --val v3A,v4A --test v3,v4 [--archive]
Writes data/feats/{val,test}/pre_<feat>.npy and preds/pre_final_{val,test}.npy (mean of within-feed z of the fe
and new head scores over configs). --archive moves all other pre_*.npy feature files to data/pretrain/feats_archive/.
"""
import argparse, shutil, sys
from pathlib import Path
import numpy as np, polars as pl

sys.path.insert(0, "<repo>/src")
from stack_val import feed_rank_z

ROOT = Path("<repo>")
P, F, FLAT = ROOT / "preds", ROOT / "data/feats", ROOT / "data/flat"
FEATS = ["fe", "fe_z", "new", "new_z", "hmax32", "hmax8", "fe_noid", "id_contrib", "idmax32"]


RAW = ["fe", "new", "fe_noid", "hmax32", "hmax8", "idmax32"]


def affine(x, ref):
    """moment matching a x + b to ref on val (label-free scale alignment of a val-side model to its test-side twin;
    OLS would shrink the spread when the two models' features are only partly correlated)."""
    mx, mr = (x > -2) & np.isfinite(x), (ref > -2) & np.isfinite(ref)
    a = ref[mr].astype(np.float64).std() / x[mx].astype(np.float64).std()
    b = ref[mr].astype(np.float64).mean() - a * x[mx].astype(np.float64).mean()
    fn = lambda v: np.where(v > -2, a * v + b, v).astype(np.float32)
    fn.slope = a
    return fn, (a, b)


def build(split, tags, calib=None):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    M, off = f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)
    acc = {k: [] for k in FEATS + ["score"]}
    for i, t in enumerate(tags):
        L0 = lambda k: np.load(P / f"pre_{t}_{k}_{split}.npy").astype(np.float32)
        if calib:  # put the val-side model on the scale of its test-side twin (fit on val, no labels)
            fns = {}
            for k in RAW:
                fns[k], ab = affine(L0(k), np.load(P / f"pre_{calib[i]}_{k}_{split}.npy").astype(np.float32))
                print(f"  calib {t}->{calib[i]} {k}: a={ab[0]:.3f} b={ab[1]:.3f}")
            L = lambda k: fns[k](L0(k)) if k in fns else L0(k)
        else:
            L = L0
        fe, new, noid = L("fe"), L("new"), L("fe_noid")
        zf, zn = feed_rank_z(off, M, fe)[:, 1], feed_rank_z(off, M, new)[:, 1]
        idm = L("idmax32")
        # parity fix (review P5): ID contribution from RAW scores with the single fe slope, so rows without an item ID
        # get exactly 0 on every split (separate affine maps for fe and fe_noid used to leave a_fe*x - a_noid*x + db on val)
        a_fe = fns["fe"].slope if calib else 1.0
        idc = (a_fe * (L0("fe").astype(np.float64) - L0("fe_noid").astype(np.float64))).astype(np.float32)
        vals = dict(fe=fe, fe_z=zf, new=new, new_z=zn, hmax32=L("hmax32"), hmax8=L("hmax8"), fe_noid=noid,
                    id_contrib=idc, idmax32=np.where(idm > -2, idm, np.nan), score=0.5 * (zf + zn))
        for k, v in vals.items():
            acc[k].append(v)
    out = {}
    for k, v in acc.items():
        if k == "idmax32":
            st = np.stack(v)
            out[k] = np.where(np.isnan(st).all(0), np.nan, np.nanmean(np.where(np.isnan(st), np.nan, st), axis=0)) \
                if len(v) > 1 else v[0]
        else:
            out[k] = np.mean(v, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--prefix", default="pre_", help="output feature prefix (pre2_ for the section-8c construction)")
    ap.add_argument("--score_name", default="pre_final")
    ap.add_argument("--calib", default="", help="test-side twins of the --val models, used to align raw-score scales on val")
    a = ap.parse_args()
    if a.archive:
        for split in ["val", "test"]:
            dst = ROOT / "data/pretrain/feats_archive" / split
            dst.mkdir(parents=True, exist_ok=True)
            for p in (F / split).glob("pre_*.npy"):
                shutil.move(str(p), dst / p.name)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cal = a.calib.split(",") if a.calib else None
    for split, tags in [("val", a.val.split(",")), ("test", a.test.split(","))]:
        out = build(split, tags, cal if split == "val" else None)
        for k in FEATS:
            np.save(F / split / f"{a.prefix}{k}.npy", out[k].astype(np.float32))
        np.save(P / f"{a.score_name}_{split}.npy", out["score"].astype(np.float32))
        print(split, tags, {k: round(float(np.nanmean(out[k])), 3) for k in FEATS})


if __name__ == "__main__":
    main()
