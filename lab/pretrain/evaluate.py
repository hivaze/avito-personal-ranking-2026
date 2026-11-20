"""Evaluate pretrained-encoder scores on val: zero-shot NDCG, write pre_* feature files, blend test with the
current best NN OOF (per-feed z-score blend, paired per-feed delta and SE, weight picked in-fold by user-grouped CV).

Usage: python pretrain/evaluate.py --tag v1 [--keys fe,new,hmax32] [--feats fe,fe_z,new,hmax32] [--write]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path("<repo>")
FLAT, FEATS, PREDS = ROOT / "data/flat", ROOT / "data/feats", ROOT / "preds"


def meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    return f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)


def load_base(name, split="val"):
    """'best' = 0.5 z(nn_final2_val_oof) + 0.5 z(stk2_res4_clean) (current clean best blend; the former
    stk2_res2_r3 was quarantined as leaky)."""
    if name == "best":
        _, M, off = meta(split)
        a = "nn_final2_val_oof" if split == "val" else "nn_final2_test"
        b = "stk2_res4_clean_val" if split == "val" else "stk2_res4_clean_test"
        return (0.5 * feed_rank_z(off, M, np.load(PREDS / f"{a}.npy").astype(np.float32))[:, 1]
                + 0.5 * feed_rank_z(off, M, np.load(PREDS / f"{b}.npy").astype(np.float32))[:, 1])
    return np.load(PREDS / f"{name}.npy")


def clean_mask():
    """Val feeds whose (non-zero) user has NO later feed in val or test (those could carry later-snapshot effects)."""
    v = pl.read_parquet(FLAT / "val/feeds.parquet").select("fidx", "user_id", "timestamp")
    t = pl.read_parquet(FLAT / "test/feeds.parquet").select("user_id", "timestamp")
    allf = pl.concat([v.select("user_id", "timestamp"), t])
    last = allf.filter(pl.col("user_id") != 0).group_by("user_id").agg(pl.col("timestamp").max().alias("tmax"))
    v = v.join(last, on="user_id", how="left").sort("fidx")
    has_later = (v["user_id"] != 0) & (v["tmax"].fill_null(0) > v["timestamp"])
    return ~has_later.to_numpy()


def blend_per_feed(base, x, y, off, M, w):
    zb = feed_rank_z(off, M, base.astype(np.float32))[:, 1]
    zx = feed_rank_z(off, M, x.astype(np.float32))[:, 1]
    zx = zx / max(zx.std(), 1e-9)
    return ndcg_per_feed(zb + w * zx, y, off, M) - ndcg_per_feed(zb, y, off, M)


def blend_eval(base, x, y, off, M, users, grid=(0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)):
    zb = feed_rank_z(off, M, base.astype(np.float32))[:, 1]
    zx = feed_rank_z(off, M, x.astype(np.float32))[:, 1]
    pb = ndcg_per_feed(zb, y, off, M)
    res = {}
    per_w = {}
    for w in grid:
        pw = ndcg_per_feed(zb + w * zx, y, off, M)
        d = pw - pb
        per_w[w] = pw
        res[w] = (float(d.mean()), float(d.std() / np.sqrt(len(d))))
    # weight chosen on the other folds (5 user-grouped folds) -> honest delta
    rng = np.random.default_rng(0)
    uu, inv = np.unique(users, return_inverse=True)
    fold_u = rng.integers(0, 5, len(uu))
    fold_u[uu == 0] = rng.integers(0, 5, (uu == 0).sum())
    fold = fold_u[inv]
    u0 = users == 0
    fold[u0] = rng.integers(0, 5, u0.sum())
    cv = np.zeros(len(M))
    for k in range(5):
        tr = fold != k
        wbest = max(grid, key=lambda w: per_w[w][tr].mean())
        cv[~tr] = per_w[wbest][~tr]
    d = cv - pb
    return float(pb.mean()), res, (float(d.mean()), float(d.std() / np.sqrt(len(d))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--keys", default="fe,fe_noid,new,new_noid,hmax32,hmax8")
    ap.add_argument("--feats", default="fe,fe_z,new,new_z,hmax32,hmax8,fe_noid,id_contrib,idmax32,ibias",
                    help="feature files to write (suffix _z = within-feed z; id_contrib = fe - fe_noid; idmax32 -2 -> NaN)")
    ap.add_argument("--prefix", default="", help="feature-name prefix (default pre_<tag>_)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--base", default="nn_ens_stage1_val_oof")
    args = ap.parse_args()
    f, M, off = meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
    users = f["user_id"].to_numpy()
    base = load_base(args.base)
    clean = clean_mask()
    out = {"tag": args.tag, "base": args.base}
    pb = ndcg_per_feed(base, y, off, M)
    print(f"base {args.base}: {pb.mean():.4f} (clean subset: {clean.sum()} feeds, {pb[clean].mean():.4f})")
    for k in args.keys.split(","):
        p = PREDS / f"pre_{args.tag}_{k}_val.npy"
        if not p.exists():
            continue
        x = np.load(p)
        zs = ndcg_per_feed(x, y, off, M).mean()
        b0, res, cv = blend_eval(base, x, y, off, M, users)
        wb = max(res, key=lambda w: res[w][0])
        fx = {}
        for w in (0.1, 0.2):
            d = blend_per_feed(base, x, y, off, M, w)
            fx[w] = (float(d.mean()), float(d.std() / np.sqrt(len(d))), float(d[clean].mean()), float(d[clean].std() / np.sqrt(clean.sum())))
        pf = ndcg_per_feed(x, y, off, M)
        zc = pf[clean].mean()
        print(f"   leak audit {k}: single-feature NDCG on the {(~clean).sum()} later-user feeds {pf[~clean].mean():.4f} vs others {zc:.4f}")
        print(f"{k:12s} zero-shot {zs:.4f} (clean {zc:.4f}) | blend best w={wb}: {res[wb][0]:+.5f} (se {res[wb][1]:.5f}) | "
              f"CV-w {cv[0]:+.5f} (se {cv[1]:.5f}) | fixed w=0.1 {fx[0.1][0]:+.5f} ({fx[0.1][1]:.5f}) clean {fx[0.1][2]:+.5f} ({fx[0.1][3]:.5f})"
              f" | w=0.2 {fx[0.2][0]:+.5f} ({fx[0.2][1]:.5f}) clean {fx[0.2][2]:+.5f} ({fx[0.2][3]:.5f})")
        out[k] = dict(zero_shot=float(zs), zero_shot_clean=float(zc), blend_best_w=wb, blend_best=res[wb], blend_cv=cv,
                      blend_fixed={str(w): v for w, v in fx.items()}, blend_grid={str(w): v for w, v in res.items()})
    with open(ROOT / "logs/pretrain_evals.jsonl", "a") as fh:
        fh.write(json.dumps(out) + "\n")
    if args.write:
        pref = args.prefix or f"pre_{args.tag}_"
        L = lambda k, split: np.load(PREDS / f"pre_{args.tag}_{k}_{split}.npy").astype(np.float32)
        wrote = []
        for split in ["val", "test"]:
            _, Ms, offs = meta(split)
            for k in args.feats.split(","):
                if k == "id_contrib":
                    x = L("fe", split) - L("fe_noid", split)
                else:
                    src = k[:-2] if k.endswith("_z") else k
                    if not (PREDS / f"pre_{args.tag}_{src}_{split}.npy").exists():
                        continue
                    x = L(src, split)
                    if k.endswith("_z"):
                        x = feed_rank_z(offs, Ms, x)[:, 1]
                    if k == "idmax32":
                        x = np.where(x > -2, x, np.nan)
                np.save(FEATS / split / f"{pref}{k}.npy", x.astype(np.float32))
                if split == "val":
                    wrote.append(pref + k)
        print("wrote", wrote)


if __name__ == "__main__":
    main()
