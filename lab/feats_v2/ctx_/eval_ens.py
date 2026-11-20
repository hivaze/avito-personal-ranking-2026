"""Supplementary evaluation (same protocol as src/eval_family.py, same folds/params/base cache), but also
(a) saves the candidate OOF per seed to feats_v2/ctx_/oof/<tag>_s<seed>.npy and
(b) reports the delta of the SEED-ENSEMBLE (mean of within-feed z-scored OOFs over seeds), which removes most of the
    LightGBM run-to-run noise that dominates single-seed deltas.
Usage: python eval_ens.py --names a,b,c | --family p1,p2  --tag T [--seeds 3] [--threads 16]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "<repo>/src")
from common import *
from eval_family import cv_oof, base_features, CACHE

OOF = Path("<repo>/feats_v2/ctx_/oof")
OUT = Path("<repo>/feats_v2/ctx_/ens_evals.jsonl")


def fz(x, fidx, M):
    x = x.astype(np.float64)
    mu = np.bincount(fidx, weights=x, minlength=len(M)) / M
    d = x - mu[fidx]
    sd = np.sqrt(np.bincount(fidx, weights=d * d, minlength=len(M)) / M)
    return d / (sd[fidx] + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    t0 = time.time()
    OOF.mkdir(exist_ok=True)
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    users = f["user_id"].to_numpy()
    base = base_features()
    fam = []
    if a.family:
        fam = [n for n in all_feature_names("val") if n.startswith(tuple(a.family.split(","))) and n not in base]
    if a.names:
        fam += [n for n in a.names.split(",") if n and n not in base and n not in fam]
    cand = base + fam
    Xc = load_feats("val", cand)
    zb, zc, per_seed = [], [], []
    for s in range(a.seeds):
        cp = CACHE / f"base_oof_s{s}.npy"
        lock = CACHE / f"base_oof_s{s}.lock"
        if not cp.exists():  # same locking protocol as src/eval_family.py
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
                ob = cv_oof(load_feats("val", base), base, y, M, fidx, users, s, a.threads)
                np.save(str(cp) + ".tmp.npy", ob)
                os.replace(str(cp) + ".tmp.npy", cp)
            except FileExistsError:
                while not cp.exists():
                    time.sleep(10)
        ob = np.load(cp)
        p = OOF / f"{a.tag}_s{s}.npy"
        if p.exists():
            oc = np.load(p)
        else:
            oc = cv_oof(Xc, cand, y, M, fidx, users, s, a.threads)
            np.save(p, oc)
        per_seed.append(float(ndcg(oc, y, off, M) - ndcg(ob, y, off, M)))
        zb.append(fz(ob, fidx, M)); zc.append(fz(oc, fidx, M))
        pb = ndcg_per_feed(np.mean(zb, 0), y, off, M); pc = ndcg_per_feed(np.mean(zc, 0), y, off, M)
        d = pc - pb
        print(f"seed {s}: single delta {per_seed[-1]:+.5f} | ens({s+1}) base {pb.mean():.5f} cand {pc.mean():.5f} "
              f"delta {d.mean():+.5f} se {d.std()/np.sqrt(len(d)):.5f}", flush=True)
    res = dict(tag=a.tag, n_family=len(fam), family=fam, ens_base=float(pb.mean()), ens_cand=float(pc.mean()),
               ens_delta=float(d.mean()), ens_se=float(d.std() / np.sqrt(len(d))), per_seed_delta=per_seed,
               mean_single_delta=float(np.mean(per_seed)), seconds=round(time.time() - t0))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
