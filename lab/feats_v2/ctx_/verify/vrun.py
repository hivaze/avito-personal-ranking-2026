"""Verifier wrapper: exactly src/eval_family.py's cv_oof/PARAMS/ROUNDS/folds; base OOF from the shared cache when present,
else computed here and stored in the scratchpad. Candidate OOFs saved per seed for later summarizing.
Usage: vrun.py --names a,b,c --tag T --seeds 0,1 [--order alpha|given|shuf<k>] [--placebo]  (placebo: within-feed permuted copies)"""
import argparse, os, sys, time
sys.path.insert(0, "<repo>/src")
from common import *
from eval_family import cv_oof, base_features, CACHE
SCR = Path("/tmp/scratchpad")
ap = argparse.ArgumentParser()
ap.add_argument("--names", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", default="0"); ap.add_argument("--order", default="given")
ap.add_argument("--placebo", action="store_true"); ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
fidx = np.load(FLAT / "val/cand_fidx.npy")
users = f["user_id"].to_numpy()
base = base_features()
fam = [n for n in a.names.split(",") if n]
if a.order == "alpha":
    fam = sorted(fam)
elif a.order.startswith("shuf"):
    rng = np.random.default_rng(int(a.order[4:])); fam = list(rng.permutation(fam))
Xb = load_feats("val", base)
Xf = load_feats("val", fam)
if a.placebo:  # permute each family column within feed (same per-feed value multiset, candidate association broken)
    rng = np.random.default_rng(12345)
    key = fidx.astype(np.float64) + rng.random(len(fidx))
    perm = np.argsort(key, kind="stable")  # sorted by feed then random -> a within-feed permutation of row indices
    Xf = Xf[perm]
Xc = np.concatenate([Xb, Xf], axis=1); cand = base + fam
for s in [int(x) for x in a.seeds.split(",")]:
    t = time.time()
    cp = CACHE / f"base_oof_s{s}.npy"; sp = SCR / "oof" / f"base_s{s}.npy"
    if cp.exists():
        ob = np.load(cp)
    elif sp.exists():
        ob = np.load(sp)
    else:
        ob = cv_oof(Xb, base, y, M, fidx, users, s, a.threads); np.save(sp, ob)
    p = SCR / "oof" / f"{a.tag}_s{s}.npy"
    oc = np.load(p) if p.exists() else cv_oof(Xc, cand, y, M, fidx, users, s, a.threads)
    np.save(p, oc)
    print(a.tag, "seed", s, "base", round(ndcg(ob, y, off, M), 5), "cand", round(ndcg(oc, y, off, M), 5),
          "delta", round(ndcg(oc, y, off, M) - ndcg(ob, y, off, M), 5), f"{time.time()-t:.0f}s", flush=True)
