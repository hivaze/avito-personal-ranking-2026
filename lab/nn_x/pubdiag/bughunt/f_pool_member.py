"""(e) pool arrays rebuilt from member files == saved preds/ens_<TAG>_*? (f) per-member val / holdout NDCG of the r6
families vs F7 (paired per-feed deltas, SE), and corr(z(ft), z(to)) on val vs test per member (does the fine-tune move
the test predictions by the same amount as the val predictions?)."""
import sys

import numpy as np

sys.path.insert(0, "<repo>/src")
from common import feed_meta  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402

R = "<repo>"
P = f"{R}/preds"
roots = range(101, 108)
FAM = {"F7": "v17_pre2s_s{r}", "F7x": "r6_cd_m02_r{r}", "F7x70": "r6_cd_m02_p70k_r{r}", "F7b": "r6_cd_c0_r{r}"}
F5cd = "v15_cd_all_e10 v15_cd_all_s81 v15_cd_all_s82 v15_cd_allprew_s83 v15_cd_allprew_s84 v15_cd_allprew_s85".split()
F6 = ["v13_lr1_mix1"] + [f"v14_s{s}" for s in range(92, 97)]
F7 = [f"v17_pre2s_s{r}" for r in roots]
F7x = [f"r6_cd_m02_r{r}" for r in roots]
F7x70 = [f"r6_cd_m02_p70k_r{r}" for r in roots]
F10 = [f"v23_gru_pre2s_s{s}" for s in range(501, 507)]
POOLS = {"Cd": F5cd + F6 + F7, "M2": F7x + F7x70, "G3": F10 + F7x + F7x70}
meta = {sp: feed_meta(sp) for sp in ("val", "test")}
yv = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)

print("== (e) pool arrays rebuilt from member files vs saved preds/ens_<TAG>_{ft,to}_{val,test}.npy")
for tag, ms in POOLS.items():
    for sp, ftn, ton in [("val", "n2cpv3_val_oof", "trainonly_val"), ("test", "n2cpv3_test", "trainonly_test")]:
        _, M, off = meta[sp]
        z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]
        for kind, fn in (("ft", ftn), ("to", ton)):
            reb = np.mean([z(np.load(f"{P}/nn_{m}_{fn}.npy")) for m in ms], 0).astype(np.float32)
            sav = np.load(f"{P}/ens_{tag}_{kind}_{sp}.npy")
            print(f"  {tag:3s} {kind} {sp:4s} n_members {len(ms)} max|rebuilt - saved| {np.abs(reb - sav).max():.2e}")

print("\n== (f) per-member NDCG vs the same-root F7 member: mean over 7 roots of the paired per-feed delta "
      "(SE = std of the root-averaged per-feed delta / sqrt(n))")
hm = np.load(f"{P}/holdout/holdout_meta.npy")
Mh, offh = hm[0].astype(np.int64), hm[1].astype(np.int64)
yh = np.load(f"{P}/holdout/holdout_y.npy").astype(np.float64)
nd = {}
for fam, pat in FAM.items():
    for r in roots:
        v = pat.format(r=r)
        _, M, off = meta["val"]
        nd[(fam, r, "val_to")] = ndcg_per_feed(np.load(f"{P}/nn_{v}_trainonly_val.npy"), yv, off, M)
        nd[(fam, r, "val_ft")] = ndcg_per_feed(np.load(f"{P}/nn_{v}_n2cpv3_val_oof.npy"), yv, off, M)
        nd[(fam, r, "hold_to")] = ndcg_per_feed(np.load(f"{P}/holdout/nn_{v}_holdout.npy"), yh, offh, Mh)
for kind in ("val_to", "val_ft", "hold_to"):
    base = np.mean([nd[("F7", r, kind)] for r in roots], 0)
    line = f"  {kind:8s} F7 {base.mean():.5f}"
    for fam in ("F7b", "F7x", "F7x70"):
        d = np.mean([nd[(fam, r, kind)] - nd[("F7", r, kind)] for r in roots], 0)
        line += f" | {fam} {d.mean():+.5f} (SE {d.std() / np.sqrt(len(d)):.5f})"
    print(line)
    for fam in ("F7b", "F7x", "F7x70"):
        per = [float((nd[(fam, r, kind)] - nd[("F7", r, kind)]).mean()) for r in roots]
        print(f"     {fam:6s} per root {np.round(per, 5).tolist()}")

print("\n== corr(z(ft), z(to)) per member (mean of per-feed Pearson): val vs test")
for fam, pat in FAM.items():
    out = []
    for sp, ftn, ton in [("val", "n2cpv3_val_oof", "trainonly_val"), ("test", "n2cpv3_test", "trainonly_test")]:
        _, M, off = meta[sp]
        cs = []
        for r in roots:
            v = pat.format(r=r)
            a = feed_rank_z(off, M, np.load(f"{P}/nn_{v}_{ftn}.npy").astype(np.float32))[:, 1]
            b = feed_rank_z(off, M, np.load(f"{P}/nn_{v}_{ton}.npy").astype(np.float32))[:, 1]
            cs.append((np.add.reduceat((a * b).astype(np.float64), off) / M).mean())
        out.append(f"{sp} {np.mean(cs):.5f} (roots {np.round(cs, 4).tolist()})")
    print(f"  {fam:6s} " + " | ".join(out))
