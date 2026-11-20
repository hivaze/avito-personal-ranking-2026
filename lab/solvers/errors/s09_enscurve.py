"""Ensemble-size curve of the nn_final5 members (train-only, fta fine-tuned, and the 0.5/0.5 mix), fit a - b/k, extrapolate.
Plus: value of adding one cross-recipe member (v13 hedges, v15 cooldown) to the 6-member mix vs adding 'one more same-recipe seed' (curve)."""
import itertools
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz
import numpy as np

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
mem = ["v12_all_e10", "v12_all_s81", "v12_all_s82", "v12_allprew_s83", "v12_allprew_s84", "v12_allprew_s85"]
TO = [fz(np.load(f"{ROOT}/preds/nn_{m}_trainonly_val.npy"), fid, M) for m in mem]
FT = [fz(np.load(f"{ROOT}/preds/nn_{m}_n2cpf_val_oof.npy"), fid, M) for m in mem]
res = {}
for k in range(1, 7):
    vals = {"TO": [], "FT": [], "mix": []}
    for sub in itertools.combinations(range(6), k):
        zt = np.mean([TO[i] for i in sub], 0)
        zf = np.mean([FT[i] for i in sub], 0)
        vals["TO"].append(ndcg_per_feed(zt, y, off, M).mean())
        vals["FT"].append(ndcg_per_feed(zf, y, off, M).mean())
        vals["mix"].append(ndcg_per_feed(0.5 * fz(zt, fid, M) + 0.5 * fz(zf, fid, M), y, off, M).mean())
    res[k] = {t: (np.mean(v), np.std(v)) for t, v in vals.items()}
    print(f"k={k}: " + "  ".join(f"{t} {m:.5f} (sd {s:.5f}, n {len(vals[t])})" for t, (m, s) in res[k].items()), flush=True)
ks = np.arange(1, 7)
for t in ("TO", "FT", "mix"):
    yy = np.array([res[k][t][0] for k in ks])
    A = np.stack([np.ones(6), 1 / ks], 1)
    a, b = np.linalg.lstsq(A, yy, rcond=None)[0]
    print(f"  fit {t}: a(inf) {a:.5f}  b {b:.5f}  -> k=12 {a + b / 12:.5f}  k=18 {a + b / 18:.5f}  k=24 {a + b / 24:.5f}  (k=6 fit {a + b / 6:.5f})")
# cross-recipe additions to the 6-member mix
zt6, zf6 = np.mean(TO, 0), np.mean(FT, 0)
base = ndcg_per_feed(0.5 * fz(zt6, fid, M) + 0.5 * fz(zf6, fid, M), y, off, M)
print(f"6-member mix {base.mean():.5f}")
extra = ["v13_lr1_mix0", "v13_lr1_mix1", "v13_lr2_mix1", "v15_cd_all_e10", "v15_cd_allprew_s83"]
for e in extra:
    try:
        et = fz(np.load(f"{ROOT}/preds/nn_{e}_trainonly_val.npy"), fid, M)
        ef = fz(np.load(f"{ROOT}/preds/nn_{e}_n2cpf_val_oof.npy"), fid, M)
    except FileNotFoundError as ex:
        print("missing", e, ex)
        continue
    zt = (6 * zt6 + et) / 7
    zf = (6 * zf6 + ef) / 7
    nd = ndcg_per_feed(0.5 * fz(zt, fid, M) + 0.5 * fz(zf, fid, M), y, off, M)
    d = nd - base
    print(f"  +{e}: {nd.mean():.5f}  d {d.mean():+.5f}+-{d.std() / np.sqrt(n):.5f}")
