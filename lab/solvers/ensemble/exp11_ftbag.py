"""More fine-tune diversity at fixed train-only members: the fine-tuned branch of v11 averaged over several fine-tune
variants of the same 6 train-only checkpoints (n2cpf = fta, n2cp = older snapshot recipe, n2cpv2 = W3 hygiene fta).
Mix stays 0.5 z(ft branch) + 0.5 z(train-only branch). Paired vs v11 on val with slices."""
import json, sys
import numpy as np
sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Split, slices, fmt_slices
from exp3_diversity import F5
S = Split("val")
ld = lambda n: S.z(np.load(f"{ROOT}/preds/{n}.npy").astype(np.float64))
to = S.z(np.mean([ld(f"nn_{m}_trainonly_val") for m in F5], 0))
ft = {v: [ld(f"nn_{m}_{v}_val_oof") for m in F5] for v in ["n2cpf", "n2cp", "n2cpv2"]}
base = S.per_feed(0.5 * S.z(np.mean(ft["n2cpf"], 0)) + 0.5 * to)
rep = {}
for name, vs in [("ft=n2cp only", ["n2cp"]), ("ft=n2cpv2 only", ["n2cpv2"]), ("ft=n2cpf+n2cpv2", ["n2cpf", "n2cpv2"]),
                 ("ft=n2cpf+n2cp", ["n2cpf", "n2cp"]), ("ft=all3", ["n2cpf", "n2cp", "n2cpv2"])]:
    f = S.z(np.mean([z for v in vs for z in ft[v]], 0))
    p = S.per_feed(0.5 * f + 0.5 * to)
    sl = slices(S, p - base)
    pf = S.per_feed(f)
    rep[name] = {"mix": float(p.mean()), "ft_alone": float(pf.mean()), "delta": {a: [float(x) for x in b] for a, b in sl.items()}}
    print(f"{name:18s} ft alone {pf.mean():.5f}  mix {p.mean():.5f}  {fmt_slices(sl)}")
# correlation between fine-tune variants of the same member
c = lambda a, b: float((np.bincount(S.fid, a * b, S.n) / S.M).mean())
print("mean per-feed corr, same member: fta vs n2cp", np.mean([c(a, b) for a, b in zip(ft["n2cpf"], ft["n2cp"])]),
      "fta vs v2", np.mean([c(a, b) for a, b in zip(ft["n2cpf"], ft["n2cpv2"])]),
      "fta seed-to-seed", np.mean([c(ft["n2cpf"][i], ft["n2cpf"][j]) for i in range(6) for j in range(i + 1, 6)]))
json.dump(rep, open(f"{ROOT}/solvers/out/ensemble/exp11.json", "w"), indent=1)
